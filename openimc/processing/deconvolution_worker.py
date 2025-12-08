# SPDX-License-Identifier: GPL-3.0-or-later
#
# OpenIMC – Interactive analysis toolkit for IMC data
#
# Copyright (C) 2025 University of Southern California
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Worker functions for high resolution IMC deconvolution using Richardson-Lucy deconvolution.
"""

import os
from typing import Tuple
import numpy as np
import tifffile

try:
    from skimage.restoration import richardson_lucy
    from skimage.util import img_as_uint
    _HAVE_SCIKIT_IMAGE = True
except ImportError:
    _HAVE_SCIKIT_IMAGE = False


def RLD_HRIMC_circle(
    image_stack: np.ndarray,
    x0: float = 7.0,
    iterations: int = 4,
    output_format: str = "float",
    passes: np.ndarray = None,
    contributions: np.ndarray = None,
    kernel: np.ndarray = None,
    passes_arr: np.ndarray = None,
    contribs_arr: np.ndarray = None,
    kernel_dim: int = None,
    region_data_full: list = None,
    I0: float = None,
) -> np.ndarray:
    """
    Apply Richardson-Lucy deconvolution to high resolution IMC image stack.
    
    There are several modes of operation for the PSF kernel:
    1. New region-based kernel generation (recommended):
       - Provide `passes_arr`, `contribs_arr`, `kernel_dim`, `region_data_full`.
       - Kernel is generated per channel using I0 from each channel's max intensity.
       - Uses `x0` and channel-specific `I0` for sigmoidal loss function.
    2. Direct kernel override:
       - Provide a kernel via `kernel` (can be any size, not just 3×3).
       - `passes` / `contributions` / `x0` are ignored for PSF construction.
    3. Custom Passes/Contributions path (backward compatibility):
       - Provide 1D arrays `passes` and `contributions` of equal shape.
       - Kernel is built from Passes_scaled = logistic(passes, x0) * contributions
         using the same index grouping as the original HR-IMC code.
    4. Legacy hard-coded HR-IMC kernel (default):
       - If none of the above are provided, uses hard-coded arrays.
    
    Args:
        image_stack: Image stack with shape (C, H, W) or (H, W, C)
        x0: Parameter for kernel calculation (default: 7.0)
        iterations: Number of Richardson-Lucy iterations (default: 4)
        output_format: Output format, either 'float' or 'uint16' (default: 'float')
        passes: Optional array of pass counts (legacy format)
        contributions: Optional array of contributions (legacy format)
        kernel: Optional direct PSF kernel (any size). If provided, overrides other methods.
        passes_arr: Optional new format passes array per region
        contribs_arr: Optional new format contributions array per region
        kernel_dim: Optional kernel dimension (for new format)
        region_data_full: Optional list of region data dicts (for new format)
        I0: Optional I0 value for sigmoidal loss. If None, uses max intensity per channel.
    
    Returns:
        Deconvolved image stack with same shape as input (except for border cropping)
    """
    if not _HAVE_SCIKIT_IMAGE:
        raise RuntimeError("scikit-image is required for deconvolution. Install with: pip install scikit-image")
    
    # Validate input
    if image_stack.size == 0:
        raise ValueError("Input image stack is empty")
    
    # Ensure input is (C, H, W) format
    # Detection heuristic: For IMC data, channels are typically < 100, while H and W are typically > 1000
    # If first dimension is smallest and < 100, it's likely (C, H, W) - keep as is
    # If last dimension is smallest and < 100, it's likely (H, W, C) - transpose
    if image_stack.ndim == 3:
        dim0, dim1, dim2 = image_stack.shape
        
        # Check if last dimension is smallest and looks like channels (< 100)
        # This would indicate (H, W, C) format
        if dim2 < 100 and dim2 < dim0 and dim2 < dim1:
            # Likely (H, W, C) format, convert to (C, H, W)
            print(f"Detected (H, W, C) format, transposing from {image_stack.shape} to (C, H, W)")
            image_stack = np.transpose(image_stack, (2, 0, 1))
            dim0, dim1, dim2 = image_stack.shape  # Update after transpose
        # Otherwise assume (C, H, W) format (most common case, especially when called from deconvolve_acquisition)
        
        # Now we have (C, H, W)
        n_channels, height, width = dim0, dim1, dim2
        
        # Validate dimensions
        if height < 1 or width < 1:
            raise ValueError(f"Invalid image dimensions: {height}x{width}")
        if n_channels < 1:
            raise ValueError(f"Invalid number of channels: {n_channels}")
    elif image_stack.ndim == 2:
        # Single channel, add channel dimension
        height, width = image_stack.shape
        if height < 1 or width < 1:
            raise ValueError(f"Invalid image dimensions: {height}x{width}")
        image_stack = image_stack[np.newaxis, :, :]
        n_channels = 1
    else:
        raise ValueError(f"Unsupported image dimensionality: {image_stack.ndim}D")
    
    # ------------------------------------------------------------------
    # Determine PSF construction mode
    # ------------------------------------------------------------------
    use_new_region_format = False
    use_direct_kernel = False
    use_legacy_format = False
    
    if kernel is not None:
        # Direct kernel override (any size)
        kernel = np.asarray(kernel, dtype=np.float32)
        if kernel.ndim != 2:
            raise ValueError(f"Provided kernel must be 2D, got {kernel.ndim}D with shape {kernel.shape}")
        if kernel.sum() <= 0:
            raise ValueError("Provided kernel has non-positive sum")
        kernel = kernel / kernel.sum()
        use_direct_kernel = True
    elif passes_arr is not None and contribs_arr is not None and kernel_dim is not None and region_data_full is not None:
        # New region-based format
        use_new_region_format = True
        from openimc.processing.hrimc_kernel_generator import compute_region_kernel
    elif passes is not None and contributions is not None:
        # Legacy custom passes+contributions format
        use_legacy_format = True
        passes = np.asarray(passes, dtype=float)
        contributions = np.asarray(contributions, dtype=float)
        if passes.shape != contributions.shape:
            raise ValueError(
                f"Passes and contributions arrays must have the same shape. "
                f"Got passes.shape={passes.shape}, contributions.shape={contributions.shape}"
            )
        Contributions = contributions / contributions.sum()
        Passes = passes
    else:
        # Legacy hard-coded arrays from original HR-IMC code
        use_legacy_format = True
        Passes = np.array([
            7,6,5,8,7,
            7,8,7,6,6,
            7,9,8,7,8,
            8,7,7,6,6,
            7,6,6,5,5,
            6,6,5,5,4,
            4,6,4,5,3,
            4,5,6,6,5,
            4,5,5,4,3,
            4,4,3,6,5,
            4,5,5,4,4,
            3,3,4,4,3,
            3,2,4,3,2,
            2,1,1,3,2,
            2,1,1,3,3,
            2,2,2,2,1,
            1,1,1,3,2,
            1,2,2,1,1,
            2,1,1,4,3,
            2,1
        ], dtype=float)
        Contributions = np.array([
            0.02,0.00108,0.00108,0.0034,0.0196,
            0.0196,0.0034,0.0034,0.0196,0.0196,
            0.0034,0.00223,0.00223,0.00223,0.0034,
            0.0034,0.0034,0.0034,0.0034,0.0034,
            0.0196,0.00106,0.0196,0.00106,0.0196,
            0.00108,0.00106,0.00106,0.00106,0.00106,
            0.00108,0.0196,0.00106,0.0196,0.00106,
            0.0196,0.0196,0.0034,0.0034,0.0196,
            0.0196,0.0034,0.0034,0.0196,0.0196,
            0.0034,0.0034,0.0196,0.00223,0.00223,
            0.00223,0.0034,0.0034,0.0034,0.0034,
            0.0034,0.0034,0.0196,0.00108,0.0196,
            0.00106,0.0196,0.00108,0.00106,0.00106,
            0.00106,0.00106,0.00108,0.0196,0.00106,
            0.0196,0.00106,0.0196,0.0034,0.0034,
            0.0196,0.0196,0.0034,0.0034,0.0196,
            0.0196,0.0034,0.0034,0.00223,0.00223,
            0.00223,0.0034,0.0034,0.0034,0.0034,
            0.00108,0.00196,0.00108,0.00219,0.00219,
            0.00219,0.00219
        ], dtype=float)
        Contributions = Contributions / Contributions.sum()
    
    # Process each channel
    processed_layers = []
    
    print(f"Processing {n_channels} channels, input shape: {image_stack.shape}")
    
    for layer_idx in range(n_channels):
        layer_data = image_stack[layer_idx, :, :]
        
        # Validate layer data
        if layer_data.size == 0:
            raise ValueError(f"Layer {layer_idx} is empty")
        
        if layer_idx == 0:
            print(f"Layer {layer_idx} shape: {layer_data.shape}, expected (H={height}, W={width})")
        
        # Check for NaN or Inf values
        if np.any(np.isnan(layer_data)) or np.any(np.isinf(layer_data)):
            print(f"Warning: Layer {layer_idx} contains NaN or Inf values, replacing with 0")
            layer_data = np.nan_to_num(layer_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Build or reuse kernel for this channel
        if use_direct_kernel:
            channel_kernel = kernel
        elif use_new_region_format:
            # New region-based kernel generation with I0 per channel
            channel_I0 = float(np.max(layer_data)) if I0 is None else I0
            if channel_I0 <= 0 or np.isnan(channel_I0) or np.isinf(channel_I0):
                print(f"Warning: Layer {layer_idx} has invalid I0={channel_I0}, using identity kernel")
                # Create identity kernel with correct dimensions
                channel_kernel = np.zeros((kernel_dim, kernel_dim), dtype=np.float32)
                center = kernel_dim // 2
                channel_kernel[center, center] = 1.0
            else:
                # Generate kernel using new compute_region_kernel function
                channel_kernel, _ = compute_region_kernel(
                    passes_arr, contribs_arr, kernel_dim, region_data_full,
                    x0=x0, I0=channel_I0
                )
                if channel_kernel.sum() <= 0 or np.isnan(channel_kernel.sum()) or np.isinf(channel_kernel.sum()):
                    print(f"Warning: Layer {layer_idx} has invalid kernel sum, using identity kernel")
                    channel_kernel = np.zeros((kernel_dim, kernel_dim), dtype=np.float32)
                    center = kernel_dim // 2
                    channel_kernel[center, center] = 1.0
        else:
            # Legacy format (passes + contributions or hard-coded)
            channel_I0 = float(np.max(layer_data))
            if channel_I0 <= 0 or np.isnan(channel_I0) or np.isinf(channel_I0):
                print(f"Warning: Layer {layer_idx} has invalid I0={channel_I0}, using identity kernel")
                channel_kernel = np.array([[0.0, 0.0, 0.0],
                                           [0.0, 1.0, 0.0],
                                           [0.0, 0.0, 0.0]], dtype=np.float32)
            else:
                Passes_scaled = channel_I0 - channel_I0 / (1 + np.exp(-(Passes - x0)))
                y_array = Passes_scaled * Contributions
                total_sum = np.sum(y_array)
                if total_sum <= 0 or np.isnan(total_sum) or np.isinf(total_sum):
                    print(f"Warning: Layer {layer_idx} has invalid kernel sum={total_sum}, using identity kernel")
                    channel_kernel = np.array([[0.0, 0.0, 0.0],
                                               [0.0, 1.0, 0.0],
                                               [0.0, 0.0, 0.0]], dtype=np.float32)
                else:
                    # Original 3×3 aggregation mapping, preserving backward compatibility
                    result = list((
                        (y_array[3] + y_array[4] + y_array[11] + y_array[14] + y_array[15] + y_array[20]) / total_sum, 
                        (y_array[0] + y_array[5] + y_array[6] + y_array[7] + y_array[8] + y_array[12] + y_array[16] + y_array[17] + y_array[22]) / total_sum, 
                        (y_array[9] + y_array[10] + y_array[13] + y_array[18] + y_array[19] + y_array[24]) / total_sum,
                        (y_array[31] + y_array[36] + y_array[37] + y_array[38] + y_array[39] + y_array[48] + y_array[51] + y_array[52] + y_array[57]) / total_sum, 
                        (y_array[33] + y_array[40] + y_array[41] + y_array[42] + y_array[43] + y_array[49] + y_array[53] + y_array[54] + y_array[59]) / total_sum, 
                        (y_array[35] + y_array[44] + y_array[45] + y_array[46] + y_array[47] + y_array[50] + y_array[55] + y_array[56] + y_array[61]) / total_sum, 
                        (y_array[68] + y_array[73] + y_array[74] + y_array[75] + y_array[83] + y_array[86]) / total_sum, 
                        (y_array[70] + y_array[76] + y_array[77] + y_array[78] + y_array[79] + y_array[84] + y_array[87] + y_array[88] + y_array[91]) / total_sum, 
                        (y_array[72] + y_array[80] + y_array[81] + y_array[82] + y_array[85] + y_array[89]) / total_sum, 
                    ))
                    channel_kernel = np.array(result, dtype=np.float32)
                    channel_kernel = channel_kernel / channel_kernel.sum()
                    channel_kernel = channel_kernel.reshape(3, 3)
        
        # Normalize layer for RL
        layer_min = layer_data.min()
        layer_max = layer_data.max()
        if layer_max > layer_min:
            layer_norm = (layer_data - layer_min) / (layer_max - layer_min)
        else:
            layer_norm = layer_data.astype(np.float32)
        layer_norm = np.clip(layer_norm, 1e-4, None)
        
        # RL deconvolution
        deconvolved_image = richardson_lucy(layer_norm, channel_kernel, num_iter=iterations)
        
        # Denormalize back
        if layer_max > layer_min:
            deconvolved_image = deconvolved_image * (layer_max - layer_min) + layer_min
        
        # Verify deconvolved image shape
        if deconvolved_image.shape != (height, width):
            print(f"Warning: Layer {layer_idx} deconvolved shape {deconvolved_image.shape} != expected ({height}, {width})")
        
        processed_layers.append(deconvolved_image)
        
        if layer_idx == 0:
            print(f"Layer {layer_idx} after deconvolution: shape={deconvolved_image.shape}")
    
    # Stack all processed layers
    if len(processed_layers) != n_channels:
        raise ValueError(f"Processed layers count mismatch: expected {n_channels}, got {len(processed_layers)}")
    
    # Verify all layers have the same shape
    if processed_layers:
        expected_layer_shape = processed_layers[0].shape
        for i, layer in enumerate(processed_layers):
            if layer.shape != expected_layer_shape:
                raise ValueError(f"Layer {i} has shape {layer.shape}, expected {expected_layer_shape}")
    
    processed_stack = np.stack(processed_layers, axis=0)  # (C, H, W)
    
    # Debug: verify stack shape
    print(f"After stacking: shape={processed_stack.shape}, expected (C={n_channels}, H={height}, W={width})")
    
    # Verify the stack shape is correct
    if processed_stack.shape[0] != n_channels:
        raise ValueError(f"Stack channel dimension mismatch: expected {n_channels}, got {processed_stack.shape[0]}")
    if processed_stack.shape[1] != height or processed_stack.shape[2] != width:
        print(f"Warning: Stack spatial dimensions mismatch: expected (H={height}, W={width}), got (H={processed_stack.shape[1]}, W={processed_stack.shape[2]})")
    
    # Remove a 2 pixel (1um) border to account for border effect
    # Check if image is large enough for cropping
    n_channels_out, height_out, width_out = processed_stack.shape
    if n_channels_out != n_channels:
        raise ValueError(f"Channel count mismatch: expected {n_channels}, got {n_channels_out}")
    
    if height_out > 6 and width_out > 6:
        processed_stack_cropped = processed_stack[:, 3:-3, 3:-3]
        expected_cropped_h = height_out - 6
        expected_cropped_w = width_out - 6
        print(f"After cropping (3px): shape={processed_stack_cropped.shape}, expected (C={n_channels}, H={expected_cropped_h}, W={expected_cropped_w})")
    else:
        # Image too small, crop less or don't crop
        if height_out > 4 and width_out > 4:
            # Crop 1 pixel instead of 3
            processed_stack_cropped = processed_stack[:, 1:-1, 1:-1]
            expected_cropped_h = height_out - 2
            expected_cropped_w = width_out - 2
            print(f"After cropping (1px): shape={processed_stack_cropped.shape}, expected (C={n_channels}, H={expected_cropped_h}, W={expected_cropped_w})")
        else:
            # Don't crop if too small
            processed_stack_cropped = processed_stack
            print(f"No cropping applied: shape={processed_stack_cropped.shape}")
    
    # Final verification before return
    final_c, final_h, final_w = processed_stack_cropped.shape
    print(f"Final output shape: (C={final_c}, H={final_h}, W={final_w})")
    if final_c != n_channels:
        raise ValueError(f"Final channel count mismatch: expected {n_channels}, got {final_c}")
    
    # Convert format if needed
    if output_format == "uint16":
        # Convert to uint16, scaling to full uint16 range
        # Find min and max across all channels
        stack_min = float(np.min(processed_stack_cropped))
        stack_max = float(np.max(processed_stack_cropped))
        
        if stack_max > stack_min:
            # Scale to [0, 65535] range for uint16
            # First normalize to [0, 1]
            normalized = (processed_stack_cropped - stack_min) / (stack_max - stack_min)
            # Then scale to uint16 range
            processed_stack_cropped = (normalized * 65535.0).astype(np.uint16)
        elif stack_max == stack_min and stack_max >= 0:
            # All values are the same and non-negative
            # Scale to uint16 range if value is in [0, 1], otherwise clip
            if stack_max <= 1.0:
                processed_stack_cropped = (processed_stack_cropped * 65535.0).astype(np.uint16)
            else:
                # Clip to uint16 max
                processed_stack_cropped = np.clip(processed_stack_cropped, 0, 65535).astype(np.uint16)
        else:
            # All values are the same and possibly negative
            # Set to zero
            processed_stack_cropped = np.zeros_like(processed_stack_cropped, dtype=np.uint16)
    else:
        # Keep as float32
        processed_stack_cropped = processed_stack_cropped.astype(np.float32)
    
    return processed_stack_cropped


def deconvolve_acquisition_from_mcd(
    mcd_path: str,
    acq_id: str,
    output_dir: str,
    x0: float = 7.0,
    iterations: int = 4,
    output_format: str = "float",
    channel_names: list = None,
    source_file_path: str = None,
    unique_acq_id: str = None,
    well_name: str = None,
    pixel_size_x: float = None,
    pixel_size_y: float = None,
    pixel_size_unit: str = "µm",
    passes: np.ndarray = None,
    contributions: np.ndarray = None,
    kernel: np.ndarray = None,
    passes_arr: np.ndarray = None,
    contribs_arr: np.ndarray = None,
    kernel_dim: int = None,
    region_data_full: list = None,
    I0: float = None
) -> str:
    """
    Deconvolve a single acquisition from an MCD file and save as OME-TIFF.
    
    Args:
        mcd_path: Path to the MCD file
        acq_id: Acquisition ID
        output_dir: Output directory for OME-TIFF files
        x0: Parameter for kernel calculation
        iterations: Number of Richardson-Lucy iterations
        output_format: Output format, either 'float' or 'uint16'
        channel_names: List of channel names for OME metadata
        source_file_path: Optional source file path for filename generation
        unique_acq_id: Optional unique acquisition ID for filename generation
        well_name: Optional well name to use in output filename (if None, will try to get from loader)
    
    Returns:
        Path to the saved OME-TIFF file
    """
    from openimc.data.mcd_loader import MCDLoader
    
    # Load the acquisition
    loader = MCDLoader()
    loader.open(mcd_path)
    
    try:
        # Get channel names from loader if not provided
        if channel_names is None:
            channel_names = loader.get_channels(acq_id) if hasattr(loader, 'get_channels') else None
        
        # Get all channels for this acquisition
        img_stack = loader.get_all_channels(acq_id)  # Returns (H, W, C)
        
        # Verify image stack is valid
        if img_stack.size == 0:
            raise ValueError(f"Image stack is empty for acquisition {acq_id}")
        
        if img_stack.ndim != 3:
            raise ValueError(f"Expected 3D array (H, W, C), got {img_stack.ndim}D array with shape {img_stack.shape}")
        
        # Check image dimensions
        height, width, n_channels = img_stack.shape
        if height < 1 or width < 1:
            raise ValueError(f"Invalid image dimensions: {height}x{width}")
        if n_channels < 1:
            raise ValueError(f"No channels found in acquisition {acq_id}")
        
        # If channel names not available, create default names
        if channel_names is None or len(channel_names) != n_channels:
            channel_names = [f"Channel_{i+1}" for i in range(n_channels)]
        
        # Extract pixel size from loader metadata if not provided
        if pixel_size_x is None or pixel_size_y is None:
            metadata = loader._acq_metadata.get(acq_id, {}) if hasattr(loader, '_acq_metadata') else {}
            if metadata:
                for key, value in metadata.items():
                    key_lower = key.lower()
                    if 'pixel' in key_lower and 'size' in key_lower:
                        if 'x' in key_lower or 'width' in key_lower:
                            try:
                                pixel_size_x = float(value)
                            except (ValueError, TypeError):
                                pass
                        elif 'y' in key_lower or 'height' in key_lower:
                            try:
                                pixel_size_y = float(value)
                            except (ValueError, TypeError):
                                pass
                    elif 'resolution' in key_lower:
                        try:
                            pixel_size_x = pixel_size_y = float(value)
                        except (ValueError, TypeError):
                            pass
                    elif key == 'PhysicalSizeX':
                        try:
                            pixel_size_x = float(value)
                        except (ValueError, TypeError):
                            pass
                    elif key == 'PhysicalSizeY':
                        try:
                            pixel_size_y = float(value)
                        except (ValueError, TypeError):
                            pass
                    elif key == 'PhysicalSizeXUnit' or key == 'PhysicalSizeYUnit':
                        pixel_size_unit = str(value)
        
        # If only one dimension found, use it for both
        if pixel_size_x is not None and pixel_size_y is None:
            pixel_size_y = pixel_size_x
        elif pixel_size_y is not None and pixel_size_x is None:
            pixel_size_x = pixel_size_y
        
        # Convert to (C, H, W) for processing
        img_stack = np.transpose(img_stack, (2, 0, 1))
        
        # Apply deconvolution
        print(f"Deconvolving acquisition {acq_id}: shape={img_stack.shape}, x0={x0}, iterations={iterations}, format={output_format}")
        deconvolved_stack = RLD_HRIMC_circle(
            img_stack,
            x0=x0,
            iterations=iterations,
            output_format=output_format,
            passes=passes,
            contributions=contributions,
            kernel=kernel,
            passes_arr=passes_arr,
            contribs_arr=contribs_arr,
            kernel_dim=kernel_dim,
            region_data_full=region_data_full,
            I0=I0
        )  # Returns (C, H, W)
        print(f"Deconvolution complete: output shape={deconvolved_stack.shape}, dtype={deconvolved_stack.dtype}")
        
        # Generate filename: source_file_well_name.ome.tif (or source_file_acquisition_id.ome.tif if no well)
        # Get source file basename (without extension)
        if source_file_path:
            source_basename = os.path.splitext(os.path.basename(source_file_path))[0]
        else:
            source_basename = os.path.splitext(os.path.basename(mcd_path))[0]
        
        # Get well name if available, otherwise use acquisition ID
        # Use provided well_name parameter first, then try loader, otherwise None
        if well_name is None:
            well_name = loader._acq_well.get(acq_id) if hasattr(loader, '_acq_well') else None
        
        # Sanitize source filename
        safe_source = "".join(c for c in source_basename if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_source = safe_source.replace(' ', '_')
        
        # Use well name if available, otherwise use acquisition ID
        if well_name:
            label_for_filename = well_name
        else:
            # Get acquisition ID for filename
            # Use unique_acq_id if provided (for multiple files), otherwise use acq_id
            if unique_acq_id:
                # Extract just the acquisition part from unique ID (remove file identifier)
                # Unique ID format is: original_id__file_hash
                if '__' in unique_acq_id:
                    acq_id_part = unique_acq_id.split('__')[0]
                else:
                    acq_id_part = unique_acq_id
            else:
                acq_id_part = acq_id
            label_for_filename = acq_id_part
        
        # Sanitize label
        safe_label = "".join(c for c in label_for_filename if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_label = safe_label.replace(' ', '_')
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create output filename: source_file_label.ome.tif
        output_filename = f"{safe_source}_{safe_label}.ome.tif"
        output_path = os.path.join(output_dir, output_filename)
        
        # Check if deconvolved stack is valid
        if deconvolved_stack.size == 0:
            raise ValueError(f"Deconvolved stack is empty for acquisition {acq_id}")
        
        # Verify deconvolved stack shape (should be C, H, W)
        print(f"Before saving: shape={deconvolved_stack.shape}, expected (C, H, W)")
        if deconvolved_stack.ndim != 3:
            raise ValueError(f"Expected 3D array (C, H, W), got {deconvolved_stack.ndim}D array with shape {deconvolved_stack.shape}")
        
        # Verify channel count matches
        expected_channels = len(channel_names) if channel_names else img_stack.shape[2]
        if deconvolved_stack.shape[0] != expected_channels:
            raise ValueError(f"Channel count mismatch: expected {expected_channels}, got {deconvolved_stack.shape[0]}")
        
        # Save as OME-TIFF in CHW format (matches GUI export format)
        # tifffile.imwrite with ome=True can handle (C, H, W) format
        # The GUI export uses this format, so we'll match it for consistency
        metadata = {}
        if channel_names:
            metadata['Channel'] = {'Name': channel_names}
        
        # Add pixel size information to metadata
        if pixel_size_x is not None:
            metadata['PhysicalSizeX'] = pixel_size_x
            metadata['PhysicalSizeXUnit'] = pixel_size_unit
        if pixel_size_y is not None:
            metadata['PhysicalSizeY'] = pixel_size_y
            metadata['PhysicalSizeYUnit'] = pixel_size_unit
        
        # Save as OME-TIFF in CHW format (same as GUI export)
        try:
            tifffile.imwrite(
                output_path,
                deconvolved_stack,  # Already in (C, H, W) format
                photometric='minisblack',
                metadata=metadata,
                ome=True
            )
            
            # Verify the file was written correctly
            if not os.path.exists(output_path):
                raise IOError(f"Output file was not created: {output_path}")
            
            # Check file size
            file_size = os.path.getsize(output_path)
            if file_size == 0:
                raise IOError(f"Output file is empty: {output_path}")
            
            # Try to read it back to verify
            with tifffile.TiffFile(output_path) as tif:
                if not tif.series:
                    raise IOError(f"TIFF file contains no image series: {output_path}")
                read_shape = tif.series[0].shape
                # tifffile may return shape in different order, so we check if dimensions match
                if set(read_shape) != set(deconvolved_stack.shape):
                    print(f"Warning: Written shape {deconvolved_stack.shape} != read shape {read_shape}")
                else:
                    print(f"File verified: written shape {deconvolved_stack.shape}, read shape {read_shape}")
            
        except Exception as e:
            # Clean up partial file if it exists
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass
            raise IOError(f"Failed to write OME-TIFF file {output_path}: {str(e)}") from e
        
        return output_path
        
    finally:
        loader.close()


def deconvolve_acquisition_from_ometiff(
    tiff_path: str,
    acq_id: str,
    output_dir: str,
    x0: float = 7.0,
    iterations: int = 4,
    output_format: str = "float",
    channel_names: list = None,
    source_file_path: str = None,
    unique_acq_id: str = None,
    channel_format: str = 'CHW',
    pixel_size_x: float = None,
    pixel_size_y: float = None,
    pixel_size_unit: str = "µm",
    passes: np.ndarray = None,
    contributions: np.ndarray = None,
    kernel: np.ndarray = None,
    passes_arr: np.ndarray = None,
    contribs_arr: np.ndarray = None,
    kernel_dim: int = None,
    region_data_full: list = None,
    I0: float = None
) -> str:
    """
    Deconvolve a single acquisition from an OME-TIFF file and save as OME-TIFF.
    
    Args:
        tiff_path: Path to the OME-TIFF file
        acq_id: Acquisition ID (not used for single file, but kept for consistency)
        output_dir: Output directory for OME-TIFF files
        x0: Parameter for kernel calculation
        iterations: Number of Richardson-Lucy iterations
        output_format: Output format, either 'float' or 'uint16'
        channel_names: List of channel names for OME metadata
        source_file_path: Optional source file path for filename generation
        unique_acq_id: Optional unique acquisition ID for filename generation
        channel_format: Format of channels in the input file ('CHW' or 'HWC')
    
    Returns:
        Path to the saved OME-TIFF file
    """
    import tifffile
    import xml.etree.ElementTree as ET
    
    # Load the OME-TIFF file and extract metadata
    pixel_size_x_from_file = None
    pixel_size_y_from_file = None
    pixel_size_unit_from_file = "µm"
    channel_names_from_file = None
    
    try:
        with tifffile.TiffFile(tiff_path) as tif:
            # Try to extract channel names and pixel size from OME metadata
            if hasattr(tif, 'ome_metadata') and tif.ome_metadata:
                ome_metadata = ET.fromstring(tif.ome_metadata)
                
                # Extract namespace
                root = ome_metadata
                namespace = None
                if root.tag.startswith('{'):
                    namespace = root.tag.split('}')[0].strip('{')
                    ns = {'ome': namespace}
                else:
                    ns = {}
                
                # Extract pixel size from Pixels element
                pixels_elem = None
                if ns:
                    pixels_elem = root.find('.//ome:Pixels', ns)
                if pixels_elem is None:
                    pixels_elem = root.find('.//Pixels')
                
                if pixels_elem is not None:
                    # Get PhysicalSizeX and PhysicalSizeY
                    if pixels_elem.get('PhysicalSizeX'):
                        try:
                            pixel_size_x_from_file = float(pixels_elem.get('PhysicalSizeX'))
                        except (ValueError, TypeError):
                            pass
                    if pixels_elem.get('PhysicalSizeY'):
                        try:
                            pixel_size_y_from_file = float(pixels_elem.get('PhysicalSizeY'))
                        except (ValueError, TypeError):
                            pass
                    if pixels_elem.get('PhysicalSizeXUnit'):
                        pixel_size_unit_from_file = pixels_elem.get('PhysicalSizeXUnit')
                    elif pixels_elem.get('PhysicalSizeYUnit'):
                        pixel_size_unit_from_file = pixels_elem.get('PhysicalSizeYUnit')
                    
                    # Extract channel names
                    channel_elements = []
                    if ns:
                        channel_elements = pixels_elem.findall('.//ome:Channel', ns)
                    if not channel_elements:
                        channel_elements = pixels_elem.findall('.//Channel')
                    
                    channel_names_from_file = []
                    for channel in channel_elements:
                        channel_name = channel.get('Name', '')
                        if not channel_name:
                            # Try to find Name as child element
                            if ns:
                                name_elem = channel.find(f'.//{{{namespace}}}Name')
                            else:
                                name_elem = channel.find('.//Name')
                            if name_elem is not None:
                                channel_name = name_elem.text or ''
                        if channel_name:
                            channel_names_from_file.append(channel_name)
    except Exception as e:
        print(f"Warning: Could not extract metadata from {tiff_path}: {e}")
    
    # Use extracted values if not provided
    if channel_names is None:
        channel_names = channel_names_from_file
    if pixel_size_x is None:
        pixel_size_x = pixel_size_x_from_file
    if pixel_size_y is None:
        pixel_size_y = pixel_size_y_from_file
    if pixel_size_unit == "µm" and pixel_size_unit_from_file != "µm":
        pixel_size_unit = pixel_size_unit_from_file
    
    # Load the OME-TIFF file
    img_stack = tifffile.imread(tiff_path)
    
    # Convert to (H, W, C) format for processing
    if img_stack.ndim == 2:
        # Single channel, add channel dimension
        img_stack = img_stack[..., np.newaxis]
    elif img_stack.ndim == 3:
        if channel_format == 'CHW':
            # Input is (C, H, W), transpose to (H, W, C)
            img_stack = np.transpose(img_stack, (1, 2, 0))
        # else: already (H, W, C), no transpose needed
    elif img_stack.ndim == 4:
        # Could be (T, C, H, W), (T, H, W, C), etc.
        if img_stack.shape[0] == 1:
            img_stack = img_stack[0]
            if img_stack.ndim == 3 and channel_format == 'CHW':
                img_stack = np.transpose(img_stack, (1, 2, 0))
        else:
            # Take first time point
            img_stack = img_stack[0]
            if img_stack.ndim == 3 and channel_format == 'CHW':
                img_stack = np.transpose(img_stack, (1, 2, 0))
    else:
        raise ValueError(f"Unsupported image dimensionality: {img_stack.ndim}D")
    
    # Verify image stack is valid
    if img_stack.size == 0:
        raise ValueError(f"Image stack is empty for file {tiff_path}")
    
    if img_stack.ndim != 3:
        raise ValueError(f"Expected 3D array (H, W, C), got {img_stack.ndim}D array with shape {img_stack.shape}")
    
    # Check image dimensions
    height, width, n_channels = img_stack.shape
    if height < 1 or width < 1:
        raise ValueError(f"Invalid image dimensions: {height}x{width}")
    if n_channels < 1:
        raise ValueError(f"No channels found in file {tiff_path}")
    
    # If channel names not available or count doesn't match, create default names
    if channel_names is None or len(channel_names) != n_channels:
        channel_names = [f"Channel_{i+1}" for i in range(n_channels)]
    
    # If only one pixel size dimension found, use it for both
    if pixel_size_x is not None and pixel_size_y is None:
        pixel_size_y = pixel_size_x
    elif pixel_size_y is not None and pixel_size_x is None:
        pixel_size_x = pixel_size_y
    
    # Convert to (C, H, W) for processing
    img_stack = np.transpose(img_stack, (2, 0, 1))
    
    # Apply deconvolution
    print(f"Deconvolving OME-TIFF {os.path.basename(tiff_path)}: shape={img_stack.shape}, x0={x0}, iterations={iterations}, format={output_format}")
    deconvolved_stack = RLD_HRIMC_circle(
        img_stack,
        x0=x0,
        iterations=iterations,
        output_format=output_format,
        passes=passes,
        contributions=contributions,
        kernel=kernel
    )  # Returns (C, H, W)
    print(f"Deconvolution complete: output shape={deconvolved_stack.shape}, dtype={deconvolved_stack.dtype}")
    
    # Generate filename: preserve original OME-TIFF filename with _deconvolved suffix
    # Get original filename without extension
    original_filename = os.path.basename(tiff_path)
    # Remove all extensions (.ome.tif, .tif, etc.)
    base_name = original_filename
    while '.' in base_name:
        base_name = os.path.splitext(base_name)[0]
    
    # Sanitize the base name
    safe_name = "".join(c for c in base_name if c.isalnum() or c in (' ', '-', '_', '.')).rstrip()
    safe_name = safe_name.replace(' ', '_')
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output filename: original_name_deconvolved.ome.tif
    # Preserve the .ome.tif extension format
    output_filename = f"{safe_name}_deconvolved.ome.tif"
    output_path = os.path.join(output_dir, output_filename)
    
    # Check if deconvolved stack is valid
    if deconvolved_stack.size == 0:
        raise ValueError(f"Deconvolved stack is empty for file {tiff_path}")
    
    # Verify deconvolved stack shape (should be C, H, W)
    print(f"Before saving: shape={deconvolved_stack.shape}, expected (C, H, W)")
    if deconvolved_stack.ndim != 3:
        raise ValueError(f"Expected 3D array (C, H, W), got {deconvolved_stack.ndim}D array with shape {deconvolved_stack.shape}")
    
    # Verify channel count matches
    expected_channels = len(channel_names) if channel_names else n_channels
    if deconvolved_stack.shape[0] != expected_channels:
        raise ValueError(f"Channel count mismatch: expected {expected_channels}, got {deconvolved_stack.shape[0]}")
    
    # Save as OME-TIFF in CHW format (matches GUI export format)
    metadata = {}
    if channel_names:
        metadata['Channel'] = {'Name': channel_names}
    
    # Add pixel size information to metadata
    if pixel_size_x is not None:
        metadata['PhysicalSizeX'] = pixel_size_x
        metadata['PhysicalSizeXUnit'] = pixel_size_unit
    if pixel_size_y is not None:
        metadata['PhysicalSizeY'] = pixel_size_y
        metadata['PhysicalSizeYUnit'] = pixel_size_unit
    
    # Save as OME-TIFF in CHW format (same as GUI export)
    try:
        tifffile.imwrite(
            output_path,
            deconvolved_stack,  # Already in (C, H, W) format
            photometric='minisblack',
            metadata=metadata,
            ome=True
        )
        
        # Verify the file was written correctly
        if not os.path.exists(output_path):
            raise IOError(f"Output file was not created: {output_path}")
        
        # Check file size
        file_size = os.path.getsize(output_path)
        if file_size == 0:
            raise IOError(f"Output file is empty: {output_path}")
        
        # Try to read it back to verify
        with tifffile.TiffFile(output_path) as tif:
            if not tif.series:
                raise IOError(f"TIFF file contains no image series: {output_path}")
            read_shape = tif.series[0].shape
            # tifffile may return shape in different order, so we check if dimensions match
            if set(read_shape) != set(deconvolved_stack.shape):
                print(f"Warning: Written shape {deconvolved_stack.shape} != read shape {read_shape}")
            else:
                print(f"File verified: written shape {deconvolved_stack.shape}, read shape {read_shape}")
        
    except Exception as e:
        # Clean up partial file if it exists
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
        raise IOError(f"Failed to write OME-TIFF file {output_path}: {str(e)}") from e
    
    return output_path


def deconvolve_acquisition(
    data_path: str,
    acq_id: str,
    output_dir: str,
    x0: float = 7.0,
    iterations: int = 4,
    output_format: str = "float",
    channel_names: list = None,
    source_file_path: str = None,
    unique_acq_id: str = None,
    loader_type: str = "mcd",
    channel_format: str = 'CHW',
    well_name: str = None,
    pixel_size_x: float = None,
    pixel_size_y: float = None,
    pixel_size_unit: str = "µm",
    passes: np.ndarray = None,
    contributions: np.ndarray = None,
    kernel: np.ndarray = None,
    passes_arr: np.ndarray = None,
    contribs_arr: np.ndarray = None,
    kernel_dim: int = None,
    region_data_full: list = None,
    I0: float = None
) -> str:
    """
    Deconvolve a single acquisition from an MCD file or OME-TIFF file and save as OME-TIFF.
    
    Args:
        data_path: Path to the MCD file or OME-TIFF file
        acq_id: Acquisition ID
        output_dir: Output directory for OME-TIFF files
        x0: Parameter for kernel calculation
        iterations: Number of Richardson-Lucy iterations
        output_format: Output format, either 'float' or 'uint16'
        channel_names: List of channel names for OME metadata
        source_file_path: Optional source file path for filename generation
        unique_acq_id: Optional unique acquisition ID for filename generation
        loader_type: Type of loader, either 'mcd' or 'ometiff'
        channel_format: Format of channels in OME-TIFF files ('CHW' or 'HWC')
        well_name: Optional well name to use in output filename (if None, will try to get from loader)
    
    Returns:
        Path to the saved OME-TIFF file
    """
    if loader_type == "mcd":
        return deconvolve_acquisition_from_mcd(
            mcd_path=data_path,
            acq_id=acq_id,
            output_dir=output_dir,
            x0=x0,
            iterations=iterations,
            output_format=output_format,
            channel_names=channel_names,
            source_file_path=source_file_path,
            unique_acq_id=unique_acq_id,
            well_name=well_name,
            pixel_size_x=pixel_size_x,
            pixel_size_y=pixel_size_y,
            pixel_size_unit=pixel_size_unit,
            passes=passes,
            contributions=contributions,
            kernel=kernel,
            passes_arr=passes_arr,
            contribs_arr=contribs_arr,
            kernel_dim=kernel_dim,
            region_data_full=region_data_full,
            I0=I0
        )
    elif loader_type == "ometiff":
        return deconvolve_acquisition_from_ometiff(
            tiff_path=data_path,
            acq_id=acq_id,
            output_dir=output_dir,
            x0=x0,
            iterations=iterations,
            output_format=output_format,
            channel_names=channel_names,
            source_file_path=source_file_path,
            unique_acq_id=unique_acq_id,
            channel_format=channel_format,
            pixel_size_x=pixel_size_x,
            pixel_size_y=pixel_size_y,
            pixel_size_unit=pixel_size_unit,
            passes=passes,
            contributions=contributions,
            kernel=kernel,
            passes_arr=passes_arr,
            contribs_arr=contribs_arr,
            kernel_dim=kernel_dim,
            region_data_full=region_data_full,
            I0=I0
        )
    else:
        raise ValueError(f"Unknown loader type: {loader_type}")

