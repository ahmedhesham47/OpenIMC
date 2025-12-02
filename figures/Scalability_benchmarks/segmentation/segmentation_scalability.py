#!/usr/bin/env python3
"""
Scalability benchmark for segmentation.

This script measures RAM usage, wall time (elapsed real-world time), maximum 
resident set size (RSS), and GPU vRAM for segmentation while varying:
1. Number of images
2. Batch size (for Cellpose batch processing)

Metrics:
- wall_time: Elapsed real-world time from start to finish (wall-clock time)
- peak_ram_mb: Peak virtual memory size (total memory allocated)
- max_rss_mb: Maximum resident set size (physical RAM actually used)
- peak_vram_mb: Peak GPU memory usage (if GPU is used)

Uses the EXACT same batch processing approach as the GUI:
- Fixed batch_size = 16 (same as GUI)
- Multiprocessing for loading/preprocessing acquisitions
- Model initialized once and reused for all batches
- model.eval() called on entire batch of images

Usage:
    python segmentation_scalability.py \
        --images-dir /path/to/images \
        --output-dir ./results \
        --nuclear-channels "Histone_1261726In113Di,Histone_473968La139Di,..." \
        --cyto-channels "Cytoker_651779Pr141Di,Cytoker_3111576Nd143Di,..." \
        --num-images 50 100 200 500

The images directory should contain:
    - Image files (e.g., .mcd, .mcdx files) OR directories of OME-TIFF files
"""

import argparse
import gc
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import psutil

# Add parent directory to path to import openimc
# Path: figures/Scalability_benchmarks/segmentation/ -> OpenIMC root
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from openimc.core import load_mcd
from openimc.data.mcd_loader import AcquisitionInfo

# Try to import torch for GPU memory tracking
try:
    import torch
    _HAVE_TORCH = True
except ImportError:
    _HAVE_TORCH = False


def get_memory_usage_mb() -> Tuple[float, float]:
    """Get current memory usage in MB.
    
    Returns:
        (rss_mb, vms_mb) - Resident Set Size and Virtual Memory Size
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    rss_mb = mem_info.rss / 1024 / 1024  # Resident Set Size (physical RAM)
    vms_mb = mem_info.vms / 1024 / 1024  # Virtual Memory Size (total virtual memory)
    return rss_mb, vms_mb


def get_gpu_memory_mb(gpu_id: Optional[Union[int, str]] = None) -> Optional[float]:
    """Get current GPU memory usage in MB.
    
    Args:
        gpu_id: GPU ID to check (optional)
    
    Returns:
        GPU memory usage in MB, or None if GPU not available
    """
    if not _HAVE_TORCH:
        return None
    
    try:
        if gpu_id is not None and gpu_id != 'auto':
            device_id = int(gpu_id) if isinstance(gpu_id, str) else gpu_id
            if torch.cuda.is_available() and device_id < torch.cuda.device_count():
                torch.cuda.set_device(device_id)
                return torch.cuda.memory_allocated(device_id) / 1024 / 1024
        elif torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # MPS (Apple Silicon) doesn't have memory tracking yet
            return None
    except Exception:
        pass
    
    return None


def reset_memory(gpu_id: Optional[Union[int, str]] = None):
    """Reset memory state before a new benchmark run.
    
    Clears caches, forces garbage collection, and allows system to settle.
    This ensures each benchmark starts with a clean memory state.
    """
    # Force multiple garbage collections to clear all references
    for _ in range(3):
        gc.collect()
    
    # Clear GPU cache if available
    if _HAVE_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Small delay to let system settle and release memory
    time.sleep(0.5)


def find_image_files(images_dir: Path) -> List[Path]:
    """Find all image files/directories - simple file finding only."""
    image_extensions = {'.mcd', '.mcdx'}
    image_files = []
    
    # Find MCD files
    for ext in image_extensions:
        image_files.extend(images_dir.glob(f'*{ext}'))
        image_files.extend(images_dir.glob(f'**/*{ext}'))
    
    # Check if images_dir itself is an OME-TIFF directory (contains TIFF files)
    tiff_files = list(images_dir.glob('*.ome.tif')) + list(images_dir.glob('*.ome.tiff')) + \
                 list(images_dir.glob('*.tif')) + list(images_dir.glob('*.tiff'))
    if tiff_files:
        image_files.append(images_dir)
    
    # Find OME-TIFF subdirectories
    for item in images_dir.iterdir():
        if item.is_dir():
            tiff_files = list(item.glob('*.ome.tif')) + list(item.glob('*.ome.tiff')) + \
                         list(item.glob('*.tif')) + list(item.glob('*.tiff'))
            if tiff_files:
                image_files.append(item)
    
    return sorted(image_files)


def _load_and_preprocess_acquisition_worker(task_data):
    """
    Worker function to load and preprocess a single acquisition for segmentation.
    This matches the GUI's _load_and_preprocess_acquisition_worker exactly.
    """
    (original_acq_id, unique_acq_id, acq_name, file_path, loader_type, preprocessing_config,
     denoise_source, custom_denoise_settings, source_file, channel_format) = task_data
    
    try:
        # Import inside function to ensure isolation
        import numpy as np
        from openimc.data.mcd_loader import MCDLoader
        from openimc.data.ometiff_loader import OMETIFFLoader
        from openimc.ui.utils import combine_channels, channelwise_minmax_normalize
        from openimc.core import _preprocess_channels_for_segmentation
        
        # Recreate loader (can't pickle loader objects)
        loader = None
        if loader_type == "mcd":
            loader = MCDLoader()
            loader.open(file_path)
        elif loader_type == "ometiff":
            loader = OMETIFFLoader(channel_format=channel_format or 'CHW')
            loader.open(file_path)
        
        if not loader:
            return None
        
        try:
            # Clear image cache before processing (same as feature extraction)
            if hasattr(loader, '_image_cache'):
                loader._image_cache.clear()
            
            config = preprocessing_config
            
            # Get nuclear channels
            nuclear_channels = config.get('nuclear_channels', [])
            if not nuclear_channels:
                return None
            
            # Get cytoplasm channels
            cyto_channels = config.get('cyto_channels', [])
            
            # Get channels, size, and metadata from loader (same as list_acquisitions does)
            channels = loader.get_channels(original_acq_id)
            
            # Get channel_metals, channel_labels, size, metadata, and well from loader's internal dictionaries
            channel_metals = []
            channel_labels = []
            size = (None, None)
            metadata = {}
            well = None
            
            if hasattr(loader, '_acq_channel_metals'):
                channel_metals = loader._acq_channel_metals.get(original_acq_id, [])
            if hasattr(loader, '_acq_channel_labels'):
                channel_labels = loader._acq_channel_labels.get(original_acq_id, [])
            if hasattr(loader, '_acq_size'):
                size = loader._acq_size.get(original_acq_id, (None, None))
            if hasattr(loader, '_acq_metadata'):
                metadata = loader._acq_metadata.get(original_acq_id, {})
            if hasattr(loader, '_acq_well'):
                well = loader._acq_well.get(original_acq_id)
            
            # Create AcquisitionInfo for preprocessing (same structure as list_acquisitions returns)
            acq_info = AcquisitionInfo(
                id=unique_acq_id,
                name=acq_name,
                well=well,
                size=size,
                channels=channels,
                channel_metals=channel_metals if isinstance(channel_metals, list) else [],
                channel_labels=channel_labels if isinstance(channel_labels, list) else [],
                metadata=metadata,
                source_file=source_file
            )
            
            # Use core preprocessing function (imported from core)
            # Note: _preprocess_channels_for_segmentation is used internally by segment()
            # We replicate the same logic here to match GUI behavior
            from openimc.core import _preprocess_channels_for_segmentation as preprocess_channels
            
            nuclear_img, cyto_img = preprocess_channels(
                loader, acq_info,
                nuclear_channels, cyto_channels,
                denoise_settings=custom_denoise_settings,
                normalization_method=config.get('normalization_method', 'channelwise_minmax'),
                arcsinh_cofactor=config.get('arcsinh_cofactor', 1.0),
                percentile_params=config.get('percentile_params', (1.0, 99.0)),
                nuclear_combo_method=config.get('nuclear_combo_method', 'mean'),
                cyto_combo_method=config.get('cyto_combo_method', 'mean'),
                nuclear_weights=config.get('nuclear_weights'),
                cyto_weights=config.get('cyto_weights')
            )
            
            # Return result with acquisition info
            return {
                'acq_id': unique_acq_id,
                'acq_name': acq_name,
                'nuclear_img': nuclear_img,
                'cyto_img': cyto_img,
                'source_file': source_file
            }
        finally:
            # Clear cache before closing (same as feature extraction)
            if hasattr(loader, '_image_cache'):
                loader._image_cache.clear()
            loader.close()
        
    except Exception as e:
        print(f"Error processing acquisition {acq_name} ({unique_acq_id}): {e}")
        import traceback
        traceback.print_exc()
        return None


def run_segmentation_benchmark(
    image_files: List[Path],
    num_images: int,
    output_dir: Path,
    nuclear_channels: List[str],
    cyto_channels: List[str],
    cellpose_model: str = 'cyto3',
    diameter: Optional[int] = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    gpu_id: Optional[Union[int, str]] = None,
    denoise_settings: Optional[Dict] = None,
    channel_format: str = 'CHW',
    batch_size: int = 16,  # Fixed batch size like GUI
    num_workers: Optional[int] = None,  # Number of workers for multiprocessing (None = auto)
    repeat: int = 0
) -> Dict:
    """
    Run segmentation benchmark using EXACT same approach as GUI.
    
    This matches the GUI's _perform_segmentation_all_acquisitions method:
    - Fixed batch_size = 16
    - Multiprocessing for loading/preprocessing
    - Model initialized once and reused
    - model.eval() called on entire batch
    
    Returns:
        Dictionary with metrics including vRAM
    """
    print(f"\n{'='*60}")
    print(f"Benchmark: {num_images} acquisitions, batch_size={batch_size}, num_workers={num_workers or 'auto'}")
    print(f"{'='*60}")
    
    # Reset memory state before starting benchmark
    reset_memory(gpu_id)
    
    # Load images to get acquisitions (metadata only, no image data)
    all_acquisitions = []
    acq_to_input_path = {}
    acq_to_loader_type = {}
    acq_to_original_id = {}
    
    loaders = []
    for img_path in image_files:
        if len(all_acquisitions) >= num_images:
            break
        
        try:
            loader, loader_type = load_mcd(str(img_path), channel_format=channel_format)
            loaders.append(loader)
            
            # Clear any image cache before getting acquisitions (same as GUI)
            if hasattr(loader, '_image_cache'):
                loader._image_cache.clear()
            
            acquisitions = loader.list_acquisitions()
            
            for acq in acquisitions:
                if len(all_acquisitions) >= num_images:
                    break
                all_acquisitions.append(acq)
                acq_to_input_path[acq.id] = str(img_path)
                acq_to_loader_type[acq.id] = loader_type
                # Extract original acquisition ID (handles multi-file unique IDs)
                from openimc.core import _extract_original_acq_id
                acq_to_original_id[acq.id] = _extract_original_acq_id(acq.id)
        except Exception as e:
            print(f"Warning: Failed to load {img_path}: {e}")
            continue
    
    # Close loaders and clear caches (will reload in workers, same as GUI)
    for loader in loaders:
        if hasattr(loader, '_image_cache'):
            loader._image_cache.clear()
        if hasattr(loader, 'close'):
            loader.close()
    loaders.clear()
    gc.collect()
    
    if len(all_acquisitions) == 0:
        return {
            'num_images': num_images,
            'batch_size': batch_size,
            'num_workers': num_workers or max(1, min(mp.cpu_count() - 2, num_images)),
            'wall_time': None,
            'peak_ram_mb': None,
            'max_rss_mb': None,
            'peak_vram_mb': None,
            'num_cells_total': 0,
            'num_acquisitions': 0,
            'success': False
        }
    
    # Limit to exactly num_images acquisitions
    all_acquisitions = all_acquisitions[:num_images]
    
    # Update mappings to only include selected acquisitions
    selected_acq_ids = {acq.id for acq in all_acquisitions}
    acq_to_input_path = {k: v for k, v in acq_to_input_path.items() if k in selected_acq_ids}
    acq_to_loader_type = {k: v for k, v in acq_to_loader_type.items() if k in selected_acq_ids}
    acq_to_original_id = {k: v for k, v in acq_to_original_id.items() if k in selected_acq_ids}
    
    print(f"Processing {len(all_acquisitions)} acquisitions (requested {num_images})")
    
    # Create output directory for masks
    masks_output_dir = output_dir / 'output_data' / 'masks' / f"{num_images}_{batch_size}_{repeat}"
    masks_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear caches before starting benchmark (same as feature extraction)
    # This ensures images are not preloaded and we measure actual segmentation time
    gc.collect()
    
    # Start timing
    start_time = time.time()
    start_rss, start_vms = get_memory_usage_mb()
    peak_vms = start_vms
    max_rss = start_rss
    peak_vram = get_gpu_memory_mb(gpu_id) or 0.0
    
    try:
        # Initialize Cellpose model once (same as GUI)
        from cellpose import models
        
        # Determine GPU usage and device (same as core.py)
        # Note: core.py uses: use_gpu = gpu_id is not None, device=gpu_id
        # But we need to handle "auto" and convert properly
        use_gpu = False
        gpu_device = None
        
        if gpu_id == "auto":
            if _HAVE_TORCH and torch.cuda.is_available():
                use_gpu = True
                gpu_device = 0
            elif _HAVE_TORCH and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                use_gpu = True
                gpu_device = 'mps'
        elif gpu_id is not None:
            use_gpu = True
            # gpu_id is already parsed in main() - should be int, str, or 'auto'
            # Cellpose expects device to be int for CUDA or 'mps' for Apple Silicon
            # Pass gpu_id directly as device (same as core.py does)
            gpu_device = gpu_id
        
        # Set CUDA device if using GPU and device is specified
        if use_gpu and _HAVE_TORCH and isinstance(gpu_device, int):
            torch.cuda.set_device(gpu_device)
        
        # Print GPU info
        if use_gpu:
            if isinstance(gpu_device, int):
                print(f"Using GPU: device={gpu_device}")
            else:
                print(f"Using GPU: device={gpu_device}")
        else:
            print("Using CPU")
        
        # Initialize model (same as GUI - only pass gpu parameter, not device)
        # Cellpose handles device selection internally when gpu=True
        # We set the CUDA device above if a specific GPU was requested
        if cellpose_model == 'nuclei':
            model_obj = models.Cellpose(gpu=use_gpu, model_type='nuclei')
        else:  # cyto3
            model_obj = models.Cellpose(gpu=use_gpu, model_type='cyto3')
        
        # Build preprocessing config (same as GUI)
        preprocessing_config = {
            'nuclear_channels': nuclear_channels,
            'cyto_channels': cyto_channels,
            'normalization_method': 'channelwise_minmax',
            'nuclear_combo_method': 'mean',
            'cyto_combo_method': 'mean',
            'nuclear_weights': None,
            'cyto_weights': None
        }
        
        # Process acquisitions in batches (same as GUI)
        successful_segmentations = 0
        all_masks = {}
        
        for batch_start in range(0, len(all_acquisitions), batch_size):
            batch_end = min(batch_start + batch_size, len(all_acquisitions))
            batch_acquisitions = all_acquisitions[batch_start:batch_end]
            
            print(f"Processing batch {batch_start//batch_size + 1} ({len(batch_acquisitions)} acquisitions)...")
            
            try:
                # Load and preprocess all acquisitions in this batch using multiprocessing (same as GUI)
                mp_args = []
                acq_id_to_acq_info = {acq.id: acq for acq in batch_acquisitions}
                
                for acq in batch_acquisitions:
                    file_path = acq_to_input_path[acq.id]
                    loader_type = acq_to_loader_type[acq.id]
                    original_acq_id = acq_to_original_id[acq.id]
                    
                    mp_args.append((
                        original_acq_id,
                        acq.id,
                        acq.name,
                        file_path,
                        loader_type,
                        preprocessing_config,
                        "none",  # denoise_source
                        denoise_settings,
                        file_path,  # source_file
                        channel_format
                    ))
                
                # Use multiprocessing for loading/preprocessing (same as GUI)
                # Use provided num_workers or auto-calculate (same as GUI default)
                if num_workers is None:
                    max_workers = max(1, min(mp.cpu_count() - 2, len(mp_args)))
                else:
                    max_workers = max(1, min(num_workers, len(mp_args)))
                batch_images = []
                batch_channels = []
                acquisition_mapping = []
                acquisition_info_list = []
                
                with mp.Pool(processes=max_workers) as pool:
                    futures = [pool.apply_async(_load_and_preprocess_acquisition_worker, (args,)) for args in mp_args]
                    
                    for future in futures:
                        try:
                            result = future.get(timeout=600)
                            if result is None:
                                continue
                            
                            acq_id = result['acq_id']
                            nuclear_img = result['nuclear_img']
                            cyto_img = result['cyto_img']
                            
                            if acq_id not in acq_id_to_acq_info:
                                continue
                            
                            acq = acq_id_to_acq_info[acq_id]
                            acq_idx = len(acquisition_info_list)
                            acquisition_info_list.append(acq)
                            
                            # Prepare input images based on model type (same as GUI)
                            if cyto_img is not None:
                                # Both nuclear and cytoplasm available
                                batch_images.extend([cyto_img, nuclear_img])
                                batch_channels.extend([0, 1])  # cyto, nuclear
                                acquisition_mapping.extend([acq_idx, acq_idx])
                            else:
                                # Only nuclear available
                                batch_images.extend([nuclear_img, nuclear_img])
                                batch_channels.extend([0, 0])  # nuclear, nuclear
                                acquisition_mapping.extend([acq_idx, acq_idx])
                        except Exception as e:
                            print(f"Error in worker: {e}")
                            continue
                
                if not batch_images:
                    continue
                
                # Monitor memory
                current_rss, current_vms = get_memory_usage_mb()
                peak_vms = max(peak_vms, current_vms)
                max_rss = max(max_rss, current_rss)
                current_vram = get_gpu_memory_mb(gpu_id)
                if current_vram is not None:
                    peak_vram = max(peak_vram, current_vram)
                
                # Run segmentation on the entire batch (same as GUI)
                masks, flows, styles, diams = model_obj.eval(
                    batch_images,
                    diameter=diameter,
                    flow_threshold=flow_threshold,
                    cellprob_threshold=cellprob_threshold,
                    channels=batch_channels
                )
                
                # Store results using acquisition mapping (same as GUI)
                processed_acquisitions = set()
                for i, mask in enumerate(masks):
                    if i < len(acquisition_mapping):
                        acq_idx = acquisition_mapping[i]
                        if acq_idx not in processed_acquisitions and acq_idx < len(acquisition_info_list):
                            acq_info = acquisition_info_list[acq_idx]
                            acq_id = acq_info.id
                            
                            if isinstance(mask, np.ndarray):
                                mask = mask.copy()
                            
                            all_masks[acq_id] = mask
                            processed_acquisitions.add(acq_idx)
                            successful_segmentations += 1
                            
                            # Save mask
                            if acq_info.well:
                                output_filename = f"{acq_info.well}_segmentation_masks.tif"
                            else:
                                output_filename = f"{acq_info.name}_segmentation_masks.tif"
                            output_path = masks_output_dir / output_filename
                            
                            import tifffile
                            tifffile.imwrite(str(output_path), mask.astype(np.uint16), compression='lzw')
                
                # Explicitly release memory (same as GUI)
                del batch_images
                del masks, flows, styles, diams
                gc.collect()
                if _HAVE_TORCH and use_gpu:
                    torch.cuda.empty_cache()
                
                # Monitor memory after batch
                current_rss, current_vms = get_memory_usage_mb()
                peak_vms = max(peak_vms, current_vms)
                max_rss = max(max_rss, current_rss)
                current_vram = get_gpu_memory_mb(gpu_id)
                if current_vram is not None:
                    peak_vram = max(peak_vram, current_vram)
                
            except Exception as e:
                print(f"Error processing batch {batch_start//batch_size + 1}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        end_time = time.time()
        wall_time = end_time - start_time
        final_rss, final_vms = get_memory_usage_mb()
        peak_vms = max(peak_vms, final_vms)
        max_rss = max(max_rss, final_rss)
        final_vram = get_gpu_memory_mb(gpu_id)
        if final_vram is not None:
            peak_vram = max(peak_vram, final_vram)
        
        # Calculate total cells
        num_cells_total = sum(len(np.unique(mask)) - 1 for mask in all_masks.values())
        
        print(f"Completed: {wall_time:.2f}s, Peak VMS: {peak_vms:.1f} MB, Max RSS: {max_rss:.1f} MB")
        if peak_vram > 0:
            print(f"  Peak vRAM: {peak_vram:.1f} MB")
        print(f"  Processed {successful_segmentations}/{len(all_acquisitions)} acquisitions successfully")
        print(f"  Total cells detected: {num_cells_total}")
        
        return {
            'num_images': num_images,
            'batch_size': batch_size,
            'num_workers': num_workers or max_workers,
            'wall_time': wall_time,
            'peak_ram_mb': peak_vms,
            'max_rss_mb': max_rss,
            'peak_vram_mb': peak_vram if peak_vram > 0 else None,
            'num_cells_total': num_cells_total,
            'num_acquisitions': successful_segmentations,
            'success': successful_segmentations > 0
        }
    
    except Exception as e:
        print(f"Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
        return {
            'num_images': num_images,
            'batch_size': batch_size,
            'num_workers': num_workers or max(1, min(mp.cpu_count() - 2, num_images)),
            'wall_time': time.time() - start_time,
            'peak_ram_mb': peak_vms,
            'max_rss_mb': max_rss,
            'peak_vram_mb': peak_vram if peak_vram > 0 else None,
            'num_cells_total': 0,
            'num_acquisitions': 0,
            'success': False,
            'error': str(e)
        }
    
    finally:
        gc.collect()
        if _HAVE_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description='Scalability benchmark for segmentation (matches GUI exactly)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--images-dir', type=str, required=True,
                       help='Directory containing image files (MCD files or OME-TIFF directories)')
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Output directory for results (default: ./results)')
    parser.add_argument('--nuclear-channels', type=str, required=True,
                       help='Comma-separated list of nuclear channel names')
    parser.add_argument('--cyto-channels', type=str, default=None,
                       help='Comma-separated list of cytoplasm channel names (optional, for cyto3 model)')
    parser.add_argument('--num-images', type=int, nargs='+', default=[50, 100, 200, 500],
                       help='Number of images to test (default: 50 100 200 500)')
    parser.add_argument('--repeats', type=int, default=3,
                       help='Number of times to repeat each configuration (default: 3)')
    parser.add_argument('--channel-format', choices=['CHW', 'HWC'], default='CHW',
                       help='Channel format for OME-TIFF files (default: CHW)')
    parser.add_argument('--cellpose-model', choices=['cyto3', 'nuclei'], default='cyto3',
                       help='Cellpose model type (default: cyto3)')
    parser.add_argument('--diameter', type=int, default=10,
                       help='Cell diameter in pixels (optional, Cellpose will estimate if not provided)')
    parser.add_argument('--flow-threshold', type=float, default=0.4,
                       help='Flow threshold for Cellpose (default: 0.4)')
    parser.add_argument('--cellprob-threshold', type=float, default=0.0,
                       help='Cell probability threshold for Cellpose (default: 0.0)')
    parser.add_argument('--gpu-id', type=str, default='0',
                       help='GPU ID to use (default: "0", use "auto" for auto-detection, or "None" for CPU)')
    parser.add_argument('--denoise-settings', type=str, default=None,
                       help='JSON file or string with denoise settings (optional)')
    parser.add_argument('--batch-size', type=int, nargs='+', default=[1,8,16],
                       help='Batch size(s) for Cellpose (default: 16). Can specify multiple values to test scalability (e.g., 1 4 8 16 32).')
    parser.add_argument('--num-workers', type=int, nargs='+', default=[None],
                       help='Number of workers for multiprocessing during loading/preprocessing (default: auto, uses cpu_count - 2). Can specify multiple values to test scalability.')
    
    args = parser.parse_args()
    
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    
    if not images_dir.exists():
        print(f"Error: Images directory does not exist: {images_dir}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse channels
    nuclear_channels = [ch.strip() for ch in args.nuclear_channels.split(',')]
    cyto_channels = None
    if args.cyto_channels:
        cyto_channels = [ch.strip() for ch in args.cyto_channels.split(',')]
    
    # Parse denoise settings if provided
    denoise_settings = None
    if args.denoise_settings:
        from openimc.core import parse_denoise_settings
        denoise_settings = parse_denoise_settings(args.denoise_settings)
    
    # Parse GPU ID
    # args.gpu_id is a string from argparse (default: '0')
    gpu_id = None
    if args.gpu_id and args.gpu_id.lower() != 'none':
        if args.gpu_id.lower() == 'auto':
            gpu_id = 'auto'
        else:
            # Try to convert to int, but keep as string if it's not numeric (e.g., 'mps')
            try:
                gpu_id = int(args.gpu_id)
            except (ValueError, TypeError):
                # Not a number, use as-is (e.g., 'mps' for Apple Silicon)
                gpu_id = args.gpu_id
    
    # Find image files
    print(f"Scanning for images in {images_dir}...")
    image_files = find_image_files(images_dir)
    
    if len(image_files) == 0:
        print(f"Error: No image files found in {images_dir}")
        sys.exit(1)
    
    print(f"Found {len(image_files)} image files/directories")
    print(f"Nuclear channels: {nuclear_channels}")
    if cyto_channels:
        print(f"Cytoplasm channels: {cyto_channels}")
    print(f"Cellpose model: {args.cellpose_model}")
    print(f"Batch sizes: {', '.join(map(str, args.batch_size))}")
    worker_str = [str(w) if w is not None else 'auto' for w in args.num_workers]
    print(f"Num workers: {', '.join(worker_str)} (auto means cpu_count - 2)")
    
    # Run benchmarks
    results = []
    first_image = args.num_images[0]
    first_worker = args.num_workers[0]
    first_batch_size = args.batch_size[0]
    for num_images in args.num_images:
        for batch_size in args.batch_size:
            for num_workers in args.num_workers:
                for repeat in range(args.repeats):
                    # Reset memory between each benchmark run
                    if repeat > 0 or (num_images != first_image or batch_size != first_batch_size or num_workers != first_worker):
                        print("\nResetting memory before next benchmark...")
                        reset_memory(gpu_id)
                    
                    result = run_segmentation_benchmark(
                        image_files, num_images, output_dir,
                        nuclear_channels, cyto_channels,
                        cellpose_model=args.cellpose_model,
                        diameter=args.diameter,
                        flow_threshold=args.flow_threshold,
                        cellprob_threshold=args.cellprob_threshold,
                        gpu_id=gpu_id,
                        denoise_settings=denoise_settings,
                        channel_format=args.channel_format,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        repeat=repeat
                    )
                    result['repeat'] = repeat
                    results.append(result)
                
                # Reset memory after each run
                reset_memory(gpu_id)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_csv = output_dir / 'scalability_results.csv'
    results_df.to_csv(results_csv, index=False)
    print(f"\nResults saved to {results_csv}")
    
    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    if 'success' in results_df.columns:
        successful = results_df[results_df['success'] == True]
        if len(successful) > 0:
            agg_dict = {
                'wall_time': ['mean', 'std'],
                'peak_ram_mb': ['mean', 'std'],
                'max_rss_mb': ['mean', 'std'],
                'num_cells_total': ['mean', 'std']
            }
            if 'peak_vram_mb' in successful.columns:
                agg_dict['peak_vram_mb'] = ['mean', 'std']
            groupby_cols = ['num_images', 'batch_size']
            if 'num_workers' in successful.columns:
                groupby_cols.append('num_workers')
            print(successful.groupby(groupby_cols).agg(agg_dict))
        else:
            print("No successful runs!")
    
    print(f"\nFull results saved to: {results_csv}")
    print(f"To generate plots, run: python generate_scalability_plots.py {results_csv}")


if __name__ == '__main__':
    main()
