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
# but WITHOUT ANY implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Module for generating passes and contributions arrays for High Resolution IMC deconvolution.

This module uses signature-based region segmentation with pole of inaccessibility
labeling to compute passes and contributions given instrument geometry and an
inverse sigmoidal loss function.
"""

import numpy as np
from typing import Callable, Tuple, Optional
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt


def example_inverse_sigmoid_loss(
    passes: np.ndarray,
    midpoint: float = 4.0,
    slope: float = 1.0,
    min_fraction: float = 0.0,
    max_fraction: float = 1.0,
) -> np.ndarray:
    """
    Example inverse sigmoidal loss function.

    Parameters
    ----------
    passes : np.ndarray
        Number of passes (>= 1) for subpixels inside the current shot.
    midpoint : float
        Pass value where signal is ~0.5 of max_fraction.
    slope : float
        Steepness of transition (positive slope = steeper drop with passes).
    min_fraction : float
        Minimum surviving fraction at very high passes.
    max_fraction : float
        Maximum surviving fraction (typically ~1.0 at 1 pass).

    Returns
    -------
    np.ndarray
        Fraction of signal remaining in [min_fraction, max_fraction].

    Note:
        Replace with your experimentally fitted inverse sigmoidal curve.
    """
    x = passes.astype(float)
    # Decreasing logistic: more passes -> less signal remaining
    frac = 1.0 / (1.0 + np.exp(slope * (x - midpoint)))
    return min_fraction + (max_fraction - min_fraction) * frac


def hr_inverse_sigmoid_loss(
    passes: np.ndarray,
    x0: float = 7.0,
    slope: float = 1.0,
    min_fraction: float = 0.0,
    max_fraction: float = 1.0,
) -> np.ndarray:
    """
    Inverse sigmoidal loss function for HR-IMC signal as a function of pass count.

    Parameters
    ----------
    passes : np.ndarray
        Integer pass counts (>= 1) for subpixels that are actually hit by the
        current beam.
    x0 : float, default 7.0
        Midpoint (in pass number) of the sigmoidal curve. Higher x0 shifts
        the drop-off in remaining signal to higher pass counts.
    slope : float, default 1.0
        Steepness of the logistic transition.
    min_fraction : float, default 0.0
        Asymptotic minimum fraction of remaining signal.
    max_fraction : float, default 1.0
        Asymptotic maximum fraction of remaining signal.

    Returns
    -------
    np.ndarray
        Fraction of signal remaining in [min_fraction, max_fraction] for each pass count.
    """
    x = passes.astype(float)
    # Decreasing logistic: more passes -> less remaining signal
    frac = 1.0 / (1.0 + np.exp(slope * (x - x0)))
    return min_fraction + (max_fraction - min_fraction) * frac


def compute_hrimc_passes_contributions(
    step_size_um: float,
    loss_fn: Callable[[np.ndarray], np.ndarray],
    pixel_size_um: float = 1.0,
    spot_diameter_um: float = 1.0,
    n_subpixels: int = 15,
    circle_resolution: int = 64,
    grid_resolution: int = 800,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Passes and Contributions for HR-IMC using signature-based region
    segmentation with pole of inaccessibility labeling.

    This method uses a high-resolution subpixel grid to identify distinct
    regions within the central crater, labels them using pole of inaccessibility,
    and computes passes and contributions per region.

    Parameters
    ----------
    step_size_um : float
        HR-IMC step size in microns (e.g., 0.333 or 0.5).
    loss_fn : Callable[[np.ndarray], np.ndarray]
        Inverse sigmoidal loss function mapping pass numbers -> fraction of
        signal remaining after repeated ablations.
    pixel_size_um : float, default 1.0
        Coarse IMC pixel size (1 µm). Not used in computation but kept for
        interface compatibility.
    spot_diameter_um : float, default 1.0
        Laser spot diameter (1 µm for HR-IMC). Used as crater radius.
    n_subpixels : int, default 15
        Not used in this implementation but kept for interface compatibility.
    circle_resolution : int, default 64
        Not used in this implementation but kept for interface compatibility.
    grid_resolution : int, default 800
        Resolution of the high-resolution subpixel grid used for region
        segmentation. Higher values provide more accurate region identification
        but increase computation time.

    Returns
    -------
    passes_flat : np.ndarray, shape (n_regions,)
        Pass count per region. Each region represents a distinct area within
        the central crater with a unique pass count.
    contributions_flat : np.ndarray, shape (n_regions,)
        Normalized contribution weight per region (sums to 1).
    psf_kernel_3x3 : np.ndarray, shape (3, 3)
        Skewed 3×3 PSF suitable for Richardson–Lucy deconvolution.
    """
    crater_radius = spot_diameter_um / 2.0
    step = step_size_um
    
    # Determine grid size to cover enough shots
    # We need enough rows/cols to include all shots that could intersect
    max_distance = 2.1 * crater_radius
    num_rows = int(np.ceil(max_distance / step) * 2) + 1
    num_cols = num_rows
    
    # ============================================
    # 1. Raster Setup
    # ============================================
    central_row = num_rows // 2
    central_col = num_cols // 2

    shot_centers = []
    for r in range(num_rows):
        for c in range(num_cols):
            x = c * step
            y = -r * step
            shot_centers.append((x, y))
    shot_centers = np.array(shot_centers, dtype=float)

    central_idx = central_row * num_cols + central_col
    central_center = shot_centers[central_idx]
    effective_shots = shot_centers[:central_idx + 1].copy()

    # ============================================
    # 2. High-Res Subpixel Grid
    # ============================================
    grid_res = grid_resolution
    xs_rel = np.linspace(-crater_radius, crater_radius, grid_res)
    ys_rel = np.linspace(-crater_radius, crater_radius, grid_res)
    XX_rel, YY_rel = np.meshgrid(xs_rel, ys_rel)

    XX = XX_rel + central_center[0]
    YY = YY_rel + central_center[1]
    points = np.c_[XX.ravel(), YY.ravel()]

    # Mask: Only care about pixels inside the main crater
    inside_mask = np.hypot(XX_rel, YY_rel) <= crater_radius
    inside_indices = np.where(inside_mask.ravel())[0]
    points_inside = points[inside_indices]

    # ============================================
    # 3. Signature-Based Region Segmentation
    # ============================================
    # Filter shots close to center
    dists_from_center = np.hypot(effective_shots[:, 0] - central_center[0],
                                 effective_shots[:, 1] - central_center[1])
    rel_shots = effective_shots[dists_from_center <= (2.1 * crater_radius)]

    # Assign random unique float ID to each shot
    rng = np.random.default_rng(42)
    shot_signatures = rng.uniform(100, 1000000, size=len(rel_shots))

    # Compute overlaps
    dists_matrix = np.sqrt(((rel_shots[None, :, :] - points_inside[:, None, :]) ** 2).sum(axis=2))
    boolean_overlaps = dists_matrix <= crater_radius

    # Pass Map (Integer count)
    pass_values_inside = np.sum(boolean_overlaps, axis=1)

    # Signature Map (Unique float per region)
    signature_values_inside = boolean_overlaps @ shot_signatures

    # Reconstruct 2D images
    pass_map = np.zeros(grid_res * grid_res, dtype=int)
    pass_map[inside_indices] = pass_values_inside
    pass_map = pass_map.reshape(grid_res, grid_res)

    sig_map = np.zeros(grid_res * grid_res, dtype=float)
    sig_map[inside_indices] = signature_values_inside
    sig_map = sig_map.reshape(grid_res, grid_res)

    # ============================================
    # 4. Labeling via Pole of Inaccessibility
    # ============================================
    region_data = []
    min_pixel_area = 0  # Filter dust

    unique_sigs = np.unique(sig_map[inside_mask.reshape(grid_res, grid_res)])

    for sig in unique_sigs:
        if sig == 0:
            continue

        # 1. Create binary mask for this specific region
        binary = (sig_map == sig).astype(np.uint8)

        # 2. Label components (handles cases where a region is split)
        labeled = label(binary, connectivity=2)
        props = regionprops(labeled)

        # Determine pass count (grab first valid pixel)
        representative_val = pass_map[binary == 1][0]

        for rp in props:
            if rp.area < min_pixel_area:
                continue

            # --- Pole of inaccessibility ---
            min_row, min_col, max_row, max_col = rp.bbox
            sub_mask = binary[min_row:max_row, min_col:max_col]

            dist_transform = distance_transform_edt(sub_mask)
            max_idx = np.argmax(dist_transform)
            max_pos = np.unravel_index(max_idx, dist_transform.shape)

            cy = max_pos[0] + min_row
            cx = max_pos[1] + min_col

            x_coord = np.interp(cx, np.arange(grid_res), xs_rel)
            y_coord = np.interp(cy, np.arange(grid_res), ys_rel)

            region_data.append({
                'val': representative_val,
                'x': x_coord,
                'y': y_coord,
                'area': float(rp.area)  # subpixel count
            })

    if len(region_data) == 0:
        raise RuntimeError("No regions found. Check step_size_um, spot_diameter_um, and geometry.")

    # ============================================
    # 5. Compute Passes, Contributions, PSF kernel
    # ============================================
    # 5.1 Passes array (one per region, in region_data order)
    passes_array = np.array([rd['val'] for rd in region_data], dtype=int)

    # 5.2 Contributions array (area-weighted, normalized)
    subpixel_area = (2 * crater_radius / grid_res) ** 2
    areas_pixels = np.array([rd['area'] for rd in region_data], dtype=float)
    raw_contributions = areas_pixels * subpixel_area
    contributions_array = raw_contributions / raw_contributions.sum()

    # 5.3 PSF kernel construction
    passes = np.array([rd['val'] for rd in region_data], dtype=float)

    if loss_fn is None:
        attenuation = np.ones_like(passes, dtype=float)
    else:
        attenuation = np.asarray(loss_fn(passes), dtype=float)

    weights = contributions_array * attenuation

    kernel = np.zeros((3, 3), dtype=float)

    # Partition central pixel into 3x3 bins in normalized coordinates
    # Normalize x,y into [-1, 1] by dividing by crater_radius
    for rd, w in zip(region_data, weights):
        x = rd['x'] / crater_radius
        y = rd['y'] / crater_radius

        # Columns: left, center, right
        if x < -1.0 / 3.0:
            col = 0
        elif x > 1.0 / 3.0:
            col = 2
        else:
            col = 1

        # Rows: top, middle, bottom (y>0 is 'up')
        if y > 1.0 / 3.0:
            row = 0
        elif y < -1.0 / 3.0:
            row = 2
        else:
            row = 1

        kernel[row, col] += w

    # Normalize PSF
    s = kernel.sum()
    if s > 0:
        kernel /= s
    else:
        raise RuntimeError("PSF kernel sum is zero. Check loss function and geometry.")

    return passes_array, contributions_array, kernel


def compute_hrimc_psf(
    step_size_um: float,
    x0: float = 7.0,
    slope: float = 1.0,
    pixel_size_um: float = 1.0,
    spot_diameter_um: float = 1.0,
    n_subpixels: int = 15,
    circle_resolution: int = 64,
    loss_fn: Callable[[np.ndarray, float, float, float, float], np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute passes, contributions, and a 3×3 PSF kernel for HR-IMC.
    
    This is a convenience wrapper around compute_hrimc_passes_contributions
    that accepts x0, slope parameters directly instead of a loss function.

    Parameters
    ----------
    step_size_um : float
        HR-IMC step size in microns (e.g. 0.333 or 0.5).
    x0 : float, default 7.0
        Midpoint of the inverse sigmoidal loss curve (in pass units).
    slope : float, default 1.0
        Slope of the inverse sigmoidal loss curve.
    pixel_size_um : float, default 1.0
        Size of a coarse IMC pixel (1 µm for HR-IMC). Not used in computation
        but kept for interface compatibility.
    spot_diameter_um : float, default 1.0
        Laser spot diameter (1 µm).
    n_subpixels : int, default 15
        Not used in this implementation but kept for interface compatibility.
    circle_resolution : int, default 64
        Not used in this implementation but kept for interface compatibility.
    loss_fn : callable, optional
        Function with signature loss_fn(passes, x0, slope, min_fraction, max_fraction)
        returning fraction of signal remaining. If None, uses hr_inverse_sigmoid_loss.

    Returns
    -------
    passes_flat : np.ndarray, shape (n_regions,)
        Pass count per region. Each region represents a distinct area within
        the central crater with a unique pass count.
    contributions_flat : np.ndarray, shape (n_regions,)
        Normalized contribution per region (sums to 1).
    psf_kernel_3x3 : np.ndarray, shape (3, 3)
        3×3 PSF kernel suitable for RL deconvolution. Row 0 corresponds to
        the "top" of the pixel, row 2 to the "bottom".
    """
    if loss_fn is None:
        def loss_fn_wrapper(passes):
            return hr_inverse_sigmoid_loss(passes, x0=x0, slope=slope, min_fraction=0.0, max_fraction=1.0)
    else:
        def loss_fn_wrapper(passes):
            return loss_fn(passes, x0=x0, slope=slope, min_fraction=0.0, max_fraction=1.0)
    
    return compute_hrimc_passes_contributions(
        step_size_um=step_size_um,
        loss_fn=loss_fn_wrapper,
        pixel_size_um=pixel_size_um,
        spot_diameter_um=spot_diameter_um,
        n_subpixels=n_subpixels,
        circle_resolution=circle_resolution
    )


def save_passes_contributions(
    passes: np.ndarray,
    contributions: np.ndarray,
    file_path: str,
    metadata: Optional[dict] = None,
    psf_kernel: Optional[np.ndarray] = None
) -> None:
    """
    Save passes, contributions arrays, and PSF kernel to a file.

    Args:
        passes: Pass count array
        contributions: Contributions array (will be normalized)
        file_path: Path to save the file (.npz format)
        metadata: Optional dictionary of metadata to save
        psf_kernel: Optional 3×3 PSF kernel array to save
    """
    # Normalize contributions
    contributions_normalized = contributions / contributions.sum()
    
    # Prepare data to save
    save_dict = {
        'passes': passes,
        'contributions': contributions_normalized
    }
    
    # Save PSF kernel if provided
    if psf_kernel is not None:
        save_dict['psf_kernel'] = psf_kernel
    
    if metadata:
        save_dict['metadata'] = metadata
    
    np.savez(file_path, **save_dict)


def load_passes_contributions(file_path: str) -> Tuple[np.ndarray, np.ndarray, Optional[dict], Optional[np.ndarray]]:
    """
    Load passes, contributions arrays, and PSF kernel from a file.

    Args:
        file_path: Path to the .npz file

    Returns:
        Tuple of (passes, contributions, metadata, psf_kernel)
        psf_kernel will be None if not present in the file
    """
    data = np.load(file_path, allow_pickle=True)
    passes = data['passes']
    contributions = data['contributions']
    
    # Normalize contributions in case they weren't normalized when saved
    contributions = contributions / contributions.sum()
    
    metadata = None
    if 'metadata' in data:
        metadata = data['metadata'].item() if hasattr(data['metadata'], 'item') else data['metadata']
    
    psf_kernel = None
    if 'psf_kernel' in data:
        psf_kernel = data['psf_kernel']
        # Ensure it's normalized
        if psf_kernel.sum() > 0:
            psf_kernel = psf_kernel / psf_kernel.sum()
    
    return passes, contributions, metadata, psf_kernel
