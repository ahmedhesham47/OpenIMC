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
from typing import Tuple, Optional, Dict, List
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt
from matplotlib.patches import Circle
import os
import json


def region_passes_and_contribs(
    crater_radius=0.5,
    step=0.333,
    num_rows=25,
    num_cols=25,
    x0=7.0,
    I0=1.0,
    grid_res_pass=800,
    grid_res_full=1000,
    min_area_pixels_full=150,
    half=4,
    rng_pass_seed=42,
    rng_full_seed=123,
):
    """
    Returns:
      passes_arr: np.ndarray, passes per region (polygon)
      contribs_arr: np.ndarray, areas per region (polygon)
      kernel_dim: int, size of kernel
      region_data_full: list, dicts for each region (cell_row/cell_col fields used in kernel function)
      Dict of spatial grid/plotting info for use by a separate plotting function
    """
    diameter = 2.0 * crater_radius

    # GLOBAL GRID
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

    # PASS MAP
    xs_rel_pass = np.linspace(-crater_radius, crater_radius, grid_res_pass)
    ys_rel_pass = np.linspace(-crater_radius, crater_radius, grid_res_pass)
    XX_rel_pass, YY_rel_pass = np.meshgrid(xs_rel_pass, ys_rel_pass)

    XX_pass = XX_rel_pass + central_center[0]
    YY_pass = YY_rel_pass + central_center[1]
    points_pass = np.c_[XX_pass.ravel(), YY_pass.ravel()]

    inside_mask_pass = np.hypot(XX_rel_pass, YY_rel_pass) <= crater_radius
    inside_idx_pass = np.where(inside_mask_pass.ravel())[0]
    points_inside_pass = points_pass[inside_idx_pass]

    effective_shots = shot_centers[:central_idx + 1].copy()
    dists_from_center = np.hypot(effective_shots[:, 0] - central_center[0],
                                 effective_shots[:, 1] - central_center[1])
    rel_shots_pass = effective_shots[dists_from_center <= (2.1 * crater_radius)]

    rng = np.random.default_rng(rng_pass_seed)
    shot_signatures_pass = rng.uniform(100, 1000000, size=len(rel_shots_pass))

    dists_matrix_pass = np.sqrt(((rel_shots_pass[None, :, :] - points_inside_pass[:, None, :])**2).sum(axis=2))
    boolean_overlaps_pass = dists_matrix_pass <= crater_radius
    pass_values_inside = np.sum(boolean_overlaps_pass, axis=1)
    signature_values_inside = boolean_overlaps_pass @ shot_signatures_pass

    pass_map = np.zeros(grid_res_pass * grid_res_pass, dtype=int)
    pass_map[inside_idx_pass] = pass_values_inside
    pass_map = pass_map.reshape(grid_res_pass, grid_res_pass)

    sig_map_pass = np.zeros(grid_res_pass * grid_res_pass, dtype=float)
    sig_map_pass[inside_idx_pass] = signature_values_inside
    sig_map_pass = sig_map_pass.reshape(grid_res_pass, grid_res_pass)

    # Helper for region pass assignment
    def sample_pass_from_map(x_rel, y_rel):
        if x_rel < -crater_radius or x_rel > crater_radius:
            return 0
        if y_rel < -crater_radius or y_rel > crater_radius:
            return 0
        ix = np.searchsorted(xs_rel_pass, x_rel)
        iy = np.searchsorted(ys_rel_pass, y_rel)
        ix = np.clip(ix, 0, grid_res_pass - 1)
        iy = np.clip(iy, 0, grid_res_pass - 1)
        return int(pass_map[iy, ix])

    # FULL RASTER POLYGONS
    rows_local = range(central_row - half, central_row + half + 1)
    cols_local = range(central_col - half, central_col + half + 1)
    centers_full = []
    for r in rows_local:
        for c in cols_local:
            idx = r * num_cols + c
            centers_full.append(shot_centers[idx])
    centers_full = np.array(centers_full, dtype=float)

    padding = crater_radius * 0.1
    x_min = central_center[0] - crater_radius - padding
    x_max = central_center[0] + crater_radius + padding
    y_min = central_center[1] - crater_radius - padding
    y_max = central_center[1] + crater_radius + padding

    x_grid_full = np.linspace(x_min, x_max, grid_res_full)
    y_grid_full = np.linspace(y_min, y_max, grid_res_full)
    X_full, Y_full = np.meshgrid(x_grid_full, y_grid_full)

    pixel_area_full = (x_max - x_min) * (y_max - y_min) / (grid_res_full * grid_res_full)
    target_mask_full = (X_full - central_center[0])**2 + (Y_full - central_center[1])**2 <= crater_radius**2

    # KERNEL DIM/GEO (d/step)
    kernel_dim_raw = diameter / step
    kernel_dim = max(int(np.round(kernel_dim_raw)), 1)
    edges = np.linspace(-crater_radius, crater_radius, kernel_dim + 1)
    eps = 1e-9

    # SIGNATURE MAP (FULL RASTER)
    sig_map_full = np.zeros_like(X_full, dtype=float)
    rng_full = np.random.default_rng(rng_full_seed)
    shot_signatures_full = rng_full.uniform(100, 1000000, size=len(centers_full))

    for i, (xc, yc) in enumerate(centers_full):
        dist_to_target = np.sqrt((xc - central_center[0])**2 + (yc - central_center[1])**2)
        if dist_to_target < 2.0 * crater_radius:
            mask = (X_full - xc)**2 + (Y_full - yc)**2 <= crater_radius**2
            sig_map_full += mask.astype(int) * shot_signatures_full[i]
    sig_map_full[~target_mask_full] = 0

    # SEGMENT REGIONS (POLYGONS)
    region_data_full = []
    unique_sigs_full = np.unique(sig_map_full[target_mask_full])
    for sig in unique_sigs_full:
        if sig == 0:
            continue
        binary = (sig_map_full == sig).astype(np.uint8)
        labeled = label(binary, connectivity=1)
        props = regionprops(labeled)
        for rp in props:
            if rp.area < min_area_pixels_full:
                continue
            coords = rp.coords
            rows_pix = coords[:, 0]
            cols_pix = coords[:, 1]
            x_pix = x_grid_full[cols_pix]
            y_pix = y_grid_full[rows_pix]
            x_rel_pix = x_pix - central_center[0]
            y_rel_pix = y_pix - central_center[1]
            on_x_grid = np.any(np.isclose(x_rel_pix[:, None], edges[1:-1], atol=eps), axis=1)
            on_y_grid = np.any(np.isclose(y_rel_pix[:, None], edges[1:-1], atol=eps), axis=1)
            if np.any(on_x_grid) or np.any(on_y_grid):
                continue
            col_idx_pix = np.searchsorted(edges, x_rel_pix, side='right') - 1
            row_idx_geom_pix = np.searchsorted(edges, y_rel_pix, side='right') - 1
            valid = (
                (col_idx_pix >= 0) & (col_idx_pix < kernel_dim) &
                (row_idx_geom_pix >= 0) & (row_idx_geom_pix < kernel_dim)
            )
            if not np.all(valid):
                continue
            row_idx_pix = kernel_dim - 1 - row_idx_geom_pix
            unique_cells = set(zip(row_idx_pix, col_idx_pix))
            if len(unique_cells) != 1:
                continue
            (cell_row, cell_col) = next(iter(unique_cells))
            min_row, min_col, max_row, max_col = rp.bbox
            sub_mask = binary[min_row:max_row, min_col:max_col]
            dist_transform = distance_transform_edt(sub_mask)
            max_idx = np.argmax(dist_transform)
            max_pos = np.unravel_index(max_idx, dist_transform.shape)
            cy = max_pos[0] + min_row
            cx = max_pos[1] + min_col
            y_phys_center = y_min + cy * (y_max - y_min) / grid_res_full
            x_phys_center = x_min + cx * (x_max - x_min) / grid_res_full
            x_rel_center = x_phys_center - central_center[0]
            y_rel_center = y_phys_center - central_center[1]
            area_um2 = rp.area * pixel_area_full
            region_data_full.append({
                'x': x_phys_center,
                'y': y_phys_center,
                'x_rel': x_rel_center,
                'y_rel': y_rel_center,
                'area': area_um2,
                'cell_row': cell_row,
                'cell_col': cell_col
            })

    # Assign passes from pass map
    for rd in region_data_full:
        rd['pass'] = sample_pass_from_map(rd['x_rel'], rd['y_rel'])

    passes_arr = np.array([rd['pass'] for rd in region_data_full], dtype=float)
    contribs_arr = np.array([rd['area'] for rd in region_data_full], dtype=float)

    # Gather plotting debug info to pass to plot function
    plot_data = {
        "inside_mask_pass": inside_mask_pass,
        "pass_map": pass_map,
        "crater_radius": crater_radius,
        "centers_full": centers_full,
        "central_center": central_center,
        "edges": edges,
        "region_data_full": region_data_full,
        "kernel_dim": kernel_dim,
        "x0": x0,
        "I0": I0,
    }

    return passes_arr, contribs_arr, kernel_dim, region_data_full, plot_data


def compute_region_kernel(passes_arr, contribs_arr, kernel_dim, region_data_full, x0=7.0, I0=1.0):
    """
    Compute RL deconvolution kernel per image channel.

    Returns: kernel (2D numpy array), region_counts (2D numpy array)
    """
    def inverse_sigmoid(p, x0=7.0, I0=1.0):
        p = np.asarray(p, dtype=float)
        return I0 - I0 / (1.0 + np.exp(-(p - x0)))

    kernel = np.zeros((kernel_dim, kernel_dim), dtype=float)
    region_counts = np.zeros((kernel_dim, kernel_dim), dtype=int)

    if len(passes_arr) == 0:
        return kernel, region_counts

    Contributions = contribs_arr / contribs_arr.sum()
    Passes_att = inverse_sigmoid(passes_arr, x0=x0, I0=I0)
    y_array = Passes_att * Contributions
    total_sum = np.sum(y_array)

    for rd, y_val in zip(region_data_full, y_array):
        row_idx = rd['cell_row']
        col_idx = rd['cell_col']
        kernel[row_idx, col_idx] += y_val / total_sum
        region_counts[row_idx, col_idx] += 1

    if kernel.sum() > 0:
        kernel /= kernel.sum()

    return kernel, region_counts


def plot_region_results(plot_data, kernel=None, region_counts=None):
    """
    Plot the pass map, full raster geometry, and kernel heatmap.

    Pass kernel and region_counts computed from compute_region_kernel, if desired.
    """
    import matplotlib.pyplot as plt

    inside_mask_pass = plot_data["inside_mask_pass"]
    pass_map = plot_data["pass_map"]
    crater_radius = plot_data["crater_radius"]
    centers_full = plot_data["centers_full"]
    central_center = plot_data["central_center"]
    edges = plot_data["edges"]
    region_data_full = plot_data["region_data_full"]
    kernel_dim = plot_data["kernel_dim"]

    # (1) Pass map
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    masked_pass = np.ma.masked_where(~inside_mask_pass, pass_map)
    im = ax.imshow(
        masked_pass,
        origin='lower',
        extent=[-crater_radius, crater_radius,
                -crater_radius, crater_radius]
    )
    plt.colorbar(im, ax=ax, label='Pass count')
    ax.set_aspect('equal')
    ax.set_xlabel('x (µm, crater-centered)')
    ax.set_ylabel('y (µm, crater-centered)')
    ax.set_title('Pass map (reference)')
    plt.tight_layout()
    plt.show()

    # (2) Full raster + kernel grid + polygon poles
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    main_circle = Circle((0, 0), crater_radius, fill=False, linewidth=2.0, color='black', zorder=20)
    ax.add_patch(main_circle)
    for xc, yc in centers_full:
        xr = xc - central_center[0]
        yr = yc - central_center[1]
        if (xr**2 + yr**2) <= (2.1 * crater_radius)**2:
            circ = Circle((xr, yr), crater_radius, fill=False, linewidth=0.8, alpha=0.3, color='#888888')
            ax.add_patch(circ)
    for e in edges:
        ax.axvline(e, color='blue', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.axhline(e, color='blue', linestyle='--', linewidth=0.8, alpha=0.7)
    sc = ax.scatter(
        [rd['x_rel'] for rd in region_data_full],
        [rd['y_rel'] for rd in region_data_full],
        c=[rd['pass'] for rd in region_data_full],
        cmap='viridis',
        s=10,
        edgecolors='k',
        lw=0.2,
        zorder=30
    )
    plt.colorbar(sc, ax=ax, label='Pass (from MAP)')
    ax.set_aspect('equal')
    ax.set_xlim(-crater_radius * 1.05, crater_radius * 1.05)
    ax.set_ylim(-crater_radius * 1.05, crater_radius * 1.05)
    ax.set_xlabel('x (µm, crater-centered)')
    ax.set_ylabel('y (µm, crater-centered)')
    ax.set_title('Full raster circle geometries + kernel grid + polygon poles')
    plt.tight_layout()
    plt.show()

    # (3) Kernel heatmap
    if kernel is not None:
        fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
        im = ax.imshow(kernel, cmap='gray', interpolation='nearest')
        plt.colorbar(im, ax=ax)
        ax.set_title(f'Kernel ({kernel_dim}x{kernel_dim})')
        ax.set_xticks(range(kernel_dim))
        ax.set_yticks(range(kernel_dim))
        ax.set_xlabel('Kernel X')
        ax.set_ylabel('Kernel Y')
        plt.tight_layout()
        plt.show()


def save_region_data(
    passes_arr: np.ndarray,
    contribs_arr: np.ndarray,
    kernel_dim: int,
    region_data_full: List[Dict],
    output_dir: str,
    x0: float = 7.0,
    I0: float = 1.0,
    crater_radius: float = 0.5,
    step: float = 0.333,
    metadata: Optional[Dict] = None
) -> Tuple[str, str]:
    """
    Save passes_arr, contribs_arr, kernel_dim, region_data_full, and sigmoidal loss parameters to a directory.
    
    Args:
        passes_arr: Array of passes per region
        contribs_arr: Array of contributions (areas) per region
        kernel_dim: Size of kernel
        region_data_full: List of dicts with region information
        output_dir: Directory to save files to
        x0: Sigmoidal loss parameter x0
        I0: Sigmoidal loss parameter I0
        crater_radius: Crater radius used
        step: Step size used
        metadata: Optional additional metadata to save
    
    Returns:
        Tuple of (npz_file_path, json_file_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save arrays and region data to NPZ file
    npz_path = os.path.join(output_dir, "region_data.npz")
    # Convert region_data_full list to numpy object array for saving
    # This ensures it can be properly loaded back
    region_data_array = np.array([region_data_full], dtype=object) if region_data_full else np.array([], dtype=object)
    save_dict = {
        'passes_arr': passes_arr,
        'contribs_arr': contribs_arr,
        'kernel_dim': np.array([kernel_dim], dtype=int),
        'region_data_full': region_data_array,  # Save as object array containing the list
    }
    np.savez(npz_path, **save_dict)
    
    # Save sigmoidal loss parameters and metadata to JSON file
    json_path = os.path.join(output_dir, "sigmoidal_loss_params.json")
    json_data = {
        'x0': float(x0),
        'I0': float(I0),
        'crater_radius': float(crater_radius),
        'step': float(step),
    }
    if metadata:
        json_data['metadata'] = metadata
    
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    return npz_path, json_path


def load_region_data(npz_path: str, json_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, int, List[Dict], Dict]:
    """
    Load passes_arr, contribs_arr, kernel_dim, region_data_full, and sigmoidal loss parameters.
    
    Args:
        npz_path: Path to NPZ file with arrays and region data
        json_path: Optional path to JSON file with sigmoidal loss parameters. If None, looks for json file in same directory.
    
    Returns:
        Tuple of (passes_arr, contribs_arr, kernel_dim, region_data_full, sigmoidal_params_dict)
    """
    # Load NPZ file
    data = np.load(npz_path, allow_pickle=True)
    passes_arr = data['passes_arr']
    contribs_arr = data['contribs_arr']
    kernel_dim = int(data['kernel_dim'][0])
    
    # Handle region_data_full - it's saved as an object array containing a list
    region_data_full = data['region_data_full']
    if isinstance(region_data_full, np.ndarray):
        # It's saved as a 1-d object array containing the list, so extract the first (and only) element
        if region_data_full.size > 0:
            region_data_full = region_data_full.item(0) if region_data_full.ndim == 0 else region_data_full[0]
        else:
            region_data_full = []
    
    # Ensure it's a list
    if not isinstance(region_data_full, list):
        # If it's somehow still not a list, try to convert
        if isinstance(region_data_full, dict):
            region_data_full = [region_data_full]
        else:
            try:
                region_data_full = list(region_data_full)
            except (TypeError, ValueError):
                raise ValueError(f"Could not convert region_data_full to list. Type: {type(region_data_full)}, value: {region_data_full}")
    
    # Load JSON file
    if json_path is None:
        # Look for json file in same directory
        json_path = os.path.join(os.path.dirname(npz_path), "sigmoidal_loss_params.json")
    
    sigmoidal_params = {}
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            sigmoidal_params = json.load(f)
    
    return passes_arr, contribs_arr, kernel_dim, region_data_full, sigmoidal_params


# Backward compatibility functions
def compute_hrimc_passes_contributions(
    step_size_um: float,
    loss_fn=None,
    pixel_size_um: float = 1.0,
    spot_diameter_um: float = 1.0,
    n_subpixels: int = 15,
    circle_resolution: int = 64,
    grid_resolution: int = 800,
    x0: float = 7.0,
    I0: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Backward compatibility wrapper for compute_hrimc_passes_contributions.
    
    This function calls region_passes_and_contribs and then compute_region_kernel
    to generate passes, contributions, and a kernel.
    
    Returns:
        passes_arr, contribs_arr, kernel (3x3 or kernel_dim x kernel_dim)
    """
    crater_radius = spot_diameter_um / 2.0
    step = step_size_um
    
    # Determine grid size
    max_distance = 2.1 * crater_radius
    num_rows = int(np.ceil(max_distance / step) * 2) + 1
    num_cols = num_rows
    
    # Call new function
    passes_arr, contribs_arr, kernel_dim, region_data_full, plot_data = region_passes_and_contribs(
        crater_radius=crater_radius,
        step=step,
        num_rows=num_rows,
        num_cols=num_cols,
        x0=x0,
        I0=I0,
        grid_res_pass=grid_resolution,
        grid_res_full=1000,
        min_area_pixels_full=150,
        half=4,
        rng_pass_seed=42,
        rng_full_seed=123,
    )
    
    # Compute kernel
    kernel, _ = compute_region_kernel(
        passes_arr, contribs_arr, kernel_dim, region_data_full, x0=x0, I0=I0
    )
    
    return passes_arr, contribs_arr, kernel


def save_passes_contributions(
    passes: np.ndarray,
    contributions: np.ndarray,
    file_path: str,
    metadata: Optional[dict] = None,
    psf_kernel: Optional[np.ndarray] = None
) -> None:
    """
    Backward compatibility function for saving passes and contributions.
    
    This saves in the old format for compatibility with existing code.
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
    Backward compatibility function for loading passes and contributions.
    
    This loads in the old format for compatibility with existing code.
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
