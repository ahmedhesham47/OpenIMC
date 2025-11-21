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

This module uses Shapely for geometry calculations to compute passes and contributions
given instrument geometry and an inverse sigmoidal loss function.
"""

import numpy as np
from typing import Callable, Tuple, Optional

try:
    from shapely.geometry import Point, Polygon
    _HAVE_SHAPELY = True
    _SHAPELY_ERROR = None
except (ImportError, AttributeError, Exception) as e:
    _HAVE_SHAPELY = False
    _SHAPELY_ERROR = str(e)


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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute subpixel Passes and Contributions for HR-IMC using Shapely, with
    a more exact raster-scan geometry that can yield up to ~9 passes for
    333 nm step size.

    Parameters
    ----------
    step_size_um : float
        HR-IMC step size in microns (e.g., 0.333 or 0.5).
    loss_fn : Callable[[np.ndarray], np.ndarray]
        Inverse sigmoidal loss function mapping pass numbers -> fraction of
        signal remaining after repeated ablations.
    pixel_size_um : float, default 1.0
        Coarse IMC pixel size (1 µm).
    spot_diameter_um : float, default 1.0
        Laser spot diameter (1 µm for HR-IMC).
    n_subpixels : int, default 15
        Number of subpixels per side inside one coarse pixel. Must be divisible
        by 3 to aggregate cleanly into a 3×3 PSF.
    circle_resolution : int, default 64
        Resolution used to approximate circles with polygons in Shapely.

    Returns
    -------
    passes_flat : np.ndarray, shape (n_subpixels * n_subpixels,)
        Pass count per subpixel for the pixel-of-interest. Subpixels outside
        the central beam have 0. Subpixels inside have pass >= 1.
    contributions_flat : np.ndarray, shape (n_subpixels * n_subpixels,)
        Normalized contribution weight per subpixel (sums to 1).
    psf_kernel_3x3 : np.ndarray, shape (3, 3)
        Skewed 3×3 PSF suitable for Richardson–Lucy deconvolution.
    """
    if not _HAVE_SHAPELY:
        error_msg = "shapely is required for exact geometry calculations. Install with: pip install shapely"
        if _SHAPELY_ERROR:
            error_msg += f"\nImport error: {_SHAPELY_ERROR}"
        raise RuntimeError(error_msg)
    
    # Ensure imports are available
    from shapely.geometry import Point, Polygon
    
    if n_subpixels < 3:
        raise ValueError("n_subpixels must be at least 3.")
    if n_subpixels % 3 != 0:
        raise ValueError("n_subpixels should be divisible by 3 for clean 3×3 binning (e.g., 15, 21, 30).")

    spot_radius = spot_diameter_um / 2.0
    pixel_half = pixel_size_um / 2.0
    subpixel_width = pixel_size_um / n_subpixels
    subpixel_area = subpixel_width ** 2

    # ------------------------------------------------------------------
    # 1. Build subpixel grid inside the pixel-of-interest
    #    Pixel spans [-0.5, +0.5] × [-0.5, +0.5], centered at (0,0).
    # ------------------------------------------------------------------
    xs = np.linspace(-pixel_half + subpixel_width / 2.0,
                     pixel_half - subpixel_width / 2.0,
                     n_subpixels)
    ys = np.linspace(-pixel_half + subpixel_width / 2.0,
                     pixel_half - subpixel_width / 2.0,
                     n_subpixels)
    xx, yy = np.meshgrid(xs, ys)  # (n_subpixels, n_subpixels)

    subpixel_polys = []
    for j in range(n_subpixels):
        for i in range(n_subpixels):
            x_center = xs[i]
            y_center = ys[j]
            x0 = x_center - subpixel_width / 2.0
            x1 = x_center + subpixel_width / 2.0
            y0 = y_center - subpixel_width / 2.0
            y1 = y_center + subpixel_width / 2.0
            poly = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
            subpixel_polys.append(poly)

    n_subpix = len(subpixel_polys)

    # ------------------------------------------------------------------
    # 2. Enumerate beam centers on the HR grid.
    #
    # Condition for a beam at (i*step, j*step) to possibly intersect the
    # central pixel [-0.5, 0.5]^2 with a circle of radius R:
    #
    #   minimal distance from (i*step, j*step) to pixel <= R
    #
    # A conservative bound is |i| * step <= pixel_half + spot_radius.
    # For 1 µm pixel, 1 µm spot, that's |i| * step <= 1.0.
    # ------------------------------------------------------------------
    max_index = int(np.floor((pixel_half + spot_radius) / step_size_um))
    if max_index < 0:
        raise RuntimeError("max_index < 0; check step_size_um / geometry.")

    # Raster order: rows from top to bottom (higher y to lower y),
    # columns left to right (lower x to higher x).
    beam_centers = []
    for j_idx in range(max_index, -max_index - 1, -1):  # e.g. 3,2,1,0,-1,-2,-3
        for i_idx in range(-max_index, max_index + 1):  # e.g. -3,-2,-1,0,1,2,3
            x = i_idx * step_size_um
            y = j_idx * step_size_um
            beam_centers.append((x, y))
    beam_centers = np.array(beam_centers)  # shape (N_beams, 2)

    # Build Shapely circles for each beam
    beam_disks = [
        Point(bx, by).buffer(spot_radius, resolution=circle_resolution)
        for (bx, by) in beam_centers
    ]

    # Identify the central shot at (0,0); this is the acquisition that produces
    # the pixel-of-interest. All beams with index < central_idx in the raster
    # are "previous passes" for regions they overlap.
    central_idx = None
    for idx, (bx, by) in enumerate(beam_centers):
        if np.isclose(bx, 0.0) and np.isclose(by, 0.0):
            central_idx = idx
            break
    if central_idx is None:
        raise RuntimeError("Central beam (0,0) not found in beam_centers.")

    # ------------------------------------------------------------------
    # 3. For each subpixel:
    #    - count how many earlier beams (index < central_idx) overlap it
    #    - compute overlap area with the central beam only
    #
    # passes = (# of overlapping beams before central) + 1, only where the
    # central beam hits. Subpixels outside the central beam get passes = 0.
    # ------------------------------------------------------------------
    previous_passes = np.zeros(n_subpix, dtype=int)
    central_overlap_area = np.zeros(n_subpix, dtype=float)

    central_disk = beam_disks[central_idx]

    for sub_idx, subpoly in enumerate(subpixel_polys):
        # Overlap with central beam
        inter_central = subpoly.intersection(central_disk).area
        central_overlap_area[sub_idx] = inter_central

        if inter_central <= 0.0:
            # This subpixel is not part of the pixel-of-interest footprint,
            # so it cannot contribute to the current shot.
            continue

        # Count previous passes (beams before central that also overlap this subpixel)
        n_prev = 0
        for b_idx in range(central_idx):
            inter_area = subpoly.intersection(beam_disks[b_idx]).area
            if inter_area > 0.0:
                n_prev += 1

        previous_passes[sub_idx] = n_prev

    passes_flat = np.zeros(n_subpix, dtype=int)
    mask_central = central_overlap_area > 0.0
    passes_flat[mask_central] = previous_passes[mask_central] + 1  # include current shot

    # ------------------------------------------------------------------
    # 4. Convert central overlap area to area fraction and apply loss_fn
    # ------------------------------------------------------------------
    area_fraction = central_overlap_area / subpixel_area

    # Only subpixels inside central beam can contribute
    passes_nonzero = passes_flat[mask_central]
    if passes_nonzero.size == 0:
        raise RuntimeError("Central beam does not intersect any subpixels. Check geometry.")

    frac_remaining = loss_fn(passes_nonzero)

    raw_contrib = np.zeros(n_subpix, dtype=float)
    raw_contrib[mask_central] = area_fraction[mask_central] * frac_remaining

    total_raw = raw_contrib.sum()
    if total_raw <= 0.0:
        raise RuntimeError(
            "Total raw contribution is zero after loss function. "
            "Check step_size_um, spot_diameter_um, and loss_fn."
        )

    contributions_flat = raw_contrib / total_raw  # normalize to sum to 1

    # ------------------------------------------------------------------
    # 5. Aggregate into skewed 3×3 PSF by binning the n_subpixels×n_subpixels grid
    #    Note: In compute_hrimc_passes_contributions, ys goes from -pixel_half to +pixel_half,
    #    so j=0 is bottom and j=n-1 is top. We need to flip vertically.
    # ------------------------------------------------------------------
    psf_kernel = np.zeros((3, 3), dtype=float)
    bin_size = n_subpixels // 3

    contrib_2d = contributions_flat.reshape(n_subpixels, n_subpixels)
    # Flip vertically: contrib_2d[0, :] (bottom in image coords) should map to PSF row 2 (bottom)
    # After flip, flipped[0, :] corresponds to the top row of subpixels
    contrib_2d_flipped = np.flipud(contrib_2d)  # Flip vertically
    for j in range(n_subpixels):
        for i in range(n_subpixels):
            by = min(j // bin_size, 2)  # 0 = top, 2 = bottom
            bx = min(i // bin_size, 2)  # 0 = left, 2 = right
            psf_kernel[by, bx] += contrib_2d_flipped[j, i]

    psf_sum = psf_kernel.sum()
    if psf_sum > 0.0:
        psf_kernel /= psf_sum

    return passes_flat, contributions_flat, psf_kernel


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

    Geometry:
    - Pixel-of-interest: a 1 µm × 1 µm square centered at (0, 0).
    - Laser spot: circle with diameter spot_diameter_um (default 1 µm).
    - HR grid: beams at (i * step_size_um, j * step_size_um) in a raster
      ordered top-to-bottom, left-to-right.
    - Only beams whose circles can intersect the pixel-of-interest are considered.
    - The beam at (0,0) is treated as the current shot; all beams with
      earlier raster indices are "previous passes".

    The pixel is discretized into n_subpixels × n_subpixels equal squares,
    each treated as a subpixel. For each subpixel:
      - passes = (# of overlapping beams before current) + 1
      - central_overlap_area = area overlapped by the current beam's disk
      - contribution ∝ central_overlap_area * loss_fn(passes, x0, slope, ...)

    Contributions are normalized to sum to 1, then aggregated into a 3×3 PSF
    by binning the n_subpixels × n_subpixels grid.

    Parameters
    ----------
    step_size_um : float
        HR-IMC step size in microns (e.g. 0.333 or 0.5).
    x0 : float, default 7.0
        Midpoint of the inverse sigmoidal loss curve (in pass units).
    slope : float, default 1.0
        Slope of the inverse sigmoidal loss curve.
    pixel_size_um : float, default 1.0
        Size of a coarse IMC pixel (1 µm for HR-IMC).
    spot_diameter_um : float, default 1.0
        Laser spot diameter (1 µm).
    n_subpixels : int, default 15
        Number of subpixels along each side. Must be divisible by 3 to
        aggregate cleanly into a 3×3 PSF.
    circle_resolution : int, default 64
        Number of segments used to approximate the circular beam footprint.
    loss_fn : callable, optional
        Function with signature loss_fn(passes, x0, slope, min_fraction, max_fraction)
        returning fraction of signal remaining. If None, uses hr_inverse_sigmoid_loss.

    Returns
    -------
    passes_flat : np.ndarray, shape (n_subpixels * n_subpixels,)
        Pass count per subpixel for the pixel-of-interest. Subpixels outside
        the current beam footprint have 0.
    contributions_flat : np.ndarray, shape (n_subpixels * n_subpixels,)
        Normalized contribution per subpixel (sums to 1).
    psf_kernel_3x3 : np.ndarray, shape (3, 3)
        3×3 PSF kernel suitable for RL deconvolution. Row 0 corresponds to
        the "top" of the pixel, row 2 to the "bottom".
    """
    if loss_fn is None:
        loss_fn = hr_inverse_sigmoid_loss
    
    if not _HAVE_SHAPELY:
        error_msg = "shapely is required for exact geometry calculations. Install with: pip install shapely"
        if _SHAPELY_ERROR:
            error_msg += f"\nImport error: {_SHAPELY_ERROR}"
        raise RuntimeError(error_msg)
    
    # Ensure imports are available
    from shapely.geometry import Point, Polygon
    
    if n_subpixels < 3:
        raise ValueError("n_subpixels must be at least 3.")
    if n_subpixels % 3 != 0:
        raise ValueError("n_subpixels should be divisible by 3 for clean 3×3 binning (e.g., 15, 21, 30).")

    spot_radius = spot_diameter_um / 2.0
    pixel_half = pixel_size_um / 2.0
    subpixel_width = pixel_size_um / n_subpixels
    subpixel_area = subpixel_width ** 2

    # ------------------------------------------------------------------
    # 1. Subpixel grid inside the pixel-of-interest
    #
    # We want: row index 0 = "top" of the pixel, row index n_subpixels-1 = "bottom".
    # So we define ys from +pixel_half down to -pixel_half.
    # ------------------------------------------------------------------
    xs = np.linspace(-pixel_half + subpixel_width / 2.0,
                     pixel_half - subpixel_width / 2.0,
                     n_subpixels)
    ys = np.linspace(pixel_half - subpixel_width / 2.0,
                     -pixel_half + subpixel_width / 2.0,
                     n_subpixels)
    xx, yy = np.meshgrid(xs, ys)  # (n_subpixels, n_subpixels)

    subpixel_polys = []
    for j in range(n_subpixels):
        for i in range(n_subpixels):
            x_center = xx[j, i]
            y_center = yy[j, i]
            x0_coord = x_center - subpixel_width / 2.0
            x1_coord = x_center + subpixel_width / 2.0
            y0_coord = y_center - subpixel_width / 2.0
            y1_coord = y_center + subpixel_width / 2.0
            poly = Polygon([(x0_coord, y0_coord), (x1_coord, y0_coord), (x1_coord, y1_coord), (x0_coord, y1_coord)])
            subpixel_polys.append(poly)

    n_subpix = len(subpixel_polys)

    # ------------------------------------------------------------------
    # 2. Enumerate beam centers in a raster grid around the pixel-of-interest.
    #
    # Condition: beams with centers (i*step, j*step) where |i|*step or |j|*step
    # is small enough that a 1 µm disk can intersect the pixel [-0.5,0.5]^2.
    # Conservative bound: |i|*step <= pixel_half + spot_radius.
    # ------------------------------------------------------------------
    max_index = int(np.floor((pixel_half + spot_radius) / step_size_um))
    if max_index < 0:
        raise RuntimeError("max_index < 0; check step_size_um / geometry.")

    beam_centers = []
    # Raster order: top row (highest y) to bottom row (lowest y),
    # left to right within each row.
    for j_idx in range(max_index, -max_index - 1, -1):  # e.g., 3,2,1,0,-1,-2,-3
        for i_idx in range(-max_index, max_index + 1):  # e.g., -3,-2,-1,0,1,2,3
            x = i_idx * step_size_um
            y = j_idx * step_size_um
            beam_centers.append((x, y))
    beam_centers = np.array(beam_centers)  # shape (N_beams, 2)

    beam_disks = [Point(bx, by).buffer(spot_radius, resolution=circle_resolution)
                  for (bx, by) in beam_centers]

    # Identify central beam (current shot) at (0,0)
    central_idx = None
    for idx, (bx, by) in enumerate(beam_centers):
        if np.isclose(bx, 0.0) and np.isclose(by, 0.0):
            central_idx = idx
            break
    if central_idx is None:
        raise RuntimeError("Central beam (0,0) not found in beam_centers.")

    central_disk = beam_disks[central_idx]

    # ------------------------------------------------------------------
    # 3. For each subpixel:
    #    - central_overlap_area = area overlapped by current beam
    #    - previous_passes = number of earlier beams overlapping this subpixel
    #
    # passes_flat = previous_passes + 1 for subpixels hit by the current beam,
    # otherwise 0 outside the beam footprint.
    # ------------------------------------------------------------------
    central_overlap_area = np.zeros(n_subpix, dtype=float)
    previous_passes = np.zeros(n_subpix, dtype=int)

    for sub_idx, subpoly in enumerate(subpixel_polys):
        inter_central = subpoly.intersection(central_disk).area
        central_overlap_area[sub_idx] = inter_central

        if inter_central <= 0.0:
            continue

        # Count previous overlapping beams
        n_prev = 0
        for b_idx in range(central_idx):
            if subpoly.intersection(beam_disks[b_idx]).area > 0.0:
                n_prev += 1
        previous_passes[sub_idx] = n_prev

    passes_flat = np.zeros(n_subpix, dtype=int)
    mask_central = central_overlap_area > 0.0
    passes_flat[mask_central] = previous_passes[mask_central] + 1

    # ------------------------------------------------------------------
    # 4. Compute contributions: area_fraction × inverse_sigmoid(passes)
    # ------------------------------------------------------------------
    area_fraction = central_overlap_area / subpixel_area

    if not np.any(mask_central):
        raise RuntimeError("Central beam does not intersect any subpixels. Check geometry.")

    passes_nonzero = passes_flat[mask_central]
    frac_remaining = loss_fn(passes_nonzero, x0=x0, slope=slope,
                             min_fraction=0.0, max_fraction=1.0)

    raw_contrib = np.zeros(n_subpix, dtype=float)
    raw_contrib[mask_central] = area_fraction[mask_central] * frac_remaining

    total_raw = raw_contrib.sum()
    if total_raw <= 0.0:
        raise RuntimeError(
            "Total raw contribution is zero after applying loss function. "
            "Check step_size_um, x0, slope, and geometry."
        )

    contributions_flat = raw_contrib / total_raw

    # ------------------------------------------------------------------
    # 5. Aggregate into 3×3 PSF by binning the n_subpixels×n_subpixels grid.
    #    Row 0 is "top" (first entries in ys), row 2 is "bottom".
    #    Note: ys goes from top (positive) to bottom (negative), so j=0 is top.
    #    When reshaping, contrib_2d[0, :] is top row, contrib_2d[n-1, :] is bottom row.
    #    We need to flip vertically so PSF row 0 (top) gets top subpixels.
    # ------------------------------------------------------------------
    psf_kernel = np.zeros((3, 3), dtype=float)
    bin_size = n_subpixels // 3

    contrib_2d = contributions_flat.reshape(n_subpixels, n_subpixels)
    # Flip vertically: contrib_2d[0, :] (top in image coords) should map to PSF row 0 (top)
    # After flip, flipped[0, :] corresponds to the top row of subpixels
    contrib_2d_flipped = np.flipud(contrib_2d)  # Flip vertically
    for j in range(n_subpixels):
        for i in range(n_subpixels):
            by = min(j // bin_size, 2)  # 0 = top, 2 = bottom
            bx = min(i // bin_size, 2)  # 0 = left, 2 = right
            psf_kernel[by, bx] += contrib_2d_flipped[j, i]

    psf_kernel /= psf_kernel.sum()

    return passes_flat, contributions_flat, psf_kernel


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
