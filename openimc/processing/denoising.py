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

"""Shared denoising helpers used across viewer, export, and processing paths."""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

try:
    from skimage import morphology
    from skimage.filters import gaussian, median
    from skimage.morphology import disk, footprint_rectangle
    from skimage.restoration import denoise_nl_means, estimate_sigma
    from scipy import ndimage as ndi

    try:
        from skimage.restoration import rolling_ball as _sk_rolling_ball  # type: ignore

        _HAVE_ROLLING_BALL = True
    except Exception:
        _HAVE_ROLLING_BALL = False

    _HAVE_SCIKIT_IMAGE = True
except ImportError:
    _HAVE_SCIKIT_IMAGE = False
    _HAVE_ROLLING_BALL = False


DEFAULT_STEP_ORDER: Tuple[str, ...] = ("hot", "speckle", "background")
BACKGROUND_METHODS: Tuple[str, ...] = ("white_tophat", "black_tophat", "rolling_ball")


def background_method_from_index(index: int) -> str:
    """Return the serialized background-method name for a combo-box index."""
    if index <= 0:
        return "white_tophat"
    if index == 1:
        return "black_tophat"
    return "rolling_ball"


def background_index_from_method(method: Optional[str]) -> int:
    """Return the combo-box index for a serialized background-method name."""
    if method == "black_tophat":
        return 1
    if method == "rolling_ball":
        return 2
    return 0


def apply_channel_denoise(
    channel_img: np.ndarray,
    denoise_settings: Optional[Dict],
    *,
    step_order: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Apply denoising to a single channel image.

    The image is always treated as raw input; callers should not feed the output
    back into this helper unless they intentionally want a second pass.
    """
    if not _HAVE_SCIKIT_IMAGE or not denoise_settings:
        return channel_img

    out = channel_img.astype(np.float32, copy=False)
    ordered_steps = tuple(step_order or DEFAULT_STEP_ORDER)

    for step in ordered_steps:
        if step == "hot":
            out = _apply_hot_pixel_step(out, denoise_settings.get("hot"))
        elif step == "speckle":
            out = _apply_speckle_step(out, denoise_settings.get("speckle"))
        elif step == "background":
            out = _apply_background_step(out, denoise_settings.get("background"))

    return _restore_output_dtype(channel_img, out)


def _apply_hot_pixel_step(channel_img: np.ndarray, hot_config: Optional[Dict]) -> np.ndarray:
    if not hot_config:
        return channel_img

    method = hot_config.get("method", "median3")
    if method == "median3":
        return _local_median_3x3(channel_img)

    if method != "n_sd_local_median":
        return channel_img

    n_sd = float(hot_config.get("n_sd", 5.0))
    neighbors = _neighbor_stack(channel_img)
    local_median = np.median(neighbors, axis=0)
    local_abs_dev = np.abs(neighbors - local_median)
    local_mad = np.median(local_abs_dev, axis=0)
    robust_scale = 1.4826 * local_mad
    neighbor_max = np.max(neighbors, axis=0)

    # Keep the threshold non-zero for perfectly flat neighborhoods while still
    # allowing isolated spikes to be corrected.
    min_scale = np.maximum(
        1e-6,
        np.finfo(np.float32).eps * np.maximum(1.0, np.abs(local_median)),
    )
    threshold = local_median + np.maximum(n_sd * robust_scale, min_scale)
    mask_hot = (channel_img > threshold) & (channel_img > neighbor_max)
    return np.where(mask_hot, local_median, channel_img)


def _apply_speckle_step(channel_img: np.ndarray, speckle_config: Optional[Dict]) -> np.ndarray:
    if not speckle_config:
        return channel_img

    method = speckle_config.get("method", "gaussian")
    sigma = float(speckle_config.get("sigma", 0.8))
    if method == "gaussian":
        return gaussian(channel_img, sigma=sigma, preserve_range=True)

    if method != "nl_means":
        return channel_img

    min_val = float(np.min(channel_img))
    max_val = float(np.max(channel_img))
    scale = max_val - min_val
    scaled = (channel_img - min_val) / scale if scale > 0 else channel_img
    sigma_est = float(np.mean(estimate_sigma(scaled, channel_axis=None)))
    smoothed = denoise_nl_means(
        scaled,
        h=1.15 * sigma_est,
        fast_mode=True,
        patch_size=5,
        patch_distance=6,
        channel_axis=None,
    )
    return smoothed * scale + min_val


def _apply_background_step(channel_img: np.ndarray, bg_config: Optional[Dict]) -> np.ndarray:
    if not bg_config:
        return channel_img

    method = bg_config.get("method", "white_tophat")
    radius = max(1, int(bg_config.get("radius", 15)))
    footprint = disk(radius)

    if method == "white_tophat":
        return _apply_morphology_op(morphology.white_tophat, channel_img, footprint)
    if method == "black_tophat":
        return _apply_morphology_op(morphology.black_tophat, channel_img, footprint)
    if method == "rolling_ball":
        if _HAVE_ROLLING_BALL:
            background = _sk_rolling_ball(channel_img, radius=radius)
        else:
            background = _apply_morphology_op(morphology.opening, channel_img, footprint)
        return np.clip(channel_img - background, 0, None)
    return channel_img


def _apply_morphology_op(op, image: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    try:
        return op(image, selem=footprint)
    except TypeError:
        return op(image, footprint=footprint)


def _local_median_3x3(image: np.ndarray) -> np.ndarray:
    try:
        return median(image, footprint=footprint_rectangle(3, 3).astype(bool))
    except Exception:
        return ndi.median_filter(image, size=3)


def _neighbor_stack(image: np.ndarray) -> np.ndarray:
    padded = np.pad(image.astype(np.float32, copy=False), 1, mode="reflect")
    height, width = image.shape
    neighbors = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neighbors.append(padded[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width])
    return np.stack(neighbors, axis=0)


def _restore_output_dtype(original_img: np.ndarray, filtered_img: np.ndarray) -> np.ndarray:
    out = filtered_img
    try:
        original_max = float(np.max(original_img))
        filtered_max = float(np.max(out))
        if filtered_max > 0 and original_max > 0:
            out = out * (original_max / filtered_max)
    except Exception:
        pass

    if np.issubdtype(original_img.dtype, np.integer):
        info = np.iinfo(original_img.dtype)
        out = np.clip(out, info.min, info.max)
    else:
        out = np.clip(out, 0, None)
    return out.astype(original_img.dtype, copy=False)
