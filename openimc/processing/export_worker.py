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
Worker functions for OME-TIFF export with multiprocessing support.
"""

from typing import Dict, Tuple
import numpy as np

from openimc.processing.denoising import apply_channel_denoise
from openimc.ui.utils import arcsinh_normalize, percentile_clip_normalize


def _apply_custom_denoise_to_channel(channel_img: np.ndarray, channel_name: str, 
                                     custom_denoise_settings: Dict) -> np.ndarray:
    """Apply custom denoise steps for a channel in raw domain.
    
    This is a module-level function that can be pickled for multiprocessing.
    """
    cfg = custom_denoise_settings.get(channel_name)
    return apply_channel_denoise(channel_img, cfg)


def process_channel_for_export(
    channel_img: np.ndarray,
    channel_name: str,
    denoise_source: str,
    custom_denoise_settings: Dict,
    normalization_method: str,
    arcsinh_cofactor: float,
    percentile_params: Tuple[float, float],
    viewer_denoise_func=None
) -> np.ndarray:
    """Process a single channel for export with denoising and normalization.
    
    This is a module-level function that can be pickled for multiprocessing.
    
    Args:
        channel_img: Raw channel image
        channel_name: Name of the channel
        denoise_source: "none", "viewer", or "custom"
        custom_denoise_settings: Dictionary of custom denoising settings per channel
        normalization_method: "None", "arcsinh", or "percentile_clip"
        arcsinh_cofactor: Cofactor for arcsinh normalization
        percentile_params: (low, high) percentiles for percentile clipping
        viewer_denoise_func: Function to apply viewer denoising (must be None for multiprocessing)
    
    Returns:
        Processed channel image
    """
    result = channel_img.copy()
    
    # Apply denoising
    if denoise_source == "viewer":
        # Note: viewer_denoise_func cannot be pickled, so this should be handled
        # in the main process before calling this function
        if viewer_denoise_func is not None:
            result = viewer_denoise_func(channel_name, result)
    elif denoise_source == "custom":
        result = _apply_custom_denoise_to_channel(result, channel_name, custom_denoise_settings)
    
    # Apply normalization
    # Note: arcsinh normalization is not applied to exported images.
    # Only denoising is applied. Arcsinh transform should be applied on extracted intensity features.
    if normalization_method == "percentile_clip":
        p_low, p_high = percentile_params
        result = percentile_clip_normalize(result, p_low=p_low, p_high=p_high)
    # Arcsinh normalization is intentionally not applied here - it should be applied on extracted features
    
    return result
