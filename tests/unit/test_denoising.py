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
Unit tests for denoising functions.
"""
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
from PyQt5 import QtWidgets

from openimc.processing.denoising import apply_channel_denoise
from openimc.processing.export_worker import process_channel_for_export
from openimc.processing.feature_worker import _apply_denoise_to_channel
from openimc.ui.main_window import MainWindow


@pytest.mark.unit
class TestDenoising:
    """Tests for denoising functions."""
    
    def test_denoise_no_settings(self, sample_image_2d):
        """Test that denoising returns original image when no settings provided."""
        result = _apply_denoise_to_channel(sample_image_2d, "test_channel", {})
        
        assert np.array_equal(result, sample_image_2d)
    
    def test_denoise_hot_pixel_median3(self, sample_image_2d):
        """Test hot pixel removal with median3 method."""
        denoise_settings = {
            'hot': {
                'method': 'median3',
                'n_sd': 5.0
            }
        }
        result = _apply_denoise_to_channel(sample_image_2d, "test_channel", denoise_settings)
        
        assert result.shape == sample_image_2d.shape
        assert result.dtype == sample_image_2d.dtype
    
    def test_denoise_speckle_gaussian(self, sample_image_2d):
        """Test speckle noise reduction with gaussian method."""
        denoise_settings = {
            'speckle': {
                'method': 'gaussian',
                'sigma': 0.8
            }
        }
        result = _apply_denoise_to_channel(sample_image_2d, "test_channel", denoise_settings)
        
        assert result.shape == sample_image_2d.shape
    
    def test_denoise_background_white_tophat(self, sample_image_2d):
        """Test background subtraction with white tophat."""
        denoise_settings = {
            'background': {
                'method': 'white_tophat',
                'radius': 15
            }
        }
        result = _apply_denoise_to_channel(sample_image_2d, "test_channel", denoise_settings)
        
        assert result.shape == sample_image_2d.shape
    
    def test_denoise_full_pipeline(self, sample_image_2d, sample_denoise_settings):
        """Test full denoising pipeline with all steps."""
        result = _apply_denoise_to_channel(sample_image_2d, "test_channel", sample_denoise_settings)
        
        assert result.shape == sample_image_2d.shape
    
    def test_denoise_partial_settings(self, sample_image_2d):
        """Test denoising with partial settings."""
        denoise_settings = {
            'hot': {
                'method': 'median3'
            }
        }
        result = _apply_denoise_to_channel(sample_image_2d, "test_channel", denoise_settings)
        
        assert result.shape == sample_image_2d.shape

    def test_n_sd_local_median_removes_isolated_hot_pixel(self):
        """The robust local-threshold hot-pixel mode should catch isolated spikes."""
        image = np.zeros((9, 9), dtype=np.uint16)
        image[4, 4] = 100
        denoise_settings = {
            'hot': {
                'method': 'n_sd_local_median',
                'n_sd': 3.0,
            }
        }

        result = _apply_denoise_to_channel(image, "test_channel", denoise_settings)

        assert result[4, 4] == 0
        assert np.count_nonzero(result) == 0

    def test_export_worker_matches_shared_denoise_helper(self, sample_image_2d):
        """Export processing should use the same denoise implementation as the shared helper."""
        denoise_settings = {
            'hot': {'method': 'median3'},
            'background': {'method': 'rolling_ball', 'radius': 3},
        }

        expected = apply_channel_denoise(sample_image_2d, denoise_settings)
        result = process_channel_for_export(
            sample_image_2d.copy(),
            "test_channel",
            "custom",
            {"test_channel": denoise_settings},
            "None",
            5.0,
            (1.0, 99.0),
            None,
        )

        assert np.array_equal(result, expected)

    def test_viewer_apply_all_channels_keeps_black_tophat_and_is_idempotent(self, qtbot, monkeypatch):
        """Repeated viewer apply-all clicks should overwrite config, not stack denoising passes."""
        monkeypatch.setattr("openimc.ui.main_window.QTimer.singleShot", lambda *_args, **_kwargs: None)

        stub = SimpleNamespace()
        stub.channel_list = QtWidgets.QListWidget()
        for channel in ("channel_a", "channel_b"):
            stub.channel_list.addItem(channel)

        stub.hot_pixel_chk = QtWidgets.QCheckBox()
        stub.hot_pixel_chk.setChecked(True)
        stub.hot_pixel_method_combo = QtWidgets.QComboBox()
        stub.hot_pixel_method_combo.addItems(["Median 3x3", ">N SD above local median"])
        stub.hot_pixel_method_combo.setCurrentIndex(0)
        stub.hot_pixel_n_spin = QtWidgets.QDoubleSpinBox()
        stub.hot_pixel_n_spin.setValue(5.0)

        stub.speckle_chk = QtWidgets.QCheckBox()
        stub.speckle_chk.setChecked(False)
        stub.speckle_method_combo = QtWidgets.QComboBox()
        stub.speckle_method_combo.addItems(["Gaussian", "Non-local means (slow)"])
        stub.gaussian_sigma_spin = QtWidgets.QDoubleSpinBox()
        stub.gaussian_sigma_spin.setValue(0.8)

        stub.bg_subtract_chk = QtWidgets.QCheckBox()
        stub.bg_subtract_chk.setChecked(True)
        stub.bg_method_combo = QtWidgets.QComboBox()
        stub.bg_method_combo.addItems(["White top-hat", "Black top-hat", "Rolling ball"])
        stub.bg_method_combo.setCurrentIndex(1)
        stub.bg_radius_spin = QtWidgets.QSpinBox()
        stub.bg_radius_spin.setValue(3)

        stub.order_combo_1 = QtWidgets.QComboBox()
        stub.order_combo_2 = QtWidgets.QComboBox()
        stub.order_combo_3 = QtWidgets.QComboBox()
        for combo, text in zip(
            (stub.order_combo_1, stub.order_combo_2, stub.order_combo_3),
            ("Hot pixel", "Speckle", "Background"),
        ):
            combo.addItems(["Hot pixel", "Speckle", "Background"])
            combo.setCurrentText(text)
        stub.step_names = ["Hot pixel", "Speckle", "Background"]

        stub.apply_all_channels_btn = QtWidgets.QPushButton()
        stub.channel_denoise = {}
        stub.denoise_enable_chk = QtWidgets.QCheckBox()
        stub.denoise_enable_chk.setChecked(True)
        stub.preserve_zoom = False
        stub._view_selected_calls = 0

        def _view_selected():
            stub._view_selected_calls += 1

        stub._view_selected = _view_selected
        stub._build_current_denoise_config = MethodType(MainWindow._build_current_denoise_config, stub)
        stub._get_denoise_step_order = MethodType(MainWindow._get_denoise_step_order, stub)
        stub._reset_apply_all_button = MethodType(MainWindow._reset_apply_all_button, stub)

        MainWindow._apply_denoise_to_all_channels(stub)
        raw = np.zeros((9, 9), dtype=np.uint16)
        raw[4, 4] = 100
        first = MainWindow._apply_denoise(stub, "channel_a", raw)

        MainWindow._apply_denoise_to_all_channels(stub)
        second = MainWindow._apply_denoise(stub, "channel_a", raw)

        assert stub.channel_denoise["channel_a"]["background"]["method"] == "black_tophat"
        assert stub.channel_denoise["channel_b"]["background"]["method"] == "black_tophat"
        assert np.array_equal(first, second)
        assert np.array_equal(raw, np.pad(np.array([[100]], dtype=np.uint16), 4))
        assert stub._view_selected_calls == 2
