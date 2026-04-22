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

import numpy as np
import pytest
from PyQt5 import QtWidgets

from openimc.data.mcd_loader import AcquisitionInfo
from openimc.ui.dialogs.feature_extraction import FeatureExtractionDialog


class _Loader:
    def __init__(self, channels):
        self._channels = list(channels)

    def get_channels(self, _acq_id):
        return list(self._channels)


class _FeatureParent(QtWidgets.QWidget):
    def __init__(self, channels):
        super().__init__()
        self.loader = _Loader(channels)

    def _get_loader_for_acquisition(self, _acq_id):
        return self.loader

    def _get_original_acq_id(self, acq_id):
        return acq_id


def _build_dialog(qtbot):
    channels = ["CD3", "CD20", "DNA"]
    acquisition = AcquisitionInfo(
        id="ROI_1",
        name="ROI 1",
        well="A1",
        size=(24, 24),
        channels=channels,
        channel_metals=[""] * len(channels),
        channel_labels=[""] * len(channels),
        metadata={},
        source_file="/tmp/panel_a.mcd",
    )
    parent = _FeatureParent(channels)
    segmentation_masks = {"ROI_1": np.ones((24, 24), dtype=np.uint32)}
    qtbot.addWidget(parent)

    dialog = FeatureExtractionDialog(parent, [acquisition], segmentation_masks)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.wait(50)
    return dialog, parent


@pytest.mark.unit
def test_feature_extraction_bulk_select_buttons_toggle_both_feature_groups(qtbot):
    dialog, _parent = _build_dialog(qtbot)

    dialog.select_none_morph_btn.click()
    dialog.select_none_intensity_btn.click()
    assert not any(checkbox.isChecked() for checkbox in dialog.morph_features.values())
    assert not any(checkbox.isChecked() for checkbox in dialog.intensity_features.values())

    dialog.select_all_morph_btn.click()
    dialog.select_all_intensity_btn.click()
    assert all(checkbox.isChecked() for checkbox in dialog.morph_features.values())
    assert all(checkbox.isChecked() for checkbox in dialog.intensity_features.values())


@pytest.mark.unit
def test_feature_extraction_custom_denoise_settings_survive_channel_switches(qtbot):
    dialog, _parent = _build_dialog(qtbot)
    dialog.denoise_source_combo.setCurrentText("Custom")
    qtbot.wait(20)

    dialog.denoise_channel_combo.setCurrentText("CD3")
    dialog.hot_pixel_chk.setChecked(True)
    dialog.hot_pixel_method_combo.setCurrentIndex(1)
    dialog.hot_pixel_n_spin.setValue(4.5)
    dialog.bg_subtract_chk.setChecked(True)
    dialog.bg_method_combo.setCurrentIndex(1)
    dialog.bg_radius_spin.setValue(7)

    dialog.denoise_channel_combo.setCurrentText("CD20")
    dialog.hot_pixel_chk.setChecked(False)
    dialog.bg_subtract_chk.setChecked(True)
    dialog.bg_method_combo.setCurrentIndex(2)
    dialog.bg_radius_spin.setValue(3)

    dialog.denoise_channel_combo.setCurrentText("CD3")
    qtbot.wait(20)

    assert dialog.hot_pixel_chk.isChecked()
    assert dialog.hot_pixel_method_combo.currentIndex() == 1
    assert dialog.hot_pixel_n_spin.value() == pytest.approx(4.5)
    assert dialog.bg_subtract_chk.isChecked()
    assert dialog.bg_method_combo.currentIndex() == 1
    assert dialog.bg_radius_spin.value() == 7

    settings = dialog.get_custom_denoise_settings()
    assert settings["CD3"]["hot"]["method"] == "n_sd_local_median"
    assert settings["CD3"]["background"]["method"] == "black_tophat"
    assert settings["CD20"]["background"]["method"] == "rolling_ball"


@pytest.mark.unit
def test_feature_extraction_dialog_uses_scroll_area_on_small_windows(qtbot):
    dialog, _parent = _build_dialog(qtbot)
    dialog.resize(520, 360)
    qtbot.wait(50)

    assert dialog.scroll_area.widget() is not None
    assert dialog.scroll_area.widgetResizable()
    assert dialog.scroll_area.widget().sizeHint().height() > dialog.scroll_area.viewport().height()
