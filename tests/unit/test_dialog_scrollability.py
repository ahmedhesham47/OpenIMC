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

import pandas as pd
import pytest
from PyQt5 import QtWidgets

from openimc.ui.dialogs.batch_correction_dialog import BatchCorrectionDialog
from openimc.ui.dialogs.pixel_correlation_dialog import PixelCorrelationDialog
from tests.unit.test_feature_extraction_dialog import _build_dialog as _build_feature_dialog
from tests.unit.test_plot_ui_regressions import _QCTestParent, _build_clustering_dialog, _build_qc_dialog


def _assert_scroll_area_overflows(scroll_area):
    assert scroll_area.widget() is not None
    assert scroll_area.widgetResizable()
    assert scroll_area.widget().sizeHint().height() > scroll_area.viewport().height()


@pytest.mark.unit
def test_feature_batch_and_pixel_dialogs_keep_settings_inside_scroll_areas(qtbot):
    feature_dialog, _feature_parent = _build_feature_dialog(qtbot)
    feature_dialog.resize(520, 360)
    qtbot.wait(50)
    _assert_scroll_area_overflows(feature_dialog.scroll_area)

    batch_df = pd.DataFrame(
        {
            "cell_id": [1, 2],
            "acquisition_id": ["ROI_1", "ROI_2"],
            "source_file": ["panel_a.mcd", "panel_b.mcd"],
            "marker_a_mean": [1.0, 2.0],
            "marker_b_median": [3.0, 4.0],
        }
    )
    batch_dialog = BatchCorrectionDialog(batch_df)
    qtbot.addWidget(batch_dialog)
    batch_dialog.resize(520, 360)
    batch_dialog.show()
    qtbot.wait(50)
    _assert_scroll_area_overflows(batch_dialog.scroll_area)

    pixel_parent = _QCTestParent(with_masks=True)
    qtbot.addWidget(pixel_parent)
    pixel_dialog = PixelCorrelationDialog(pixel_parent)
    qtbot.addWidget(pixel_dialog)
    pixel_dialog.resize(520, 420)
    pixel_dialog.show()
    qtbot.wait(50)
    _assert_scroll_area_overflows(pixel_dialog.options_scroll_area)


@pytest.mark.unit
def test_clustering_and_qc_settings_dialogs_keep_controls_scrollable(qtbot):
    clustering_dialog = _build_clustering_dialog(qtbot)
    clustering_dialog.clustering_settings_dialog.resize(380, 260)
    clustering_dialog.clustering_settings_dialog.show()
    qtbot.wait(50)
    _assert_scroll_area_overflows(clustering_dialog.clustering_settings_scroll_area)

    qc_parent = _QCTestParent(with_masks=True)
    qc_dialog = _build_qc_dialog(qtbot, parent=qc_parent)
    qc_dialog.qc_settings_dialog.resize(380, 260)
    qc_dialog.qc_settings_dialog.show()
    qtbot.wait(50)
    _assert_scroll_area_overflows(qc_dialog.qc_settings_scroll_area)


@pytest.mark.unit
def test_batch_correction_metadata_section_is_compact_by_default(qtbot):
    batch_df = pd.DataFrame(
        {
            "cell_id": [1],
            "acquisition_id": ["ROI_1"],
            "source_file": ["panel_a.mcd"],
            "marker_a_mean": [1.0],
        }
    )
    dialog = BatchCorrectionDialog(batch_df)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.wait(30)

    assert dialog.metadata_summary_label.text() == "No metadata files configured."
    assert not dialog.metadata_toggle_btn.isChecked()
    assert not dialog.metadata_content.isVisible()
