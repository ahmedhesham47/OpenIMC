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
from PyQt5.QtCore import Qt

from openimc.data.mcd_loader import AcquisitionInfo
from openimc.ui.dialogs.comparison_dialog import DynamicComparisonDialog


class _CountingLoader:
    def __init__(self, data_by_key, channel_order_by_acq):
        self.data_by_key = data_by_key
        self.channel_order_by_acq = channel_order_by_acq
        self.get_all_calls = []
        self.get_image_calls = []

    def get_all_channels(self, acq_id: str) -> np.ndarray:
        self.get_all_calls.append(acq_id)
        channels = self.channel_order_by_acq[acq_id]
        return np.stack([self.data_by_key[(acq_id, channel)] for channel in channels], axis=-1)

    def get_image(self, acq_id: str, channel: str) -> np.ndarray:
        self.get_image_calls.append((acq_id, channel))
        return self.data_by_key[(acq_id, channel)]


def _build_dialog(qtbot, *, channels, data_by_key, acq_ids=("r1", "r2")):
    acquisitions = [
        AcquisitionInfo(
            id=acq_id,
            name=acq_id,
            well=f"A{idx + 1}",
            size=next(iter(data_by_key.values())).shape[:2],
            channels=list(channels),
            channel_metals=[""] * len(channels),
            channel_labels=[""] * len(channels),
            metadata={},
            source_file="test.mcd",
        )
        for idx, acq_id in enumerate(acq_ids)
    ]
    loader = _CountingLoader(data_by_key, {acq.id: list(channels) for acq in acquisitions})
    dialog = DynamicComparisonDialog(acquisitions, loader)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.wait(50)
    return dialog, loader


def _add_acquisitions(dialog, qtbot, *rows):
    for row in rows:
        dialog.available_acq_list.setCurrentRow(row)
        dialog._add_acquisition()
        qtbot.wait(20)


def _rgb_row(dialog, channel: str) -> int:
    for row in range(dialog.channel_color_table.rowCount()):
        item = dialog.channel_color_table.item(row, 0)
        if item is not None and item.text() == channel:
            return row
    raise AssertionError(f"Could not find RGB row for {channel!r}")


def _set_rgb_channel(dialog, qtbot, channel: str, *, enabled: bool = True, color: str = "Red"):
    row = _rgb_row(dialog, channel)
    color_combo = dialog.channel_color_table.cellWidget(row, 1)
    color_combo.setCurrentText(color)
    qtbot.wait(10)
    dialog.channel_color_table.item(row, 0).setCheckState(Qt.Checked if enabled else Qt.Unchecked)
    qtbot.wait(20)


def _canvas_image(dialog, index: int):
    canvas = dialog.image_layout.itemAt(index).widget()
    return canvas.ax.get_images()[0]


def test_comparison_preserves_selected_marker_when_adding_roi(qtbot):
    channels = ["c1", "c2", "c3"]
    data = {
        ("r1", "c1"): np.ones((4, 4), dtype=np.float32),
        ("r1", "c2"): np.full((4, 4), 2.0, dtype=np.float32),
        ("r1", "c3"): np.full((4, 4), 3.0, dtype=np.float32),
        ("r2", "c1"): np.full((4, 4), 4.0, dtype=np.float32),
        ("r2", "c2"): np.full((4, 4), 5.0, dtype=np.float32),
        ("r2", "c3"): np.full((4, 4), 6.0, dtype=np.float32),
    }
    dialog, _loader = _build_dialog(qtbot, channels=channels, data_by_key=data)

    _add_acquisitions(dialog, qtbot, 0)
    dialog.channel_combo.setCurrentText("c2")
    qtbot.wait(20)

    _add_acquisitions(dialog, qtbot, 1)

    assert dialog.channel_combo.currentText() == "c2"


def test_comparison_preserves_rgb_assignments_when_roi_changes(qtbot):
    channels = ["c1", "c2", "c3"]
    data = {
        (acq_id, channel): np.full((4, 4), idx + 1, dtype=np.float32)
        for acq_id in ("r1", "r2")
        for idx, channel in enumerate(channels)
    }
    dialog, _loader = _build_dialog(qtbot, channels=channels, data_by_key=data)

    _add_acquisitions(dialog, qtbot, 0)
    dialog.rgb_mode_chk.setChecked(True)
    qtbot.wait(20)
    _set_rgb_channel(dialog, qtbot, "c1", enabled=True, color="Red")
    _set_rgb_channel(dialog, qtbot, "c2", enabled=True, color="Green")

    _add_acquisitions(dialog, qtbot, 1)

    row_c1 = _rgb_row(dialog, "c1")
    row_c2 = _rgb_row(dialog, "c2")
    assert dialog.channel_color_table.item(row_c1, 0).checkState() == Qt.Checked
    assert dialog.channel_color_table.cellWidget(row_c1, 1).currentText() == "Red"
    assert dialog.channel_color_table.item(row_c2, 0).checkState() == Qt.Checked
    assert dialog.channel_color_table.cellWidget(row_c2, 1).currentText() == "Green"


def test_comparison_preserves_scaling_channel_when_rgb_color_changes(qtbot):
    channels = ["c1", "c2", "c3"]
    data = {
        (acq_id, channel): np.full((4, 4), idx + 1, dtype=np.float32)
        for acq_id in ("r1", "r2")
        for idx, channel in enumerate(channels)
    }
    dialog, _loader = _build_dialog(qtbot, channels=channels, data_by_key=data)

    _add_acquisitions(dialog, qtbot, 0, 1)
    dialog.rgb_mode_chk.setChecked(True)
    dialog.custom_scaling_chk.setChecked(True)
    qtbot.wait(20)
    _set_rgb_channel(dialog, qtbot, "c1", enabled=True, color="Red")
    _set_rgb_channel(dialog, qtbot, "c2", enabled=True, color="Green")

    dialog.scaling_channel_combo.setCurrentText("c2")
    qtbot.wait(20)

    row_c1 = _rgb_row(dialog, "c1")
    dialog.channel_color_table.cellWidget(row_c1, 1).setCurrentText("Blue")
    qtbot.wait(20)

    assert dialog.scaling_channel_combo.currentText() == "c2"


def test_comparison_single_channel_linked_scaling_shares_clims(qtbot):
    channels = ["c1", "c2"]
    data = {
        ("r1", "c1"): np.array([[0, 10], [20, 30]], dtype=np.float32),
        ("r2", "c1"): np.array([[100, 110], [120, 130]], dtype=np.float32),
        ("r1", "c2"): np.zeros((2, 2), dtype=np.float32),
        ("r2", "c2"): np.zeros((2, 2), dtype=np.float32),
    }
    dialog, _loader = _build_dialog(qtbot, channels=channels, data_by_key=data)

    _add_acquisitions(dialog, qtbot, 0, 1)
    dialog.channel_combo.setCurrentText("c1")
    qtbot.wait(20)

    left_clim = _canvas_image(dialog, 0).get_clim()
    right_clim = _canvas_image(dialog, 1).get_clim()

    assert left_clim == pytest.approx((0.0, 130.0))
    assert right_clim == pytest.approx((0.0, 130.0))


def test_comparison_rgb_linked_scaling_uses_global_range_without_custom_scaling(qtbot):
    channels = ["c1", "c2", "c3"]
    data = {
        ("r1", "c1"): np.array([[0, 10], [20, 30]], dtype=np.float32),
        ("r2", "c1"): np.array([[100, 110], [120, 130]], dtype=np.float32),
        ("r1", "c2"): np.zeros((2, 2), dtype=np.float32),
        ("r2", "c2"): np.zeros((2, 2), dtype=np.float32),
        ("r1", "c3"): np.zeros((2, 2), dtype=np.float32),
        ("r2", "c3"): np.zeros((2, 2), dtype=np.float32),
    }
    dialog, _loader = _build_dialog(qtbot, channels=channels, data_by_key=data)

    _add_acquisitions(dialog, qtbot, 0, 1)
    dialog.rgb_mode_chk.setChecked(True)
    qtbot.wait(20)
    _set_rgb_channel(dialog, qtbot, "c1", enabled=True, color="Red")

    left_rgb = _canvas_image(dialog, 0).get_array()
    right_rgb = _canvas_image(dialog, 1).get_array()

    assert np.max(left_rgb[..., 0]) == pytest.approx(30.0 / 130.0, abs=1e-6)
    assert np.min(right_rgb[..., 0]) == pytest.approx(100.0 / 130.0, abs=1e-6)
    assert np.allclose(left_rgb[..., 1:], 0.0)
    assert np.allclose(right_rgb[..., 1:], 0.0)


def test_comparison_prefetch_cache_serves_early_channels_without_get_image_calls(qtbot):
    channels = [f"c{i:02d}" for i in range(60)]
    data = {
        (acq_id, channel): np.full((4, 4), idx, dtype=np.float32)
        for acq_id in ("r1", "r2")
        for idx, channel in enumerate(channels)
    }
    dialog, loader = _build_dialog(qtbot, channels=channels, data_by_key=data)

    _add_acquisitions(dialog, qtbot, 0, 1)
    initial_get_all_calls = list(loader.get_all_calls)
    assert loader.get_image_calls == []

    dialog.rgb_mode_chk.setChecked(True)
    qtbot.wait(20)
    _set_rgb_channel(dialog, qtbot, "c00", enabled=True, color="Red")
    _set_rgb_channel(dialog, qtbot, "c01", enabled=True, color="Green")
    _set_rgb_channel(dialog, qtbot, "c02", enabled=True, color="Blue")

    assert loader.get_image_calls == []
    assert loader.get_all_calls == initial_get_all_calls


def test_comparison_tiles_keep_equal_visual_size_when_adding_rois(qtbot):
    channels = ["c1", "c2"]
    data = {
        (acq_id, channel): np.full((32, 32), idx + 1, dtype=np.float32)
        for acq_id in ("r1", "r2", "r3")
        for idx, channel in enumerate(channels)
    }
    dialog, _loader = _build_dialog(qtbot, channels=channels, data_by_key=data, acq_ids=("r1", "r2", "r3"))

    _add_acquisitions(dialog, qtbot, 0, 1, 2)
    qtbot.wait(50)

    widths = []
    heights = []
    for index in range(dialog.image_layout.count()):
        widget = dialog.image_layout.itemAt(index).widget()
        widths.append(widget.width())
        heights.append(widget.height())

    assert len(set(widths)) == 1
    assert len(set(heights)) == 1
