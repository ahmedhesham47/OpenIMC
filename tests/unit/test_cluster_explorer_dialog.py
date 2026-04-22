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
import pandas as pd
import pytest
from PyQt5 import QtWidgets

from openimc.data.mcd_loader import AcquisitionInfo
from openimc.ui.dialogs import clustering as clustering_module
from openimc.ui.dialogs.clustering import (
    ClusterExplorerDialog,
    ClusterExplorerSettingsDialog,
)


class _CountingLoader:
    def __init__(self, stacks_by_acq, channel_order_by_acq):
        self.stacks_by_acq = stacks_by_acq
        self.channel_order_by_acq = channel_order_by_acq
        self.get_all_calls = []
        self.get_image_calls = []
        self.get_channels_calls = []

    def get_all_channels(self, acq_id: str) -> np.ndarray:
        self.get_all_calls.append(acq_id)
        return self.stacks_by_acq[acq_id]

    def get_image(self, acq_id: str, channel: str) -> np.ndarray:
        self.get_image_calls.append((acq_id, channel))
        channel_index = self.channel_order_by_acq[acq_id].index(channel)
        return self.stacks_by_acq[acq_id][..., channel_index]

    def get_channels(self, acq_id: str):
        self.get_channels_calls.append(acq_id)
        return list(self.channel_order_by_acq[acq_id])


class _ParentWindow(QtWidgets.QWidget):
    def __init__(self, loaders_by_unique, segmentation_masks, acq_to_file, unique_acq_to_original, acquisitions):
        super().__init__()
        self.loaders_by_unique = loaders_by_unique
        self.loader = next(iter(loaders_by_unique.values()))
        self.mcd_loaders = {path: loaders_by_unique[key] for key, path in acq_to_file.items()}
        self.segmentation_masks = segmentation_masks
        self.acq_to_file = acq_to_file
        self.unique_acq_to_original = unique_acq_to_original
        self.acquisitions = acquisitions

    def _get_loader_for_acquisition(self, acq_id: str):
        return self.loaders_by_unique.get(acq_id)

    def _get_original_acq_id(self, acq_id: str) -> str:
        return self.unique_acq_to_original.get(acq_id, acq_id)

    def _get_pixel_size_um(self, acq_id, acq_info=None):
        if "file_a" in acq_id or "file_b" in acq_id:
            return 0.5
        return 1.0


class _LabelProvider:
    def __init__(self, clustered_data: pd.DataFrame):
        self.clustered_data = clustered_data

    def _get_cluster_display_name(self, cluster_id):
        return f"Phenotype {cluster_id}"


def _build_cluster_explorer(qtbot, with_masks: bool = True):
    channels = ["marker_a", "marker_b", "marker_c"]
    source_files = [
        "file_a.mcd",
        "file_b.mcd",
        "file_c.mcd",
        "file_d.mcd",
        "file_e.mcd",
        "file_f.mcd",
    ]

    loaders_by_unique = {}
    segmentation_masks = {}
    acq_to_file = {}
    unique_acq_to_original = {}
    acquisitions = []
    rows = []
    cluster_zero_cells = []
    cluster_one_cells = []
    row_index = 0
    next_cell_id = 101

    x_gradient = np.tile(np.arange(64, dtype=np.float32), (64, 1))
    y_gradient = x_gradient.T

    for acq_idx, source_file in enumerate(source_files):
        unique_acq_id = f"shared_acq__file_{chr(ord('a') + acq_idx)}"
        original_acq_id = "shared_acq"
        acq_to_file[unique_acq_id] = f"/tmp/{source_file}"
        unique_acq_to_original[unique_acq_id] = original_acq_id

        base = float((acq_idx + 1) * 25.0)
        stack = np.stack(
            [
                base + 0.8 * x_gradient,
                base / 2.0 + 1.2 * y_gradient,
                base / 3.0 + 0.6 * (x_gradient + y_gradient),
            ],
            axis=-1,
        ).astype(np.float32)
        loaders_by_unique[unique_acq_id] = _CountingLoader(
            {original_acq_id: stack},
            {original_acq_id: channels},
        )

        acquisitions.append(
            AcquisitionInfo(
                id=unique_acq_id,
                name=f"ROI {acq_idx + 1}",
                well=f"A{acq_idx + 1}",
                size=stack.shape[:2],
                channels=list(channels),
                channel_metals=[""] * len(channels),
                channel_labels=[""] * len(channels),
                metadata={"pixel_size_x": 0.5 if acq_idx < 2 else 1.0},
                source_file=f"/tmp/{source_file}",
            )
        )

        mask = np.zeros((64, 64), dtype=np.int32)
        cell_centers = [(16, 16), (46, 46)]
        for local_idx, (center_y, center_x) in enumerate(cell_centers):
            cell_id = next_cell_id
            next_cell_id += 1
            mask[center_y - 4 : center_y + 4, center_x - 4 : center_x + 4] = cell_id
            cluster_id = 0 if row_index < 10 else 1
            rows.append(
                {
                    "row_index": row_index,
                    "cluster": cluster_id,
                    "acquisition_id": original_acq_id,
                    "source_file": source_file,
                    "well": f"A{acq_idx + 1}",
                    "source_well": f"A{acq_idx + 1}",
                    "cell_id": cell_id,
                    "centroid_x": center_x,
                    "centroid_y": center_y,
                    "area": float((acq_idx + 1) * 100 + local_idx * 7),
                    "marker_a_mean": base + 5.0 + local_idx,
                    "marker_b_mean": base + 20.0 + local_idx * 2.0,
                    "marker_c_mean": base / 2.0 + local_idx * 3.0,
                }
            )
            if cluster_id == 0:
                cluster_zero_cells.append(row_index)
            else:
                cluster_one_cells.append(row_index)
            row_index += 1

        if with_masks:
            segmentation_masks[unique_acq_id] = mask

    feature_dataframe = pd.DataFrame(rows).set_index("row_index")
    cluster_info = [
        {"cluster_id": 0, "size": len(cluster_zero_cells), "cells": cluster_zero_cells},
        {"cluster_id": 1, "size": len(cluster_one_cells), "cells": cluster_one_cells},
    ]

    parent = _ParentWindow(
        loaders_by_unique,
        segmentation_masks,
        acq_to_file,
        unique_acq_to_original,
        acquisitions,
    )
    provider = _LabelProvider(feature_dataframe.copy())
    qtbot.addWidget(parent)

    dialog = ClusterExplorerDialog(cluster_info, feature_dataframe, parent=parent, label_provider=provider)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.wait(120)
    return dialog, parent, loaders_by_unique


def _sum_loader_calls(loaders_by_unique, attr_name: str) -> int:
    return sum(len(getattr(loader, attr_name)) for loader in loaders_by_unique.values())


def test_cluster_explorer_opens_without_preloading_first_cluster_preview(qtbot):
    dialog, _parent, loaders_by_unique = _build_cluster_explorer(qtbot)

    assert dialog.current_preview_records == []
    assert dialog._current_render_specs == []
    assert _sum_loader_calls(loaders_by_unique, "get_all_calls") == 0
    assert "Load/Refresh" in dialog.status_label.text()


def test_cluster_explorer_is_grayed_out_without_segmentation_masks(qtbot):
    dialog, _parent, loaders_by_unique = _build_cluster_explorer(qtbot, with_masks=False)

    assert not dialog.scroll_area.isEnabled()
    assert not dialog.settings_btn.isEnabled()
    assert not dialog.load_btn.isEnabled()
    assert not dialog.export_grid_btn.isEnabled()
    assert not dialog.export_btn.isEnabled()
    assert "Segmentation masks are not loaded" in dialog.status_label.text()
    assert _sum_loader_calls(loaders_by_unique, "get_all_calls") == 0


def test_cluster_explorer_keeps_edge_cells_centered_with_black_padding(qtbot):
    dialog, parent, _loaders = _build_cluster_explorer(qtbot)

    cell_idx = dialog.current_cluster["cells"][0]
    cell_row = dialog.feature_dataframe.loc[cell_idx]
    unique_acq_id = dialog._resolve_unique_acq_id(cell_row["acquisition_id"], cell_row)
    cell_id = int(cell_row["cell_id"])

    updated_mask = parent.segmentation_masks[unique_acq_id].copy()
    updated_mask[updated_mask == cell_id] = 0
    updated_mask[0:6, 0:6] = cell_id
    parent.segmentation_masks[unique_acq_id] = updated_mask
    dialog.feature_dataframe.loc[cell_idx, "centroid_x"] = 2
    dialog.feature_dataframe.loc[cell_idx, "centroid_y"] = 2
    dialog._cell_preview_cache.pop(cell_idx, None)
    dialog._crop_cache = {
        key: value for key, value in dialog._crop_cache.items() if key[0] != cell_idx
    }

    dialog._settings.sample_count = 1
    order_key = dialog._cluster_order_key()
    remaining = [idx for idx in dialog.current_cluster["cells"] if idx != cell_idx]
    dialog._sample_orders[order_key] = [cell_idx] + remaining
    dialog._load_cell_images()
    qtbot.wait(80)

    record = dialog.current_preview_records[0]
    coords = np.argwhere(record["cropped_mask"])
    crop_center = (dialog.PREVIEW_CROP_SIZE - 1) / 2.0

    assert abs(coords[:, 0].mean() - crop_center) <= 2.0
    assert abs(coords[:, 1].mean() - crop_center) <= 2.0


def test_cluster_explorer_settings_dialog_applies_state_from_popup(qtbot):
    dialog, _parent, _loaders = _build_cluster_explorer(qtbot)

    settings_dialog = ClusterExplorerSettingsDialog(dialog, dialog._settings)
    qtbot.addWidget(settings_dialog)
    settings_dialog.show()
    qtbot.wait(50)

    assert not hasattr(dialog, "cluster_combo")
    assert not hasattr(dialog, "channel_combo")
    assert settings_dialog.cluster_combo.currentData() == dialog._settings.cluster_id
    assert settings_dialog.channel_combo.currentText() == dialog._settings.channel
    assert settings_dialog.sample_mode_combo.currentData() == dialog._settings.sample_mode
    assert settings_dialog.sample_feature_combo.currentText() == dialog._settings.sample_feature
    assert settings_dialog.link_intensity_scale_chk.isChecked() == dialog._settings.link_intensity_scale
    assert settings_dialog.show_mask_outline_chk.isChecked() == dialog._settings.show_mask_outline
    assert settings_dialog.show_tile_titles_chk.isChecked() == dialog._settings.show_tile_titles

    settings_dialog.cluster_combo.setCurrentIndex(1)
    settings_dialog.channel_combo.setCurrentText("marker_b")
    settings_dialog.sample_count_spin.setValue(2)
    settings_dialog.column_count_spin.setValue(2)
    settings_dialog.sample_mode_combo.setCurrentIndex(
        settings_dialog.sample_mode_combo.findData("top_feature")
    )
    settings_dialog.sample_feature_combo.setCurrentText("area")
    settings_dialog.link_intensity_scale_chk.setChecked(False)
    settings_dialog.show_mask_outline_chk.setChecked(True)
    settings_dialog.show_tile_titles_chk.setChecked(True)
    settings_dialog._apply_only()
    qtbot.wait(80)

    assert dialog._settings.cluster_id == 1
    assert dialog._settings.channel == "marker_b"
    assert dialog._settings.sample_count == 2
    assert dialog._settings.column_count == 2
    assert dialog._settings.sample_mode == "top_feature"
    assert dialog._settings.sample_feature == "area"
    assert dialog._settings.link_intensity_scale is False
    assert dialog._settings.show_mask_outline is True
    assert dialog._settings.show_tile_titles is True
    assert "Phenotype 1" in dialog.summary_label.text()
    assert "Top by area" in dialog.summary_label.text()
    assert "scale per crop" in dialog.summary_label.text()
    assert "mask outline on" in dialog.summary_label.text()
    assert "titles on" in dialog.summary_label.text()


def test_cluster_explorer_settings_cancel_does_not_mutate_state(qtbot):
    dialog, _parent, _loaders = _build_cluster_explorer(qtbot)
    original = dialog._settings.clone()

    settings_dialog = ClusterExplorerSettingsDialog(dialog, dialog._settings)
    qtbot.addWidget(settings_dialog)
    settings_dialog.show()
    qtbot.wait(50)

    settings_dialog.channel_combo.setCurrentText("marker_c")
    settings_dialog.sample_count_spin.setValue(3)
    settings_dialog.balance_sources_chk.setChecked(True)
    settings_dialog.reject()
    qtbot.wait(20)

    assert dialog._settings.channel == original.channel
    assert dialog._settings.sample_count == original.sample_count
    assert dialog._settings.balance_sources == original.balance_sources


def test_cluster_explorer_lock_sampling_preserves_preview_cells_across_marker_changes(qtbot):
    dialog, _parent, _loaders = _build_cluster_explorer(qtbot)
    settings = dialog._settings.clone()
    settings.sample_count = 6
    settings.lock_sampling = True
    dialog.apply_settings(settings)
    qtbot.wait(80)
    initial_ids = dialog._current_preview_cell_ids()

    updated = dialog._settings.clone()
    updated.channel = "marker_b"
    dialog.apply_settings(updated)
    qtbot.wait(80)
    assert dialog._current_preview_cell_ids() == initial_ids

    reopened = dialog._settings.clone()
    reopened.column_count = 2
    reopened.show_scale_bar = True
    dialog.apply_settings(reopened)
    qtbot.wait(80)
    assert dialog._current_preview_cell_ids() == initial_ids


def test_cluster_explorer_balanced_sampling_prefers_distinct_source_files(qtbot):
    dialog, _parent, _loaders = _build_cluster_explorer(qtbot)
    settings = dialog._settings.clone()
    settings.sample_count = 4
    settings.balance_sources = True
    dialog.apply_settings(settings, force_resample=True)
    qtbot.wait(80)

    source_files = [record["source_file"] for record in dialog.current_preview_records]
    assert len(source_files) == 4
    assert len(set(source_files)) == 4


def test_cluster_explorer_can_rank_top_cells_by_independent_feature(qtbot):
    dialog, _parent, _loaders = _build_cluster_explorer(qtbot)
    settings = dialog._settings.clone()
    settings.sample_count = 3
    settings.sample_mode = "top_feature"
    settings.sample_feature = "area"
    settings.channel = "marker_b"
    dialog.apply_settings(settings, force_resample=True)
    qtbot.wait(80)

    preview_indices = [record["cell_index"] for record in dialog.current_preview_records]
    expected = (
        dialog.feature_dataframe.loc[dialog.current_cluster["cells"], "area"]
        .sort_values(ascending=False)
        .index[:3]
        .tolist()
    )

    assert preview_indices == expected
    assert dialog._settings.channel == "marker_b"


def test_cluster_explorer_uses_cached_stacks_without_get_image_calls_on_marker_change(qtbot):
    dialog, _parent, loaders_by_unique = _build_cluster_explorer(qtbot)
    dialog._load_cell_images()
    qtbot.wait(80)

    initial_get_all = _sum_loader_calls(loaders_by_unique, "get_all_calls")
    initial_get_image = _sum_loader_calls(loaders_by_unique, "get_image_calls")
    assert initial_get_all > 0
    assert initial_get_image == 0

    updated = dialog._settings.clone()
    updated.channel = "marker_b"
    dialog.apply_settings(updated)
    qtbot.wait(80)

    assert _sum_loader_calls(loaders_by_unique, "get_image_calls") == 0
    assert _sum_loader_calls(loaders_by_unique, "get_all_calls") == initial_get_all


def test_cluster_explorer_lock_sampling_preserves_ranked_preview_cells_across_marker_changes(qtbot):
    dialog, _parent, _loaders = _build_cluster_explorer(qtbot)
    settings = dialog._settings.clone()
    settings.sample_count = 4
    settings.sample_mode = "top_feature"
    settings.sample_feature = "area"
    settings.lock_sampling = True
    dialog.apply_settings(settings)
    qtbot.wait(80)
    initial_ids = dialog._current_preview_cell_ids()

    updated = dialog._settings.clone()
    updated.channel = "marker_c"
    dialog.apply_settings(updated)
    qtbot.wait(80)

    assert dialog._current_preview_cell_ids() == initial_ids


def test_cluster_explorer_single_channel_tiles_show_intensity_bars_and_shared_scaling(qtbot):
    dialog, _parent, _loaders = _build_cluster_explorer(qtbot)
    dialog._load_cell_images()
    qtbot.wait(50)

    first_canvas = dialog._tile_canvases[0]
    second_canvas = dialog._tile_canvases[1]
    assert len(first_canvas.figure.axes) == 2
    assert len(second_canvas.figure.axes) == 2
    assert first_canvas.image_ax.images[0].colorbar is not None
    assert first_canvas.image_ax.images[0].colorbar.ax in first_canvas.intensity_ax.child_axes
    assert first_canvas.intensity_ax.yaxis.get_ticks_position() == "left"
    assert first_canvas.image_ax.get_title() == ""
    assert first_canvas.image_ax.images[0].get_clim() == pytest.approx(
        second_canvas.image_ax.images[0].get_clim()
    )


def test_cluster_explorer_tile_titles_can_be_enabled(qtbot):
    dialog, _parent, _loaders = _build_cluster_explorer(qtbot)
    updated = dialog._settings.clone()
    updated.sample_count = 1
    updated.show_tile_titles = True
    dialog.apply_settings(updated)
    qtbot.wait(80)

    first_canvas = dialog._tile_canvases[0]
    assert first_canvas.image_ax.get_title() != ""


def test_cluster_explorer_single_channel_tiles_can_unlink_scaling(qtbot):
    dialog, _parent, _loaders = _build_cluster_explorer(qtbot)
    ordered_cells = list(dialog.current_cluster["cells"])
    key = dialog._cluster_order_key()
    dialog._sample_orders[key] = [ordered_cells[0], ordered_cells[2]] + [
        cell_idx for cell_idx in ordered_cells if cell_idx not in {ordered_cells[0], ordered_cells[2]}
    ]

    updated = dialog._settings.clone()
    updated.sample_count = 2
    dialog.apply_settings(updated)
    qtbot.wait(80)

    first_canvas = dialog._tile_canvases[0]
    second_canvas = dialog._tile_canvases[1]
    assert first_canvas.image_ax.images[0].get_clim() == pytest.approx(
        second_canvas.image_ax.images[0].get_clim()
    )

    unlinked = dialog._settings.clone()
    unlinked.link_intensity_scale = False
    dialog.apply_settings(unlinked)
    qtbot.wait(80)

    first_canvas = dialog._tile_canvases[0]
    second_canvas = dialog._tile_canvases[1]
    assert first_canvas.image_ax.images[0].get_clim() != pytest.approx(
        second_canvas.image_ax.images[0].get_clim()
    )


def test_cluster_explorer_rgb_mode_hides_intensity_bars(qtbot):
    dialog, _parent, _loaders = _build_cluster_explorer(qtbot)
    updated = dialog._settings.clone()
    updated.rgb_mode = True
    updated.rgb_channels = {"R": "marker_a", "G": "marker_b", "B": "marker_c"}
    dialog.apply_settings(updated)
    qtbot.wait(80)

    first_canvas = dialog._tile_canvases[0]
    assert len(first_canvas.figure.axes) == 1
    assert first_canvas.intensity_ax is None


def test_cluster_explorer_scale_bar_renders_on_each_tile_when_enabled(qtbot):
    dialog, _parent, _loaders = _build_cluster_explorer(qtbot)
    updated = dialog._settings.clone()
    updated.show_scale_bar = True
    updated.scale_bar_length_um = 8.0
    dialog.apply_settings(updated)
    qtbot.wait(80)

    visible_canvases = dialog._tile_canvases[: len(dialog._current_render_specs)]
    assert visible_canvases
    assert all(len(canvas.image_ax.lines) > 0 for canvas in visible_canvases)


def test_cluster_explorer_mask_outline_renders_when_enabled(qtbot):
    dialog, _parent, _loaders = _build_cluster_explorer(qtbot)
    updated = dialog._settings.clone()
    updated.sample_count = 1
    updated.show_mask_outline = True
    dialog.apply_settings(updated)
    qtbot.wait(80)

    first_canvas = dialog._tile_canvases[0]
    assert len(first_canvas.image_ax.collections) > 0


def test_cluster_explorer_preview_retains_context_outside_mask(qtbot):
    dialog, _parent, _loaders = _build_cluster_explorer(qtbot)
    updated = dialog._settings.clone()
    updated.sample_count = 1
    dialog.apply_settings(updated)
    qtbot.wait(80)

    first_canvas = dialog._tile_canvases[0]
    image = first_canvas.image_ax.images[0].get_array()
    record = dialog.current_preview_records[0]

    assert np.any(~record["cropped_mask"])
    assert float(np.max(image[~record["cropped_mask"]])) > 0.0


def test_cluster_explorer_export_grid_uses_current_preview_layout(qtbot, monkeypatch):
    dialog, _parent, _loaders = _build_cluster_explorer(qtbot)
    updated = dialog._settings.clone()
    updated.sample_count = 3
    updated.column_count = 2
    dialog.apply_settings(updated)
    qtbot.wait(80)

    captured = {}

    def _fake_save(figure, default_filename, parent=None):
        captured["figure"] = figure
        captured["default_filename"] = default_filename
        return True

    monkeypatch.setattr(clustering_module, "save_figure_with_options", _fake_save)
    dialog._export_grid()

    assert captured["default_filename"].startswith("cluster_")
    image_axes = [ax for ax in captured["figure"].axes if ax.images]
    assert len(image_axes) == 3
