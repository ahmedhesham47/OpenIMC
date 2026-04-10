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

import sys
import numpy as np
import pandas as pd
from matplotlib.text import Text

from openimc.ui.dialogs.clustering import CellClusteringDialog


_CLUSTER_MAP_WAIT_TIMEOUT_MS = 15000 if sys.platform == 'darwin' else 5000


def _cluster_map_axes_ready(fig):
    try:
        heatmap_axis = _get_heatmap_axis(fig)
        _get_colorbar_axis(fig, heatmap_axis)
        return True
    except Exception:
        return False


def _build_clustered_dataframe(n_clusters: int = 7, cells_per_cluster: int = 8, n_features: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for cluster_id in range(1, n_clusters + 1):
        for offset in range(cells_per_cluster):
            row = {
                'cell_id': (cluster_id - 1) * cells_per_cluster + offset + 1,
                'centroid_x': float(cluster_id * 10 + offset),
                'centroid_y': float(cluster_id * 5 + offset),
                'cluster': cluster_id,
            }
            for feature_idx in range(n_features):
                row[f'marker_{feature_idx}_mean'] = float(rng.normal(loc=cluster_id + feature_idx / 3.0, scale=0.15))
            rows.append(row)
    return pd.DataFrame(rows)


def _measure_text_overflow(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_bbox = fig.bbox
    fig_width = max(1.0, float(fig_bbox.width))
    fig_height = max(1.0, float(fig_bbox.height))
    overflow = {'left': 0.0, 'right': 0.0, 'top': 0.0, 'bottom': 0.0}

    for text_artist in fig.findobj(match=Text):
        if not text_artist.get_visible():
            continue
        text_value = text_artist.get_text()
        if not isinstance(text_value, str) or not text_value.strip():
            continue
        bbox = text_artist.get_window_extent(renderer=renderer)
        if bbox.width <= 0 or bbox.height <= 0:
            continue
        if bbox.x0 < fig_bbox.x0:
            overflow['left'] = max(overflow['left'], (fig_bbox.x0 - bbox.x0) / fig_width)
        if bbox.x1 > fig_bbox.x1:
            overflow['right'] = max(overflow['right'], (bbox.x1 - fig_bbox.x1) / fig_width)
        if bbox.y0 < fig_bbox.y0:
            overflow['bottom'] = max(overflow['bottom'], (fig_bbox.y0 - bbox.y0) / fig_height)
        if bbox.y1 > fig_bbox.y1:
            overflow['top'] = max(overflow['top'], (bbox.y1 - fig_bbox.y1) / fig_height)

    return overflow


def _get_heatmap_axis(fig):
    for axis in fig.axes:
        if axis.get_xlabel() in {'Clusters', 'Features'} and axis.get_ylabel() in {'Features', 'Clusters'}:
            return axis
    raise AssertionError('Could not find cluster-map heatmap axis')


def _get_colorbar_axis(fig, heatmap_axis):
    candidates = []
    heatmap_pos = heatmap_axis.get_position()
    for axis in fig.axes:
        if axis is heatmap_axis:
            continue
        axis_pos = axis.get_position()
        visible_ticklabels = [lbl for lbl in axis.get_yticklabels() if lbl.get_visible() and lbl.get_text()]
        visible_ticklabels.extend([lbl for lbl in axis.get_xticklabels() if lbl.get_visible() and lbl.get_text()])
        is_vertical_bar = axis_pos.width < heatmap_pos.width * 0.2
        is_horizontal_bar = axis_pos.height < heatmap_pos.height * 0.2
        if visible_ticklabels and (is_vertical_bar or is_horizontal_bar):
            candidates.append(axis)
    if not candidates:
        raise AssertionError('Could not find cluster-map colorbar axis')
    return min(candidates, key=lambda axis: axis.get_position().width * axis.get_position().height)


def _build_dialog(qtbot, *, cbar_position='Upper right', cbar_orientation='Vertical'):
    df = _build_clustered_dataframe()
    dialog = CellClusteringDialog(df, clustered_cells_dataframe=df.copy())
    qtbot.addWidget(dialog)
    dialog.feature_label_map = {
        f'marker_{idx}_mean': f'Feature {idx} with a deliberately long display label for GUI layout checks'
        for idx in range(12)
    }
    dialog.cluster_annotation_map = {
        cluster_id: f'Cluster {cluster_id} with an intentionally long descriptive name'
        for cluster_id in range(1, 8)
    }
    dialog.cluster_map_orientation = 'landscape'
    dialog.cluster_map_dendrogram = 'Both rows and columns'
    dialog.cluster_map_colorbar_position = cbar_position
    dialog.cluster_map_colorbar_orientation = cbar_orientation
    dialog.cluster_map_cell_size = 22
    dialog.resize(1220, 900)
    dialog.show()
    qtbot.wait(150)
    dialog._show_cluster_map()
    qtbot.waitUntil(lambda: _cluster_map_axes_ready(dialog.figure), timeout=_CLUSTER_MAP_WAIT_TIMEOUT_MS)
    if hasattr(dialog, '_flush_canvas'):
        dialog._flush_canvas(force_layout_refresh=True)
    qtbot.wait(150)
    return dialog


def _cluster_map_layout_is_clear(fig, *, expect_horizontal=False):
    try:
        overflow = _measure_text_overflow(fig)
        if max(overflow.values()) > 0.01:
            return False

        heatmap_axis = _get_heatmap_axis(fig)
        colorbar_axis = _get_colorbar_axis(fig, heatmap_axis)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        rightmost_ylabel_px = heatmap_axis.bbox.x1
        for label in heatmap_axis.get_yticklabels():
            if not label.get_visible() or not label.get_text():
                continue
            rightmost_ylabel_px = max(rightmost_ylabel_px, label.get_window_extent(renderer=renderer).x1)

        if colorbar_axis.bbox.x0 < rightmost_ylabel_px - 1.0:
            return False
        if colorbar_axis.bbox.y0 < heatmap_axis.bbox.y1 - 1.0:
            return False

        if expect_horizontal:
            colorbar_pos = colorbar_axis.get_position()
            if colorbar_pos.width <= colorbar_pos.height:
                return False
            for text_artist in fig.findobj(match=Text):
                if getattr(text_artist, 'axes', None) is colorbar_axis:
                    continue
                if not text_artist.get_visible():
                    continue
                text_value = text_artist.get_text()
                if not isinstance(text_value, str) or not text_value.strip():
                    continue
                bbox = text_artist.get_window_extent(renderer=renderer)
                if bbox.width <= 0 or bbox.height <= 0:
                    continue
                if _bboxes_overlap(bbox, colorbar_axis.bbox):
                    return False

        return True
    except Exception:
        return False


def test_cluster_map_reflows_after_resize(qtbot):
    dialog = _build_dialog(qtbot)

    dialog.resize(920, 720)
    qtbot.wait(250)
    dialog._show_cluster_map()
    qtbot.wait(150)

    dialog.resize(1420, 1040)
    qtbot.waitUntil(
        lambda: abs(dialog.figure.get_size_inches()[0] - (dialog.canvas.width() / max(dialog.figure.get_dpi(), 1.0))) < 0.2,
        timeout=max(4000, _CLUSTER_MAP_WAIT_TIMEOUT_MS),
    )
    qtbot.waitUntil(lambda: _cluster_map_layout_is_clear(dialog.figure), timeout=_CLUSTER_MAP_WAIT_TIMEOUT_MS)
    qtbot.wait(150)

    overflow = _measure_text_overflow(dialog.figure)
    assert max(overflow.values()) <= 0.01


def _bboxes_overlap(a, b, pad_px=1.0):
    return not (
        a.x1 <= (b.x0 + pad_px)
        or a.x0 >= (b.x1 - pad_px)
        or a.y1 <= (b.y0 + pad_px)
        or a.y0 >= (b.y1 - pad_px)
    )


def test_cluster_map_supports_horizontal_colorbar_without_covering_text(qtbot):
    dialog = _build_dialog(qtbot, cbar_position='Upper right', cbar_orientation='Horizontal')
    qtbot.waitUntil(
        lambda: _cluster_map_layout_is_clear(dialog.figure, expect_horizontal=True),
        timeout=_CLUSTER_MAP_WAIT_TIMEOUT_MS,
    )

    overflow = _measure_text_overflow(dialog.figure)
    assert max(overflow.values()) <= 0.01

    heatmap_axis = _get_heatmap_axis(dialog.figure)
    colorbar_axis = _get_colorbar_axis(dialog.figure, heatmap_axis)
    colorbar_pos = colorbar_axis.get_position()
    assert colorbar_pos.width > colorbar_pos.height
    assert colorbar_axis.bbox.y0 >= heatmap_axis.bbox.y1 - 1.0

    dialog.figure.canvas.draw()
    renderer = dialog.figure.canvas.get_renderer()
    for text_artist in dialog.figure.findobj(match=Text):
        if getattr(text_artist, 'axes', None) is colorbar_axis:
            continue
        if not text_artist.get_visible():
            continue
        text_value = text_artist.get_text()
        if not isinstance(text_value, str) or not text_value.strip():
            continue
        bbox = text_artist.get_window_extent(renderer=renderer)
        if bbox.width <= 0 or bbox.height <= 0:
            continue
        assert not _bboxes_overlap(bbox, colorbar_axis.bbox)
