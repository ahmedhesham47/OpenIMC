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
from PyQt5 import QtWidgets

from openimc.ui.cluster_utils import build_cluster_annotation_map, get_cluster_display_name
from openimc.ui.dialogs.clustering import CellClusteringDialog
from openimc.ui.dialogs.simple_spatial_analysis import SimpleSpatialAnalysisDialog


def _build_clustered_features():
    base_df = pd.DataFrame(
        {
            "cell_id": [1, 2, 3, 4],
            "centroid_x": [0.0, 1.0, 2.0, 3.0],
            "centroid_y": [0.0, 1.0, 2.0, 3.0],
            "acquisition_id": ["ROI_1"] * 4,
            "marker_a_mean": [0.1, 0.2, 0.8, 0.9],
            "cluster": [1, 1, 2, 2],
        }
    )
    clustered_df = base_df.copy()
    clustered_df["cluster_phenotype"] = [
        "Edited T cells",
        "Edited T cells",
        "Edited B cells",
        "Edited B cells",
    ]
    return base_df, clustered_df


def test_clustering_dialog_uses_cluster_phenotypes_from_loaded_clustered_dataframe(qtbot):
    feature_df, clustered_df = _build_clustered_features()

    dialog = CellClusteringDialog(feature_df, clustered_cells_dataframe=clustered_df.copy())
    qtbot.addWidget(dialog)

    assert dialog.cluster_annotation_map == {
        1: "Edited T cells",
        2: "Edited B cells",
    }
    assert dialog.cluster_backend_names == dialog.cluster_annotation_map
    assert dialog._get_cluster_display_name(1) == "Edited T cells"


def test_clustering_output_exports_current_edited_cluster_names(qtbot, monkeypatch, tmp_path):
    feature_df, clustered_df = _build_clustered_features()

    dialog = CellClusteringDialog(feature_df, clustered_cells_dataframe=clustered_df.copy())
    qtbot.addWidget(dialog)
    dialog.cluster_annotation_map = {
        1: "Renamed T cells",
        2: "Renamed B cells",
    }
    dialog.cluster_backend_names = {
        1: "T_cell",
        2: "B_cell",
    }

    dialog._apply_cluster_annotations()

    output_path = tmp_path / "clustered_output.csv"
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(output_path), "CSV Files (*.csv)"),
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        lambda *args, **kwargs: QtWidgets.QMessageBox.Ok,
    )

    dialog._save_clustering_output()

    exported_df = pd.read_csv(output_path, index_col=0)
    assert set(exported_df.loc[exported_df["cluster"] == 1, "cluster_phenotype"]) == {"Renamed T cells"}
    assert set(exported_df.loc[exported_df["cluster"] == 2, "cluster_phenotype"]) == {"Renamed B cells"}


def test_simple_spatial_uses_cluster_phenotypes_from_loaded_clustered_dataframe(qtbot):
    feature_df, clustered_df = _build_clustered_features()

    dialog = SimpleSpatialAnalysisDialog(feature_df, clustered_cells_dataframe=clustered_df.copy())
    qtbot.addWidget(dialog)

    assert dialog.cluster_annotation_map == {
        1: "Edited T cells",
        2: "Edited B cells",
    }
    assert set(dialog.clustered_cells_dataframe["cluster_phenotype"]) == {
        "Edited T cells",
        "Edited B cells",
    }


def test_loaded_feature_dataframe_cluster_phenotypes_are_available_for_display():
    _, clustered_df = _build_clustered_features()

    annotation_map = build_cluster_annotation_map(
        {},
        clustered_df.copy(),
    )

    assert annotation_map == {
        1: "Edited T cells",
        2: "Edited B cells",
    }
    assert get_cluster_display_name(1, annotation_map=annotation_map) == "Edited T cells"


def test_simple_spatial_uses_local_cluster_names_without_parent_round_trip(qtbot):
    _, clustered_df = _build_clustered_features()

    class _FailingParent(QtWidgets.QWidget):
        def _get_cluster_display_name(self, cluster_id):
            raise AssertionError("Simple spatial dialog should not ask the parent for each cluster label.")

    parent = _FailingParent()
    qtbot.addWidget(parent)

    dialog = SimpleSpatialAnalysisDialog(
        clustered_df.copy(),
        clustered_cells_dataframe=clustered_df.copy(),
        parent=parent,
    )
    qtbot.addWidget(dialog)

    assert set(dialog.feature_dataframe["cluster_phenotype"]) == {
        "Edited T cells",
        "Edited B cells",
    }
