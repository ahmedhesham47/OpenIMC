from typing import Optional

import pandas as pd
from PyQt5 import QtWidgets

from openimc.ui.main_window import MainWindow
from openimc.ui.dialogs.clustering import CellClusteringDialog
from openimc.ui.dialogs.simple_spatial_analysis import SimpleSpatialAnalysisDialog


def _build_feature_dataframe(scale: float = 1.0) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "cell_id": [1, 2, 3, 4, 5, 6],
            "label": [1, 2, 3, 4, 5, 6],
            "centroid_x": [10, 20, 30, 40, 50, 60],
            "centroid_y": [15, 25, 35, 45, 55, 65],
            "area_um2": [100, 120, 140, 160, 180, 200],
            "mean_CD3": [1, 2, 3, 4, 5, 6],
            "mean_CD45": [6, 5, 4, 3, 2, 1],
            "cluster": [1, 1, 2, 2, 1, 2],
            "cluster_id": [1, 1, 2, 2, 1, 2],
            "cluster_phenotype": ["A", "A", "B", "B", "A", "B"],
            "source_file": ["roi_1.csv", "roi_1.csv", "roi_1.csv", "roi_2.csv", "roi_2.csv", "roi_2.csv"],
            "source_well": ["roi_1", "roi_1", "roi_1", "roi_2", "roi_2", "roi_2"],
            "acquisition_id": ["roi_1", "roi_1", "roi_1", "roi_2", "roi_2", "roi_2"],
        }
    )
    scaled = base.copy()
    scaled["mean_CD3"] = scaled["mean_CD3"] * scale
    scaled["mean_CD45"] = scaled["mean_CD45"] * scale
    return scaled


class _FeatureSetHost(QtWidgets.QWidget):
    _has_batch_corrected_feature_source = MainWindow._has_batch_corrected_feature_source
    _normalize_analysis_feature_set_preference = MainWindow._normalize_analysis_feature_set_preference
    _get_default_analysis_feature_set_preference = MainWindow._get_default_analysis_feature_set_preference
    _get_effective_analysis_feature_set_preference = MainWindow._get_effective_analysis_feature_set_preference
    _apply_analysis_feature_set_to_dialog = MainWindow._apply_analysis_feature_set_to_dialog
    _sync_analysis_feature_source_dialogs = MainWindow._sync_analysis_feature_source_dialogs
    _set_analysis_feature_set_preference = MainWindow._set_analysis_feature_set_preference

    def __init__(self, feature_dataframe: pd.DataFrame, batch_corrected_dataframe: Optional[pd.DataFrame] = None):
        super().__init__()
        self.feature_dataframe = feature_dataframe.copy()
        self.batch_corrected_dataframe = (
            batch_corrected_dataframe.copy()
            if batch_corrected_dataframe is not None
            else None
        )
        self.analysis_feature_set_preference = None
        self._analysis_feature_sync_in_progress = False
        self.clustering_dialog = None
        self.simple_spatial_dialog = None
        self.advanced_spatial_dialog = None


def test_clustering_dialog_refreshes_feature_sources_after_batch_correction(qtbot):
    original_df = _build_feature_dataframe(scale=1.0)
    corrected_df = _build_feature_dataframe(scale=10.0)
    host = _FeatureSetHost(original_df)
    qtbot.addWidget(host)

    dialog = CellClusteringDialog(original_df, parent=host)
    host.clustering_dialog = dialog
    qtbot.addWidget(dialog)

    assert dialog.feature_set_combo.currentText() == "Loaded Features"

    host.batch_corrected_dataframe = corrected_df.copy()
    host._set_analysis_feature_set_preference("batch_corrected")

    assert [dialog.feature_set_combo.itemText(i) for i in range(dialog.feature_set_combo.count())] == [
        "Original Features",
        "Batch-Corrected Features",
    ]
    assert dialog.feature_set_combo.currentText() == "Batch-Corrected Features"
    pd.testing.assert_frame_equal(
        dialog.feature_dataframe.reset_index(drop=True),
        corrected_df.reset_index(drop=True),
    )


def test_clustering_feature_set_selection_syncs_to_simple_spatial(qtbot):
    original_df = _build_feature_dataframe(scale=1.0)
    corrected_df = _build_feature_dataframe(scale=10.0)
    host = _FeatureSetHost(original_df, corrected_df)
    qtbot.addWidget(host)

    clustering_dialog = CellClusteringDialog(
        original_df,
        batch_corrected_dataframe=corrected_df,
        parent=host,
    )
    simple_spatial_dialog = SimpleSpatialAnalysisDialog(
        original_df,
        batch_corrected_dataframe=corrected_df,
        parent=host,
    )
    host.clustering_dialog = clustering_dialog
    host.simple_spatial_dialog = simple_spatial_dialog
    qtbot.addWidget(clustering_dialog)
    qtbot.addWidget(simple_spatial_dialog)

    host._set_analysis_feature_set_preference("batch_corrected")
    assert simple_spatial_dialog.feature_set_combo.currentText() == "Batch-Corrected Features"

    clustering_dialog.feature_set_combo.setCurrentText("Original Features")
    qtbot.wait(10)

    assert host.analysis_feature_set_preference == "original"
    assert simple_spatial_dialog.feature_set_combo.currentText() == "Original Features"
    pd.testing.assert_frame_equal(
        simple_spatial_dialog.feature_dataframe.reset_index(drop=True),
        original_df.reset_index(drop=True),
    )
