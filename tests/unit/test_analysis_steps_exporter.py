import json
from datetime import datetime

from openimc.ui.analysis_steps_exporter import AnalysisStepsExporter
from openimc.utils.logger import get_logger, set_log_file


def test_export_analysis_steps_expands_detailed_lists(tmp_path):
    log_file = tmp_path / "methods_log.jsonl"
    output_file = tmp_path / "analysis_steps.txt"

    set_log_file(str(log_file))
    logger = get_logger()
    logger.log_segmentation(
        method="cellpose",
        parameters={
            "nuclear_channels": ["DNA1", "DNA2", "Histone"],
            "cyto_channels": ["CD45", "PanCK", "Ecad", "SMA", "CD3", "CD20"],
            "denoise_settings": {
                "DNA1": {"hot": {"method": "median3"}},
                "CD45": {"background": {"method": "rolling_ball", "radius": 15}},
            },
        },
        acquisitions=["acq_1"],
        source_file="example.mcd",
    )
    logger.log_feature_extraction(
        parameters={
            "feature_categories": ["morphology", "intensity"],
            "selected_morphology_features": ["area_um2", "eccentricity"],
            "selected_intensity_features": ["mean", "median", "std", "mad", "p10", "p90"],
        },
        acquisitions=["acq_1"],
        features_extracted=["area_um2", "eccentricity", "mean", "median", "std", "mad", "p10", "p90"],
        source_file="example.mcd",
    )

    exporter = AnalysisStepsExporter()
    assert exporter.export_analysis_steps(str(output_file))

    content = output_file.read_text()
    assert "- nuclear_channels:" in content
    assert "* DNA1" in content
    assert "* CD20" in content
    assert "- selected_intensity_features:" in content
    assert "* p90" in content
    assert "- denoise_settings:" in content
    assert "rolling_ball" in content
    assert "8 features" not in content


def test_has_entries_respects_session_start_time(tmp_path):
    log_file = tmp_path / "methods_log.jsonl"
    set_log_file(str(log_file))

    old_timestamp = "2026-03-25T10:00:00"
    new_timestamp = "2026-03-25T12:00:00"
    entries = [
        {
            "type": "log_metadata",
            "timestamp": "2026-03-25T09:00:00",
            "description": "test",
            "format": "JSON Lines",
        },
        {
            "timestamp": old_timestamp,
            "type": "segmentation",
            "operation": "cellpose",
            "parameters": {},
            "acquisitions": ["old_acq"],
            "output_path": None,
            "notes": None,
            "source_file": "old.mcd",
        },
        {
            "timestamp": new_timestamp,
            "type": "feature_extraction",
            "operation": "extract_features",
            "parameters": {},
            "acquisitions": ["new_acq"],
            "output_path": None,
            "notes": None,
            "source_file": "new.mcd",
        },
    ]
    log_file.write_text("".join(json.dumps(entry) + "\n" for entry in entries))

    exporter = AnalysisStepsExporter()
    assert exporter.has_entries(session_start_time=datetime.fromisoformat("2026-03-25T11:00:00"))
    assert not exporter.has_entries(session_start_time=datetime.fromisoformat("2026-03-25T13:00:00"))


def test_export_analysis_steps_includes_pca_clustering_metadata(tmp_path):
    log_file = tmp_path / "methods_log.jsonl"
    output_file = tmp_path / "analysis_steps.txt"

    set_log_file(str(log_file))
    logger = get_logger()
    logger.log_clustering(
        method="leiden",
        parameters={
            "feature_representation": "principal_components",
            "pca_selection_mode": "variance",
            "pca_requested_variance": 0.95,
            "pca_requested_n_components": None,
            "pca_n_components_retained": 8,
            "pca_variance_retained": 0.963,
            "pca_input_feature_count": 42,
        },
        features_used=["CD45_mean", "CD3_mean"],
        n_clusters=5,
        acquisitions=["ROI_1"],
        source_file="example.mcd",
    )

    exporter = AnalysisStepsExporter()
    assert exporter.export_analysis_steps(str(output_file))

    content = output_file.read_text()
    assert "feature_representation: principal_components" in content
    assert "pca_selection_mode: variance" in content
    assert "pca_n_components_retained: 8" in content
    assert "pca_variance_retained: 0.963" in content
    assert "pca_input_feature_count: 42" in content
