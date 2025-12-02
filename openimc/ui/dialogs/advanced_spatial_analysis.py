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
Advanced Spatial Analysis Dialog for OpenIMC

This module provides the advanced spatial analysis dialog using squidpy.
"""

import os
import sys
# CRITICAL: Configure dask BEFORE any imports that might trigger dask.dataframe import
# This must be done at the very top, before any other imports
os.environ.setdefault('DASK_DATAFRAME__QUERY_PLANNING', 'False')

# Also try direct config if dask is available (must be before squidpy import)
try:
    import dask
    # Check if dask.dataframe has already been imported (too late to configure)
    dask_dataframe_imported = 'dask.dataframe' in sys.modules
    if dask_dataframe_imported:
        print("[DEBUG] WARNING: dask.dataframe already imported - configuration may be too late!")
    else:
        print("[DEBUG] dask.dataframe not yet imported - configuring dask...")
    # Set configuration before dask.dataframe is imported
    dask.config.set({'dataframe.query-planning': False})
    print("[DEBUG] Configured dask: dataframe.query-planning = False")
except (ImportError, AttributeError) as e:
    print(f"[DEBUG] Could not configure dask: {e}")
    pass
except Exception as e:
    print(f"[DEBUG] Unexpected error configuring dask: {e}")
    import traceback
    traceback.print_exc()
    pass

from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial import Delaunay
import seaborn as sns
from openimc.ui.dialogs.figure_save_dialog import save_figure_with_options
from openimc.ui.dialogs.spatial_analysis import (
    SourceFileFilterDialog,
    _HAVE_SQUIDPY,
    _HAVE_SPARSE,
    _get_vivid_colors
)
from openimc.core import (
    dataframe_to_anndata,
    build_spatial_graph_anndata,
    spatial_neighborhood_enrichment,
    spatial_cooccurrence,
    spatial_autocorrelation,
    spatial_ripley,
    export_anndata
)

print("[DEBUG] advanced_spatial_analysis.py: Starting squidpy import check...")
print(f"[DEBUG] _HAVE_SQUIDPY (from spatial_analysis): {_HAVE_SQUIDPY}")
try:
    # Suppress FutureWarning about anndata.read_text deprecation and squidpy __version__ deprecation
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning, message='.*read_text.*')
        warnings.filterwarnings('ignore', category=FutureWarning, message='.*__version__.*')
        print("[DEBUG] Attempting to import squidpy...")
        import squidpy as sq
        print(f"[DEBUG] Successfully imported squidpy, version: {getattr(sq, '__version__', 'unknown')}")
        print("[DEBUG] Attempting to import anndata...")
        import anndata as ad
        print(f"[DEBUG] Successfully imported anndata, version: {getattr(ad, '__version__', 'unknown')}")
    _HAVE_SQUIDPY_LOCAL = True
    _SQUIDPY_IMPORT_ERROR = None
    print("[DEBUG] _HAVE_SQUIDPY_LOCAL set to True")
except (ImportError, RuntimeError) as e:
    sq = None
    ad = None
    _HAVE_SQUIDPY_LOCAL = False
    _SQUIDPY_IMPORT_ERROR = str(e)
    print(f"[DEBUG] FAILED to import squidpy/anndata: {e}")
    print(f"[DEBUG] Exception type: {type(e)}")
    import traceback
    print(f"[DEBUG] Traceback:")
    traceback.print_exc()
    # Log the error for debugging
    import warnings
    warnings.warn(f"Failed to import squidpy in advanced_spatial_analysis: {e}. Squidpy features will be disabled.", ImportWarning)
except Exception as e:
    sq = None
    ad = None
    _HAVE_SQUIDPY_LOCAL = False
    _SQUIDPY_IMPORT_ERROR = str(e)
    print(f"[DEBUG] UNEXPECTED ERROR importing squidpy/anndata: {e}")
    print(f"[DEBUG] Exception type: {type(e)}")
    import traceback
    print(f"[DEBUG] Traceback:")
    traceback.print_exc()

try:
    from scipy import sparse as sp
except Exception:
    sp = None


class AdvancedSpatialAnalysisDialog(QtWidgets.QDialog):
    """Advanced Spatial Analysis Dialog using squidpy for all analyses."""
    def __init__(self, feature_dataframe: pd.DataFrame, batch_corrected_dataframe=None, parent=None):
        print("[DEBUG] AdvancedSpatialAnalysisDialog.__init__: Starting initialization...")
        print(f"[DEBUG] _HAVE_SQUIDPY (from spatial_analysis): {_HAVE_SQUIDPY}")
        print(f"[DEBUG] _HAVE_SQUIDPY_LOCAL (local import): {_HAVE_SQUIDPY_LOCAL}")
        print(f"[DEBUG] _SQUIDPY_IMPORT_ERROR: {_SQUIDPY_IMPORT_ERROR if '_SQUIDPY_IMPORT_ERROR' in globals() else 'N/A'}")
        print(f"[DEBUG] sq is None: {sq is None}")
        print(f"[DEBUG] ad is None: {ad is None}")
        
        # Check both the imported flag and local import status
        if not _HAVE_SQUIDPY and not _HAVE_SQUIDPY_LOCAL:
            print("[DEBUG] Both _HAVE_SQUIDPY and _HAVE_SQUIDPY_LOCAL are False - raising error")
            error_msg = "squidpy is required for AdvancedSpatialAnalysisDialog. Please install with: pip install squidpy anndata"
            if '_SQUIDPY_IMPORT_ERROR' in globals() and _SQUIDPY_IMPORT_ERROR:
                error_msg += f"\n\nImport error details: {_SQUIDPY_IMPORT_ERROR}"
            raise RuntimeError(error_msg)
        
        print("[DEBUG] Squidpy check passed, continuing initialization...")
        
        super().__init__(parent)
        self.setWindowTitle("Advanced Spatial Analysis (Squidpy)")
        self.setMinimumSize(900, 650)
        
        # Set size to 90% of parent window size if parent exists
        if parent is not None:
            parent_size = parent.size()
            dialog_width = int(parent_size.width() * 0.9)
            dialog_height = int(parent_size.height() * 0.9)
            self.resize(dialog_width, dialog_height)

        self.original_feature_dataframe = feature_dataframe
        self.batch_corrected_dataframe = batch_corrected_dataframe
        if batch_corrected_dataframe is not None and not batch_corrected_dataframe.empty:
            self.feature_dataframe = batch_corrected_dataframe.copy()
        else:
            self.feature_dataframe = feature_dataframe.copy()
        
        # Store AnnData objects per ROI
        self.anndata_cache: Dict[str, 'ad.AnnData'] = {}
        self.spatial_graph_built = False
        
        # Track which analyses have been run per ROI
        self.analysis_status: Dict[str, Dict[str, bool]] = {}  # {roi_id: {analysis_type: bool}}
        
        
        # Cluster annotation mapping
        self.cluster_annotation_map = {}
        
        # Source file filtering
        self.selected_source_files = set()
        self.available_source_files = set()
        
        # Track processed ROIs
        self.processed_rois: Dict[str, Dict[str, Any]] = {}  # {roi_id: {graph_built: bool, analyses: []}}
        
        # Store aggregated results for plotting
        self.aggregated_results: Dict[str, Any] = {}  # {analysis_type: aggregated_data}
        
        self._create_ui()
        
        if hasattr(self, 'source_file_status_label'):
            self._update_source_file_status_label()
    
    def _get_roi_column(self):
        """Get the appropriate ROI column name."""
        if self.feature_dataframe is not None and 'source_well' in self.feature_dataframe.columns:
            return 'source_well'
        return 'acquisition_id'
    
    def _get_filtered_dataframe(self):
        """Get the filtered dataframe based on selected source files and cell filters."""
        df = self.feature_dataframe.copy()
        
        # Apply source file filtering
        if ('source_file' in df.columns and 
            hasattr(self, 'selected_source_files') and 
            self.selected_source_files and 
            len(self.selected_source_files) > 0):
            df = df[df['source_file'].isin(self.selected_source_files)]
        
        # Apply cell filters from clustering dialog if available
        filter_settings = self._get_filter_settings_from_parent()
        if filter_settings:
            df = self._apply_cell_filters(df, filter_settings)
        
        return df
    
    def _get_filter_settings_from_parent(self):
        """Get filter settings from parent's clustering dialog if available."""
        parent = self.parent()
        if parent is not None and hasattr(parent, 'clustering_dialog'):
            clustering_dialog = parent.clustering_dialog
            if clustering_dialog is not None and hasattr(clustering_dialog, 'filter_settings'):
                return clustering_dialog.filter_settings
        return None
    
    def _apply_cell_filters(self, df, filter_settings):
        """Apply cell filtering based on filter settings.
        
        Args:
            df: DataFrame to filter
            filter_settings: Dictionary with filter settings
        
        Returns:
            Filtered DataFrame
        """
        if filter_settings is None:
            return df
        
        filtered_df = df.copy()
        
        # Exclude cells touching edge
        if filter_settings.get('exclude_edge_cells', False):
            if 'touches_edge' in filtered_df.columns:
                # Use .eq(False) instead of ~ to avoid numpy boolean subtraction issues
                filtered_df = filtered_df[filtered_df['touches_edge'].astype(bool).eq(False)]
            elif 'touches_border' in filtered_df.columns:
                # Fallback to touches_border if touches_edge not available
                filtered_df = filtered_df[filtered_df['touches_border'].astype(bool).eq(False)]
        
        # Filter by area
        if 'area_um2' in filtered_df.columns:
            min_area = filter_settings.get('min_area')
            max_area = filter_settings.get('max_area')
            
            if min_area is not None:
                filtered_df = filtered_df[filtered_df['area_um2'] >= min_area]
            if max_area is not None:
                filtered_df = filtered_df[filtered_df['area_um2'] <= max_area]
        
        return filtered_df
    
    def _get_pixel_size_um(self, roi_id: str) -> float:
        """Get pixel size in micrometers for a specific ROI."""
        pixel_size_um = 1.0  # Default
        parent = self.parent()
        if parent is not None:
            try:
                if hasattr(parent, '_get_pixel_size_um'):
                    pixel_size_um = float(parent._get_pixel_size_um(roi_id))
            except Exception:
                pass
        return pixel_size_um
    
    def _get_or_create_anndata(self, roi_id: str) -> Optional['ad.AnnData']:
        """Get or create AnnData object for a specific ROI."""
        print(f"[DEBUG] _get_or_create_anndata: roi_id={roi_id}")
        if roi_id in self.anndata_cache:
            print(f"[DEBUG] Using cached AnnData for ROI {roi_id}")
            return self.anndata_cache[roi_id]
        
        roi_col = self._get_roi_column()
        print(f"[DEBUG] ROI column: {roi_col}")
        filtered_df = self._get_filtered_dataframe()
        print(f"[DEBUG] Filtered dataframe shape: {filtered_df.shape}")
        print(f"[DEBUG] Filtered dataframe columns: {list(filtered_df.columns)[:10]}...")
        
        # Get pixel size
        pixel_size_um = self._get_pixel_size_um(roi_id)
        
        # Use core function
        adata = dataframe_to_anndata(
            filtered_df,
            roi_id=roi_id,
            roi_column=roi_col,
            pixel_size_um=pixel_size_um
        )
        
        if adata is not None:
            print(f"[DEBUG] Created AnnData: shape={adata.shape}, var_names={list(adata.var_names)[:5]}...")
            # Ensure cluster columns are categorical (required by squidpy)
            for col in ['cluster', 'cluster_phenotype', 'cluster_id']:
                if col in adata.obs.columns:
                    adata.obs[col] = adata.obs[col].astype('category')
                    print(f"[DEBUG] Set {col} as categorical, categories: {list(adata.obs[col].cat.categories)}")
            self.anndata_cache[roi_id] = adata
        else:
            print(f"[DEBUG] Failed to create AnnData for ROI {roi_id}")
        
        return adata
    
    def _create_ui(self):
        """Create the UI with squidpy-specific tabs."""
        layout = QtWidgets.QVBoxLayout(self)
        
        # Feature set selector
        if self.batch_corrected_dataframe is not None and not self.batch_corrected_dataframe.empty:
            feature_set_layout = QtWidgets.QHBoxLayout()
            feature_set_layout.addWidget(QtWidgets.QLabel("Feature Set:"))
            self.feature_set_combo = QtWidgets.QComboBox()
            self.feature_set_combo.addItem("Original Features")
            self.feature_set_combo.addItem("Batch-Corrected Features")
            self.feature_set_combo.setCurrentText("Batch-Corrected Features")
            self.feature_set_combo.currentTextChanged.connect(self._on_feature_set_changed)
            feature_set_layout.addWidget(self.feature_set_combo)
            feature_set_layout.addStretch()
            layout.addLayout(feature_set_layout)
        
        # Source file filter
        if 'source_file' in self.feature_dataframe.columns:
            source_files = sorted(self.feature_dataframe['source_file'].dropna().unique())
            self.available_source_files = set(source_files)
            if len(source_files) > 1:
                source_file_layout = QtWidgets.QHBoxLayout()
                source_file_layout.addWidget(QtWidgets.QLabel("Source Files:"))
                self.source_file_status_label = QtWidgets.QLabel("All files")
                source_file_layout.addWidget(self.source_file_status_label)
                self.source_file_config_btn = QtWidgets.QPushButton("Configure...")
                self.source_file_config_btn.clicked.connect(self._open_source_file_filter_dialog)
                source_file_layout.addWidget(self.source_file_config_btn)
                source_file_layout.addStretch()
                layout.addLayout(source_file_layout)
        
        # Graph creation section - Spatial Graph Construction
        params_group = QtWidgets.QGroupBox("Spatial Graph Construction")
        params_group_layout = QtWidgets.QVBoxLayout(params_group)
        params_layout = QtWidgets.QGridLayout()

        self.graph_method_combo = QtWidgets.QComboBox()
        self.graph_method_combo.addItems(["kNN", "Radius", "Delaunay"])
        self.graph_method_combo.currentTextChanged.connect(self._on_graph_method_changed)
        
        self.graph_k_spin = QtWidgets.QSpinBox()
        self.graph_k_spin.setRange(1, 64)
        self.graph_k_spin.setValue(20)
        
        self.graph_radius_spin = QtWidgets.QDoubleSpinBox()
        self.graph_radius_spin.setRange(0.1, 500.0)
        self.graph_radius_spin.setDecimals(1)
        self.graph_radius_spin.setValue(20.0)
        
        params_layout.addWidget(QtWidgets.QLabel("Mode:"), 0, 0)
        params_layout.addWidget(self.graph_method_combo, 0, 1)
        
        self.k_label = QtWidgets.QLabel("k:")
        params_layout.addWidget(self.k_label, 0, 2)
        params_layout.addWidget(self.graph_k_spin, 0, 3)
        
        self.radius_label = QtWidgets.QLabel("Radius (µm):")
        params_layout.addWidget(self.radius_label, 0, 4)
        params_layout.addWidget(self.graph_radius_spin, 0, 5)
        
        params_layout.addWidget(QtWidgets.QLabel("Random Seed:"), 0, 6)
        self.seed_spinbox = QtWidgets.QSpinBox()
        self.seed_spinbox.setRange(0, 2**31 - 1)
        self.seed_spinbox.setValue(42)
        self.seed_spinbox.setToolTip("Random seed for reproducibility (default: 42)")
        params_layout.addWidget(self.seed_spinbox, 0, 7)
        
        self.create_graph_btn = QtWidgets.QPushButton("Build Graph")
        self.create_graph_btn.setToolTip("Build the spatial graph using the selected mode and parameters")
        params_layout.addWidget(self.create_graph_btn, 0, 8)
        
        # Status label below the grid
        status_layout = QtWidgets.QHBoxLayout()
        self.graph_status_label = QtWidgets.QLabel("Graph not created")
        status_layout.addWidget(self.graph_status_label)
        status_layout.addStretch()
        
        params_group_layout.addLayout(params_layout)
        params_group_layout.addLayout(status_layout)
        
        self._on_graph_method_changed()
        layout.addWidget(params_group)
        
        # Actions
        action_row = QtWidgets.QHBoxLayout()
        self.export_anndata_btn = QtWidgets.QPushButton("Export to AnnData…")
        self.export_anndata_btn.setToolTip("Export data to AnnData format")
        action_row.addWidget(self.export_anndata_btn)
        action_row.addStretch(1)
        layout.addLayout(action_row)
        
        # Tabs for squidpy-specific analyses
        self.tabs = QtWidgets.QTabWidget()
        
        # Create squidpy-specific tabs
        self._create_squidpy_tabs()
        
        layout.addWidget(self.tabs, 1)
        
        # Wire signals
        self.export_anndata_btn.clicked.connect(self._export_to_anndata)
        self.create_graph_btn.clicked.connect(self._create_spatial_graph)
        
        # Initialize
        self._load_cluster_annotations()
        self._populate_roi_combo()
        self._update_button_states()
    
    def _create_squidpy_tabs(self):
        """Create squidpy-specific analysis tabs."""
        # Note: Spatial Graph tab removed - merged into Spatial Visualization tab
        
        # Add other squidpy tabs
        self._create_sq_nhood_tab()
        self._create_sq_cooccur_tab()
        self._create_sq_autocorr_tab()
        self._create_sq_ripley_tab()
    
    def _create_sq_nhood_tab(self):
        """Create neighborhood enrichment tab."""
        self.sq_nhood_tab = QtWidgets.QWidget()
        sq_nhood_layout = QtWidgets.QVBoxLayout(self.sq_nhood_tab)
        
        sq_nhood_info = QtWidgets.QLabel(
            "Test whether certain cell types are observed as neighbors more or less than expected by chance.\n"
            "Produces a cluster-cluster enrichment matrix."
        )
        sq_nhood_info.setWordWrap(True)
        sq_nhood_layout.addWidget(sq_nhood_info)
        
        sq_nhood_params = QtWidgets.QHBoxLayout()
        sq_nhood_params.addWidget(QtWidgets.QLabel("ROI (optional, 'All' for aggregation):"))
        self.sq_nhood_roi_combo = QtWidgets.QComboBox()
        self.sq_nhood_roi_combo.addItem("All ROIs", None)  # None means all ROIs
        sq_nhood_params.addWidget(self.sq_nhood_roi_combo)
        
        # Aggregation label and combo - will be shown/hidden based on ROI selection
        self.sq_nhood_agg_label = QtWidgets.QLabel("Aggregation:")
        self.sq_nhood_agg_combo = QtWidgets.QComboBox()
        self.sq_nhood_agg_combo.addItems(["Mean", "Sum"])
        sq_nhood_params.addWidget(self.sq_nhood_agg_label)
        sq_nhood_params.addWidget(self.sq_nhood_agg_combo)
        
        sq_nhood_params.addWidget(QtWidgets.QLabel("Cluster column:"))
        self.sq_nhood_cluster_combo = QtWidgets.QComboBox()
        self.sq_nhood_cluster_combo.addItems(["cluster", "cluster_phenotype", "cluster_id"])
        sq_nhood_params.addWidget(self.sq_nhood_cluster_combo)
        sq_nhood_params.addStretch()
        
        sq_nhood_btn_layout = QtWidgets.QHBoxLayout()
        self.sq_nhood_run_btn = QtWidgets.QPushButton("Run Neighborhood Enrichment")
        self.sq_nhood_save_btn = QtWidgets.QPushButton("Save Plot")
        self.sq_nhood_save_btn.setEnabled(False)
        sq_nhood_btn_layout.addWidget(self.sq_nhood_run_btn)
        sq_nhood_btn_layout.addWidget(self.sq_nhood_save_btn)
        sq_nhood_btn_layout.addStretch()
        
        sq_nhood_layout.addLayout(sq_nhood_params)
        sq_nhood_layout.addLayout(sq_nhood_btn_layout)
        
        # Add navigation toolbar
        self.sq_nhood_canvas = FigureCanvas(Figure(figsize=(8, 6)))
        self.sq_nhood_toolbar = NavigationToolbar(self.sq_nhood_canvas, self)
        sq_nhood_layout.addWidget(self.sq_nhood_toolbar)
        sq_nhood_layout.addWidget(self.sq_nhood_canvas)
        self.tabs.addTab(self.sq_nhood_tab, "Neighborhood Enrichment")
        
        self.sq_nhood_run_btn.clicked.connect(self._run_sq_nhood_enrichment)
        self.sq_nhood_save_btn.clicked.connect(self._save_sq_nhood_plot)
        self.sq_nhood_roi_combo.currentIndexChanged.connect(self._on_sq_nhood_roi_changed)
        self.sq_nhood_agg_combo.currentIndexChanged.connect(self._on_sq_nhood_agg_changed)
        self.sq_nhood_cluster_combo.currentIndexChanged.connect(self._on_sq_nhood_cluster_changed)
    
    def _create_sq_cooccur_tab(self):
        """Create co-occurrence tab."""
        self.sq_cooccur_tab = QtWidgets.QWidget()
        sq_cooccur_layout = QtWidgets.QVBoxLayout(self.sq_cooccur_tab)
        
        sq_cooccur_info = QtWidgets.QLabel(
            "Measure how likely two cell types are to co-occur within neighborhoods of increasing radii.\n"
            "Gives co-occurrence curves across distance scales."
        )
        sq_cooccur_info.setWordWrap(True)
        sq_cooccur_layout.addWidget(sq_cooccur_info)
        
        sq_cooccur_params = QtWidgets.QHBoxLayout()
        sq_cooccur_params.addWidget(QtWidgets.QLabel("ROI (optional, 'All' for aggregation):"))
        self.sq_cooccur_roi_combo = QtWidgets.QComboBox()
        self.sq_cooccur_roi_combo.addItem("All ROIs", None)
        sq_cooccur_params.addWidget(self.sq_cooccur_roi_combo)
        
        sq_cooccur_params.addWidget(QtWidgets.QLabel("Aggregation:"))
        self.sq_cooccur_agg_combo = QtWidgets.QComboBox()
        self.sq_cooccur_agg_combo.addItems(["Mean", "Sum"])
        sq_cooccur_params.addWidget(self.sq_cooccur_agg_combo)
        
        sq_cooccur_params.addWidget(QtWidgets.QLabel("Cluster column:"))
        self.sq_cooccur_cluster_combo = QtWidgets.QComboBox()
        self.sq_cooccur_cluster_combo.addItems(["cluster", "cluster_phenotype", "cluster_id"])
        sq_cooccur_params.addWidget(self.sq_cooccur_cluster_combo)
        
        sq_cooccur_params.addWidget(QtWidgets.QLabel("Reference cluster:"))
        self.sq_cooccur_ref_cluster_combo = QtWidgets.QComboBox()
        self.sq_cooccur_ref_cluster_combo.addItem("All clusters", None)
        sq_cooccur_params.addWidget(self.sq_cooccur_ref_cluster_combo)
        
        interval_label = QtWidgets.QLabel("Interval (comma-separated distances, e.g., '10,20,30,50'):")
        interval_label.setToolTip("Co-occurrence requires multiple distances to create curves. Provide at least 2 distances separated by commas (e.g., '10,20,30,50').")
        sq_cooccur_params.addWidget(interval_label)
        self.sq_cooccur_sizes_edit = QtWidgets.QLineEdit("10,20,30,50,100")
        self.sq_cooccur_sizes_edit.setToolTip("Enter multiple distances separated by commas. At least 2 distances are required (e.g., '10,20,30,50').")
        sq_cooccur_params.addWidget(self.sq_cooccur_sizes_edit)
        sq_cooccur_params.addStretch()
        
        sq_cooccur_btn_layout = QtWidgets.QHBoxLayout()
        self.sq_cooccur_run_btn = QtWidgets.QPushButton("Run Co-occurrence Analysis")
        self.sq_cooccur_save_btn = QtWidgets.QPushButton("Save Plot")
        self.sq_cooccur_save_btn.setEnabled(False)
        sq_cooccur_btn_layout.addWidget(self.sq_cooccur_run_btn)
        sq_cooccur_btn_layout.addWidget(self.sq_cooccur_save_btn)
        sq_cooccur_btn_layout.addStretch()
        
        # Plot type selection
        sq_cooccur_plot_layout = QtWidgets.QHBoxLayout()
        sq_cooccur_plot_layout.addWidget(QtWidgets.QLabel("Plot type:"))
        self.sq_cooccur_plot_type_combo = QtWidgets.QComboBox()
        self.sq_cooccur_plot_type_combo.addItems(["Curves", "Heatmap"])
        self.sq_cooccur_plot_type_combo.setToolTip("Curves: Show co-occurrence across all distances. Heatmap: Show co-occurrence matrix at a selected distance.")
        sq_cooccur_plot_layout.addWidget(self.sq_cooccur_plot_type_combo)
        
        sq_cooccur_plot_layout.addWidget(QtWidgets.QLabel("Distance for heatmap:"))
        self.sq_cooccur_distance_combo = QtWidgets.QComboBox()
        self.sq_cooccur_distance_combo.setToolTip("Select a distance from the interval to display in the heatmap.")
        self.sq_cooccur_distance_combo.setEnabled(False)  # Enabled only when heatmap is selected
        sq_cooccur_plot_layout.addWidget(self.sq_cooccur_distance_combo)
        sq_cooccur_plot_layout.addStretch()
        
        sq_cooccur_layout.addLayout(sq_cooccur_params)
        sq_cooccur_layout.addLayout(sq_cooccur_btn_layout)
        sq_cooccur_layout.addLayout(sq_cooccur_plot_layout)
        
        # Add navigation toolbar
        self.sq_cooccur_canvas = FigureCanvas(Figure(figsize=(8, 6)))
        self.sq_cooccur_toolbar = NavigationToolbar(self.sq_cooccur_canvas, self)
        sq_cooccur_layout.addWidget(self.sq_cooccur_toolbar)
        sq_cooccur_layout.addWidget(self.sq_cooccur_canvas)
        self.tabs.addTab(self.sq_cooccur_tab, "Co-occurrence")
        
        self.sq_cooccur_run_btn.clicked.connect(self._run_sq_cooccurrence)
        self.sq_cooccur_save_btn.clicked.connect(self._save_sq_cooccur_plot)
        self.sq_cooccur_roi_combo.currentIndexChanged.connect(self._on_sq_cooccur_roi_changed)
        self.sq_cooccur_ref_cluster_combo.currentIndexChanged.connect(self._on_sq_cooccur_ref_cluster_changed)
        self.sq_cooccur_cluster_combo.currentIndexChanged.connect(self._on_sq_cooccur_cluster_column_changed)
        self.sq_cooccur_plot_type_combo.currentTextChanged.connect(self._on_sq_cooccur_plot_type_changed)
        self.sq_cooccur_distance_combo.currentIndexChanged.connect(self._on_sq_cooccur_distance_changed)
        
        # Store interval for distance selection
        self.sq_cooccur_interval = None
    
    def _create_sq_autocorr_tab(self):
        """Create spatial autocorrelation tab."""
        self.sq_autocorr_tab = QtWidgets.QWidget()
        sq_autocorr_layout = QtWidgets.QVBoxLayout(self.sq_autocorr_tab)
        
        sq_autocorr_info = QtWidgets.QLabel(
            "Compute Moran's I spatial autocorrelation for marker expression. "
            "Tests whether expression of markers is spatially clustered. "
            "Positive values indicate spatial clustering, negative values indicate spatial dispersion. "
            "Red bars indicate statistically significant results (p < 0.05)."
        )
        sq_autocorr_info.setWordWrap(True)
        sq_autocorr_layout.addWidget(sq_autocorr_info)
        
        sq_autocorr_params = QtWidgets.QHBoxLayout()
        sq_autocorr_params.addWidget(QtWidgets.QLabel("ROI (optional, 'All' for aggregation):"))
        self.sq_autocorr_roi_combo = QtWidgets.QComboBox()
        self.sq_autocorr_roi_combo.addItem("All ROIs", None)
        sq_autocorr_params.addWidget(self.sq_autocorr_roi_combo)
        
        sq_autocorr_params.addWidget(QtWidgets.QLabel("Aggregation:"))
        self.sq_autocorr_agg_combo = QtWidgets.QComboBox()
        self.sq_autocorr_agg_combo.addItems(["Mean", "Sum"])
        sq_autocorr_params.addWidget(self.sq_autocorr_agg_combo)
        
        sq_autocorr_params.addWidget(QtWidgets.QLabel("Markers (comma-separated, or 'all'):"))
        self.sq_autocorr_markers_edit = QtWidgets.QLineEdit("all")
        sq_autocorr_params.addWidget(self.sq_autocorr_markers_edit)
        
        sq_autocorr_params.addWidget(QtWidgets.QLabel("Top K:"))
        self.sq_autocorr_topk_spin = QtWidgets.QSpinBox()
        self.sq_autocorr_topk_spin.setRange(1, 100)
        self.sq_autocorr_topk_spin.setValue(20)
        sq_autocorr_params.addWidget(self.sq_autocorr_topk_spin)
        sq_autocorr_params.addStretch()
        
        sq_autocorr_btn_layout = QtWidgets.QHBoxLayout()
        self.sq_autocorr_run_btn = QtWidgets.QPushButton("Run Spatial Autocorrelation")
        self.sq_autocorr_save_btn = QtWidgets.QPushButton("Save Plot")
        self.sq_autocorr_save_btn.setEnabled(False)
        sq_autocorr_btn_layout.addWidget(self.sq_autocorr_run_btn)
        sq_autocorr_btn_layout.addWidget(self.sq_autocorr_save_btn)
        sq_autocorr_btn_layout.addStretch()
        
        sq_autocorr_layout.addLayout(sq_autocorr_params)
        sq_autocorr_layout.addLayout(sq_autocorr_btn_layout)
        
        # Add navigation toolbar
        self.sq_autocorr_canvas = FigureCanvas(Figure(figsize=(8, 6)))
        self.sq_autocorr_toolbar = NavigationToolbar(self.sq_autocorr_canvas, self)
        sq_autocorr_layout.addWidget(self.sq_autocorr_toolbar)
        sq_autocorr_layout.addWidget(self.sq_autocorr_canvas)
        self.tabs.addTab(self.sq_autocorr_tab, "Spatial Autocorrelation")
        
        self.sq_autocorr_run_btn.clicked.connect(self._run_sq_autocorrelation)
        self.sq_autocorr_save_btn.clicked.connect(self._save_sq_autocorr_plot)
        self.sq_autocorr_roi_combo.currentIndexChanged.connect(self._on_sq_autocorr_roi_changed)
        self.sq_autocorr_topk_spin.valueChanged.connect(self._on_sq_autocorr_topk_changed)
    
    def _create_sq_ripley_tab(self):
        """Create Ripley tab."""
        self.sq_ripley_tab = QtWidgets.QWidget()
        sq_ripley_layout = QtWidgets.QVBoxLayout(self.sq_ripley_tab)
        
        sq_ripley_info = QtWidgets.QLabel(
            "Compute Ripley's F, G, and L functions using squidpy for spatial clustering analysis."
        )
        sq_ripley_info.setWordWrap(True)
        sq_ripley_layout.addWidget(sq_ripley_info)
        
        sq_ripley_params = QtWidgets.QHBoxLayout()
        sq_ripley_params.addWidget(QtWidgets.QLabel("ROI (optional, 'All' for aggregation):"))
        self.sq_ripley_roi_combo = QtWidgets.QComboBox()
        self.sq_ripley_roi_combo.addItem("All ROIs", None)
        sq_ripley_params.addWidget(self.sq_ripley_roi_combo)
        
        sq_ripley_params.addWidget(QtWidgets.QLabel("Aggregation:"))
        self.sq_ripley_agg_combo = QtWidgets.QComboBox()
        self.sq_ripley_agg_combo.addItems(["Mean", "Sum"])
        sq_ripley_params.addWidget(self.sq_ripley_agg_combo)
        
        sq_ripley_params.addWidget(QtWidgets.QLabel("Mode:"))
        self.sq_ripley_mode_combo = QtWidgets.QComboBox()
        self.sq_ripley_mode_combo.addItems(["F", "G", "L"])
        self.sq_ripley_mode_combo.setCurrentText("L")  # Set L as default
        self.sq_ripley_mode_combo.currentIndexChanged.connect(self._on_sq_ripley_mode_changed)
        sq_ripley_params.addWidget(self.sq_ripley_mode_combo)
        
        sq_ripley_params.addWidget(QtWidgets.QLabel("Max distance (µm):"))
        self.sq_ripley_r_max_spin = QtWidgets.QDoubleSpinBox()
        self.sq_ripley_r_max_spin.setRange(1.0, 200.0)
        self.sq_ripley_r_max_spin.setDecimals(1)
        self.sq_ripley_r_max_spin.setValue(50.0)
        sq_ripley_params.addWidget(self.sq_ripley_r_max_spin)
        
        sq_ripley_params.addWidget(QtWidgets.QLabel("Cluster column:"))
        self.sq_ripley_cluster_combo = QtWidgets.QComboBox()
        self.sq_ripley_cluster_combo.addItems(["cluster", "cluster_phenotype", "cluster_id"])
        sq_ripley_params.addWidget(self.sq_ripley_cluster_combo)
        sq_ripley_params.addStretch()
        
        sq_ripley_btn_layout = QtWidgets.QHBoxLayout()
        self.sq_ripley_run_btn = QtWidgets.QPushButton("Run Ripley Analysis")
        self.sq_ripley_save_btn = QtWidgets.QPushButton("Save Plot")
        self.sq_ripley_save_btn.setEnabled(False)
        sq_ripley_btn_layout.addWidget(self.sq_ripley_run_btn)
        sq_ripley_btn_layout.addWidget(self.sq_ripley_save_btn)
        sq_ripley_btn_layout.addStretch()
        
        sq_ripley_layout.addLayout(sq_ripley_params)
        sq_ripley_layout.addLayout(sq_ripley_btn_layout)
        
        # Add navigation toolbar
        self.sq_ripley_canvas = FigureCanvas(Figure(figsize=(8, 6)))
        self.sq_ripley_toolbar = NavigationToolbar(self.sq_ripley_canvas, self)
        sq_ripley_layout.addWidget(self.sq_ripley_toolbar)
        sq_ripley_layout.addWidget(self.sq_ripley_canvas)
        self.tabs.addTab(self.sq_ripley_tab, "Ripley Functions")
        
        self.sq_ripley_run_btn.clicked.connect(self._run_sq_ripley)
        self.sq_ripley_save_btn.clicked.connect(self._save_sq_ripley_plot)
        self.sq_ripley_roi_combo.currentIndexChanged.connect(self._on_sq_ripley_roi_changed)
        self.sq_ripley_cluster_combo.currentIndexChanged.connect(self._on_sq_ripley_cluster_changed)
    
    def _on_graph_method_changed(self):
        """Handle method change for graph creation."""
        method = self.graph_method_combo.currentText()
        is_knn = method == "kNN"
        is_delaunay = method == "Delaunay"

        self.k_label.setVisible(is_knn)
        self.graph_k_spin.setVisible(is_knn)
        
        self.radius_label.setVisible(not is_knn and not is_delaunay)
        self.graph_radius_spin.setVisible(not is_knn and not is_delaunay)
    
    def _open_source_file_filter_dialog(self):
        """Open source file filter dialog."""
        dlg = SourceFileFilterDialog(
            self.available_source_files,
            self.selected_source_files,
            self
        )
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            selected = dlg.get_selected_files()
            if len(selected) == len(self.available_source_files):
                self.selected_source_files = set()
            else:
                self.selected_source_files = selected
            self._update_source_file_status_label()
    
    def _update_source_file_status_label(self):
        """Update source file status label."""
        if not hasattr(self, 'source_file_status_label'):
            return
        if not self.selected_source_files:
            self.source_file_status_label.setText("All files")
        else:
            count = len(self.selected_source_files)
            total = len(self.available_source_files)
            if count == 1:
                file_name = list(self.selected_source_files)[0]
                self.source_file_status_label.setText(f"1 file: {file_name}")
            else:
                self.source_file_status_label.setText(f"{count} of {total} files")
    
    def _load_cluster_annotations(self):
        """Load cluster annotations from parent."""
        try:
            parent = self.parent()
            if parent is not None and hasattr(parent, 'cluster_annotation_map'):
                self.cluster_annotation_map = parent.cluster_annotation_map.copy()
        except Exception:
            pass
    
    def _populate_roi_combo(self):
        """Populate all ROI combo boxes."""
        filtered_df = self._get_filtered_dataframe()
        roi_col = self._get_roi_column()
        unique_rois = sorted(filtered_df[roi_col].unique())
        
        # List of ROI combo boxes that need "All ROIs" option (analyses)
        analysis_roi_combos = [
            'sq_nhood_roi_combo',
            'sq_cooccur_roi_combo',
            'sq_autocorr_roi_combo',
            'sq_ripley_roi_combo'
        ]
        
        # Populate analysis ROI combos with "All ROIs" option
        for combo_name in analysis_roi_combos:
            if hasattr(self, combo_name):
                combo = getattr(self, combo_name)
                # Check if "All ROIs" already exists
                has_all = False
                for i in range(combo.count()):
                    if combo.itemData(i) is None:
                        has_all = True
                        break
                if not has_all:
                    combo.insertItem(0, "All ROIs", None)
                # Update ROI list
                for i in range(combo.count() - 1, -1, -1):  # Iterate backwards to avoid index issues
                    if combo.itemData(i) is not None:
                        combo.removeItem(i)
                for roi_id in unique_rois:
                    combo.addItem(str(roi_id), roi_id)
    
    def _get_cluster_display_name(self, cluster_id):
        """Get display name for cluster with annotation if available."""
        if cluster_id in self.cluster_annotation_map:
            annotation = self.cluster_annotation_map[cluster_id]
            return f"{cluster_id} ({annotation})"
        return str(cluster_id)
    
    def _on_feature_set_changed(self):
        """Handle feature set change - invalidate cache and refresh."""
        # Update feature dataframe
        if self.feature_set_combo.currentText() == "Batch-Corrected Features":
            if self.batch_corrected_dataframe is not None and not self.batch_corrected_dataframe.empty:
                self.feature_dataframe = self.batch_corrected_dataframe.copy()
            else:
                QtWidgets.QMessageBox.warning(self, "No Batch-Corrected Data", 
                    "Batch-corrected features are not available. Using original features.")
                self.feature_set_combo.setCurrentText("Original Features")
                return
        else:
            self.feature_dataframe = self.original_feature_dataframe.copy()
        
        # Clear cache since data changed
        self.anndata_cache = {}
        self.processed_rois = {}
        self.analysis_status = {}
        self.aggregated_results = {}
        self.spatial_graph_built = False
        
        # Refresh UI
        self._populate_roi_combo()
        self._update_button_states()
        
        # Clear all plots
        for canvas_name in ['sq_nhood_canvas', 'sq_cooccur_canvas', 'sq_autocorr_canvas', 'sq_ripley_canvas']:
            if hasattr(self, canvas_name):
                canvas = getattr(self, canvas_name)
                canvas.figure.clear()
                canvas.draw()
        
        # Update status
        if hasattr(self, 'graph_status_label'):
            self.graph_status_label.setText("Graph not created (feature set changed)")
            self.graph_status_label.setStyleSheet("")
    
    def _update_button_states(self):
        """Update button enabled/disabled states based on prerequisites."""
        graph_built = self.spatial_graph_built
        
        # Buttons that require graph
        graph_dependent_buttons = [
            'sq_nhood_run_btn', 'sq_cooccur_run_btn',
            'sq_autocorr_run_btn', 'sq_ripley_run_btn'
        ]
        
        for btn_name in graph_dependent_buttons:
            if hasattr(self, btn_name):
                btn = getattr(self, btn_name)
                btn.setEnabled(graph_built)
    
    def _get_selected_roi(self, combo):
        """Get selected ROI from combo box."""
        if combo is None:
            return None
        return combo.currentData()
    
    def _on_sq_nhood_roi_changed(self):
        """Handle ROI change in neighborhood enrichment tab."""
        roi_id = self._get_selected_roi(self.sq_nhood_roi_combo)
        
        # Show/hide aggregation controls based on ROI selection
        if roi_id is None:
            # "All ROIs" selected - show aggregation options
            self.sq_nhood_agg_label.setVisible(True)
            self.sq_nhood_agg_combo.setVisible(True)
            # Use aggregated result if available
            if 'nhood_enrichment' in self.aggregated_results:
                self._plot_sq_nhood_enrichment(self.aggregated_results['nhood_enrichment'])
                self.sq_nhood_save_btn.setEnabled(True)
        else:
            # Single ROI selected - hide aggregation options and re-run analysis
            self.sq_nhood_agg_label.setVisible(False)
            self.sq_nhood_agg_combo.setVisible(False)
            
            # Auto-refresh plot for this single ROI
            if roi_id in self.anndata_cache:
                adata = self.anndata_cache[roi_id]
                if 'spatial_connectivities' in adata.obsp:
                    # Re-run enrichment for this ROI
                    cluster_key = self.sq_nhood_cluster_combo.currentText()
                    if cluster_key in adata.obs.columns:
                        try:
                            if not hasattr(adata.obs[cluster_key], 'cat'):
                                adata.obs[cluster_key] = adata.obs[cluster_key].astype('category')
                            sq.gr.nhood_enrichment(adata, cluster_key=cluster_key)
                            if 'nhood_enrichment' in adata.uns:
                                self._plot_sq_nhood_enrichment(adata)
                                self.sq_nhood_save_btn.setEnabled(True)
                        except Exception as e:
                            print(f"[DEBUG] Error re-running enrichment for ROI {roi_id}: {e}")
                elif 'nhood_enrichment' in adata.uns:
                    # Use existing results if available
                    self._plot_sq_nhood_enrichment(adata)
                    self.sq_nhood_save_btn.setEnabled(True)
    
    def _on_sq_nhood_agg_changed(self):
        """Handle aggregation change in neighborhood enrichment tab."""
        roi_id = self._get_selected_roi(self.sq_nhood_roi_combo)
        if roi_id is None and 'nhood_enrichment' in self.aggregated_results:
            # Re-aggregate and plot
            self._run_sq_nhood_enrichment()
    
    def _on_sq_cooccur_roi_changed(self):
        """Handle ROI change in co-occurrence tab."""
        roi_id = self._get_selected_roi(self.sq_cooccur_roi_combo)
        if roi_id and roi_id in self.anndata_cache:
            adata = self.anndata_cache[roi_id]
            # Update reference cluster combo (even if no co-occurrence data yet)
            self._update_cooccur_ref_cluster_combo(adata)
            # Check for co-occurrence data with alternative key names
            has_cooccur = False
            for key in adata.uns.keys():
                if 'co' in key.lower() and 'occur' in key.lower():
                    has_cooccur = True
                    break
            if has_cooccur:
                self._plot_sq_cooccurrence(adata)
                self.sq_cooccur_save_btn.setEnabled(True)
        elif roi_id is None:
            # "All ROIs" selected - try to populate from first available ROI
            if self.anndata_cache:
                first_adata = list(self.anndata_cache.values())[0]
                self._update_cooccur_ref_cluster_combo(first_adata)
    
    def _on_sq_cooccur_ref_cluster_changed(self):
        """Handle reference cluster change in co-occurrence tab."""
        roi_id = self._get_selected_roi(self.sq_cooccur_roi_combo)
        if roi_id and roi_id in self.anndata_cache:
            adata = self.anndata_cache[roi_id]
            # Check for co-occurrence data with alternative key names
            has_cooccur = False
            for key in adata.uns.keys():
                if 'co' in key.lower() and 'occur' in key.lower():
                    has_cooccur = True
                    break
            if has_cooccur:
                self._plot_sq_cooccurrence(adata)
                self.sq_cooccur_save_btn.setEnabled(True)
    
    def _on_sq_cooccur_cluster_column_changed(self):
        """Handle cluster column change in co-occurrence tab."""
        roi_id = self._get_selected_roi(self.sq_cooccur_roi_combo)
        if roi_id and roi_id in self.anndata_cache:
            adata = self.anndata_cache[roi_id]
            # Update reference cluster combo
            self._update_cooccur_ref_cluster_combo(adata)
            # Check for co-occurrence data with alternative key names
            has_cooccur = False
            for key in adata.uns.keys():
                if 'co' in key.lower() and 'occur' in key.lower():
                    has_cooccur = True
                    break
            if has_cooccur:
                self._plot_sq_cooccurrence(adata)
                self.sq_cooccur_save_btn.setEnabled(True)
    
    def _on_sq_cooccur_plot_type_changed(self):
        """Handle plot type change in co-occurrence tab."""
        plot_type = self.sq_cooccur_plot_type_combo.currentText()
        # Enable/disable distance combo based on plot type
        self.sq_cooccur_distance_combo.setEnabled(plot_type == "Heatmap")
        
        # Replot if data is available
        roi_id = self._get_selected_roi(self.sq_cooccur_roi_combo)
        if roi_id and roi_id in self.anndata_cache:
            adata = self.anndata_cache[roi_id]
            # Check for co-occurrence data
            has_cooccur = False
            for key in adata.uns.keys():
                if 'co' in key.lower() and 'occur' in key.lower():
                    has_cooccur = True
                    break
            if has_cooccur:
                self._plot_sq_cooccurrence(adata)
    
    def _on_sq_cooccur_distance_changed(self):
        """Handle distance selection change for heatmap in co-occurrence tab."""
        # Replot if heatmap is selected and data is available
        plot_type = self.sq_cooccur_plot_type_combo.currentText()
        if plot_type == "Heatmap":
            roi_id = self._get_selected_roi(self.sq_cooccur_roi_combo)
            if roi_id and roi_id in self.anndata_cache:
                adata = self.anndata_cache[roi_id]
                # Check for co-occurrence data
                has_cooccur = False
                for key in adata.uns.keys():
                    if 'co' in key.lower() and 'occur' in key.lower():
                        has_cooccur = True
                        break
                if has_cooccur:
                    self._plot_sq_cooccurrence(adata)
    
    def _on_sq_nhood_cluster_changed(self):
        """Handle cluster column change in neighborhood enrichment tab."""
        roi_id = self._get_selected_roi(self.sq_nhood_roi_combo)
        if roi_id is None:
            # Re-run aggregation if "All ROIs" is selected
            if 'nhood_enrichment' in self.aggregated_results:
                self._run_sq_nhood_enrichment()
        elif roi_id in self.anndata_cache:
            adata = self.anndata_cache[roi_id]
            if 'spatial_connectivities' in adata.obsp:
                cluster_key = self.sq_nhood_cluster_combo.currentText()
                if cluster_key in adata.obs.columns:
                    try:
                        if not hasattr(adata.obs[cluster_key], 'cat'):
                            adata.obs[cluster_key] = adata.obs[cluster_key].astype('category')
                        sq.gr.nhood_enrichment(adata, cluster_key=cluster_key)
                        if 'nhood_enrichment' in adata.uns:
                            self._plot_sq_nhood_enrichment(adata)
                            self.sq_nhood_save_btn.setEnabled(True)
                    except Exception as e:
                        print(f"[DEBUG] Error re-running enrichment for ROI {roi_id}: {e}")
    
    def _update_cooccur_ref_cluster_combo(self, adata: 'ad.AnnData', preserve_selection: bool = True):
        """Update the reference cluster combo box with available clusters."""
        cluster_key = self.sq_cooccur_cluster_combo.currentText()
        
        # Preserve current selection if requested
        current_selection = None
        if preserve_selection:
            current_selection = self.sq_cooccur_ref_cluster_combo.currentData()
        
        self.sq_cooccur_ref_cluster_combo.clear()
        self.sq_cooccur_ref_cluster_combo.addItem("All clusters", None)
        
        if cluster_key in adata.obs.columns:
            if hasattr(adata.obs[cluster_key], 'cat'):
                categories = list(adata.obs[cluster_key].cat.categories)
            else:
                categories = sorted(adata.obs[cluster_key].unique())
            
            for cat in categories:
                self.sq_cooccur_ref_cluster_combo.addItem(
                    self._get_cluster_display_name(cat), cat
                )
            
            # Restore previous selection if it still exists, otherwise set default
            if preserve_selection and current_selection is not None:
                # Try to restore the previous selection
                for i in range(self.sq_cooccur_ref_cluster_combo.count()):
                    if self.sq_cooccur_ref_cluster_combo.itemData(i) == current_selection:
                        self.sq_cooccur_ref_cluster_combo.setCurrentIndex(i)
                        return
            
            # Set default to "1" if it exists, otherwise use first cluster
            default_cluster = None
            # Try to find cluster "1" (as string or int)
            for cat in categories:
                if str(cat) == "1" or cat == 1:
                    default_cluster = cat
                    break
            
            # If "1" not found, use first cluster
            if default_cluster is None and len(categories) > 0:
                default_cluster = categories[0]
            
            # Set the default selection
            if default_cluster is not None:
                for i in range(self.sq_cooccur_ref_cluster_combo.count()):
                    if self.sq_cooccur_ref_cluster_combo.itemData(i) == default_cluster:
                        self.sq_cooccur_ref_cluster_combo.setCurrentIndex(i)
                        break
    
    def _on_sq_autocorr_roi_changed(self):
        """Handle ROI change in autocorrelation tab."""
        roi_id = self._get_selected_roi(self.sq_autocorr_roi_combo)
        if roi_id is None:
            # "All ROIs" selected - use aggregated result if available
            if 'autocorrelation' in self.aggregated_results:
                self._plot_sq_autocorrelation(self.aggregated_results['autocorrelation'])
                self.sq_autocorr_save_btn.setEnabled(True)
        elif roi_id in self.anndata_cache:
            adata = self.anndata_cache[roi_id]
            if 'moranI' in adata.uns:
                self._plot_sq_autocorrelation(adata)
                self.sq_autocorr_save_btn.setEnabled(True)
    
    def _on_sq_ripley_roi_changed(self):
        """Handle ROI change in Ripley tab."""
        roi_id = self._get_selected_roi(self.sq_ripley_roi_combo)
        if roi_id and roi_id in self.anndata_cache:
            adata = self.anndata_cache[roi_id]
            cluster_key = self.sq_ripley_cluster_combo.currentText()
            # Check for any ripley key
            has_ripley = False
            for key in adata.uns.keys():
                if 'ripley' in key.lower():
                    has_ripley = True
                    break
            if has_ripley:
                self._plot_sq_ripley(adata, cluster_key)
                self.sq_ripley_save_btn.setEnabled(True)
    
    def _on_sq_ripley_mode_changed(self):
        """Handle mode change (F/G/L) in Ripley tab - auto-refresh if data exists."""
        roi_id = self._get_selected_roi(self.sq_ripley_roi_combo)
        if roi_id and roi_id in self.anndata_cache:
            adata = self.anndata_cache[roi_id]
            cluster_key = self.sq_ripley_cluster_combo.currentText()
            
            # Check if any ripley data exists (for any mode)
            has_ripley = False
            for key in adata.uns.keys():
                if 'ripley' in key.lower():
                    has_ripley = True
                    break
            
            if has_ripley:
                self._plot_sq_ripley(adata, cluster_key)
                self.sq_ripley_save_btn.setEnabled(True)
    
    def _on_sq_ripley_cluster_changed(self):
        """Handle cluster column change in Ripley tab."""
        roi_id = self._get_selected_roi(self.sq_ripley_roi_combo)
        if roi_id and roi_id in self.anndata_cache:
            adata = self.anndata_cache[roi_id]
            cluster_key = self.sq_ripley_cluster_combo.currentText()
            # Check for any ripley key
            has_ripley = False
            for key in adata.uns.keys():
                if 'ripley' in key.lower():
                    has_ripley = True
                    break
            if has_ripley:
                self._plot_sq_ripley(adata, cluster_key)
                self.sq_ripley_save_btn.setEnabled(True)
    
    def _on_sq_autocorr_topk_changed(self):
        """Handle top K change in autocorrelation tab - auto-refresh plot."""
        roi_id = self._get_selected_roi(self.sq_autocorr_roi_combo)
        if roi_id is None:
            # "All ROIs" selected - use aggregated result if available
            if 'autocorrelation' in self.aggregated_results:
                self._plot_sq_autocorrelation(self.aggregated_results['autocorrelation'])
                self.sq_autocorr_save_btn.setEnabled(True)
        elif roi_id in self.anndata_cache:
            adata = self.anndata_cache[roi_id]
            if 'moranI' in adata.uns:
                self._plot_sq_autocorrelation(adata)
                self.sq_autocorr_save_btn.setEnabled(True)
    
    def _get_all_rois(self):
        """Get all ROI IDs from filtered dataframe."""
        filtered_df = self._get_filtered_dataframe()
        roi_col = self._get_roi_column()
        return sorted(filtered_df[roi_col].unique())
    
    def _create_spatial_graph(self):
        """Create spatial graph for all ROIs using core function."""
        print(f"[DEBUG] _create_spatial_graph: Starting")
        if not self._validate_data():
            print(f"[DEBUG] Data validation failed")
            return
        
        try:
            method = self.graph_method_combo.currentText()
            k = int(self.graph_k_spin.value()) if method == "kNN" else None
            radius = float(self.graph_radius_spin.value()) if method == "Radius" else None
            seed = int(self.seed_spinbox.value())
            print(f"[DEBUG] Graph method: {method}, k={k}, radius={radius}")
            
            roi_col = self._get_roi_column()
            filtered_df = self._get_filtered_dataframe()
            
            # Get pixel size (use first ROI as default)
            all_rois = self._get_all_rois()
            if not all_rois:
                QtWidgets.QMessageBox.warning(self, "No ROIs", "No ROIs found in the data.")
                return
            
            pixel_size_um = self._get_pixel_size_um(all_rois[0])
            
            # Use core function to build graph
            anndata_dict = build_spatial_graph_anndata(
                features_df=filtered_df,
                method=method,
                k_neighbors=k if k else 20,
                radius=radius,
                pixel_size_um=pixel_size_um,
                roi_column=roi_col,
                roi_id=None,  # Process all ROIs
                seed=seed
            )
            
            if anndata_dict:
                # Update cache with new AnnData objects
                self.anndata_cache.update(anndata_dict)
                self.spatial_graph_built = True
                success_count = len(anndata_dict)
                
                self.graph_status_label.setText(f"Graph created for {success_count} ROI(s)")
                self.graph_status_label.setStyleSheet("color: green;")
                
                # Update processed ROIs tracking
                for roi_id in anndata_dict.keys():
                    if roi_id not in self.processed_rois:
                        self.processed_rois[roi_id] = {}
                    self.processed_rois[roi_id]['graph_built'] = True
                
                # Populate reference cluster combo if we have data
                if self.anndata_cache:
                    first_adata = list(self.anndata_cache.values())[0]
                    cluster_key = self.sq_cooccur_cluster_combo.currentText()
                    if cluster_key in first_adata.obs.columns:
                        self._update_cooccur_ref_cluster_combo(first_adata)
                
                self._update_button_states()
                QtWidgets.QMessageBox.information(self, "Graph Created", 
                    f"Spatial graph created successfully for {success_count} ROI(s).")
            else:
                self.graph_status_label.setText("Graph creation failed")
                self.graph_status_label.setStyleSheet("color: red;")
                QtWidgets.QMessageBox.warning(self, "Graph Creation Failed", 
                    "Failed to create spatial graph for any ROI.")
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Error", f"Error creating spatial graph: {str(e)}")
    
    def _run_sq_nhood_enrichment(self):
        """Run neighborhood enrichment analysis using core function."""
        print(f"[DEBUG] _run_sq_nhood_enrichment: Starting", flush=True)
        import sys
        sys.stdout.flush()
        if not self.spatial_graph_built:
            print(f"[DEBUG] Spatial graph not built", flush=True)
            QtWidgets.QMessageBox.warning(self, "Graph Required", 
                "Please create the spatial graph first (Step 1 at the top).")
            return
        
        try:
            cluster_key = self.sq_nhood_cluster_combo.currentText()
            roi_id = self._get_selected_roi(self.sq_nhood_roi_combo)
            agg_method = self.sq_nhood_agg_combo.currentText().lower()  # "mean" or "sum"
            print(f"[DEBUG] Cluster key: {cluster_key}, ROI: {roi_id}, Aggregation: {agg_method}", flush=True)
            sys.stdout.flush()
            
            # Check if cluster column exists
            filtered_df = self._get_filtered_dataframe()
            if cluster_key not in filtered_df.columns:
                print(f"[DEBUG] Cluster column '{cluster_key}' not found in dataframe")
                QtWidgets.QMessageBox.warning(self, "Missing Column", 
                    f"Cluster column '{cluster_key}' not found in data.")
                return
            
            # Get AnnData dict - filter to selected ROI if needed
            if roi_id is None:
                # Use all cached AnnData objects with graphs
                anndata_dict = {rid: adata for rid, adata in self.anndata_cache.items() 
                               if 'spatial_connectivities' in adata.obsp}
            else:
                # Use only selected ROI
                if roi_id not in self.anndata_cache:
                    QtWidgets.QMessageBox.warning(self, "No Data", f"No data found for ROI {roi_id}.")
                    return
                adata = self.anndata_cache[roi_id]
                if 'spatial_connectivities' not in adata.obsp:
                    QtWidgets.QMessageBox.warning(self, "No Graph", f"No spatial graph found for ROI {roi_id}.")
                    return
                anndata_dict = {roi_id: adata}
            
            if not anndata_dict:
                QtWidgets.QMessageBox.warning(self, "No Data", "No AnnData objects with spatial graphs found.")
                return
            
            # Use core function
            print(f"[DEBUG] Calling spatial_neighborhood_enrichment with {len(anndata_dict)} ROI(s)", flush=True)
            import sys
            sys.stdout.flush()
            try:
                print(f"[DEBUG] About to call spatial_neighborhood_enrichment...", flush=True)
                results = spatial_neighborhood_enrichment(
                    anndata_dict=anndata_dict,
                    cluster_key=cluster_key,
                    aggregation=agg_method
                )
                print(f"[DEBUG] spatial_neighborhood_enrichment returned", flush=True)
                sys.stdout.flush()
                print(f"[DEBUG] Results keys: {list(results.keys())}", flush=True)
                print(f"[DEBUG] Results has 'results': {'results' in results}", flush=True)
                print(f"[DEBUG] Results has 'aggregated': {'aggregated' in results}", flush=True)
                print(f"[DEBUG] Results has 'significant_counts': {'significant_counts' in results}", flush=True)
                if 'results' in results:
                    print(f"[DEBUG] Number of result ROIs: {len(results['results'])}", flush=True)
                if 'aggregated' in results:
                    print(f"[DEBUG] Aggregated is None: {results['aggregated'] is None}", flush=True)
                    if results['aggregated'] is not None:
                        print(f"[DEBUG] Aggregated shape: {results['aggregated'].shape}", flush=True)
                sys.stdout.flush()
            except Exception as e:
                print(f"[DEBUG] ERROR in spatial_neighborhood_enrichment: {e}", flush=True)
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
                raise
            
            # Update cache with results
            print(f"[DEBUG] Updating cache with results")
            self.anndata_cache.update(results['results'])
            
            # Update status
            for roi_id in results['results'].keys():
                if roi_id not in self.analysis_status:
                    self.analysis_status[roi_id] = {}
                self.analysis_status[roi_id]['nhood_enrichment'] = True
            
            # Create temporary adata for aggregated plotting
            if results['aggregated'] is not None:
                print(f"[DEBUG] Creating TempAnnData for aggregated results")
                print(f"[DEBUG] Aggregated matrix type: {type(results['aggregated'])}, shape: {results['aggregated'].shape if hasattr(results['aggregated'], 'shape') else 'N/A'}")
                print(f"[DEBUG] Cluster categories: {results['cluster_categories']}")
                
                class TempAnnData:
                    def __init__(self, matrix, cluster_key, obs, significant_counts=None):
                        self.uns = {'nhood_enrichment': {'zscore': matrix}}
                        self.obs = obs
                        self._cluster_key = cluster_key
                        self._significant_counts = significant_counts
                
                import pandas as pd
                cluster_categories = results['cluster_categories']
                obs_df = pd.DataFrame({cluster_key: cluster_categories})
                obs_df.index = [str(c) for c in cluster_categories]
                obs_df[cluster_key] = obs_df[cluster_key].astype('category')
                
                temp_adata = TempAnnData(
                    results['aggregated'], 
                    cluster_key, 
                    obs_df,
                    significant_counts=results.get('significant_counts')
                )
                print(f"[DEBUG] TempAnnData created: uns keys={list(temp_adata.uns.keys())}, obs shape={obs_df.shape}")
                print(f"[DEBUG] TempAnnData.uns['nhood_enrichment'] keys={list(temp_adata.uns['nhood_enrichment'].keys())}")
                print(f"[DEBUG] TempAnnData.uns['nhood_enrichment']['zscore'] shape={temp_adata.uns['nhood_enrichment']['zscore'].shape}")
                
                self.aggregated_results['nhood_enrichment'] = temp_adata
                
                QtWidgets.QMessageBox.information(self, "Enrichment Complete", 
                    f"Neighborhood enrichment completed for {len(results['results'])} ROI(s). "
                    f"Aggregated using {agg_method}.")
                
                # Plot aggregated results
                print(f"[DEBUG] Calling _plot_sq_nhood_enrichment with TempAnnData")
                print(f"[DEBUG] TempAnnData.uns keys: {list(temp_adata.uns.keys())}")
                print(f"[DEBUG] TempAnnData has significant_counts: {hasattr(temp_adata, '_significant_counts')}")
                if hasattr(temp_adata, '_significant_counts') and temp_adata._significant_counts is not None:
                    print(f"[DEBUG] significant_counts shape: {temp_adata._significant_counts.shape}")
                # Switch to neighborhood enrichment tab to show the plot
                nhood_tab_idx = None
                for i in range(self.tabs.count()):
                    if self.tabs.tabText(i) == "Neighborhood Enrichment":
                        nhood_tab_idx = i
                        break
                if nhood_tab_idx is not None:
                    self.tabs.setCurrentIndex(nhood_tab_idx)
                    print(f"[DEBUG] Switched to Neighborhood Enrichment tab (index {nhood_tab_idx})")
                
                self._plot_sq_nhood_enrichment(temp_adata)
                # Ensure canvas is visible
                self.sq_nhood_canvas.setVisible(True)
                self.sq_nhood_canvas.show()
                self.sq_nhood_canvas.update()
                self.sq_nhood_save_btn.setEnabled(True)
            else:
                # Plot first ROI result
                if results['results']:
                    first_adata = list(results['results'].values())[0]
                    print(f"[DEBUG] Plotting first ROI result (no aggregated result)")
                    print(f"[DEBUG] First ROI adata.uns keys: {list(first_adata.uns.keys())}")
                    if 'nhood_enrichment' in first_adata.uns:
                        print(f"[DEBUG] First ROI nhood_enrichment keys: {list(first_adata.uns['nhood_enrichment'].keys()) if isinstance(first_adata.uns['nhood_enrichment'], dict) else 'N/A'}")
                    # Switch to neighborhood enrichment tab to show the plot
                    nhood_tab_idx = None
                    for i in range(self.tabs.count()):
                        if self.tabs.tabText(i) == "Neighborhood Enrichment":
                            nhood_tab_idx = i
                            break
                    if nhood_tab_idx is not None:
                        self.tabs.setCurrentIndex(nhood_tab_idx)
                        print(f"[DEBUG] Switched to Neighborhood Enrichment tab (index {nhood_tab_idx})")
                    
                    self._plot_sq_nhood_enrichment(first_adata)
                    # Ensure canvas is visible
                    self.sq_nhood_canvas.setVisible(True)
                    self.sq_nhood_canvas.show()
                    self.sq_nhood_canvas.update()
                    self.sq_nhood_save_btn.setEnabled(True)
                    QtWidgets.QMessageBox.information(self, "Enrichment Complete", 
                        f"Neighborhood enrichment completed for {len(results['results'])} ROI(s).")
                
        except Exception as e:
            print(f"[DEBUG] EXCEPTION in _run_sq_nhood_enrichment: {e}")
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Error", f"Error running enrichment: {str(e)}")
    
    def _plot_sq_nhood_enrichment(self, adata: 'ad.AnnData'):
        """Plot neighborhood enrichment results."""
        print(f"[DEBUG] _plot_sq_nhood_enrichment: Starting")
        print(f"[DEBUG] adata: {adata}")
        if adata is None:
            print(f"[DEBUG] adata is None")
            self.sq_nhood_canvas.figure.clear()
            ax = self.sq_nhood_canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No data provided for plotting.', 
                   ha='center', va='center', transform=ax.transAxes)
            self.sq_nhood_canvas.draw()
            return
        
        print(f"[DEBUG] adata.uns keys: {list(adata.uns.keys()) if hasattr(adata, 'uns') else 'N/A'}")
        
        if 'nhood_enrichment' not in adata.uns:
            print(f"[DEBUG] 'nhood_enrichment' not found in adata.uns")
            self.sq_nhood_canvas.figure.clear()
            ax = self.sq_nhood_canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No enrichment data found.\nPlease run enrichment analysis first.', 
                   ha='center', va='center', transform=ax.transAxes)
            self.sq_nhood_canvas.draw()
            return
        
        # Manual plotting (more reliable than squidpy's plotting with custom canvas)
        self.sq_nhood_canvas.figure.clear()
        ax = self.sq_nhood_canvas.figure.add_subplot(111)
        
        enrichment_data = adata.uns['nhood_enrichment']
        cluster_key = self.sq_nhood_cluster_combo.currentText()
        
        print(f"[DEBUG] enrichment_data type: {type(enrichment_data)}")
        if isinstance(enrichment_data, dict):
            print(f"[DEBUG] enrichment_data keys: {list(enrichment_data.keys())}")
        
        # Convert to numpy array if needed
        matrix = None
        if isinstance(enrichment_data, dict):
            # Squidpy stores the result in a nested structure
            # Try common keys
            if 'zscore' in enrichment_data:
                matrix = enrichment_data['zscore']
                print(f"[DEBUG] Found 'zscore' key, shape: {matrix.shape if hasattr(matrix, 'shape') else 'N/A'}")
            elif 'count' in enrichment_data:
                matrix = enrichment_data['count']
                print(f"[DEBUG] Found 'count' key, shape: {matrix.shape if hasattr(matrix, 'shape') else 'N/A'}")
            elif 'stat' in enrichment_data:
                matrix = enrichment_data['stat']
                print(f"[DEBUG] Found 'stat' key, shape: {matrix.shape if hasattr(matrix, 'shape') else 'N/A'}")
            else:
                # Try to find a matrix-like value
                print(f"[DEBUG] Searching for matrix-like value in dict")
                for key, value in enrichment_data.items():
                    print(f"[DEBUG]   Checking key '{key}': type={type(value)}, ndim={getattr(value, 'ndim', 'N/A')}")
                    if isinstance(value, np.ndarray) and value.ndim == 2:
                        matrix = value
                        print(f"[DEBUG]   Found matrix at key '{key}', shape: {matrix.shape}")
                        break
                # If still not found, try first value
                if matrix is None and len(enrichment_data) > 0:
                    first_val = list(enrichment_data.values())[0]
                    print(f"[DEBUG] Trying first value: type={type(first_val)}, ndim={getattr(first_val, 'ndim', 'N/A')}")
                    if isinstance(first_val, np.ndarray) and first_val.ndim == 2:
                        matrix = first_val
                        print(f"[DEBUG]   Using first value as matrix, shape: {matrix.shape}")
        elif isinstance(enrichment_data, np.ndarray):
            matrix = enrichment_data
            print(f"[DEBUG] enrichment_data is numpy array, shape: {matrix.shape}")
        
        print(f"[DEBUG] Final matrix: {matrix is not None}, type: {type(matrix) if matrix is not None else 'None'}")
        if matrix is not None:
            print(f"[DEBUG] Matrix shape: {matrix.shape}, ndim: {matrix.ndim}, dtype: {matrix.dtype}")
        
        if matrix is not None and isinstance(matrix, np.ndarray) and matrix.ndim == 2:
            print(f"[DEBUG] Plotting matrix with shape {matrix.shape}")
            print(f"[DEBUG] Matrix dtype: {matrix.dtype}, min: {np.nanmin(matrix)}, max: {np.nanmax(matrix)}")
            try:
                # Handle NaN/inf values
                if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):
                    print(f"[DEBUG] Found NaN/inf values, replacing")
                    # Replace with 0 for display
                    matrix = np.nan_to_num(matrix, nan=0.0, posinf=3.0, neginf=-3.0)
                
                # Determine vmin/vmax from data if reasonable
                vmin, vmax = -3, 3
                if matrix.size > 0:
                    finite_vals = matrix[np.isfinite(matrix)]
                    if len(finite_vals) > 0:
                        data_min, data_max = np.min(finite_vals), np.max(finite_vals)
                        print(f"[DEBUG] Matrix value range: [{data_min}, {data_max}]")
                        if abs(data_min) < 10 and abs(data_max) < 10:
                            vmin, vmax = data_min, data_max
                        else:
                            # Use percentiles for extreme values
                            vmin = np.percentile(finite_vals, 5)
                            vmax = np.percentile(finite_vals, 95)
                        print(f"[DEBUG] Using vmin={vmin}, vmax={vmax}")
                
                # Create square heatmap with aspect='equal' for square cells
                # Use extent to ensure proper square cells
                n_rows, n_cols = matrix.shape
                im = ax.imshow(matrix, cmap='RdBu_r', aspect='equal', vmin=vmin, vmax=vmax, 
                              interpolation='nearest', origin='upper', 
                              extent=[-0.5, n_cols - 0.5, n_rows - 0.5, -0.5])
                print(f"[DEBUG] imshow created with extent=[-0.5, {n_cols-0.5}, {n_rows-0.5}, -0.5]")
                
                # Add colorbar with proper spacing
                # Use pad to control spacing between plot and colorbar
                cbar = self.sq_nhood_canvas.figure.colorbar(im, ax=ax, label='Z-Score', pad=0.02, shrink=0.8)
                print(f"[DEBUG] Colorbar created")
                
                # Set axis limits to ensure proper display
                ax.set_xlim(-0.5, n_cols - 0.5)
                ax.set_ylim(n_rows - 0.5, -0.5)
                print(f"[DEBUG] Axis limits set: x=[-0.5, {n_cols-0.5}], y=[{n_rows-0.5}, -0.5]")
                print(f"[DEBUG] imshow and colorbar created successfully")
                
                # Add circles for significant interactions if available
                significant_counts = None
                # Try to get significant_counts from adata object
                if hasattr(adata, '_significant_counts') and adata._significant_counts is not None:
                    significant_counts = adata._significant_counts
                    print(f"[DEBUG] Found significant_counts in adata, shape: {significant_counts.shape}")
                else:
                    # For single ROI, compute from matrix (threshold = 2.0)
                    # This is a fallback - ideally we'd have it from the core function
                    significance_threshold = 2.0
                    significant_counts = (np.abs(matrix) > significance_threshold).astype(int)
                    print(f"[DEBUG] Computed significant_counts from matrix (single ROI), threshold={significance_threshold}")
                
                if significant_counts is not None and significant_counts.shape == matrix.shape:
                    print(f"[DEBUG] Adding circles for significant interactions")
                    # Normalize circle sizes (0 to max count)
                    max_count = np.max(significant_counts) if significant_counts.size > 0 else 1
                    if max_count > 0:
                        # Circle size will be proportional to count, scaled to fit within cell
                        # Use a fraction of cell size (e.g., 0.3 to 0.8 of cell width)
                        n_rows, n_cols = matrix.shape
                        # Calculate cell size in data coordinates
                        cell_size = 1.0  # Since we're using integer indices
                        min_radius = 0.15 * cell_size  # Minimum circle radius
                        max_radius = 0.4 * cell_size   # Maximum circle radius
                        
                        # Draw circles for each cell
                        # With extent=[-0.5, n_cols-0.5, n_rows-0.5, -0.5] and origin='upper',
                        # cell (i, j) has center at (j, i) in data coordinates
                        circles_added = 0
                        for i in range(n_rows):
                            for j in range(n_cols):
                                count = significant_counts[i, j]
                                if count > 0:
                                    # Scale radius based on count
                                    radius = min_radius + (max_radius - min_radius) * (count / max_count)
                                    # Center of cell in data coordinates
                                    # Row i corresponds to y = i (with origin='upper', y decreases downward)
                                    # Column j corresponds to x = j
                                    circle = Circle((j, i), radius, color='black', 
                                                   fill=False, linewidth=1.5, alpha=0.7)
                                    ax.add_patch(circle)
                                    circles_added += 1
                        print(f"[DEBUG] Added {circles_added} circles for cells with significant interactions")
                        
                        # Add legend for circle sizes
                        # Show legend for 1, max_count//2, and max_count (or closest values)
                        legend_counts = [1]
                        if max_count > 1:
                            if max_count > 2:
                                mid_count = max(1, max_count // 2)
                                if mid_count not in legend_counts:
                                    legend_counts.append(mid_count)
                            if max_count not in legend_counts:
                                legend_counts.append(max_count)
                        
                        # Remove duplicates and sort
                        legend_counts = sorted(set(legend_counts))
                        
                        # Create legend using a custom handler that properly scales circles
                        from matplotlib.legend_handler import HandlerPatch
                        
                        # Store radius information for the handler
                        class CircleLegendHandler(HandlerPatch):
                            def __init__(self, radius_data, max_radius_data, ax_ref):
                                super().__init__()
                                self.radius_data = radius_data
                                self.max_radius_data = max_radius_data
                                self.ax_ref = ax_ref
                            
                            def create_artists(self, legend, orig_handle,
                                             xdescent, ydescent, width, height, fontsize, trans):
                                # Convert data radius to display coordinates
                                # Get the transform from data to display coordinates
                                data_to_display = self.ax_ref.transData
                                
                                # Use a reference point in the plot (center of a cell at (0, 0))
                                # Convert the radius from data coordinates to display coordinates
                                p_center = data_to_display.transform((0, 0))
                                p_radius = data_to_display.transform((self.radius_data, 0))
                                
                                # Calculate display radius in pixels
                                display_radius = abs(p_radius[0] - p_center[0])
                                
                                # Now convert display radius to legend box coordinates
                                # The legend box uses a transform that maps to display coordinates
                                # We need to scale the display radius to fit in the legend box
                                # Use a reasonable fraction of the legend box size
                                max_legend_size = min(width, height) * 0.4  # Max circle size in legend box
                                
                                # Scale the display radius to fit in legend box
                                # We want the largest circle to be max_legend_size
                                # First, get the max radius in display coordinates for reference
                                p_max_center = data_to_display.transform((0, 0))
                                p_max_radius = data_to_display.transform((self.max_radius_data, 0))
                                max_display_radius = abs(p_max_radius[0] - p_max_center[0])
                                
                                # Scale proportionally
                                if max_display_radius > 0:
                                    scale_factor = max_legend_size / max_display_radius
                                    legend_radius = display_radius * scale_factor
                                else:
                                    # Fallback: use proportional scaling based on data radius
                                    legend_radius = max_legend_size * (self.radius_data / self.max_radius_data) if self.max_radius_data > 0 else max_legend_size
                                
                                center_x = xdescent + width / 2
                                center_y = ydescent + height / 2
                                circle = Circle((center_x, center_y), legend_radius,
                                              color='black', fill=False, linewidth=1.5, alpha=0.7,
                                              transform=trans)
                                return [circle]
                        
                        # Create legend handles with radius information
                        legend_handles = []
                        legend_labels = []
                        handler_map = {}
                        
                        for count_val in legend_counts:
                            # Calculate the actual radius used in the plot (in data coordinates)
                            radius_data = min_radius + (max_radius - min_radius) * (count_val / max_count)
                            
                            # Create a dummy circle patch (size doesn't matter, handler will override)
                            legend_circle = Circle((0, 0), 1.0, 
                                                  color='black', fill=False, 
                                                  linewidth=1.5, alpha=0.7)
                            legend_circle._radius_data = radius_data  # Store actual radius
                            legend_handles.append(legend_circle)
                            label = f"{count_val} image{'s' if count_val > 1 else ''}"
                            legend_labels.append(label)
                            
                            # Create handler for this circle with radius info
                            handler_map[legend_circle] = CircleLegendHandler(radius_data, max_radius, ax)
                        
                        # Create legend positioned on top of the plot, horizontal layout
                        legend_created = False
                        if legend_handles:
                            # Position legend on top, centered, horizontal layout
                            # Move it higher to avoid overlapping with title
                            legend = ax.legend(legend_handles, legend_labels,
                                             title='Significant Interactions',
                                             loc='lower center',
                                             bbox_to_anchor=(0.5, 1.08),
                                             frameon=True,
                                             fancybox=True,
                                             shadow=True,
                                             fontsize=9,
                                             ncol=len(legend_handles),  # Horizontal layout
                                             handler_map=handler_map)
                            legend.get_frame().set_facecolor('white')
                            legend.get_frame().set_alpha(0.95)
                            legend.get_title().set_fontsize(9)
                            legend_created = True
                            print(f"[DEBUG] Added legend with {len(legend_handles)} entries")
                
            except Exception as e:
                print(f"[DEBUG] Error during plotting: {e}")
                import traceback
                traceback.print_exc()
                raise
            
            # Add labels if available
            if hasattr(adata, 'obs') and cluster_key in adata.obs.columns:
                if hasattr(adata.obs[cluster_key], 'cat'):
                    categories = list(adata.obs[cluster_key].cat.categories)
                else:
                    categories = sorted(adata.obs[cluster_key].unique())
                
                print(f"[DEBUG] Found {len(categories)} categories: {categories[:5]}...")
                n_cats = len(categories)
                if matrix.shape[0] == n_cats and matrix.shape[1] == n_cats:
                    ax.set_xticks(np.arange(n_cats))
                    ax.set_yticks(np.arange(n_cats))
                    ax.set_xticklabels([self._get_cluster_display_name(c) for c in categories], rotation=45, ha='right')
                    ax.set_yticklabels([self._get_cluster_display_name(c) for c in categories])
                    print(f"[DEBUG] Added cluster labels")
                else:
                    print(f"[DEBUG] Shape mismatch: matrix {matrix.shape} vs {n_cats} categories")
                    # Still plot, but without labels
                    ax.set_xticks(np.arange(min(matrix.shape[1], 10)))
                    ax.set_yticks(np.arange(min(matrix.shape[0], 10)))
            else:
                print(f"[DEBUG] No cluster labels available (obs exists: {hasattr(adata, 'obs')}, cluster_key: {cluster_key})")
                # Plot without labels
                ax.set_xticks(np.arange(min(matrix.shape[1], 10)))
                ax.set_yticks(np.arange(min(matrix.shape[0], 10)))
            
            # Set title with padding to avoid overlap with legend
            has_legend_at_title = ax.get_legend() is not None
            ax.set_title("Neighborhood Enrichment", pad=25 if has_legend_at_title else 10)
            ax.set_xlabel("Neighbor Cluster")
            ax.set_ylabel("Cell Cluster")
            print(f"[DEBUG] Plot completed successfully")
        else:
            # Debug: show what we got
            print(f"[DEBUG] Failed to extract matrix for plotting")
            debug_info = f"Data type: {type(enrichment_data)}\n"
            if isinstance(enrichment_data, dict):
                debug_info += f"Keys: {list(enrichment_data.keys())}\n"
                for k, v in enrichment_data.items():
                    debug_info += f"  {k}: {type(v)}, shape: {getattr(v, 'shape', 'N/A')}\n"
            else:
                debug_info += f"Value: {enrichment_data}\n"
            print(f"[DEBUG] Debug info:\n{debug_info}")
            ax.text(0.5, 0.5, f'Unable to plot enrichment data.\nData format not recognized.\n\n{debug_info}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=8)
        
        try:
            print(f"[DEBUG] Applying layout adjustments...")
            # Check if legend exists on the axis
            has_legend = ax.get_legend() is not None
            
            # Adjust subplot parameters to center the plot and make room for legend/colorbar
            # Legend is now on top, so adjust top margin accordingly
            if has_legend:
                top_margin = 0.85  # Less space on top to make room for legend above title
            else:
                top_margin = 0.95   # Normal top margin without legend
            
            self.sq_nhood_canvas.figure.subplots_adjust(
                left=0.10,           # Left margin
                right=0.88,          # Right margin (for colorbar)
                top=top_margin,      # Top margin (adjusted for legend on top)
                bottom=0.10,         # Bottom margin
                wspace=0.1,          # Width space between subplots
                hspace=0.1           # Height space between subplots
            )
            print(f"[DEBUG] subplots_adjust completed")
            print(f"[DEBUG] Drawing canvas...")
            self.sq_nhood_canvas.draw()
            print(f"[DEBUG] Canvas draw completed")
            # Force update and repaint
            print(f"[DEBUG] Updating canvas...")
            self.sq_nhood_canvas.update()
            self.sq_nhood_canvas.repaint()
            print(f"[DEBUG] Canvas update and repaint completed")
            print(f"[DEBUG] Canvas size: {self.sq_nhood_canvas.size()}")
            print(f"[DEBUG] Figure size: {self.sq_nhood_canvas.figure.get_size_inches()}")
        except Exception as e:
            print(f"[DEBUG] Error during canvas update: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_sq_nhood_plot(self):
        """Save the neighborhood enrichment plot."""
        if save_figure_with_options(self.sq_nhood_canvas.figure, "squidpy_nhood_enrichment.png", self):
            QtWidgets.QMessageBox.information(self, "Success", "Plot saved successfully")
    
    def _run_sq_cooccurrence(self):
        """Run co-occurrence analysis using core function."""
        print(f"[DEBUG] _run_sq_cooccurrence: Starting")
        if not self.spatial_graph_built:
            print(f"[DEBUG] Spatial graph not built")
            QtWidgets.QMessageBox.warning(self, "Graph Required", 
                "Please create the spatial graph first (Step 1 at the top).")
            return
        
        try:
            cluster_key = self.sq_cooccur_cluster_combo.currentText()
            roi_id = self._get_selected_roi(self.sq_cooccur_roi_combo)
            print(f"[DEBUG] Cluster key: {cluster_key}, ROI: {roi_id}")
            
            # Parse neighborhood sizes
            sizes_str = self.sq_cooccur_sizes_edit.text().strip()
            if sizes_str.lower() == 'all' or not sizes_str:
                QtWidgets.QMessageBox.warning(self, "Invalid Input", 
                    "Co-occurrence analysis requires multiple distances.\n"
                    "Please enter at least 2 distances separated by commas (e.g., '10,20,30,50').")
                return
            
            try:
                # Parse as comma-separated list
                nhood_sizes = [float(x.strip()) for x in sizes_str.split(',') if x.strip()]
                if len(nhood_sizes) < 2:
                    QtWidgets.QMessageBox.warning(self, "Invalid Input", 
                        f"Co-occurrence analysis requires at least 2 distances.\n"
                        f"You provided {len(nhood_sizes)} distance(s): {nhood_sizes}\n"
                        f"Please enter multiple distances separated by commas (e.g., '10,20,30,50').")
                    return
            except ValueError as e:
                QtWidgets.QMessageBox.warning(self, "Invalid Input", 
                    f"Could not parse distances. Please enter comma-separated numbers (e.g., '10,20,30,50').\n"
                    f"Error: {str(e)}")
                return
            
            print(f"[DEBUG] Neighborhood sizes: {nhood_sizes} (count: {len(nhood_sizes)})")
            
            # Store interval for distance selection
            self.sq_cooccur_interval = nhood_sizes
            
            # Update distance combo
            self.sq_cooccur_distance_combo.clear()
            for dist in nhood_sizes:
                self.sq_cooccur_distance_combo.addItem(f"{dist} µm", dist)
            
            # Check if cluster column exists
            filtered_df = self._get_filtered_dataframe()
            if cluster_key not in filtered_df.columns:
                print(f"[DEBUG] Cluster column '{cluster_key}' not found in dataframe")
                QtWidgets.QMessageBox.warning(self, "Missing Column", 
                    f"Cluster column '{cluster_key}' not found in data.")
                return
            
            # Get AnnData dict - filter to selected ROI if needed
            if roi_id is None:
                # Use all cached AnnData objects with graphs
                anndata_dict = {rid: adata for rid, adata in self.anndata_cache.items() 
                               if 'spatial_connectivities' in adata.obsp}
            else:
                # Use only selected ROI
                if roi_id not in self.anndata_cache:
                    QtWidgets.QMessageBox.warning(self, "No Data", f"No data found for ROI {roi_id}.")
                    return
                adata = self.anndata_cache[roi_id]
                if 'spatial_connectivities' not in adata.obsp:
                    QtWidgets.QMessageBox.warning(self, "No Graph", f"No spatial graph found for ROI {roi_id}.")
                    return
                anndata_dict = {roi_id: adata}
            
            if not anndata_dict:
                QtWidgets.QMessageBox.warning(self, "No Data", "No AnnData objects with spatial graphs found.")
                return
            
            # Use core function
            results = spatial_cooccurrence(
                anndata_dict=anndata_dict,
                cluster_key=cluster_key,
                interval=nhood_sizes,
                reference_cluster=self.sq_cooccur_ref_cluster_combo.currentData()
            )
            
            # Update cache with results
            self.anndata_cache.update(results)
            
            # Update status
            for roi_id in results.keys():
                if roi_id not in self.analysis_status:
                    self.analysis_status[roi_id] = {}
                self.analysis_status[roi_id]['co_occurrence'] = True
            
            if results:
                # Use first ROI for plotting and reference cluster combo
                plot_adata = list(results.values())[0]
                # Update reference cluster combo (preserve selection if combo already has items)
                preserve = self.sq_cooccur_ref_cluster_combo.count() > 0
                self._update_cooccur_ref_cluster_combo(plot_adata, preserve_selection=preserve)
                
                QtWidgets.QMessageBox.information(self, "Co-occurrence Complete", 
                    f"Co-occurrence analysis completed for {len(results)} ROI(s).")
                
                # Plot results (will use the default reference cluster "1")
                self._plot_sq_cooccurrence(plot_adata)
                self.sq_cooccur_save_btn.setEnabled(True)
            else:
                QtWidgets.QMessageBox.warning(self, "No Results", 
                    "Co-occurrence analysis completed but no results to plot.")
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Error", f"Error running co-occurrence: {str(e)}")
    
    def _plot_sq_cooccurrence(self, adata: 'ad.AnnData'):
        """Plot co-occurrence results using squidpy's plotting function."""
        print(f"[DEBUG] _plot_sq_cooccurrence: Starting")
        print(f"[DEBUG] adata: {adata}")
        if adata is None:
            print(f"[DEBUG] adata is None")
            self.sq_cooccur_canvas.figure.clear()
            ax = self.sq_cooccur_canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No data provided for plotting.', 
                   ha='center', va='center', transform=ax.transAxes)
            self.sq_cooccur_canvas.draw()
            return
        
        print(f"[DEBUG] adata.uns keys: {list(adata.uns.keys())}")
        
        # Check for co_occurrence key or alternatives
        cooccur_key = None
        if 'co_occurrence' in adata.uns:
            cooccur_key = 'co_occurrence'
            print(f"[DEBUG] Found 'co_occurrence' key")
        else:
            # Try alternative key names
            for key in adata.uns.keys():
                if 'co' in key.lower() and 'occur' in key.lower():
                    cooccur_key = key
                    print(f"[DEBUG] Found alternative key: {key}")
                    break
        
        if cooccur_key is None:
            print(f"[DEBUG] No co-occurrence data found in adata.uns")
            self.sq_cooccur_canvas.figure.clear()
            ax = self.sq_cooccur_canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No co-occurrence data found.\nPlease run co-occurrence analysis first.', 
                   ha='center', va='center', transform=ax.transAxes)
            self.sq_cooccur_canvas.draw()
            return
        
        # Use manual plotting (squidpy's plotting functions don't work well with custom canvases)
        # sq.pl.co_occurrence() doesn't support custom axes and creates its own figure
        cooccur_data = adata.uns[cooccur_key]
        cluster_key = self.sq_cooccur_cluster_combo.currentText()
        plot_type = self.sq_cooccur_plot_type_combo.currentText()
        
        print(f"[DEBUG] Co-occurrence data type: {type(cooccur_data)}")
        print(f"[DEBUG] Plot type: {plot_type}")
        
        # Get cluster categories
        if cluster_key in adata.obs.columns:
            if hasattr(adata.obs[cluster_key], 'cat'):
                categories = list(adata.obs[cluster_key].cat.categories)
            else:
                categories = sorted(adata.obs[cluster_key].unique())
        else:
            categories = []
        
        print(f"[DEBUG] Categories: {categories}")
        
        # Handle heatmap plotting
        if plot_type == "Heatmap":
            # Try to populate distance combo if empty but interval data is available
            if self.sq_cooccur_distance_combo.count() == 0:
                interval_to_populate = self.sq_cooccur_interval
                if interval_to_populate is None and isinstance(cooccur_data, dict):
                    if 'interval' in cooccur_data:
                        interval_to_populate = cooccur_data['interval']
                    elif 'distances' in cooccur_data:
                        interval_to_populate = cooccur_data['distances']
                
                if interval_to_populate is not None:
                    # Convert to list if needed
                    if isinstance(interval_to_populate, np.ndarray):
                        interval_to_populate = interval_to_populate.tolist()
                    # Populate combo
                    self.sq_cooccur_distance_combo.clear()
                    for dist in interval_to_populate:
                        self.sq_cooccur_distance_combo.addItem(f"{dist} µm", dist)
                    # Store for future use
                    if self.sq_cooccur_interval is None:
                        self.sq_cooccur_interval = interval_to_populate
                else:
                    ax = self.sq_cooccur_canvas.figure.add_subplot(111)
                    ax.text(0.5, 0.5, 'No distances available.\nPlease run co-occurrence analysis first.', 
                           ha='center', va='center', transform=ax.transAxes)
                    self.sq_cooccur_canvas.draw()
                    return
            
            selected_distance = self.sq_cooccur_distance_combo.currentData()
            if selected_distance is None:
                ax = self.sq_cooccur_canvas.figure.add_subplot(111)
                ax.text(0.5, 0.5, 'Please select a distance for the heatmap.', 
                       ha='center', va='center', transform=ax.transAxes)
                self.sq_cooccur_canvas.draw()
                return
            
            # Find the index of the selected distance
            # Try to get interval from stored value or extract from data
            interval_to_use = self.sq_cooccur_interval
            if interval_to_use is None and isinstance(cooccur_data, dict):
                if 'interval' in cooccur_data:
                    interval_to_use = cooccur_data['interval']
                elif 'distances' in cooccur_data:
                    interval_to_use = cooccur_data['distances']
            
            if interval_to_use is None:
                ax = self.sq_cooccur_canvas.figure.add_subplot(111)
                ax.text(0.5, 0.5, 'Interval data not available.\nPlease run co-occurrence analysis first.', 
                       ha='center', va='center', transform=ax.transAxes)
                self.sq_cooccur_canvas.draw()
                return
            
            # Convert to list if it's a numpy array
            if isinstance(interval_to_use, np.ndarray):
                interval_to_use = interval_to_use.tolist()
            
            try:
                distance_idx = interval_to_use.index(selected_distance)
            except ValueError:
                ax = self.sq_cooccur_canvas.figure.add_subplot(111)
                ax.text(0.5, 0.5, f'Distance {selected_distance} not found in interval.', 
                       ha='center', va='center', transform=ax.transAxes)
                self.sq_cooccur_canvas.draw()
                return
            
            # Extract heatmap data
            if isinstance(cooccur_data, dict) and 'occ' in cooccur_data:
                occ_array = cooccur_data['occ']
                if occ_array.ndim == 3:
                    # Extract 2D slice at the selected distance
                    heatmap_data = occ_array[:, :, distance_idx]
                    ax = self.sq_cooccur_canvas.figure.add_subplot(111)
                    
                    # Create heatmap
                    im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto', interpolation='nearest')
                    
                    # Set labels
                    if len(categories) == heatmap_data.shape[0] and len(categories) == heatmap_data.shape[1]:
                        ax.set_xticks(np.arange(len(categories)))
                        ax.set_yticks(np.arange(len(categories)))
                        ax.set_xticklabels([self._get_cluster_display_name(c) for c in categories], rotation=45, ha='right')
                        ax.set_yticklabels([self._get_cluster_display_name(c) for c in categories])
                    
                    ax.set_xlabel('To Phenotype')
                    ax.set_ylabel('From Phenotype')
                    ax.set_title(f'Co-occurrence Heatmap at {selected_distance} µm')
                    
                    # Add colorbar
                    cbar = self.sq_cooccur_canvas.figure.colorbar(im, ax=ax)
                    cbar.set_label('Co-occurrence Score')
                    
                    self.sq_cooccur_canvas.figure.tight_layout()
                    self.sq_cooccur_canvas.draw()
                    return
                else:
                    ax = self.sq_cooccur_canvas.figure.add_subplot(111)
                    ax.text(0.5, 0.5, 'Co-occurrence data format not suitable for heatmap.\nExpected 3D array.', 
                           ha='center', va='center', transform=ax.transAxes)
                    self.sq_cooccur_canvas.draw()
                    return
            else:
                ax = self.sq_cooccur_canvas.figure.add_subplot(111)
                ax.text(0.5, 0.5, 'Co-occurrence data format not recognized for heatmap.', 
                       ha='center', va='center', transform=ax.transAxes)
                self.sq_cooccur_canvas.draw()
                return
        
        # Plot using squidpy's exact approach
        # Based on sq.pl.co_occurrence source code
        occurrence_data = cooccur_data
        out = occurrence_data["occ"]
        interval = occurrence_data["interval"][1:]  # Skip first value like squidpy does
        
        # IMPORTANT: The 'out' array shape is determined by the categories used when
        # co-occurrence was computed. We need to use those categories, not necessarily
        # the current ROI's categories. The array shape is (n_clusters, n_clusters, n_distances)
        # So we need to get the categories that match the array dimensions.
        out_shape = out.shape
        print(f"[DEBUG] Co-occurrence 'out' array shape: {out_shape}")
        
        # The out array should be (n_clusters, n_clusters, n_distances)
        # Use the actual dimensions from the array
        n_clusters_in_data = out_shape[0] if len(out_shape) >= 2 else 0
        
        # Try to get categories from the data structure if available
        # Otherwise, use categories from current ROI but ensure they match array size
        if n_clusters_in_data > 0:
            # Check if categories match the array size
            if len(categories) == n_clusters_in_data:
                categories_list = categories
            else:
                # Categories don't match - try to get from adata.obs categories
                # or use indices if categories are missing
                print(f"[DEBUG] Category mismatch: {len(categories)} categories but array has {n_clusters_in_data} clusters")
                # Use the categories from the adata that was used to compute co-occurrence
                # If the current ROI has fewer categories, we need to handle this
                if len(categories) < n_clusters_in_data:
                    # Current ROI has fewer categories - this means some clusters are missing
                    # We should use the categories from when co-occurrence was computed
                    # For now, create placeholder categories or use indices
                    print(f"[DEBUG] Warning: Current ROI has fewer categories than co-occurrence data")
                    # Try to get all possible categories from the cluster column's categories
                    if cluster_key in adata.obs.columns and hasattr(adata.obs[cluster_key], 'cat'):
                        all_categories = list(adata.obs[cluster_key].cat.categories)
                        if len(all_categories) >= n_clusters_in_data:
                            categories_list = all_categories[:n_clusters_in_data]
                        else:
                            # Still not enough - use indices as fallback
                            categories_list = [f"Cluster_{i}" for i in range(n_clusters_in_data)]
                    else:
                        categories_list = [f"Cluster_{i}" for i in range(n_clusters_in_data)]
                else:
                    # Current ROI has more categories - use first n_clusters_in_data
                    categories_list = categories[:n_clusters_in_data]
        else:
            categories_list = categories
        
        print(f"[DEBUG] Using {len(categories_list)} categories for plotting: {categories_list[:5]}...")
        
        # Get clusters to plot (use reference cluster if selected, otherwise all)
        clusters_to_plot = None
        ref_cluster = self.sq_cooccur_ref_cluster_combo.currentData()
        if ref_cluster is not None:
            # Plot only the reference cluster (like squidpy's clusters parameter)
            clusters_to_plot = [ref_cluster] if ref_cluster in categories_list else categories_list
        else:
            # Plot all clusters
            clusters_to_plot = categories_list
        
        # Filter to valid clusters
        clusters_to_plot = [c for c in clusters_to_plot if c in categories_list]
        
        if not clusters_to_plot:
            ax = self.sq_cooccur_canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No valid clusters to plot.', 
                   ha='center', va='center', transform=ax.transAxes)
            self.sq_cooccur_canvas.draw()
            return
        
        # Create subplots like squidpy does (1 row, len(clusters) columns)
        n_clusters = len(clusters_to_plot)
        self.sq_cooccur_canvas.figure.clear()
        
        # Get palette using tab20/tab60 for consistent cluster coloring
        # Use _get_vivid_colors which provides tab20, tab20b, tab20c for up to 60 clusters
        n_categories = len(categories_list)
        if n_categories > 0:
            # Get colors using tab20/tab60 scheme
            color_array = _get_vivid_colors(n_categories)
            # Convert to dict mapping category to color (RGBA tuple)
            palette = {cat: tuple(color_array[i]) for i, cat in enumerate(categories_list)}
        else:
            palette = {}
        
        # Create subplots
        if n_clusters == 1:
            axs = [self.sq_cooccur_canvas.figure.add_subplot(111)]
        else:
            # Create grid of subplots
            n_cols = min(n_clusters, 4)  # Limit columns for readability
            n_rows = (n_clusters + n_cols - 1) // n_cols
            axs = []
            for i in range(n_clusters):
                ax = self.sq_cooccur_canvas.figure.add_subplot(n_rows, n_cols, i + 1)
                axs.append(ax)
        
        # Plot each cluster like squidpy does
        for g, ax in zip(clusters_to_plot, axs):
            # Find index of cluster in categories_list
            try:
                idx = categories_list.index(g)
            except ValueError:
                print(f"[DEBUG] Warning: Cluster {g} not found in categories_list, skipping")
                continue
            
            # Create DataFrame like squidpy: out[idx, :, :].T with columns=categories
            # Shape: (n_distances, n_clusters)
            cluster_data = out[idx, :, :].T
            actual_n_cols = cluster_data.shape[1]
            
            # Ensure the number of columns matches categories_list length
            if actual_n_cols != len(categories_list):
                print(f"[DEBUG] Warning: cluster_data has {actual_n_cols} columns but categories_list has {len(categories_list)} items")
                # Use categories that match the actual data shape
                if actual_n_cols <= len(categories_list):
                    # Use first actual_n_cols categories
                    df_columns = categories_list[:actual_n_cols]
                else:
                    # More columns than categories - pad with placeholder names
                    df_columns = list(categories_list) + [f"Cluster_{i}" for i in range(len(categories_list), actual_n_cols)]
                df = pd.DataFrame(cluster_data, columns=df_columns)
            else:
                df = pd.DataFrame(cluster_data, columns=categories_list)
            
            # Melt to long format like squidpy
            df_melted = df.melt(var_name=cluster_key, value_name="probability")
            df_melted["distance"] = np.tile(interval, len(categories_list))
            
            # Use seaborn lineplot like squidpy
            # Convert palette dict to list if needed (seaborn can handle both)
            plot_palette = palette
            if isinstance(palette, dict):
                # Convert dict to list in the order of categories_list
                plot_palette = [palette.get(cat, '#000000') for cat in categories_list]
            
            sns.lineplot(
                x="distance",
                y="probability",
                data=df_melted,
                dashes=False,
                hue=cluster_key,
                hue_order=categories_list,
                palette=plot_palette,
                ax=ax,
                legend='full',  # Ensure all entries are shown
            )
            
            # Set legend like squidpy - ensure all entries are shown
            legend_kwargs = {"loc": "center left", "bbox_to_anchor": (1, 0.5)}
            
            # Get all handles and labels from the axes (works across matplotlib versions)
            handles, labels = ax.get_legend_handles_labels()
            
            # Remove duplicates while preserving order
            seen = set()
            unique_handles = []
            unique_labels = []
            for h, l in zip(handles, labels):
                if l not in seen:
                    seen.add(l)
                    unique_handles.append(h)
                    unique_labels.append(l)
            
            # Recreate legend with all unique entries
            if unique_handles:
                ax.legend(unique_handles, unique_labels, **legend_kwargs)
            else:
                # If no handles, create default legend
                ax.legend(**legend_kwargs)
            
            # Set title like squidpy (LaTeX format)
            ax.set_title(rf"$\frac{{p(exp|{g})}}{{p(exp)}}$")
            ax.set_ylabel("value")
        
        self.sq_cooccur_canvas.figure.tight_layout()
        self.sq_cooccur_canvas.draw()
    
    def _save_sq_cooccur_plot(self):
        """Save the co-occurrence plot."""
        if save_figure_with_options(self.sq_cooccur_canvas.figure, "squidpy_cooccurrence.png", self):
            QtWidgets.QMessageBox.information(self, "Success", "Plot saved successfully")
    
    def _run_sq_autocorrelation(self):
        """Run spatial autocorrelation analysis using core function."""
        print(f"[DEBUG] _run_sq_autocorrelation: Starting")
        if not self.spatial_graph_built:
            print(f"[DEBUG] Spatial graph not built")
            QtWidgets.QMessageBox.warning(self, "Graph Required", 
                "Please create the spatial graph first (Step 1 at the top).")
            return
        
        try:
            markers_str = self.sq_autocorr_markers_edit.text().strip()
            roi_id = self._get_selected_roi(self.sq_autocorr_roi_combo)
            agg_method = self.sq_autocorr_agg_combo.currentText().lower()
            print(f"[DEBUG] Markers: {markers_str}, ROI: {roi_id}, Aggregation: {agg_method}")
            
            # Parse markers
            markers = None
            if markers_str.lower() != 'all' and markers_str:
                markers = [m.strip() for m in markers_str.split(',')]
            
            # Get AnnData dict - filter to selected ROI if needed
            if roi_id is None:
                # Use all cached AnnData objects with graphs
                anndata_dict = {rid: adata for rid, adata in self.anndata_cache.items() 
                               if 'spatial_connectivities' in adata.obsp}
            else:
                # Use only selected ROI
                if roi_id not in self.anndata_cache:
                    QtWidgets.QMessageBox.warning(self, "No Data", f"No data found for ROI {roi_id}.")
                    return
                adata = self.anndata_cache[roi_id]
                if 'spatial_connectivities' not in adata.obsp:
                    QtWidgets.QMessageBox.warning(self, "No Graph", f"No spatial graph found for ROI {roi_id}.")
                    return
                anndata_dict = {roi_id: adata}
            
            if not anndata_dict:
                QtWidgets.QMessageBox.warning(self, "No Data", "No AnnData objects with spatial graphs found.")
                return
            
            # Use core function
            results = spatial_autocorrelation(
                anndata_dict=anndata_dict,
                markers=markers,
                aggregation=agg_method
            )
            
            # Update cache with results
            self.anndata_cache.update(results['results'])
            
            # Update status
            for roi_id in results['results'].keys():
                if roi_id not in self.analysis_status:
                    self.analysis_status[roi_id] = {}
                self.analysis_status[roi_id]['autocorrelation'] = True
            
            # Store aggregated result if available
            if results['aggregated'] is not None:
                self.aggregated_results['autocorrelation'] = results['aggregated']
                plot_adata = results['aggregated']
            elif results['results']:
                plot_adata = list(results['results'].values())[0]
            else:
                QtWidgets.QMessageBox.warning(self, "No Results", "No autocorrelation results to plot.")
                return
            
            QtWidgets.QMessageBox.information(self, "Autocorrelation Complete", 
                f"Spatial autocorrelation completed for {len(results['results'])} ROI(s). "
                f"{'Aggregated using ' + agg_method if len(results['results']) > 1 else ''}")
            
            # Plot results
            self._plot_sq_autocorrelation(plot_adata)
            self.sq_autocorr_save_btn.setEnabled(True)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Error", f"Error running autocorrelation: {str(e)}")
    
    def _plot_sq_autocorrelation(self, adata: 'ad.AnnData'):
        """Plot spatial autocorrelation results using squidpy's plotting function."""
        print(f"[DEBUG] _plot_sq_autocorrelation: Starting")
        print(f"[DEBUG] adata: {adata}")
        if adata is None:
            print(f"[DEBUG] adata is None")
            self.sq_autocorr_canvas.figure.clear()
            ax = self.sq_autocorr_canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No data provided for plotting.', 
                   ha='center', va='center', transform=ax.transAxes)
            self.sq_autocorr_canvas.draw()
            return
        
        print(f"[DEBUG] adata.uns keys: {list(adata.uns.keys())}")
        
        # Check for moranI key or alternatives
        moran_key = None
        if 'moranI' in adata.uns:
            moran_key = 'moranI'
            print(f"[DEBUG] Found 'moranI' key")
        else:
            # Try alternative key names
            for key in adata.uns.keys():
                if 'moran' in key.lower() or 'autocorr' in key.lower():
                    moran_key = key
                    print(f"[DEBUG] Found alternative key: {key}")
                    break
        
        if moran_key is None:
            print(f"[DEBUG] No Moran's I data found in adata.uns")
            self.sq_autocorr_canvas.figure.clear()
            ax = self.sq_autocorr_canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No autocorrelation data found.\nPlease run autocorrelation analysis first.', 
                   ha='center', va='center', transform=ax.transAxes)
            self.sq_autocorr_canvas.draw()
            return
        
        # Squidpy doesn't have a spatial_autocorr plotting function
        # Use manual plotting which works well
        self.sq_autocorr_canvas.figure.clear()
        ax = self.sq_autocorr_canvas.figure.add_subplot(111)
        moran_data = adata.uns[moran_key]
        
        print(f"[DEBUG] moran_data type: {type(moran_data)}")
        
        # Extract I values and p-values
        I_values = None
        p_values = None
        gene_names = None
        
        if isinstance(moran_data, pd.DataFrame):
            print(f"[DEBUG] moran_data is DataFrame, shape: {moran_data.shape}, columns: {list(moran_data.columns)}")
            # DataFrame format - extract columns
            if 'I' in moran_data.columns:
                I_values = moran_data['I'].values
                print(f"[DEBUG] Extracted I values from DataFrame, shape: {I_values.shape}")
            elif 'moranI' in moran_data.columns:
                I_values = moran_data['moranI'].values
                print(f"[DEBUG] Extracted moranI values from DataFrame, shape: {I_values.shape}")
            
            if 'pval_norm' in moran_data.columns:
                p_values = moran_data['pval_norm'].values
            elif 'pval' in moran_data.columns:
                p_values = moran_data['pval'].values
            
            # Get gene names from index or var_names column
            if 'var_names' in moran_data.columns:
                gene_names = moran_data['var_names'].values
            elif moran_data.index.name == 'var_names' or all(isinstance(x, str) for x in moran_data.index):
                gene_names = moran_data.index.values
            else:
                # Try to infer from index
                gene_names = [str(x) for x in moran_data.index]
        
        elif isinstance(moran_data, dict):
            I_values = None
            p_values = None
            gene_names = None
            
            # Try different possible keys
            if 'I' in moran_data:
                I_values = moran_data['I']
                if isinstance(I_values, np.ndarray):
                    pass  # Good
                elif isinstance(I_values, (list, tuple)):
                    I_values = np.array(I_values)
                else:
                    I_values = None
            
            if 'pval_norm' in moran_data:
                p_values = moran_data['pval_norm']
                if isinstance(p_values, np.ndarray):
                    pass
                elif isinstance(p_values, (list, tuple)):
                    p_values = np.array(p_values)
                else:
                    p_values = None
            elif 'pval' in moran_data:
                p_values = moran_data['pval']
                if isinstance(p_values, np.ndarray):
                    pass
                elif isinstance(p_values, (list, tuple)):
                    p_values = np.array(p_values)
                else:
                    p_values = None
            
            # Get gene names
            if 'var_names' in moran_data:
                gene_names = moran_data['var_names']
            elif hasattr(adata, 'var_names'):
                gene_names = adata.var_names.values
            else:
                gene_names = [f"Feature_{i}" for i in range(len(I_values))] if I_values is not None else []
        
        # Plot the data (works for both DataFrame and dict formats)
        if I_values is not None and len(I_values) > 0:
            print(f"[DEBUG] Plotting {len(I_values)} Moran's I values")
            # Create bar plot
            sorted_idx = np.argsort(I_values)[::-1]  # Sort by I value descending
            top_n = min(self.sq_autocorr_topk_spin.value(), len(sorted_idx))  # Use spinbox value
            
            top_I = I_values[sorted_idx[:top_n]]
            top_p = p_values[sorted_idx[:top_n]] if p_values is not None and len(p_values) > 0 else None
            top_genes = [gene_names[i] for i in sorted_idx[:top_n]] if len(gene_names) > 0 else [f"Feature_{i}" for i in sorted_idx[:top_n]]
            
            colors = ['red' if (top_p is not None and p < 0.05) else 'gray' for p in (top_p if top_p is not None else [1.0] * top_n)]
            ax.barh(range(top_n), top_I, color=colors)
            ax.set_yticks(range(top_n))
            ax.set_yticklabels(top_genes, fontsize=8)
            ax.set_xlabel("Moran's I")
            ax.set_title(f"Spatial Autocorrelation (Top {top_n})")
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
            ax.grid(True, alpha=0.3, axis='x')
            print(f"[DEBUG] Plot completed successfully")
        else:
            print(f"[DEBUG] Failed to plot - I_values is None or empty")
            ax.text(0.5, 0.5, 'Unable to extract Moran\'s I values.\nData format not recognized.\nCheck debug output for details.', 
                   ha='center', va='center', transform=ax.transAxes)
        
        self.sq_autocorr_canvas.figure.tight_layout()
        self.sq_autocorr_canvas.draw()
    
    def _save_sq_autocorr_plot(self):
        """Save the spatial autocorrelation plot."""
        if save_figure_with_options(self.sq_autocorr_canvas.figure, "squidpy_autocorrelation.png", self):
            QtWidgets.QMessageBox.information(self, "Success", "Plot saved successfully")
    
    def _run_sq_ripley(self):
        """Run Ripley analysis using core function."""
        print(f"[DEBUG] _run_sq_ripley: Starting")
        if not self.spatial_graph_built:
            print(f"[DEBUG] Spatial graph not built")
            QtWidgets.QMessageBox.warning(self, "Graph Required", 
                "Please create the spatial graph first (Step 1 at the top).")
            return
        
        try:
            mode = self.sq_ripley_mode_combo.currentText()  # F, G, or L
            max_dist = float(self.sq_ripley_r_max_spin.value())
            cluster_key = self.sq_ripley_cluster_combo.currentText()
            roi_id = self._get_selected_roi(self.sq_ripley_roi_combo)
            print(f"[DEBUG] Mode: {mode}, max_dist: {max_dist}, cluster_key: {cluster_key}, ROI: {roi_id}")
            
            # Check if cluster column exists
            filtered_df = self._get_filtered_dataframe()
            if cluster_key not in filtered_df.columns:
                print(f"[DEBUG] Cluster column '{cluster_key}' not found in dataframe")
                QtWidgets.QMessageBox.warning(self, "Missing Column", 
                    f"Cluster column '{cluster_key}' not found in data.")
                return
            
            # Get AnnData dict - filter to selected ROI if needed
            if roi_id is None:
                # Use all cached AnnData objects with graphs
                anndata_dict = {rid: adata for rid, adata in self.anndata_cache.items() 
                               if 'spatial_connectivities' in adata.obsp}
            else:
                # Use only selected ROI
                if roi_id not in self.anndata_cache:
                    QtWidgets.QMessageBox.warning(self, "No Data", f"No data found for ROI {roi_id}.")
                    return
                adata = self.anndata_cache[roi_id]
                if 'spatial_connectivities' not in adata.obsp:
                    QtWidgets.QMessageBox.warning(self, "No Graph", f"No spatial graph found for ROI {roi_id}.")
                    return
                anndata_dict = {roi_id: adata}
            
            if not anndata_dict:
                QtWidgets.QMessageBox.warning(self, "No Data", "No AnnData objects with spatial graphs found.")
                return
            
            # Use core function
            results = spatial_ripley(
                anndata_dict=anndata_dict,
                cluster_key=cluster_key,
                mode=mode,
                max_dist=max_dist
            )
            
            # Update cache with results
            self.anndata_cache.update(results)
            
            # Update status
            for roi_id in results.keys():
                if roi_id not in self.analysis_status:
                    self.analysis_status[roi_id] = {}
                self.analysis_status[roi_id]['ripley'] = True
            
            if results:
                # Check if aggregation is needed (multiple ROIs and aggregation selected)
                agg_method = self.sq_ripley_agg_combo.currentText().lower()
                should_aggregate = (roi_id is None and len(results) > 1 and agg_method in ['mean', 'sum'])
                
                if should_aggregate:
                    # Aggregate results across ROIs
                    print(f"[DEBUG] Aggregating Ripley results using {agg_method} for {len(results)} ROIs")
                    plot_adata = self._aggregate_ripley_results(results, cluster_key, mode, agg_method)
                else:
                    # Use first ROI for plotting
                    plot_adata = list(results.values())[0]
                
                QtWidgets.QMessageBox.information(self, "Ripley Complete", 
                    f"Ripley analysis completed for {len(results)} ROI(s).")
                
                # Plot results
                self._plot_sq_ripley(plot_adata, cluster_key)
                self.sq_ripley_save_btn.setEnabled(True)
            else:
                QtWidgets.QMessageBox.warning(self, "No Results", 
                    "Ripley analysis completed but no results to plot. "
                    "This can happen when clusters are too small.")
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Error", f"Error running Ripley: {str(e)}")
    
    def _aggregate_ripley_results(self, results: Dict[str, 'ad.AnnData'], cluster_key: str, mode: str, agg_method: str) -> 'ad.AnnData':
        """Aggregate Ripley results across multiple ROIs."""
        from squidpy._constants._constants import RipleyStat
        
        mode_enum = RipleyStat(mode)
        stat_key = f"{mode_enum.s}_stat"
        sims_stat_key = "sims_stat"
        
        # Collect all stat dataframes
        stat_dfs = []
        sims_stat_dfs = []
        all_bins = None
        all_clusters = set()
        
        for roi_id, adata in results.items():
            if 'ripley' not in adata.uns:
                continue
            
            ripley_data = adata.uns['ripley']
            if stat_key not in ripley_data:
                continue
            
            stat_df = ripley_data[stat_key]
            stat_dfs.append(stat_df)
            
            # Collect unique clusters and bins
            if cluster_key in stat_df.columns:
                all_clusters.update(stat_df[cluster_key].unique())
            if 'bins' in stat_df.columns:
                if all_bins is None:
                    all_bins = stat_df['bins'].unique()
                else:
                    # Ensure bins match across ROIs
                    bins_this_roi = stat_df['bins'].unique()
                    if len(bins_this_roi) != len(all_bins) or not np.allclose(bins_this_roi, all_bins):
                        print(f"[DEBUG] Warning: Bins mismatch for ROI {roi_id}, using first ROI's bins")
            
            # Collect simulation stats if available
            if sims_stat_key in ripley_data:
                sims_stat_dfs.append(ripley_data[sims_stat_key])
        
        if not stat_dfs:
            # Fallback: return first result
            return list(results.values())[0]
        
        # Aggregate stat dataframes
        # Combine all dataframes
        combined_stat_df = pd.concat(stat_dfs, ignore_index=True)
        
        # Group by bins and cluster_key, then aggregate stats
        grouped = combined_stat_df.groupby(['bins', cluster_key], as_index=False)
        if agg_method == 'mean':
            aggregated_stat_df = grouped.agg({'stats': 'mean'}).reset_index(drop=True)
        else:  # sum
            aggregated_stat_df = grouped.agg({'stats': 'sum'}).reset_index(drop=True)
        
        # Aggregate simulation stats if available
        aggregated_sims_stat = None
        if sims_stat_dfs:
            combined_sims_df = pd.concat(sims_stat_dfs, ignore_index=True)
            grouped_sims = combined_sims_df.groupby('bins', as_index=False)
            if agg_method == 'mean':
                aggregated_sims_stat = grouped_sims.agg({'stats': 'mean'}).reset_index(drop=True)
            else:  # sum
                aggregated_sims_stat = grouped_sims.agg({'stats': 'sum'}).reset_index(drop=True)
        
        # Create aggregated result structure
        aggregated_ripley_data = {
            stat_key: aggregated_stat_df
        }
        if aggregated_sims_stat is not None:
            aggregated_ripley_data[sims_stat_key] = aggregated_sims_stat
        
        # Create a new AnnData object with aggregated results
        # Use the first ROI's structure as template
        first_adata = list(results.values())[0]
        aggregated_adata = first_adata.copy()
        aggregated_adata.uns['ripley'] = aggregated_ripley_data
        
        print(f"[DEBUG] Aggregated Ripley results: {len(aggregated_stat_df)} rows, {len(all_clusters)} clusters")
        
        return aggregated_adata
    
    def _plot_sq_ripley(self, adata: 'ad.AnnData', cluster_key: str):
        """Plot Ripley results using squidpy's plotting function."""
        print(f"[DEBUG] _plot_sq_ripley: Starting")
        print(f"[DEBUG] adata: {adata}, cluster_key: {cluster_key}")
        if adata is None:
            print(f"[DEBUG] adata is None")
            self.sq_ripley_canvas.figure.clear()
            ax = self.sq_ripley_canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No data provided for plotting.', 
                   ha='center', va='center', transform=ax.transAxes)
            self.sq_ripley_canvas.draw()
            return
        
        print(f"[DEBUG] adata.uns keys: {list(adata.uns.keys())}")
        
        # Check for ripley key or alternatives
        ripley_key = None
        if 'ripley' in adata.uns:
            ripley_key = 'ripley'
            print(f"[DEBUG] Found 'ripley' key")
        else:
            # Try alternative key names
            for key in adata.uns.keys():
                if 'ripley' in key.lower():
                    ripley_key = key
                    print(f"[DEBUG] Found alternative key: {key}")
                    break
        
        if ripley_key is None:
            print(f"[DEBUG] No Ripley data found in adata.uns")
            self.sq_ripley_canvas.figure.clear()
            ax = self.sq_ripley_canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No Ripley data found.\nPlease run Ripley analysis first.', 
                   ha='center', va='center', transform=ax.transAxes)
            self.sq_ripley_canvas.draw()
            return
        
        # Use manual plotting (squidpy's plotting functions don't work well with custom canvases)
        # sq.pl.ripley() doesn't support custom axes properly
        mode = self.sq_ripley_mode_combo.currentText()
        self.sq_ripley_canvas.figure.clear()
        ax = self.sq_ripley_canvas.figure.add_subplot(111)
        ripley_data = adata.uns[ripley_key]
        
        print(f"[DEBUG] ripley_data type: {type(ripley_data)}")
        
        # Plot using squidpy's exact approach
        # Based on sq.pl.ripley source code
        from squidpy._constants._constants import RipleyStat
        
        res = ripley_data
        mode_enum = RipleyStat(mode)
        
        # Get categories and palette like squidpy does
        if cluster_key in adata.obs.columns:
            if hasattr(adata.obs[cluster_key], 'cat'):
                categories_list = list(adata.obs[cluster_key].cat.categories)
            else:
                categories_list = sorted(adata.obs[cluster_key].unique())
        else:
            categories_list = []
        
        # Get palette using tab20/tab60 for consistent cluster coloring
        # Use _get_vivid_colors which provides tab20, tab20b, tab20c for up to 60 clusters
        n_categories = len(categories_list)
        if n_categories > 0:
            # Get colors using tab20/tab60 scheme
            color_array = _get_vivid_colors(n_categories)
            # Convert to dict mapping category to color (RGBA tuple)
            palette = {cat: tuple(color_array[i]) for i, cat in enumerate(categories_list)}
        else:
            palette = {}
        
        # Get the statistic DataFrame
        stat_key = f"{mode_enum.s}_stat"
        if stat_key not in res:
            ax.text(0.5, 0.5, f'No {stat_key} found in Ripley data.', 
                   ha='center', va='center', transform=ax.transAxes)
            self.sq_ripley_canvas.draw()
            return
        
        stat_df = res[stat_key]
        
        # Convert palette dict to list if needed (seaborn can handle both)
        plot_palette = palette
        if isinstance(palette, dict):
            # Convert dict to list in the order of categories_list
            plot_palette = [palette.get(cat, '#000000') for cat in categories_list]
        
        # Use seaborn lineplot like squidpy does
        sns.lineplot(
            y="stats",
            x="bins",
            hue=cluster_key,
            data=stat_df,
            hue_order=categories_list,
            palette=plot_palette,
            ax=ax,
        )
        
        # Plot simulations if available (like squidpy's plot_sims parameter)
        if "sims_stat" in res:
            sns.lineplot(
                y="stats", 
                x="bins", 
                errorbar="sd", 
                alpha=0.01, 
                color="gray", 
                data=res["sims_stat"], 
                ax=ax
            )
        
        # Set legend like squidpy
        legend_kwargs = {"loc": "center left", "bbox_to_anchor": (1, 0.5)}
        ax.legend(**legend_kwargs)
        
        # Set labels like squidpy
        ax.set_ylabel("value")
        ax.set_title(f"Ripley's {mode_enum.s}")
        
        self.sq_ripley_canvas.figure.tight_layout()
        self.sq_ripley_canvas.draw()
    
    def _save_sq_ripley_plot(self):
        """Save the Ripley plot."""
        if save_figure_with_options(self.sq_ripley_canvas.figure, "squidpy_ripley.png", self):
            QtWidgets.QMessageBox.information(self, "Success", "Plot saved successfully")
    
    def _export_to_anndata(self):
        """Export data to AnnData format using core function."""
        from PyQt5.QtWidgets import QFileDialog
        
        if not self.anndata_cache:
            QtWidgets.QMessageBox.warning(self, "No Data", 
                "No AnnData objects available. Please build spatial graphs first.")
            return
        
        # Ask user if they want combined or separate files
        reply = QtWidgets.QMessageBox.question(
            self,
            "Export Format",
            "Export as:\n\nYes = Combined file (all ROIs)\nNo = Separate files (one per ROI)",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Yes
        )
        
        if reply == QtWidgets.QMessageBox.Cancel:
            return
        
        try:
            if reply == QtWidgets.QMessageBox.Yes:
                # Combined export
                file_path, _ = QFileDialog.getSaveFileName(
                    self, "Export Combined AnnData", "combined_anndata.h5ad", 
                    "AnnData Files (*.h5ad);;All Files (*)"
                )
                if not file_path:
                    return
                
                # Use core function
                export_anndata(self.anndata_cache, file_path, combined=True)
                QtWidgets.QMessageBox.information(self, "Export Complete", 
                    f"Combined AnnData exported to:\n{file_path}")
            else:
                # Separate files
                export_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
                if not export_dir:
                    return
                
                # Use core function
                export_anndata(self.anndata_cache, export_dir, combined=False)
                QtWidgets.QMessageBox.information(self, "Export Complete", 
                    f"Exported {len(self.anndata_cache)} AnnData file(s) to:\n{export_dir}")
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Export Error", f"Failed to export: {str(e)}")
    
    def _validate_data(self):
        """Validate that required data is available for analysis."""
        if self.feature_dataframe is None or self.feature_dataframe.empty:
            QtWidgets.QMessageBox.warning(self, "No Data", "Feature dataframe is empty.")
            return False
        
        filtered_df = self._get_filtered_dataframe()
        if filtered_df.empty:
            QtWidgets.QMessageBox.warning(self, "No Data", 
                "No data available for selected source files.")
            return False
        
        roi_col = self._get_roi_column()
        required_cols = {roi_col, "cell_id", "centroid_x", "centroid_y"}
        missing = [c for c in required_cols if c not in self.feature_dataframe.columns]
        if missing:
            QtWidgets.QMessageBox.critical(self, "Missing columns", 
                f"Missing required columns: {', '.join(missing)}")
            return False
        
        return True
    
    def reset_analysis_state(self):
        """Reset all analysis state - clear results and allow restart."""
        # Clear all caches
        self.anndata_cache = {}
        self.processed_rois = {}
        self.analysis_status = {}
        self.aggregated_results = {}
        
        # Reset analysis flags
        self.spatial_graph_built = False
        
        # Clear all canvas figures
        canvas_names = [
            'sq_nhood_canvas', 'sq_cooccur_canvas', 'sq_autocorr_canvas', 'sq_ripley_canvas'
        ]
        for canvas_name in canvas_names:
            if hasattr(self, canvas_name):
                canvas = getattr(self, canvas_name)
                canvas.figure.clear()
                canvas.draw()
        
        # Disable save buttons
        save_button_names = [
            'sq_nhood_save_btn', 'sq_cooccur_save_btn', 'sq_autocorr_save_btn', 'sq_ripley_save_btn'
        ]
        for btn_name in save_button_names:
            if hasattr(self, btn_name):
                btn = getattr(self, btn_name)
                btn.setEnabled(False)
        
        # Update status labels
        if hasattr(self, 'graph_status_label'):
            self.graph_status_label.setText("Graph not created")
            self.graph_status_label.setStyleSheet("")
        
        # Update button states
        self._update_button_states()
    
    def refresh_dataframe(self):
        """Refresh the feature dataframe from parent window."""
        parent = self.parent()
        if parent is None:
            return
        
        # Get updated dataframes from parent
        if hasattr(parent, 'feature_dataframe') and parent.feature_dataframe is not None:
            self.original_feature_dataframe = parent.feature_dataframe.copy()
            
            # Update feature_dataframe based on batch correction preference
            if hasattr(parent, 'batch_corrected_dataframe') and parent.batch_corrected_dataframe is not None and not parent.batch_corrected_dataframe.empty:
                self.batch_corrected_dataframe = parent.batch_corrected_dataframe.copy()
                if hasattr(self, 'feature_set_combo') and self.feature_set_combo.currentText() == "Batch-Corrected Features":
                    self.feature_dataframe = self.batch_corrected_dataframe.copy()
                else:
                    self.feature_dataframe = self.original_feature_dataframe.copy()
            else:
                self.batch_corrected_dataframe = None
                self.feature_dataframe = self.original_feature_dataframe.copy()
        
        # Update cluster annotation map if available
        if hasattr(parent, 'cluster_annotation_map'):
            self.cluster_annotation_map = parent.cluster_annotation_map.copy()
        
        # Clear AnnData cache since dataframe changed
        self.anndata_cache = {}
        
        # Refresh ROI combo boxes and other UI elements that depend on dataframe
        self._populate_roi_combo()
    
    def on_clusters_changed(self):
        """Handle cluster changes - reset analysis and refresh dataframe."""
        # Show info message
        reply = QtWidgets.QMessageBox.information(
            self,
            "Clusters Changed",
            "Cluster assignments have been updated. All spatial analysis results will be cleared.\n\n"
            "You can now rebuild the spatial graph and rerun analyses with the new cluster assignments.",
            QtWidgets.QMessageBox.Ok
        )
        
        # Refresh dataframe first
        self.refresh_dataframe()
        
        # Reset all analysis state
        self.reset_analysis_state()
    
    def closeEvent(self, event):
        """Handle dialog closing."""
        event.accept()

