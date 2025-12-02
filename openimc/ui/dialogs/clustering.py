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

from typing import List
import os
import pandas as pd
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PyQt5 import QtWidgets, QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from scipy.stats import mannwhitneyu
from skimage.measure import regionprops
import json
import math
from openimc.utils.logger import get_logger
from openimc.ui.dialogs.figure_save_dialog import save_figure_with_options
from openimc.ui.dialogs.plot_config_dialog import PlotConfigDialog
from openimc.core import cluster

# Optional seaborn for enhanced clustering visualization
try:
    import seaborn as sns
    _HAVE_SEABORN = True
except ImportError:
    _HAVE_SEABORN = False

# Optional leidenalg for Louvain clustering
try:
    import leidenalg
    import igraph as ig
    _HAVE_LEIDEN = True
except ImportError:
    _HAVE_LEIDEN = False

# Optional UMAP for dimensionality reduction
try:
    import umap
    _HAVE_UMAP = True
except ImportError:
    _HAVE_UMAP = False

# Optional HDBSCAN for density-based clustering
try:
    import hdbscan
    _HAVE_HDBSCAN = True
except ImportError:
    _HAVE_HDBSCAN = False

# Optional scikit-learn for k-means and metrics
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, silhouette_samples
    _HAVE_SKLEARN = True
except ImportError:
    _HAVE_SKLEARN = False

# Optional t-SNE for dimensionality reduction
try:
    from sklearn.manifold import TSNE
    _HAVE_TSNE = True
except ImportError:
    _HAVE_TSNE = False


def _get_vivid_colors(n):
    """
    Generate n vivid, distinct colors suitable for cluster visualization.
    Uses tab20, tab20b, tab20c for first 60 colors, then hsv for additional colors.
    
    Args:
        n: Number of colors needed
        
    Returns:
        Array of RGBA colors (n, 4)
    """
    colors = []
    
    # Use tab20, tab20b, tab20c for first 60 colors (vivid and distinct)
    if n <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, n))
    elif n <= 40:
        colors = np.vstack([
            plt.cm.tab20(np.linspace(0, 1, 20)),
            plt.cm.tab20b(np.linspace(0, 1, n - 20))
        ])
    elif n <= 60:
        colors = np.vstack([
            plt.cm.tab20(np.linspace(0, 1, 20)),
            plt.cm.tab20b(np.linspace(0, 1, 20)),
            plt.cm.tab20c(np.linspace(0, 1, n - 40))
        ])
    else:
        # For more than 60 colors, use tab20 series + hsv for the rest
        colors = np.vstack([
            plt.cm.tab20(np.linspace(0, 1, 20)),
            plt.cm.tab20b(np.linspace(0, 1, 20)),
            plt.cm.tab20c(np.linspace(0, 1, 20))
        ])
        # Use hsv colormap for additional colors, avoiding very dark/light values
        remaining = n - 60
        hsv_colors = plt.cm.hsv(np.linspace(0.1, 0.9, remaining))
        colors = np.vstack([colors, hsv_colors])
    
    return colors


def _get_patient_colors(n):
    """
    Generate n distinct colors for patient/source annotation.
    Uses a different color palette than clusters (Set3, Pastel1, Pastel2) to ensure
    patient annotations are visually distinct from cluster colors.
    
    Args:
        n: Number of colors needed
        
    Returns:
        Array of RGBA colors (n, 4)
    """
    colors = []
    
    # Use Set3, Pastel1, Pastel2 for first 36 colors (different from tab20 used for clusters)
    if n <= 12:
        colors = plt.cm.Set3(np.linspace(0, 1, n))
    elif n <= 24:
        colors = np.vstack([
            plt.cm.Set3(np.linspace(0, 1, 12)),
            plt.cm.Pastel1(np.linspace(0, 1, n - 12))
        ])
    elif n <= 36:
        colors = np.vstack([
            plt.cm.Set3(np.linspace(0, 1, 12)),
            plt.cm.Pastel1(np.linspace(0, 1, 9)),
            plt.cm.Pastel2(np.linspace(0, 1, n - 21))
        ])
    else:
        # For more than 36 colors, use Set3/Pastel series + hsv for the rest
        colors = np.vstack([
            plt.cm.Set3(np.linspace(0, 1, 12)),
            plt.cm.Pastel1(np.linspace(0, 1, 9)),
            plt.cm.Pastel2(np.linspace(0, 1, 8))
        ])
        # Use hsv colormap for additional colors, with different range than cluster colors
        remaining = n - 29
        # Use a different hue range to ensure distinction from cluster colors
        hsv_colors = plt.cm.hsv(np.linspace(0.15, 0.85, remaining))
        colors = np.vstack([colors, hsv_colors])
    
    return colors


# --------------------------
# Cell Clustering Dialog
# --------------------------
class CellClusteringDialog(QtWidgets.QDialog):
    def __init__(self, feature_dataframe, normalization_config=None, batch_corrected_dataframe=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cell Clustering Analysis")
        self.setModal(True)
        
        # Set size to 90% of parent window if available
        if parent is not None:
            parent_size = parent.size()
            dialog_width = int(parent_size.width() * 0.9)
            dialog_height = int(parent_size.height() * 0.9)
            self.resize(dialog_width, dialog_height)
        
        self.setMinimumSize(800, 600)
        self.original_feature_dataframe = feature_dataframe  # Store original
        self.batch_corrected_dataframe = batch_corrected_dataframe  # Store batch-corrected
        # Default to batch-corrected features if available, otherwise use original
        if batch_corrected_dataframe is not None and not batch_corrected_dataframe.empty:
            self.feature_dataframe = batch_corrected_dataframe.copy()
        else:
            self.feature_dataframe = feature_dataframe  # Active dataframe (can be switched)
        self.normalization_config = normalization_config
        self.cluster_labels = None
        self.clustered_data = None
        self.clustered_data_unscaled = None  # Store original unscaled data for heatmap display
        self.umap_embedding = None
        self.tsne_embedding = None
        self.cluster_annotation_map = {}
        self.cluster_backend_names = {}  # Store normalized names for CSV export
        self.original_cluster_assignments = None  # Store original cluster assignments before merging
        self.clustering_scaling_method = None  # Store scaling method used for clustering
        self.patient_annotation_map = {}  # Store custom patient/source file labels
        self.feature_label_map = {}  # Store custom feature labels for y-axis ticks (friendly names)
        self.patient_legend_label = 'Patient/Source'  # Custom label for patient annotation legend
        # Initialize patient annotation column with default priority (source_file, batch_group, source_well, then metadata columns)
        self.patient_annotation_column = None
        if self.feature_dataframe is not None:
            # First check standard columns
            for col in ['source_file', 'batch_group', 'source_well']:
                if col in self.feature_dataframe.columns:
                    self.patient_annotation_column = col
                    break
            
            # If no standard column found, check for metadata columns
            if self.patient_annotation_column is None:
                metadata_cols = self._get_metadata_columns(self.feature_dataframe)
                if metadata_cols:
                    # Prefer columns that might be batch identifiers (PID, patient_id, etc.)
                    priority_metadata = [col for col in metadata_cols 
                                       if any(keyword in col.lower() for keyword in ['pid', 'patient', 'batch', 'sample', 'subject'])]
                    if priority_metadata:
                        self.patient_annotation_column = priority_metadata[0]
                    else:
                        self.patient_annotation_column = metadata_cols[0]
        self.patient_annotation_enabled = False  # Track whether patient annotation is enabled
        self.feature_tick_fontsize = 8  # Font size for feature labels on y-axis
        self.gating_rules = []  # list of dict: {name, logic, conditions: [{column, op, threshold}]}
        self.llm_phenotype_cache = {}  # Cache for LLM phenotype suggestions
        self.seed = 42  # Default seed for reproducibility
        self.statistical_results = {}  # Store statistical test results for export: {marker: [(cluster1, cluster2, p_val, adj_p_val)]}
        self.filter_settings = None  # Store filter settings from feature selector
        
        self._create_ui()
        self._setup_plot()
        self._on_clustering_type_changed()  # Initialize UI state
        self._on_leiden_mode_changed()  # Initialize Leiden mode state
        
    def _create_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # Title
        title_label = QtWidgets.QLabel("Cell Clustering Analysis")
        title_label.setStyleSheet("QLabel { font-weight: bold; font-size: 12pt; }")
        layout.addWidget(title_label)
        
        # Options panel
        options_group = QtWidgets.QGroupBox("Clustering Options")
        options_layout = QtWidgets.QHBoxLayout(options_group)
        
        # Feature set selector (if batch-corrected data is available)
        if self.batch_corrected_dataframe is not None and not self.batch_corrected_dataframe.empty:
            options_layout.addWidget(QtWidgets.QLabel("Feature Set:"))
            self.feature_set_combo = QtWidgets.QComboBox()
            self.feature_set_combo.addItem("Original Features")
            self.feature_set_combo.addItem("Batch-Corrected Features")
            self.feature_set_combo.setToolTip("Choose between original or batch-corrected feature sets")
            self.feature_set_combo.currentTextChanged.connect(self._on_feature_set_changed)
            # Default to batch-corrected features if available (block signals to avoid triggering callback during init)
            self.feature_set_combo.blockSignals(True)
            self.feature_set_combo.setCurrentText("Batch-Corrected Features")
            self.feature_set_combo.blockSignals(False)
            options_layout.addWidget(self.feature_set_combo)
            options_layout.addSpacing(10)
        
        # (Aggregation and morphometric inclusion moved to Feature Selector dialog)
        
        # Clustering method type (first)
        options_layout.addWidget(QtWidgets.QLabel("Clustering Method:"))
        self.clustering_type = QtWidgets.QComboBox()
        clustering_types = []
        if _HAVE_LEIDEN:
            clustering_types.append("Leiden")
            clustering_types.append("Louvain")
        if _HAVE_SKLEARN:
            clustering_types.append("K-means")
        if _HAVE_HDBSCAN:
            clustering_types.append("HDBSCAN")
        clustering_types.append("Hierarchical")
        self.clustering_type.addItems(clustering_types)
        # Set Leiden as default if available, otherwise fall back to Hierarchical
        if _HAVE_LEIDEN:
            self.clustering_type.setCurrentText("Leiden")
        else:
            self.clustering_type.setCurrentText("Hierarchical")
        self.clustering_type.currentTextChanged.connect(self._on_clustering_type_changed)
        options_layout.addWidget(self.clustering_type)
        
        # Feature scaling method (for clustering)
        options_layout.addWidget(QtWidgets.QLabel("Feature Scaling:"))
        self.clustering_scaling_combo = QtWidgets.QComboBox()
        self.clustering_scaling_combo.addItems(["None (no scaling)", "Z-score", "MAD (Median Absolute Deviation)"])
        self.clustering_scaling_combo.setCurrentText("None (no scaling)")
        self.clustering_scaling_combo.setToolTip("Scaling method applied to features before clustering")
        options_layout.addWidget(self.clustering_scaling_combo)
        
        # Random seed
        options_layout.addWidget(QtWidgets.QLabel("Random Seed:"))
        self.seed_spinbox = QtWidgets.QSpinBox()
        self.seed_spinbox.setRange(0, 2**31 - 1)
        self.seed_spinbox.setValue(42)
        self.seed_spinbox.setToolTip("Random seed for reproducibility (default: 42)")
        options_layout.addWidget(self.seed_spinbox)
        
        # Number of clusters (for hierarchical and k-means)
        self.n_clusters_label = QtWidgets.QLabel("Number of clusters:")
        options_layout.addWidget(self.n_clusters_label)
        self.n_clusters = QtWidgets.QSpinBox()
        self.n_clusters.setRange(2, 50)
        self.n_clusters.setValue(5)
        options_layout.addWidget(self.n_clusters)
        
        # K-range search button (for hierarchical and k-means)
        self.k_range_btn = QtWidgets.QPushButton("Find Optimal K")
        self.k_range_btn.setToolTip("Search over a range of k values and plot elbow/silhouette scores")
        self.k_range_btn.clicked.connect(self._open_k_range_dialog)
        options_layout.addWidget(self.k_range_btn)
        
        # Hierarchical method selection (initially visible)
        self.hierarchical_label = QtWidgets.QLabel("Linkage Method:")
        self.hierarchical_method = QtWidgets.QComboBox()
        self.hierarchical_method.addItems(["ward", "complete", "average", "single"])
        self.hierarchical_method.setCurrentText("ward")
        options_layout.addWidget(self.hierarchical_label)
        options_layout.addWidget(self.hierarchical_method)
        
        # Leiden clustering options (initially hidden)
        self.leiden_options_group = QtWidgets.QGroupBox("Leiden/Louvain Options")
        leiden_options_layout = QtWidgets.QHBoxLayout(self.leiden_options_group)
        
        # Resolution vs Modularity choice
        self.leiden_mode_group = QtWidgets.QButtonGroup()
        self.resolution_radio = QtWidgets.QRadioButton("Resolution")
        self.modularity_radio = QtWidgets.QRadioButton("Modularity")
        self.resolution_radio.setChecked(True)
        self.leiden_mode_group.addButton(self.resolution_radio)
        self.leiden_mode_group.addButton(self.modularity_radio)
        leiden_options_layout.addWidget(self.resolution_radio)
        leiden_options_layout.addWidget(self.modularity_radio)
        
        # N neighbors parameter for graph construction
        self.n_neighbors_label = QtWidgets.QLabel("N neighbors:")
        self.n_neighbors_spinbox = QtWidgets.QSpinBox()
        self.n_neighbors_spinbox.setRange(5, 100)
        self.n_neighbors_spinbox.setValue(15)
        self.n_neighbors_spinbox.setToolTip("Number of neighbors for k-NN graph construction")
        leiden_options_layout.addWidget(self.n_neighbors_label)
        leiden_options_layout.addWidget(self.n_neighbors_spinbox)
        
        # Resolution parameter
        self.resolution_label = QtWidgets.QLabel("Resolution:")
        self.resolution_spinbox = QtWidgets.QDoubleSpinBox()
        self.resolution_spinbox.setRange(0.1, 5.0)
        self.resolution_spinbox.setSingleStep(0.1)
        self.resolution_spinbox.setValue(1.0)
        self.resolution_spinbox.setDecimals(1)
        leiden_options_layout.addWidget(self.resolution_label)
        leiden_options_layout.addWidget(self.resolution_spinbox)
        
        # Distance metric selection
        self.leiden_metric_label = QtWidgets.QLabel("Distance metric:")
        self.leiden_metric_combo = QtWidgets.QComboBox()
        self.leiden_metric_combo.addItems(["euclidean", "manhattan", "cosine"])
        self.leiden_metric_combo.setCurrentText("euclidean")
        self.leiden_metric_combo.setToolTip("Distance metric to use for k-NN graph construction")
        leiden_options_layout.addWidget(self.leiden_metric_label)
        leiden_options_layout.addWidget(self.leiden_metric_combo)
        
        # Jaccard weighting option (PhenoGraph-like)
        self.jaccard_checkbox = QtWidgets.QCheckBox("Use Jaccard weighting")
        self.jaccard_checkbox.setToolTip("Weight graph edges with Jaccard similarity (PhenoGraph-like implementation)")
        self.jaccard_checkbox.setChecked(False)
        leiden_options_layout.addWidget(self.jaccard_checkbox)
        
        leiden_options_layout.addStretch()
        
        # Connect radio button changes
        self.resolution_radio.toggled.connect(self._on_leiden_mode_changed)
        self.modularity_radio.toggled.connect(self._on_leiden_mode_changed)
        
        self.leiden_options_group.setVisible(False)
        options_layout.addWidget(self.leiden_options_group)

        # HDBSCAN clustering options (initially hidden)
        self.hdbscan_options_group = QtWidgets.QGroupBox("HDBSCAN Clustering Options")
        hdbscan_options_layout = QtWidgets.QHBoxLayout(self.hdbscan_options_group)
        
        # Min cluster size
        self.min_cluster_size_label = QtWidgets.QLabel("Min cluster size:")
        self.min_cluster_size_spinbox = QtWidgets.QSpinBox()
        self.min_cluster_size_spinbox.setRange(2, 1000)
        self.min_cluster_size_spinbox.setValue(10)
        self.min_cluster_size_spinbox.setToolTip("Minimum size of clusters; smaller clusters will be discarded as noise")
        hdbscan_options_layout.addWidget(self.min_cluster_size_label)
        hdbscan_options_layout.addWidget(self.min_cluster_size_spinbox)
        
        # Min samples
        self.min_samples_label = QtWidgets.QLabel("Min samples:")
        self.min_samples_spinbox = QtWidgets.QSpinBox()
        self.min_samples_spinbox.setRange(1, 100)
        self.min_samples_spinbox.setValue(5)
        self.min_samples_spinbox.setToolTip("Number of samples in a neighborhood for a point to be considered a core point")
        hdbscan_options_layout.addWidget(self.min_samples_label)
        hdbscan_options_layout.addWidget(self.min_samples_spinbox)
        
        # Cluster selection method (EOM vs Leaf)
        self.cluster_selection_label = QtWidgets.QLabel("Cluster selection method:")
        self.cluster_selection_combo = QtWidgets.QComboBox()
        self.cluster_selection_combo.addItems(["eom", "leaf"])
        self.cluster_selection_combo.setCurrentText("eom")
        self.cluster_selection_combo.setToolTip("eom: Excess of Mass (default, more conservative)\nleaf: Leaf (more aggressive, creates smaller clusters)")
        hdbscan_options_layout.addWidget(self.cluster_selection_label)
        hdbscan_options_layout.addWidget(self.cluster_selection_combo)
        
        # Metric selection (only euclidean and manhattan for HDBSCAN)
        self.metric_label = QtWidgets.QLabel("Distance metric:")
        self.metric_combo = QtWidgets.QComboBox()
        self.metric_combo.addItems(["euclidean", "manhattan"])
        self.metric_combo.setCurrentText("euclidean")
        self.metric_combo.setToolTip("Distance metric to use for clustering")
        hdbscan_options_layout.addWidget(self.metric_label)
        hdbscan_options_layout.addWidget(self.metric_combo)
        
        hdbscan_options_layout.addStretch()
        
        self.hdbscan_options_group.setVisible(False)
        options_layout.addWidget(self.hdbscan_options_group)

        # Dendrogram mode (only for hierarchical methods)
        self.dendro_label = QtWidgets.QLabel("Dendrogram:")
        self.dendro_mode = QtWidgets.QComboBox()
        self.dendro_mode.addItems(["Rows only", "Rows and columns"]) 
        self.dendro_mode.setCurrentText("Rows and columns")  # Default to both dendrograms
        options_layout.addWidget(self.dendro_label)
        options_layout.addWidget(self.dendro_mode)
        
        # Run clustering button
        self.run_btn = QtWidgets.QPushButton("Run Clustering")
        self.run_btn.clicked.connect(self._run_clustering)
        options_layout.addWidget(self.run_btn)
        
        # Save clustering output button
        self.save_output_btn = QtWidgets.QPushButton("Save Clustering Output")
        self.save_output_btn.clicked.connect(self._save_clustering_output)
        self.save_output_btn.setEnabled(False)
        self.save_output_btn.setToolTip("Save CSV with all features, cluster labels, and manual annotations")
        options_layout.addWidget(self.save_output_btn)
        
        options_layout.addStretch()
        layout.addWidget(options_group)
        
        # Plot area (Step 2: Visualization)
        plot_group = QtWidgets.QGroupBox("Visualization")
        plot_layout = QtWidgets.QVBoxLayout(plot_group)
        
        # Create matplotlib canvas
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)
        
        # Visualization controls
        viz_layout = QtWidgets.QHBoxLayout()
        viz_layout.addWidget(QtWidgets.QLabel("View:"))
        self.view_combo = QtWidgets.QComboBox()
        view_items = ["Heatmap", "UMAP", "Stacked Bars", "Differential Expression", "Boxplot/Violin Plot"]
        if _HAVE_TSNE:
            view_items.insert(2, "t-SNE")  # Insert after UMAP
        self.view_combo.addItems(view_items)
        self.view_combo.currentTextChanged.connect(self._on_view_changed)
        viz_layout.addWidget(self.view_combo)

        # Color-by control (UMAP/t-SNE only) - multi-select for faceted plotting
        self.color_by_label = QtWidgets.QLabel("Color by (select multiple for faceted plots):")
        viz_layout.addWidget(self.color_by_label)
        # Search/filter box for color-by options
        self.color_by_search = QtWidgets.QLineEdit()
        self.color_by_search.setPlaceholderText("Search/filter options...")
        self.color_by_search.textChanged.connect(self._filter_color_by_options)
        viz_layout.addWidget(self.color_by_search)
        self.color_by_listwidget = QtWidgets.QListWidget()
        self.color_by_listwidget.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.color_by_listwidget.setMaximumHeight(100)
        self.color_by_listwidget.itemSelectionChanged.connect(self._on_color_by_changed)
        viz_layout.addWidget(self.color_by_listwidget)
        # Keep combo for backward compatibility but hide it
        self.color_by_combo = QtWidgets.QComboBox()
        self.color_by_combo.setVisible(False)

        # Point size control (UMAP/t-SNE only)
        self.point_size_label = QtWidgets.QLabel("Point size:")
        viz_layout.addWidget(self.point_size_label)
        self.point_size_spinbox = QtWidgets.QSpinBox()
        self.point_size_spinbox.setMinimum(1)
        self.point_size_spinbox.setMaximum(200)
        self.point_size_spinbox.setValue(18)
        self.point_size_spinbox.setToolTip("Size of points in scatter plot")
        self.point_size_spinbox.valueChanged.connect(self._on_point_style_changed)
        viz_layout.addWidget(self.point_size_spinbox)

        # Point alpha control (UMAP/t-SNE only)
        self.point_alpha_label = QtWidgets.QLabel("Point alpha:")
        viz_layout.addWidget(self.point_alpha_label)
        self.point_alpha_spinbox = QtWidgets.QDoubleSpinBox()
        self.point_alpha_spinbox.setMinimum(0.0)
        self.point_alpha_spinbox.setMaximum(1.0)
        self.point_alpha_spinbox.setSingleStep(0.1)
        self.point_alpha_spinbox.setValue(0.8)
        self.point_alpha_spinbox.setDecimals(2)
        self.point_alpha_spinbox.setToolTip("Transparency of points (0.0 = transparent, 1.0 = opaque)")
        self.point_alpha_spinbox.valueChanged.connect(self._on_point_style_changed)
        viz_layout.addWidget(self.point_alpha_spinbox)

        # Show legend checkbox (UMAP/t-SNE/Stacked Bars)
        self.show_legend_checkbox = QtWidgets.QCheckBox("Show legend")
        self.show_legend_checkbox.setChecked(True)
        self.show_legend_checkbox.setToolTip("Show/hide legend in plots (legend is also shown in heatmap)")
        self.show_legend_checkbox.stateChanged.connect(self._on_legend_changed)
        viz_layout.addWidget(self.show_legend_checkbox)

        # Remake UMAP button (UMAP only)
        self.remake_umap_btn = QtWidgets.QPushButton("Remake UMAP")
        self.remake_umap_btn.setToolTip("Regenerate UMAP with new parameters (features, scaling, n_neighbors)")
        self.remake_umap_btn.clicked.connect(self._remake_umap)
        viz_layout.addWidget(self.remake_umap_btn)

        # Group-by for stacked bars (Stacked Bars only)
        self.group_by_label = QtWidgets.QLabel("Group by:")
        viz_layout.addWidget(self.group_by_label)
        self.group_by_combo = QtWidgets.QComboBox()
        candidate_cols = [
            'roi', 'ROI', 'slide', 'Slide', 'condition', 'Condition',
            'acquisition_name', 'well', 'acquisition_id'
        ]
        available_group_cols = [c for c in candidate_cols if c in self.feature_dataframe.columns]
        # Add source_file explicitly if it exists
        if 'source_file' in self.feature_dataframe.columns and 'source_file' not in available_group_cols:
            available_group_cols.insert(0, 'source_file')
        # Add source_file_acquisition_id if both source_file and acquisition_id exist
        if 'source_file' in self.feature_dataframe.columns and 'acquisition_id' in self.feature_dataframe.columns:
            # Create merged column if it doesn't exist
            if 'source_file_acquisition_id' not in self.feature_dataframe.columns:
                # Use assign() to avoid DataFrame fragmentation warnings
                self.feature_dataframe = self.feature_dataframe.assign(
                    source_file_acquisition_id=(
                        self.feature_dataframe['source_file'].astype(str) + '_' + 
                        self.feature_dataframe['acquisition_id'].astype(str)
                    )
                )
            if 'source_file_acquisition_id' not in available_group_cols:
                available_group_cols.insert(0, 'source_file_acquisition_id')
        if not available_group_cols:
            available_group_cols = ['acquisition_name'] if 'acquisition_name' in self.feature_dataframe.columns else []
        
        # Add metadata columns for grouping
        metadata_cols = self._get_metadata_columns(self.feature_dataframe)
        if metadata_cols:
            if available_group_cols:
                available_group_cols.extend(metadata_cols)
            else:
                available_group_cols = metadata_cols
        
        for col in available_group_cols:
            self.group_by_combo.addItem(col)
        viz_layout.addWidget(self.group_by_combo)
        
        # View type selector for stacked bars (Fraction vs Total enumeration)
        self.stacked_bars_view_type_label = QtWidgets.QLabel("View type:")
        viz_layout.addWidget(self.stacked_bars_view_type_label)
        self.stacked_bars_view_type_combo = QtWidgets.QComboBox()
        self.stacked_bars_view_type_combo.addItems(["Fraction", "Total enumeration"])
        self.stacked_bars_view_type_combo.setCurrentText("Fraction")
        self.stacked_bars_view_type_combo.currentTextChanged.connect(self._on_stacked_bars_view_type_changed)
        viz_layout.addWidget(self.stacked_bars_view_type_combo)
        
        # Cluster filter button for stacked bars
        self.stacked_bars_filter_btn = QtWidgets.QPushButton("Filter Clusters...")
        self.stacked_bars_filter_btn.setToolTip("Select which clusters to display in the stacked bars plot")
        self.stacked_bars_filter_btn.clicked.connect(self._open_stacked_bars_filter_dialog)
        viz_layout.addWidget(self.stacked_bars_filter_btn)
        
        # Initialize stacked bars filter selection (None means show all clusters)
        self.stacked_bars_filter_selection = None
        
        # Connect group_by_combo to refresh plot when changed
        self.group_by_combo.currentTextChanged.connect(self._on_group_by_changed)

        # Colormap selector (for heatmaps and differential expression)
        # Note: For Heatmap, colormap is also available in PlotConfigDialog
        # But we keep it here for Differential Expression since Configure Plot only shows for Heatmap
        self.colormap_label = QtWidgets.QLabel("Colormap:")
        viz_layout.addWidget(self.colormap_label)
        self.colormap_combo = QtWidgets.QComboBox()
        self.colormap_combo.addItems([
            "RdBu_r (Red-White-Blue)",
            "viridis (Purple-Green-Yellow)", 
            "plasma (Purple-Pink-Yellow)",
            "inferno (Purple-Red-Yellow)",
            "Blues (Light-Dark Blue)",
            "Reds (Light-Dark Red)",
            "Greens (Light-Dark Green)",
            "Oranges (Light-Dark Orange)",
            "Purples (Light-Dark Purple)"
        ])
        self.colormap_combo.setCurrentText("RdBu_r (Red-White-Blue)")
        self.colormap_combo.currentTextChanged.connect(self._on_colormap_changed)
        viz_layout.addWidget(self.colormap_combo)

        # Heatmap scaling selector (for heatmap only)
        self.heatmap_scaling_label = QtWidgets.QLabel("Heatmap Scaling:")
        viz_layout.addWidget(self.heatmap_scaling_label)
        self.heatmap_scaling_combo = QtWidgets.QComboBox()
        self.heatmap_scaling_combo.addItems(["None (no scaling)", "Z-score", "MAD (Median Absolute Deviation)"])
        self.heatmap_scaling_combo.setCurrentText("None (no scaling)")
        self.heatmap_scaling_combo.setToolTip("Scaling method applied to features in the heatmap display")
        self.heatmap_scaling_combo.currentTextChanged.connect(self._on_heatmap_scaling_changed)
        viz_layout.addWidget(self.heatmap_scaling_combo)

        # Configure plot button (opens PlotConfigDialog)
        # Note: Heatmap source, heatmap filter, heatmap scaling, patient annotation,
        # and patient label customization are now available in PlotConfigDialog (Heatmap only)
        self.configure_plot_btn = QtWidgets.QPushButton("Configure Plot...")
        self.configure_plot_btn.setToolTip("Open plot configuration dialog to customize font sizes, labels, and other plot settings")
        self.configure_plot_btn.clicked.connect(self._open_plot_config_dialog)
        viz_layout.addWidget(self.configure_plot_btn)

        # Customize feature labels button (for Differential Expression, Stacked Bars, Boxplot/Violin Plot)
        self.feature_labels_btn = QtWidgets.QPushButton("Customize Feature Labels...")
        self.feature_labels_btn.setToolTip("Set custom display names for features in visualizations (e.g., 'Vimentin_mean' -> 'Mean Vimentin')")
        self.feature_labels_btn.clicked.connect(self._open_feature_labels_dialog)
        viz_layout.addWidget(self.feature_labels_btn)

        # Top N markers selector (for differential expression only)
        # Note: Also available in PlotConfigDialog, but kept here for quick access
        self.top_n_label = QtWidgets.QLabel("Top N:")
        viz_layout.addWidget(self.top_n_label)
        self.top_n_spinbox = QtWidgets.QSpinBox()
        self.top_n_spinbox.setMinimum(1)
        self.top_n_spinbox.setMaximum(20)
        self.top_n_spinbox.setValue(5)
        self.top_n_spinbox.valueChanged.connect(self._on_top_n_changed)
        viz_layout.addWidget(self.top_n_spinbox)

        # Note: Marker selection, plot type, and statistical testing options
        # are also available in PlotConfigDialog, but kept here for quick access
        # Marker selection for boxplot/violin plot
        self.marker_select_label = QtWidgets.QLabel("Markers:")
        viz_layout.addWidget(self.marker_select_label)
        self.marker_select_btn = QtWidgets.QPushButton("Select Markers...")
        self.marker_select_btn.setToolTip("Select markers to visualize")
        self.marker_select_btn.clicked.connect(self._open_marker_selection_dialog)
        viz_layout.addWidget(self.marker_select_btn)
        self.selected_markers = []  # Store selected markers

        # Plot type selector (for boxplot/violin plot only)
        self.plot_type_label = QtWidgets.QLabel("Plot type:")
        viz_layout.addWidget(self.plot_type_label)
        self.plot_type_combo = QtWidgets.QComboBox()
        self.plot_type_combo.addItems(["Violin Plot", "Boxplot"])
        self.plot_type_combo.setCurrentText("Violin Plot")
        self.plot_type_combo.currentTextChanged.connect(self._on_plot_type_changed)
        viz_layout.addWidget(self.plot_type_combo)

        # Statistical testing checkbox (for boxplot/violin plot only)
        self.stats_test_checkbox = QtWidgets.QCheckBox("Show statistical tests")
        self.stats_test_checkbox.setToolTip("Perform statistical tests with BH correction")
        self.stats_test_checkbox.setChecked(False)
        self.stats_test_checkbox.stateChanged.connect(self._on_stats_test_changed)
        viz_layout.addWidget(self.stats_test_checkbox)

        # Statistical test mode selector
        self.stats_mode_label = QtWidgets.QLabel("Test mode:")
        viz_layout.addWidget(self.stats_mode_label)
        self.stats_mode_combo = QtWidgets.QComboBox()
        self.stats_mode_combo.addItems(["Pairwise (all pairs)", "One vs Others"])
        self.stats_mode_combo.currentTextChanged.connect(self._on_stats_mode_changed)
        viz_layout.addWidget(self.stats_mode_combo)

        # Cluster selector for one-vs-others mode
        self.stats_cluster_label = QtWidgets.QLabel("Reference cluster:")
        viz_layout.addWidget(self.stats_cluster_label)
        self.stats_cluster_combo = QtWidgets.QComboBox()
        self.stats_cluster_combo.currentTextChanged.connect(self._on_stats_cluster_changed)
        viz_layout.addWidget(self.stats_cluster_combo)

        # Export statistical results button
        self.stats_export_btn = QtWidgets.QPushButton("Export Stats")
        self.stats_export_btn.setToolTip("Export statistical test results (raw and adjusted p-values)")
        self.stats_export_btn.clicked.connect(self._export_statistical_results)
        self.stats_export_btn.setEnabled(False)
        viz_layout.addWidget(self.stats_export_btn)

        viz_layout.addStretch()

        # Save current plot
        self.save_plot_btn = QtWidgets.QPushButton("Save Plot")
        self.save_plot_btn.clicked.connect(self._save_current_plot)
        self.save_plot_btn.setEnabled(False)
        viz_layout.addWidget(self.save_plot_btn)

        # Close
        self.close_btn = QtWidgets.QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        viz_layout.addWidget(self.close_btn)

        plot_layout.addLayout(viz_layout)
        layout.addWidget(plot_group)

        # Step 3: Phenotype tools
        phenotype_group = QtWidgets.QGroupBox("Phenotype Annotation / Exploration")
        phenotype_layout = QtWidgets.QHBoxLayout(phenotype_group)
        self.annotate_btn = QtWidgets.QPushButton("Annotate Phenotypes")
        self.annotate_btn.clicked.connect(self._open_annotation_dialog)
        self.annotate_btn.setEnabled(False)
        phenotype_layout.addWidget(self.annotate_btn)
        
        # Merge clusters button
        self.merge_clusters_btn = QtWidgets.QPushButton("Merge Clusters")
        self.merge_clusters_btn.clicked.connect(self._open_merge_clusters_dialog)
        self.merge_clusters_btn.setEnabled(False)
        self.merge_clusters_btn.setToolTip("Merge two clusters into one")
        phenotype_layout.addWidget(self.merge_clusters_btn)
        

        self.explore_btn = QtWidgets.QPushButton("Explore Clusters")
        self.explore_btn.clicked.connect(self._explore_clusters)
        self.explore_btn.setEnabled(False)
        phenotype_layout.addWidget(self.explore_btn)

        # Manual gating entry point (Step 1/3 entry kept here for linear flow)
        self.gating_btn = QtWidgets.QPushButton("Manual Gating")
        self.gating_btn.clicked.connect(self._open_gating_dialog)
        phenotype_layout.addWidget(self.gating_btn)

        phenotype_layout.addStretch()
        layout.addWidget(phenotype_group)
        
    def _setup_plot(self):
        """Setup the matplotlib plot."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, "Click 'Run Clustering' to generate heatmap", 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        self.canvas.draw()
        self._update_viz_controls_visibility()
        
    def _on_clustering_type_changed(self):
        """Handle clustering type change to show/hide relevant controls."""
        clustering_type = self.clustering_type.currentText()
        is_leiden = clustering_type == "Leiden"
        is_louvain = clustering_type == "Louvain"
        is_hierarchical = clustering_type == "Hierarchical"
        is_hdbscan = clustering_type == "HDBSCAN"
        is_kmeans = clustering_type == "K-means"
        
        # Show/hide Leiden options group (also used for Louvain)
        self.leiden_options_group.setVisible(is_leiden or is_louvain)
        
        # Show/hide HDBSCAN options group
        self.hdbscan_options_group.setVisible(is_hdbscan)
        
        # Show/hide hierarchical method selection
        self.hierarchical_label.setVisible(is_hierarchical)
        self.hierarchical_method.setVisible(is_hierarchical)
        
        # Show/hide number of clusters for hierarchical and k-means
        if hasattr(self, 'n_clusters_label'):
            self.n_clusters_label.setVisible(is_hierarchical or is_kmeans)
        if hasattr(self, 'n_clusters'):
            self.n_clusters.setVisible(is_hierarchical or is_kmeans)
        
        # Show/hide k-range search button for hierarchical and k-means
        if hasattr(self, 'k_range_btn'):
            self.k_range_btn.setVisible(is_hierarchical or is_kmeans)
        
        # Show/hide dendrogram controls for hierarchical methods
        self.dendro_label.setVisible(is_hierarchical)
        self.dendro_mode.setVisible(is_hierarchical)
        self._update_viz_controls_visibility()
    
    def _on_feature_set_changed(self):
        """Handle feature set selection change."""
        if not hasattr(self, 'feature_set_combo'):
            return
        
        selected = self.feature_set_combo.currentText()
        if selected == "Batch-Corrected Features" and self.batch_corrected_dataframe is not None:
            self.feature_dataframe = self.batch_corrected_dataframe.copy()
        else:
            self.feature_dataframe = self.original_feature_dataframe.copy()
        
        # Clear existing clustering results when switching feature sets
        self.cluster_labels = None
        self.clustered_data = None
        self.clustered_data_unscaled = None
        self.umap_embedding = None
        self.tsne_embedding = None
        
        # Clear plots if they exist
        if hasattr(self, 'figure') and hasattr(self, 'canvas'):
            self.figure.clear()
            # Add placeholder text
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "Feature set changed. Click 'Run Clustering' to generate heatmap", 
                    ha='center', va='center', transform=ax.transAxes, fontsize=14)
            self.canvas.draw()
    
    def _on_leiden_mode_changed(self):
        """Handle Leiden clustering mode change (resolution vs modularity)."""
        use_resolution = self.resolution_radio.isChecked()
        self.resolution_label.setVisible(use_resolution)
        self.resolution_spinbox.setVisible(use_resolution)
        
    def _run_clustering(self):
        """Run the clustering analysis."""
        try:
            # Reset cluster merges when re-clustering (restore original assignments if they exist)
            if self.original_cluster_assignments is not None:
                self._reset_cluster_merges()
            
            # Reset custom cluster names when re-clustering
            self.cluster_annotation_map = {}
            self.cluster_backend_names = {}
            # Clear LLM phenotype cache when re-clustering
            self.llm_phenotype_cache = {}
            
            # Clear any existing cluster phenotype data
            if hasattr(self, 'clustered_data') and self.clustered_data is not None and 'cluster_phenotype' in self.clustered_data.columns:
                self.clustered_data = self.clustered_data.drop('cluster_phenotype', axis=1)
            if 'cluster_phenotype' in self.feature_dataframe.columns:
                self.feature_dataframe = self.feature_dataframe.drop('cluster_phenotype', axis=1)
            
            # Get options
            # Defaults for backward compatibility (now controlled by selector)
            agg_method = "mean"
            include_morpho = True
            n_clusters = self.n_clusters.value()
            clustering_type = self.clustering_type.currentText()
            
            # Determine the actual clustering method
            if clustering_type == "Leiden":
                cluster_method = "leiden"
            elif clustering_type == "Louvain":
                cluster_method = "louvain"
            elif clustering_type == "HDBSCAN":
                cluster_method = "hdbscan"
            else:  # Hierarchical
                cluster_method = self.hierarchical_method.currentText()
            
            # Prepare data
            # Allow user to select features interactively
            available_cols = self._list_available_feature_columns(include_morpho)
            from openimc.ui.dialogs.feature_selector_dialog import FeatureSelectorDialog
            selector = FeatureSelectorDialog(available_cols, self)
            # Pre-populate filter settings if available
            if self.filter_settings is not None:
                selector.set_filter_settings(self.filter_settings)
            if selector.exec_() != QtWidgets.QDialog.Accepted:
                return
            selected_columns = selector.get_selected_columns()
            
            # Get filter settings
            filter_settings = selector.get_filter_settings()
            self.filter_settings = filter_settings  # Store for use in UMAP/spatial analyses

            # Apply filters to feature dataframe before clustering
            filtered_df = self._apply_filters(self.feature_dataframe.copy(), filter_settings)
            if filtered_df.empty:
                QtWidgets.QMessageBox.warning(self, "No Data", "No cells remain after applying filters.")
                return
            
            # Get scaling method
            scaling_text = self.clustering_scaling_combo.currentText()
            scaling_map = {
                "None (no scaling)": "none",
                "Z-score": "zscore",
                "MAD (Median Absolute Deviation)": "mad"
            }
            scaling_method = scaling_map.get(scaling_text, "zscore")
            
            result = self._prepare_clustering_data(agg_method, include_morpho, selected_columns, scaling_method, filtered_df, filter_settings)
            
            if result is None:
                QtWidgets.QMessageBox.warning(self, "No Data", "No suitable data found for clustering.")
                return
            
            data, data_unscaled = result
            
            if data is None or data.empty:
                QtWidgets.QMessageBox.warning(self, "No Data", "No suitable data found for clustering.")
                return
            
            # Clear canvas before clustering
            self.figure.clear()
            self.canvas.draw()
            
            # Perform clustering
            self.clustered_data, self.cluster_labels = self._perform_clustering(data, n_clusters, cluster_method)
            
            # Ensure cluster column is integer type to avoid boolean subtraction issues
            if self.clustered_data is not None and 'cluster' in self.clustered_data.columns:
                self.clustered_data['cluster'] = self.clustered_data['cluster'].astype(int)
            
            # Store original cluster assignments before any merging
            if self.clustered_data is not None and 'cluster' in self.clustered_data.columns:
                self.original_cluster_assignments = self.clustered_data['cluster'].copy()
            
            # Store unscaled data with same structure as clustered_data
            if self.clustered_data is not None:
                # Align unscaled data with clustered_data indices and add cluster column
                self.clustered_data_unscaled = data_unscaled.loc[self.clustered_data.index].copy()
                if 'cluster' in self.clustered_data.columns:
                    self.clustered_data_unscaled['cluster'] = self.clustered_data['cluster'].astype(int).values
                # Copy any other non-feature columns from clustered_data
                for col in self.clustered_data.columns:
                    if col not in self.clustered_data_unscaled.columns and col != 'cluster':
                        if col in ['acquisition_id', 'manual_phenotype']:
                            self.clustered_data_unscaled[col] = self.clustered_data[col].values
            
            # Automatically add cluster column to main feature dataframe
            if self.clustered_data is not None and 'cluster' in self.clustered_data.columns:
                # Ensure cluster column exists in main dataframe
                if 'cluster' not in self.feature_dataframe.columns:
                    self.feature_dataframe['cluster'] = 0  # Initialize with default value
                
                # Update cluster assignments for the clustered cells (ensure integer type)
                self.feature_dataframe.loc[self.clustered_data.index, 'cluster'] = self.clustered_data['cluster'].astype(int).values
            else:
                pass
            
            # Log clustering operation
            logger = get_logger()
            n_clusters_found = len(np.unique(self.cluster_labels)) if self.cluster_labels is not None else n_clusters
            
            # Get scaling method for logging
            scaling_text = self.clustering_scaling_combo.currentText()
            scaling_map = {
                "None (no scaling)": "none",
                "Z-score": "zscore",
                "MAD (Median Absolute Deviation)": "mad"
            }
            scaling_method = scaling_map.get(scaling_text, "zscore")
            
            params = {
                "method": cluster_method,
                "n_clusters": n_clusters,
                "n_clusters_found": int(n_clusters_found),
                "aggregation_method": agg_method,
                "include_morphological": include_morpho,
                "scaling_method": scaling_method,
                "distance_metric": "euclidean",
                "n_cells": int(len(self.clustered_data)) if self.clustered_data is not None else 0
            }
            
            if cluster_method == "leiden":
                if self.resolution_radio.isChecked():
                    params["resolution_parameter"] = self.resolution_spinbox.value()
                else:
                    params["optimization_method"] = "modularity"
                params["seed"] = self.seed_spinbox.value()
                params["n_neighbors"] = self.n_neighbors_spinbox.value()
                params["distance_metric"] = self.leiden_metric_combo.currentText()
            elif cluster_method == "louvain":
                params["seed"] = self.seed_spinbox.value()
                params["n_neighbors"] = self.n_neighbors_spinbox.value()
                params["distance_metric"] = self.leiden_metric_combo.currentText()
            elif cluster_method == "hdbscan":
                params["min_cluster_size"] = self.min_cluster_size_spinbox.value()
                params["min_samples"] = self.min_samples_spinbox.value()
                params["cluster_selection_method"] = self.cluster_selection_combo.currentText()
                params["metric"] = self.metric_combo.currentText()
                params["distance_metric"] = self.metric_combo.currentText()
                params["seed"] = self.seed_spinbox.value()
            else:
                params["linkage_method"] = cluster_method
                # Hierarchical clustering is deterministic, but we log seed for consistency
                params["seed"] = self.seed_spinbox.value()
            
            # Get acquisition IDs from clustered data
            acquisitions = []
            if self.clustered_data is not None and 'acquisition_id' in self.clustered_data.columns:
                acquisitions = list(self.clustered_data['acquisition_id'].unique())
            
            # Get source file name from parent if available
            source_file = None
            if self.parent() is not None and hasattr(self.parent(), 'current_path'):
                import os
                source_file = os.path.basename(self.parent().current_path) if self.parent().current_path else None
            
            logger.log_clustering(
                method=cluster_method,
                parameters=params,
                features_used=selected_columns,
                n_clusters=int(n_clusters_found),
                acquisitions=acquisitions,
                notes=f"Clustered {len(self.clustered_data) if self.clustered_data is not None else 0} cells into {n_clusters_found} clusters",
                source_file=source_file
            )
            
            # Store the scaling method used for clustering and sync heatmap scaling
            clustering_scaling_text = self.clustering_scaling_combo.currentText()
            self.clustering_scaling_method = clustering_scaling_text
            if hasattr(self, 'heatmap_scaling_combo'):
                # Update heatmap scaling to match clustering scaling
                self.heatmap_scaling_combo.blockSignals(True)
                self.heatmap_scaling_combo.setCurrentText(clustering_scaling_text)
                self.heatmap_scaling_combo.blockSignals(False)
            
            # Default to heatmap view after clustering
            try:
                print(f"[DEBUG PLOT] About to create heatmap after clustering")
                print(f"[DEBUG PLOT] clustered_data type: {type(self.clustered_data)}")
                if self.clustered_data is not None:
                    print(f"[DEBUG PLOT] clustered_data shape: {self.clustered_data.shape}")
                    print(f"[DEBUG PLOT] clustered_data columns: {list(self.clustered_data.columns)[:10]}...")
                    if 'cluster' in self.clustered_data.columns:
                        print(f"[DEBUG PLOT] cluster column dtype: {self.clustered_data['cluster'].dtype}")
                        print(f"[DEBUG PLOT] cluster column sample values: {self.clustered_data['cluster'].head().values}")
                        print(f"[DEBUG PLOT] cluster column unique values (first 10): {self.clustered_data['cluster'].unique()[:10]}")
                self._create_heatmap()
                print(f"[DEBUG PLOT] Heatmap created successfully")
            except Exception as e:
                import traceback
                print(f"[DEBUG PLOT] ERROR in _create_heatmap: {str(e)}")
                print(f"[DEBUG PLOT] Error type: {type(e)}")
                print(f"[DEBUG PLOT] Traceback:")
                traceback.print_exc()
                QtWidgets.QMessageBox.critical(self, "Plot Generation Error", 
                    f"Error generating heatmap after clustering:\n{str(e)}\n\nSee console for details.")
                # Still try to show something
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                ax.text(0.5, 0.5, f"Clustering completed but plot generation failed.\nError: {str(e)}", 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                self.canvas.draw()
            
            # Force canvas refresh
            try:
                self.canvas.draw()
            except Exception as e:
                print(f"[DEBUG PLOT] ERROR in canvas.draw(): {str(e)}")
                import traceback
                traceback.print_exc()
            
            # Enable buttons
            self.explore_btn.setEnabled(True)
            self.annotate_btn.setEnabled(True)
            self.merge_clusters_btn.setEnabled(True)
            self.save_plot_btn.setEnabled(True)
            self.save_output_btn.setEnabled(True)
            
            # Update statistical cluster combo if it exists
            if hasattr(self, 'stats_cluster_combo'):
                self._update_stats_cluster_combo()
            # If UMAP was previously run, keep that available
            # Otherwise, selecting UMAP will prompt to run

            # Auto-apply annotations if already loaded for these cluster ids
            if self.cluster_annotation_map:
                self._apply_cluster_annotations()
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Clustering Error", f"Error during clustering: {str(e)}")
    
    def _list_available_feature_columns(self, include_morpho):
        marker_cols = [col for col in self.feature_dataframe.columns 
                      if any(col.endswith(suffix) for suffix in ['_mean', '_median', '_std', '_mad', '_p10', '_p90', '_integrated', '_frac_pos'])]
        morpho_cols = []
        if include_morpho:
            morpho_cols = [col for col in self.feature_dataframe.columns 
                          if col in ['area_um2', 'perimeter_um', 'equivalent_diameter_um', 'eccentricity', 
                                   'solidity', 'extent', 'circularity', 'major_axis_len_um', 'minor_axis_len_um', 
                                   'aspect_ratio', 'bbox_area_um2', 'touches_border', 'touches_edge', 'holes_count']]
        # Note: centroid_x and centroid_y are excluded from clustering as they are spatial coordinates
        return sorted(set(marker_cols + morpho_cols))

    def _apply_filters(self, df, filter_settings):
        """Apply cell filtering based on filter settings.
        
        Args:
            df: DataFrame to filter
            filter_settings: Dictionary with filter settings from FeatureSelectorDialog
        
        Returns:
            Filtered DataFrame
        """
        if filter_settings is None:
            return df
        
        initial_count = len(df)
        filtered_df = df.copy()
        
        # Exclude cells touching edge
        if filter_settings.get('exclude_edge_cells', False):
            before_edge_filter = len(filtered_df)
            if 'touches_edge' in filtered_df.columns:
                # Count cells touching edge before filtering
                edge_cells_count = filtered_df['touches_edge'].astype(bool).sum()
                print(f"[FILTER DEBUG] Excluding cells touching edge: {edge_cells_count} cells will be removed")
                # Use .eq(False) instead of ~ to avoid numpy boolean subtraction issues
                filtered_df = filtered_df[filtered_df['touches_edge'].astype(bool).eq(False)]
                after_edge_filter = len(filtered_df)
                print(f"[FILTER DEBUG] Edge filter: {before_edge_filter} -> {after_edge_filter} cells ({before_edge_filter - after_edge_filter} removed)")
            elif 'touches_border' in filtered_df.columns:
                # Count cells touching border before filtering
                border_cells_count = filtered_df['touches_border'].astype(bool).sum()
                print(f"[FILTER DEBUG] Excluding cells touching border: {border_cells_count} cells will be removed")
                # Fallback to touches_border if touches_edge not available
                filtered_df = filtered_df[filtered_df['touches_border'].astype(bool).eq(False)]
                after_edge_filter = len(filtered_df)
                print(f"[FILTER DEBUG] Border filter: {before_edge_filter} -> {after_edge_filter} cells ({before_edge_filter - after_edge_filter} removed)")
        
        # Filter by area
        if 'area_um2' in filtered_df.columns:
            before_area_filter = len(filtered_df)
            min_area = filter_settings.get('min_area')
            max_area = filter_settings.get('max_area')
            
            if min_area is not None:
                filtered_df = filtered_df[filtered_df['area_um2'] >= min_area]
                print(f"[FILTER DEBUG] Min area filter (>= {min_area}): {before_area_filter} -> {len(filtered_df)} cells")
            if max_area is not None:
                before_max = len(filtered_df)
                filtered_df = filtered_df[filtered_df['area_um2'] <= max_area]
                print(f"[FILTER DEBUG] Max area filter (<= {max_area}): {before_max} -> {len(filtered_df)} cells")
        
        final_count = len(filtered_df)
        if initial_count != final_count:
            print(f"[FILTER DEBUG] Total filtering: {initial_count} -> {final_count} cells ({initial_count - final_count} removed, {100*(initial_count - final_count)/initial_count:.1f}%)")
        else:
            print(f"[FILTER DEBUG] No cells removed by filters (total: {final_count} cells)")
        
        return filtered_df
    
    def _apply_percentile_censoring(self, data, filter_settings):
        """Apply percentile censoring to data to remove outliers.
        
        Per-channel censoring: For each channel/feature, compute the 99th percentile across all cells,
        then set any value above that threshold to the 99th percentile.
        
        Args:
            data: pandas DataFrame with feature data
            filter_settings: Dictionary with filter settings including percentile censoring options
        
        Returns:
            Censored pandas DataFrame
        """
        if filter_settings is None:
            return data
        
        if not filter_settings.get('enable_percentile_censoring', False):
            return data
        
        print(f"[CENSORING DEBUG] Starting percentile censoring on {len(data)} cells, {len(data.columns)} features")
        data_censored = data.copy()
        censor_both_ends = filter_settings.get('censor_both_ends', False)
        
        censored_cols = []
        total_values_censored = 0
        total_values = 0
        
        # Apply censoring column by column (per-channel censoring)
        for col in data_censored.columns:
            col_data = data_censored[col].values
            
            # Skip non-numeric columns (including boolean)
            if col_data.dtype == bool:
                continue
            try:
                # Try to convert to float to check if numeric
                test_data = col_data.astype(np.float64)
            except (ValueError, TypeError):
                # Not numeric, skip this column
                continue
            
            # Convert to float64 to avoid dtype issues
            col_data = col_data.astype(np.float64)
            
            # Skip if all values are NaN or infinite
            finite_mask = np.isfinite(col_data)
            if not np.any(finite_mask):
                continue
            
            finite_data = col_data[finite_mask]
            total_values += len(finite_data)
            
            try:
                if censor_both_ends:
                    # Censor at both 1st and 99th percentiles
                    p1 = np.percentile(finite_data, 1)
                    p99 = np.percentile(finite_data, 99)
                    original_max = np.max(finite_data)
                    original_min = np.min(finite_data)
                    
                    # Count values that will be censored
                    values_above_p99 = np.sum(col_data > p99)
                    values_below_p1 = np.sum(col_data < p1)
                    values_censored = values_above_p99 + values_below_p1
                    
                    data_censored[col] = np.clip(col_data, p1, p99).astype(np.float64)
                    
                    if values_censored > 0:
                        censored_cols.append(col)
                        total_values_censored += values_censored
                        print(f"[CENSORING DEBUG] {col}: 1st={p1:.4f}, 99th={p99:.4f}, "
                              f"range=[{original_min:.4f}, {original_max:.4f}], "
                              f"censored {values_censored} values ({values_above_p99} above p99, {values_below_p1} below p1)")
                else:
                    # Censor at 99th percentile only (cap values at 99th percentile)
                    # This is the standard IMC approach: "censored at the 99th percentile"
                    p99 = np.percentile(finite_data, 99)
                    original_max = np.max(finite_data)
                    
                    # Count values that will be censored (values above p99)
                    values_above_p99 = np.sum(col_data > p99)
                    
                    if values_above_p99 > 0:
                        censored_cols.append(col)
                        total_values_censored += values_above_p99
                        print(f"[CENSORING DEBUG] {col}: 99th percentile={p99:.4f}, "
                              f"max={original_max:.4f}, censored {values_above_p99} values above p99 "
                              f"({100*values_above_p99/len(finite_data):.2f}% of cells)")
                    
                    # Use np.minimum to clip from above (equivalent to clip with None lower bound)
                    data_censored[col] = np.minimum(col_data, p99).astype(np.float64)
            except (ValueError, TypeError) as e:
                # If percentile calculation fails, skip this column
                print(f"[CENSORING WARNING] Failed to apply percentile censoring to column {col}: {e}")
                continue
        
        if censored_cols:
            print(f"[CENSORING DEBUG] Censoring complete: {len(censored_cols)}/{len(data.columns)} features censored, "
                  f"{total_values_censored} values total ({100*total_values_censored/total_values:.2f}% of all values)")
        else:
            print(f"[CENSORING DEBUG] Censoring complete: No values needed censoring (all values within percentiles)")
        
        return data_censored
    
    def _prepare_clustering_data(self, agg_method, include_morpho, selected_columns, scaling_method="zscore", filtered_df=None, filter_settings=None):
        """Prepare data for clustering.
        
        Args:
            agg_method: Aggregation method (not used currently)
            include_morpho: Whether to include morphological features (not used currently)
            selected_columns: List of column names to use for clustering
            scaling_method: Scaling method - 'zscore' or 'mad'
            filtered_df: Optional pre-filtered dataframe (if None, uses self.feature_dataframe)
        """
        # Use filtered dataframe if provided, otherwise use original
        working_df = filtered_df if filtered_df is not None else self.feature_dataframe
        
        feature_cols = list(selected_columns or [])
        
        if not feature_cols:
            return None
        
        # Check if all selected columns exist in the dataframe
        missing_cols = [col for col in feature_cols if col not in working_df.columns]
        if missing_cols:
            return None
        
        # Extract data
        data = working_df[feature_cols].copy()
        # Handle missing/infinite values safely
        data = data.replace([np.inf, -np.inf], np.nan).fillna(data.median(numeric_only=True))
        
        # Apply percentile censoring if enabled (before scaling)
        if filter_settings is not None:
            data = self._apply_percentile_censoring(data, filter_settings)
        
        # Ensure all columns are numeric (float64) to avoid boolean subtraction issues
        for col in data.columns:
            if data[col].dtype == bool:
                # Convert boolean to int then float
                data[col] = data[col].astype(int).astype(np.float64)
            elif not np.issubdtype(data[col].dtype, np.number):
                # Convert non-numeric to float64 if possible
                try:
                    data[col] = pd.to_numeric(data[col], errors='coerce').astype(np.float64)
                except (ValueError, TypeError):
                    # If conversion fails, drop the column
                    data = data.drop(columns=[col])
        
        # Store unscaled data for heatmap display (before scaling)
        data_unscaled = data.copy()
        
        # Apply selected scaling method (Z-score or MAD)
        data = self._apply_scaling(data, scaling_method)
        
        # Drop any residual non-finite rows/cols
        data = data.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any').dropna(axis=1, how='any')
        
        # Guard: require at least 2 rows and 2 columns to compute distances
        if data.shape[0] < 2 or data.shape[1] < 2:
            return None
        
        # Store unscaled data, aligning indices and columns with scaled data after dropna
        # Only keep rows and columns that remain after dropna
        data_unscaled = data_unscaled.loc[data.index, data.columns]
        
        return data, data_unscaled
    
    def _apply_scaling(self, data, scaling_method):
        """Apply scaling to data for UMAP.
        
        Args:
            data: pandas DataFrame to scale
            scaling_method: str, one of 'none', 'zscore', 'mad'
        
        Returns:
            Scaled pandas DataFrame
        """
        print(f"[DEBUG SCALING] _apply_scaling called with method: '{scaling_method}'")
        print(f"[DEBUG SCALING] Input data shape: {data.shape}")
        if scaling_method == 'none':
            print(f"[DEBUG SCALING] No scaling applied, returning copy")
            return data.copy()
        
        data_scaled = data.copy()
        
        if scaling_method == 'zscore':
            print(f"[DEBUG SCALING] Applying Z-score normalization")
            # Z-score normalization: (x - mean) / std
            data_means = data_scaled.mean()
            data_stds = data_scaled.std(ddof=0)
            
            # Handle columns with zero variance or NaN std/mean
            zero_var_cols = (data_stds == 0) | data_stds.isna() | data_means.isna()
            if zero_var_cols.any():
                # Set zero variance/NaN columns to 0 (centered but not scaled)
                data_scaled.loc[:, zero_var_cols] = 0
                non_zero_var_cols = ~zero_var_cols
                if non_zero_var_cols.any():
                    normalized_data = (data_scaled.loc[:, non_zero_var_cols] - data_means[non_zero_var_cols]) / data_stds[non_zero_var_cols]
                    data_scaled.loc[:, non_zero_var_cols] = normalized_data
            else:
                # Normalize all columns
                data_scaled = (data_scaled - data_means) / data_stds
        
        elif scaling_method == 'mad':
            print(f"[DEBUG SCALING] Applying MAD normalization")
            # MAD (Median Absolute Deviation) scaling: (x - median) / MAD
            # MAD = median(|x - median(x)|)
            data_medians = data_scaled.median()
            
            # Calculate MAD for each column
            mad_values = {}
            for col in data_scaled.columns:
                col_data = data_scaled[col].values
                median_val = data_medians[col]
                # Handle NaN median
                if pd.isna(median_val):
                    mad_values[col] = 0.0
                else:
                    mad = np.median(np.abs(col_data - median_val))
                    # Handle NaN MAD
                    if pd.isna(mad):
                        mad_values[col] = 0.0
                    else:
                        mad_values[col] = mad
            
            # Convert to Series for vectorized operations
            mad_series = pd.Series(mad_values)
            
            # Handle columns with zero MAD or NaN (all values are the same or invalid)
            zero_mad_cols = (mad_series == 0) | mad_series.isna() | data_medians.isna()
            if zero_mad_cols.any():
                # Set zero MAD/NaN columns to 0 (centered but not scaled)
                data_scaled.loc[:, zero_mad_cols] = 0
                non_zero_mad_cols = ~zero_mad_cols
                if non_zero_mad_cols.any():
                    # Scale non-zero MAD columns
                    for col in data_scaled.columns[non_zero_mad_cols]:
                        data_scaled[col] = (data_scaled[col] - data_medians[col]) / mad_series[col]
            else:
                # Scale all columns
                for col in data_scaled.columns:
                    data_scaled[col] = (data_scaled[col] - data_medians[col]) / mad_series[col]
        
        # Handle any infinities that might have been introduced
        data_scaled = data_scaled.replace([np.inf, -np.inf], np.nan)
        
        print(f"[DEBUG SCALING] Output data shape: {data_scaled.shape}")
        if data_scaled.shape[0] > 0 and data_scaled.shape[1] > 0:
            print(f"[DEBUG SCALING] Output data sample (first row, first 3 cols): {data_scaled.iloc[0, :3].values}")
            print(f"[DEBUG SCALING] Output data stats - mean: {data_scaled.iloc[:, :3].mean().values}, std: {data_scaled.iloc[:, :3].std().values}")
        
        return data_scaled
    
    def _perform_clustering(self, data, n_clusters, method):
        """Perform clustering using specified method."""
        import time
        print(f"[CLUSTERING DEBUG] _perform_clustering called: method={method}, n_clusters={n_clusters}")
        print(f"[CLUSTERING DEBUG] Input data shape: {data.shape}")
        print(f"[CLUSTERING DEBUG] Input data columns: {list(data.columns)[:10]}...")  # First 10 columns
        
        t0 = time.time()
        clustering_type = self.clustering_type.currentText()
        
        # Get selected columns from data
        selected_columns = list(data.columns)
        print(f"[CLUSTERING DEBUG] Selected columns count: {len(selected_columns)}")
        
        # Get scaling method
        scaling_text = self.clustering_scaling_combo.currentText()
        scaling_map = {
            "None (no scaling)": "none",
            "Z-score": "zscore",
            "MAD (Median Absolute Deviation)": "mad"
        }
        scaling_method = scaling_map.get(scaling_text, "zscore")
        print(f"[CLUSTERING DEBUG] Scaling method: {scaling_method}")
        
        # Get seed
        seed = self.seed_spinbox.value()
        
        # Prepare full dataframe with all columns (for core function)
        # The core function needs the full dataframe but will use only selected columns
        # Use the indices from the filtered/scaled data to get the corresponding rows from the original dataframe
        full_data = self.feature_dataframe.loc[data.index].copy()
        print(f"[CLUSTERING DEBUG] Full dataframe shape: {full_data.shape}")
        print(f"[CLUSTERING DEBUG] Time to prepare data: {time.time() - t0:.3f}s")
        
        if clustering_type == "Leiden":
            # Use core.cluster for Leiden
            resolution = self.resolution_spinbox.value() if self.resolution_radio.isChecked() else 1.0
            n_neighbors = self.n_neighbors_spinbox.value()
            metric = self.leiden_metric_combo.currentText()
            print(f"[CLUSTERING DEBUG] Calling core.cluster with Leiden method")
            print(f"[CLUSTERING DEBUG] Parameters: resolution={resolution}, seed={seed}, scaling={scaling_method}")
            print(f"[CLUSTERING DEBUG] Parameters: n_neighbors={n_neighbors}, metric={metric}")
            print(f"[CLUSTERING DEBUG] Selected columns: {len(selected_columns)} columns")
            
            use_jaccard = self.jaccard_checkbox.isChecked()
            t1 = time.time()
            clustered_df = cluster(
                features_df=full_data,
                method="leiden",
                columns=selected_columns,
                scaling=scaling_method,
                output_path=None,  # Don't save here
                resolution=resolution,
                seed=seed,
                n_neighbors=n_neighbors,
                metric=metric,
                use_jaccard=use_jaccard
            )
            print(f"[CLUSTERING DEBUG] core.cluster returned in {time.time() - t1:.3f}s")
            print(f"[CLUSTERING DEBUG] Result shape: {clustered_df.shape}")
            print(f"[CLUSTERING DEBUG] Unique clusters: {clustered_df['cluster'].nunique()}")
            # Extract cluster labels and ensure integer type to avoid boolean subtraction issues
            cluster_labels = clustered_df['cluster'].astype(int).values
            # Get data subset with clusters
            clustered_data = data.copy()
            clustered_data['cluster'] = cluster_labels.astype(int)
            clustered_data = clustered_data.sort_values('cluster')
            return clustered_data, cluster_labels
            
        elif clustering_type == "Louvain":
            # Use core.cluster for Louvain
            n_neighbors = self.n_neighbors_spinbox.value()
            metric = self.leiden_metric_combo.currentText()
            print(f"[CLUSTERING DEBUG] Calling core.cluster with Louvain method")
            print(f"[CLUSTERING DEBUG] Parameters: seed={seed}, scaling={scaling_method}")
            print(f"[CLUSTERING DEBUG] Parameters: n_neighbors={n_neighbors}, metric={metric}")
            print(f"[CLUSTERING DEBUG] Selected columns: {len(selected_columns)} columns")
            
            use_jaccard = self.jaccard_checkbox.isChecked()
            t1 = time.time()
            clustered_df = cluster(
                features_df=full_data,
                method="louvain",
                columns=selected_columns,
                scaling=scaling_method,
                output_path=None,  # Don't save here
                seed=seed,
                n_neighbors=n_neighbors,
                metric=metric,
                use_jaccard=use_jaccard
            )
            print(f"[CLUSTERING DEBUG] core.cluster returned in {time.time() - t1:.3f}s")
            print(f"[CLUSTERING DEBUG] Result shape: {clustered_df.shape}")
            print(f"[CLUSTERING DEBUG] Unique clusters: {clustered_df['cluster'].nunique()}")
            # Extract cluster labels and ensure integer type to avoid boolean subtraction issues
            cluster_labels = clustered_df['cluster'].astype(int).values
            # Get data subset with clusters
            clustered_data = data.copy()
            clustered_data['cluster'] = cluster_labels.astype(int)
            clustered_data = clustered_data.sort_values('cluster')
            return clustered_data, cluster_labels
            
        elif clustering_type == "HDBSCAN":
            # Use core.cluster for HDBSCAN
            min_cluster_size = self.min_cluster_size_spinbox.value()
            min_samples = self.min_samples_spinbox.value()
            cluster_selection_method = self.cluster_selection_combo.currentText()
            hdbscan_metric = self.metric_combo.currentText()
            print(f"[CLUSTERING DEBUG] Calling core.cluster with HDBSCAN method")
            print(f"[CLUSTERING DEBUG] Parameters: seed={seed}, scaling={scaling_method}")
            print(f"[CLUSTERING DEBUG] Parameters: min_cluster_size={min_cluster_size}, min_samples={min_samples}")
            print(f"[CLUSTERING DEBUG] Parameters: cluster_selection_method={cluster_selection_method}, metric={hdbscan_metric}")
            print(f"[CLUSTERING DEBUG] Selected columns: {len(selected_columns)} columns")
            
            t1 = time.time()
            clustered_df = cluster(
                features_df=full_data,
                method="hdbscan",
                columns=selected_columns,
                scaling=scaling_method,
                output_path=None,  # Don't save here
                seed=seed,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_method=cluster_selection_method,
                hdbscan_metric=hdbscan_metric
            )
            print(f"[CLUSTERING DEBUG] core.cluster returned in {time.time() - t1:.3f}s")
            # Extract cluster labels and ensure integer type to avoid boolean subtraction issues
            cluster_labels = clustered_df['cluster'].astype(int).values
            # Get data subset with clusters
            clustered_data = data.copy()
            clustered_data['cluster'] = cluster_labels.astype(int)
            clustered_data = clustered_data.sort_values('cluster')
            return clustered_data, cluster_labels
            
        elif clustering_type == "K-means":
            # Use core.cluster for K-means
            print(f"[CLUSTERING DEBUG] Calling core.cluster with K-means method")
            print(f"[CLUSTERING DEBUG] Parameters: n_clusters={n_clusters}, seed={seed}, scaling={scaling_method}")
            print(f"[CLUSTERING DEBUG] Selected columns: {len(selected_columns)} columns")
            
            t1 = time.time()
            clustered_df = cluster(
                features_df=full_data,
                method="kmeans",
                columns=selected_columns,
                scaling=scaling_method,
                output_path=None,  # Don't save here
                n_clusters=n_clusters,
                seed=seed,
                n_init=10  # Use 10 initializations (efficient default)
            )
            print(f"[CLUSTERING DEBUG] core.cluster returned in {time.time() - t1:.3f}s")
            print(f"[CLUSTERING DEBUG] Result shape: {clustered_df.shape}")
            print(f"[CLUSTERING DEBUG] Unique clusters: {clustered_df['cluster'].nunique()}")
            # Extract cluster labels and ensure integer type to avoid boolean subtraction issues
            cluster_labels = clustered_df['cluster'].astype(int).values
            # Get data subset with clusters
            clustered_data = data.copy()
            clustered_data['cluster'] = cluster_labels.astype(int)
            clustered_data = clustered_data.sort_values('cluster')
            return clustered_data, cluster_labels
        else:  # Hierarchical
            # Use core.cluster for hierarchical
            linkage_method = method if isinstance(method, str) else "ward"
            print(f"[CLUSTERING DEBUG] Calling core.cluster with Hierarchical method")
            print(f"[CLUSTERING DEBUG] Parameters: n_clusters={n_clusters}, linkage={linkage_method}, seed={seed}, scaling={scaling_method}")
            print(f"[CLUSTERING DEBUG] Selected columns: {len(selected_columns)} columns")
            
            t1 = time.time()
            clustered_df = cluster(
                features_df=full_data,
                method="hierarchical",
                columns=selected_columns,
                scaling=scaling_method,
                output_path=None,  # Don't save here
                n_clusters=n_clusters,
                linkage=linkage_method,
                seed=seed
            )
            print(f"[CLUSTERING DEBUG] core.cluster returned in {time.time() - t1:.3f}s")
            # Extract cluster labels and ensure integer type to avoid boolean subtraction issues
            cluster_labels = clustered_df['cluster'].astype(int).values
            # Get data subset with clusters
            clustered_data = data.copy()
            clustered_data['cluster'] = cluster_labels.astype(int)
            clustered_data = clustered_data.sort_values('cluster')
            return clustered_data, cluster_labels
    
    def _perform_hierarchical_clustering(self, data, n_clusters, method):
        """Perform hierarchical clustering."""
        # Calculate distance matrix
        distances = pdist(data.values, metric='euclidean')
        
        # Perform linkage
        linkage_matrix = linkage(distances, method=method)
        
        # Get cluster labels and ensure integer type to avoid boolean subtraction issues
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust').astype(int)
        
        # Sort data by cluster
        data_with_clusters = data.copy()
        data_with_clusters['cluster'] = cluster_labels.astype(int)
        
        # Sort by cluster
        clustered_data = data_with_clusters.sort_values('cluster')
        
        return clustered_data, cluster_labels
    
    def _perform_kmeans_clustering(self, data, n_clusters):
        """Perform K-means clustering."""
        if not _HAVE_SKLEARN:
            raise ImportError("scikit-learn is required for K-means clustering")
        
        # Get seed from UI
        seed = self.seed_spinbox.value()
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        cluster_labels = kmeans.fit_predict(data.values)
        
        # Convert to 1-based labels and ensure integer type to avoid boolean subtraction issues
        cluster_labels = (cluster_labels + 1).astype(int)
        
        # Sort data by cluster
        data_with_clusters = data.copy()
        data_with_clusters['cluster'] = cluster_labels.astype(int)
        
        # Sort by cluster
        clustered_data = data_with_clusters.sort_values('cluster')
        
        return clustered_data, cluster_labels
    
    def _perform_leiden_clustering(self, data):
        """Perform Leiden clustering using k-NN graph."""
        if not _HAVE_LEIDEN:
            raise ImportError("leidenalg and igraph are required for Leiden clustering")
        if not _HAVE_SKLEARN:
            raise ImportError("scikit-learn is required for k-NN graph construction")
        
        from sklearn.neighbors import NearestNeighbors
        
        # Get n_neighbors, metric, and Jaccard option from UI
        n_neighbors = self.n_neighbors_spinbox.value()
        metric = self.leiden_metric_combo.currentText()
        use_jaccard = self.jaccard_checkbox.isChecked()
        
        # Build k-NN graph
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=metric).fit(data.values)
        distances, indices = nbrs.kneighbors(data.values)
        
        # Create graph from k-NN
        n = data.shape[0]
        edges = []
        weights = []
        
        if use_jaccard:
            # Compute neighbor sets for Jaccard similarity (PhenoGraph-like)
            # Each node's neighbor set includes itself and its k-nearest neighbors
            neighbor_sets = [set(indices[i]) | {i} for i in range(n)]
            
            for i in range(n):
                for j_idx, neighbor_idx in enumerate(indices[i]):
                    if neighbor_idx != i:  # Don't add self-loops
                        edges.append((i, neighbor_idx))
                        # Compute Jaccard similarity: |N(i) ∩ N(j)| / |N(i) ∪ N(j)|
                        intersection = len(neighbor_sets[i] & neighbor_sets[neighbor_idx])
                        union = len(neighbor_sets[i] | neighbor_sets[neighbor_idx])
                        jaccard = intersection / union if union > 0 else 0.0
                        weights.append(jaccard)
        else:
            # Use inverse distance weighting (default)
            for i in range(n):
                for j_idx, neighbor_idx in enumerate(indices[i]):
                    if neighbor_idx != i:  # Don't add self-loops
                        edges.append((i, neighbor_idx))
                        # Convert distance to similarity (inverse, normalized)
                        weight = 1.0 / (1.0 + distances[i][j_idx])
                        weights.append(weight)
        
        # Create igraph (undirected - convert to symmetric)
        edge_set = set()
        symmetric_edges = []
        symmetric_weights = []
        for (i, j), w in zip(edges, weights):
            if (i, j) not in edge_set and (j, i) not in edge_set:
                edge_set.add((i, j))
                symmetric_edges.append((i, j))
                symmetric_weights.append(w)
        
        g = ig.Graph(n)
        g.add_edges(symmetric_edges)
        g.es['weight'] = symmetric_weights
        
        # Get seed from UI
        seed = self.seed_spinbox.value()
        
        # Perform Leiden clustering
        if self.resolution_radio.isChecked():
            # Use resolution parameter
            resolution = self.resolution_spinbox.value()
            partition = leidenalg.find_partition(
                g,
                leidenalg.RBConfigurationVertexPartition,
                weights='weight',
                resolution_parameter=resolution,
                seed=seed,
            )
        else:
            # Use modularity optimization
            partition = leidenalg.find_partition(
                g,
                leidenalg.ModularityVertexPartition,
                weights='weight',
                seed=seed,
            )
        
        # Get cluster labels
        cluster_labels = np.array(partition.membership) + 1  # Start from 1
        
        # Sort data by cluster
        data_with_clusters = data.copy()
        data_with_clusters['cluster'] = cluster_labels
        
        # Sort by cluster
        clustered_data = data_with_clusters.sort_values('cluster')
        
        return clustered_data, cluster_labels
    
    def _perform_louvain_clustering(self, data):
        """Perform Louvain clustering using k-NN graph."""
        if not _HAVE_LEIDEN:
            raise ImportError("leidenalg and igraph are required for Louvain clustering")
        if not _HAVE_SKLEARN:
            raise ImportError("scikit-learn is required for k-NN graph construction")
        
        from sklearn.neighbors import NearestNeighbors
        
        # Get n_neighbors, metric, and Jaccard option from UI
        n_neighbors = self.n_neighbors_spinbox.value()
        metric = self.leiden_metric_combo.currentText()
        use_jaccard = self.jaccard_checkbox.isChecked()
        
        # Build k-NN graph
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=metric).fit(data.values)
        distances, indices = nbrs.kneighbors(data.values)
        
        # Create graph from k-NN
        n = data.shape[0]
        edges = []
        weights = []
        
        if use_jaccard:
            # Compute neighbor sets for Jaccard similarity (PhenoGraph-like)
            # Each node's neighbor set includes itself and its k-nearest neighbors
            neighbor_sets = [set(indices[i]) | {i} for i in range(n)]
            
            for i in range(n):
                for j_idx, neighbor_idx in enumerate(indices[i]):
                    if neighbor_idx != i:  # Don't add self-loops
                        edges.append((i, neighbor_idx))
                        # Compute Jaccard similarity: |N(i) ∩ N(j)| / |N(i) ∪ N(j)|
                        intersection = len(neighbor_sets[i] & neighbor_sets[neighbor_idx])
                        union = len(neighbor_sets[i] | neighbor_sets[neighbor_idx])
                        jaccard = intersection / union if union > 0 else 0.0
                        weights.append(jaccard)
        else:
            # Use inverse distance weighting (default)
            for i in range(n):
                for j_idx, neighbor_idx in enumerate(indices[i]):
                    if neighbor_idx != i:  # Don't add self-loops
                        edges.append((i, neighbor_idx))
                        # Convert distance to similarity (inverse, normalized)
                        weight = 1.0 / (1.0 + distances[i][j_idx])
                        weights.append(weight)
        
        # Create igraph (undirected - convert to symmetric)
        edge_set = set()
        symmetric_edges = []
        symmetric_weights = []
        for (i, j), w in zip(edges, weights):
            if (i, j) not in edge_set and (j, i) not in edge_set:
                edge_set.add((i, j))
                symmetric_edges.append((i, j))
                symmetric_weights.append(w)
        
        g = ig.Graph(n)
        g.add_edges(symmetric_edges)
        g.es['weight'] = symmetric_weights
        
        # Get seed from UI
        seed = self.seed_spinbox.value()
        
        # Perform Louvain clustering (using ModularityVertexPartition)
        # Louvain is essentially modularity optimization
        partition = leidenalg.find_partition(
            g,
            leidenalg.ModularityVertexPartition,
            weights='weight',
            seed=seed,
        )
        
        # Get cluster labels
        cluster_labels = np.array(partition.membership) + 1  # Start from 1
        
        # Sort data by cluster
        data_with_clusters = data.copy()
        data_with_clusters['cluster'] = cluster_labels
        
        # Sort by cluster
        clustered_data = data_with_clusters.sort_values('cluster')
        
        return clustered_data, cluster_labels
    
    def _perform_hdbscan_clustering(self, data):
        """Perform HDBSCAN clustering."""
        if not _HAVE_HDBSCAN:
            raise ImportError("hdbscan is required for HDBSCAN clustering")
        
        # Get parameters from UI
        min_cluster_size = self.min_cluster_size_spinbox.value()
        min_samples = self.min_samples_spinbox.value()
        cluster_selection_method = self.cluster_selection_combo.currentText()
        metric = self.metric_combo.currentText()
        seed = self.seed_spinbox.value()
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Create HDBSCAN clusterer
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=cluster_selection_method,
            metric=metric,
            core_dist_n_jobs=1  # Use single thread for stability
        )
        
        # Fit and get cluster labels
        cluster_labels = clusterer.fit_predict(data.values)
        
        # HDBSCAN uses -1 for noise points, convert to 0-based then 1-based
        # First convert -1 to 0, then add 1 to all labels
        cluster_labels = cluster_labels + 1  # -1 becomes 0, others become 1-based
        
        # Sort data by cluster
        data_with_clusters = data.copy()
        data_with_clusters['cluster'] = cluster_labels
        
        # Sort by cluster (noise points will be at the beginning with cluster 0)
        clustered_data = data_with_clusters.sort_values('cluster')
        
        return clustered_data, cluster_labels
    
    def _create_heatmap(self):
        """Create the heatmap visualization."""
        try:
            print(f"[DEBUG PLOT] _create_heatmap called")
            self.figure.clear()
            
            # Check if clustered_data exists
            if self.clustered_data is None:
                ax = self.figure.add_subplot(111)
                ax.text(0.5, 0.5, "No clustered data available.\nPlease run clustering first.", 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title("Heatmap")
                self.canvas.draw()
                return
            
            print(f"[DEBUG PLOT] clustered_data exists, shape: {self.clustered_data.shape}")
            if 'cluster' in self.clustered_data.columns:
                print(f"[DEBUG PLOT] cluster column dtype: {self.clustered_data['cluster'].dtype}")
                print(f"[DEBUG PLOT] cluster column type info: {type(self.clustered_data['cluster'])}")
            
            # Get selected scaling method
            scaling_text = "None (no scaling)"  # Default
            if hasattr(self, 'heatmap_scaling_combo'):
                scaling_text = self.heatmap_scaling_combo.currentText()
                print(f"[DEBUG PLOT] Heatmap scaling selected from combo: {scaling_text}")
            else:
                print(f"[DEBUG PLOT] WARNING: heatmap_scaling_combo not found, using default: {scaling_text}")
            
            # Map UI text to method string
            scaling_map = {
                "None (no scaling)": "none",
                "Z-score": "zscore",
                "MAD (Median Absolute Deviation)": "mad"
            }
            scaling_method = scaling_map.get(scaling_text, "none")
            print(f"[DEBUG PLOT] Scaling method mapped to: {scaling_method}")
            
            # Use unscaled data if available, otherwise use clustered_data
            base_data = self.clustered_data_unscaled if self.clustered_data_unscaled is not None else self.clustered_data
            print(f"[DEBUG PLOT] base_data type: {type(base_data)}, shape: {base_data.shape if base_data is not None else None}")
            if base_data is not None and 'cluster' in base_data.columns:
                print(f"[DEBUG PLOT] base_data cluster dtype: {base_data['cluster'].dtype}")
                print(f"[DEBUG PLOT] base_data cluster sample: {base_data['cluster'].head(3).values}")
        except Exception as e:
            print(f"[DEBUG PLOT] ERROR at start of _create_heatmap: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
        # Determine source and prepare data ordering and optional grouping
        source = self.heatmap_source_combo.currentText() if hasattr(self, 'heatmap_source_combo') else 'Clusters'
        data_to_plot = base_data.copy()
        
        # Ensure patient annotation column is available
        # Determine which column to use (selected or default priority)
        patient_col = None
        if hasattr(self, 'patient_annotation_column') and self.patient_annotation_column:
            patient_col = self.patient_annotation_column
        else:
            # Default priority order
            for col in ['source_file', 'batch_group', 'source_well']:
                if col in self.feature_dataframe.columns:
                    patient_col = col
                    break
        
        if patient_col and patient_col not in data_to_plot.columns and patient_col in self.feature_dataframe.columns:
            # Merge patient annotation column from feature_dataframe by index
            patient_col_series = self.feature_dataframe[patient_col].reindex(data_to_plot.index)
            data_to_plot[patient_col] = patient_col_series.values
        
        group_col = 'cluster'
        legend_labels = None
        if source == 'Manual Gates' and 'manual_phenotype' in data_to_plot.columns:
            groups = self._get_manual_groups_series()
            if groups is not None:
                data_to_plot = data_to_plot.copy()
                data_to_plot['__group__'] = groups.values
                group_col = '__group__'
                # Apply filter by names if set
                if hasattr(self, 'heatmap_filter_selection') and self.heatmap_filter_selection:
                    data_to_plot = self._apply_heatmap_filter(data_to_plot, group_col)
                # Sort by group label
                data_to_plot = data_to_plot.sort_values(group_col)
                legend_labels = sorted(data_to_plot[group_col].unique())
            else:
                group_col = 'cluster'
        else:
            # Clusters source: optionally filter by selected clusters (by display name or id)
            try:
                print(f"[DEBUG PLOT] Checking heatmap filter selection")
                if hasattr(self, 'heatmap_filter_selection') and self.heatmap_filter_selection:
                    print(f"[DEBUG PLOT] Filter selection exists: {self.heatmap_filter_selection}")
                    # Ensure cluster column is integer before filtering
                    if data_to_plot['cluster'].dtype == bool:
                        print(f"[DEBUG PLOT] WARNING: data_to_plot cluster column is boolean! Converting to int")
                        data_to_plot['cluster'] = data_to_plot['cluster'].astype(int)
                    elif data_to_plot['cluster'].dtype.name.startswith('object'):
                        print(f"[DEBUG PLOT] WARNING: data_to_plot cluster column is object! Converting to int")
                        data_to_plot['cluster'] = pd.to_numeric(data_to_plot['cluster'], errors='coerce').fillna(0).astype(int)
                    else:
                        data_to_plot['cluster'] = data_to_plot['cluster'].astype(int)
                    
                    wanted_ids = set()
                    for cid in sorted(base_data['cluster'].unique()):
                        name = self._get_cluster_display_name(cid)
                        if name in self.heatmap_filter_selection or str(cid) in self.heatmap_filter_selection:
                            wanted_ids.add(int(cid))  # Ensure integer
                    print(f"[DEBUG PLOT] wanted_ids: {wanted_ids}")
                    if wanted_ids:
                        mask = data_to_plot['cluster'].isin(sorted(wanted_ids))
                        print(f"[DEBUG PLOT] mask type: {type(mask)}, dtype: {mask.dtype if hasattr(mask, 'dtype') else 'N/A'}")
                        data_to_plot = data_to_plot[mask]
                        print(f"[DEBUG PLOT] Filtered data_to_plot shape: {data_to_plot.shape}")
                # Ensure cluster is integer before sorting
                if data_to_plot['cluster'].dtype != int and not data_to_plot['cluster'].dtype.name.startswith('int'):
                    print(f"[DEBUG PLOT] Converting cluster to int before sorting, current dtype: {data_to_plot['cluster'].dtype}")
                    data_to_plot['cluster'] = data_to_plot['cluster'].astype(int)
                data_to_plot = data_to_plot.sort_values('cluster')
                print(f"[DEBUG PLOT] Sorted data_to_plot, shape: {data_to_plot.shape}")
            except Exception as e:
                print(f"[DEBUG PLOT] ERROR in cluster filtering/sorting: {str(e)}")
                import traceback
                traceback.print_exc()
                raise

        # Prepare feature columns before scaling
        feature_cols = self._select_feature_columns(data_to_plot)
        print(f"[DEBUG PLOT] Selected {len(feature_cols)} feature columns for scaling")
        
        # Apply selected scaling to feature data
        feature_data = data_to_plot[feature_cols].copy()
        print(f"[DEBUG PLOT] Applying scaling method '{scaling_method}' to feature data")
        print(f"[DEBUG PLOT] Feature data shape before scaling: {feature_data.shape}")
        if feature_data.shape[0] > 0 and feature_data.shape[1] > 0:
            print(f"[DEBUG PLOT] Feature data sample (first row, first 3 cols) before scaling: {feature_data.iloc[0, :3].values}")
        feature_data_scaled = self._apply_scaling(feature_data, scaling_method)
        print(f"[DEBUG PLOT] Feature data shape after scaling: {feature_data_scaled.shape}")
        if feature_data_scaled.shape[0] > 0 and feature_data_scaled.shape[1] > 0:
            print(f"[DEBUG PLOT] Feature data sample (first row, first 3 cols) after scaling: {feature_data_scaled.iloc[0, :3].values}")
        feature_data_scaled = feature_data_scaled.fillna(0)  # Handle any NaN from scaling
        
        # Create data_to_plot with scaled features
        data_to_plot_scaled = data_to_plot.copy()
        for col in feature_cols:
            if col in feature_data_scaled.columns:
                data_to_plot_scaled[col] = feature_data_scaled[col].values

        # Use Scanpy-style heatmap (replaces seaborn)
        self._create_scanpy_style_heatmap(data_to_plot, data_to_plot_scaled, feature_cols, 
                                          group_col, source, scaling_method)
        return
    
    def _create_scanpy_style_heatmap(self, data_to_plot, data_to_plot_scaled, feature_cols, 
                                     group_col, source, scaling_method):
        """Create a Scanpy-style heatmap with improved layout and spacing."""
        self.figure.clear()
        
        # Prepare heatmap data
        heatmap_data = data_to_plot_scaled[feature_cols].values
        
        # Ensure heatmap_data is numeric (convert boolean/object to float)
        print(f"[DEBUG PLOT] heatmap_data dtype before conversion: {heatmap_data.dtype}")
        print(f"[DEBUG PLOT] heatmap_data shape: {heatmap_data.shape}")
        if heatmap_data.dtype == bool:
            print(f"[DEBUG PLOT] WARNING: heatmap_data is boolean! Converting to float")
            heatmap_data = heatmap_data.astype(float)
        elif heatmap_data.dtype.name.startswith('object'):
            print(f"[DEBUG PLOT] WARNING: heatmap_data is object type! Converting to float")
            # Try to convert to numeric, fill NaN with 0
            heatmap_data = pd.DataFrame(heatmap_data).apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)
        elif not np.issubdtype(heatmap_data.dtype, np.number):
            print(f"[DEBUG PLOT] WARNING: heatmap_data is not numeric (dtype: {heatmap_data.dtype})! Converting to float")
            heatmap_data = heatmap_data.astype(float)
        print(f"[DEBUG PLOT] heatmap_data dtype after conversion: {heatmap_data.dtype}")
        print(f"[DEBUG PLOT] heatmap_data min/max: {np.nanmin(heatmap_data)}, {np.nanmax(heatmap_data)}")
        
        # Determine if dendrograms should be applied (but don't show them)
        clustering_type = self.clustering_type.currentText() if hasattr(self, 'clustering_type') else 'Hierarchical'
        is_leiden = clustering_type == "Leiden"
        is_louvain = clustering_type == "Louvain"
        is_hdbscan = clustering_type == "HDBSCAN"
        
        # Apply clustering for reordering but don't show dendrograms
        if is_leiden or is_louvain or is_hdbscan:
            row_cluster = False
            col_cluster = False
            linkage_method = None
        else:
            row_cluster = True
            dendro_mode = self.dendro_mode.currentText() if hasattr(self, 'dendro_mode') else "Rows and columns"
            col_cluster = (dendro_mode == "Rows and columns")
            linkage_method = self.hierarchical_method.currentText() if hasattr(self, 'hierarchical_method') else "ward"
        
        # Apply hierarchical clustering if needed (for reordering only)
        from scipy.cluster.hierarchy import linkage, leaves_list
        
        row_indices = np.arange(len(feature_cols))
        col_indices = np.arange(heatmap_data.shape[0])
        
        if row_cluster:
            # Cluster features (rows)
            row_linkage = linkage(heatmap_data.T, method=linkage_method, metric='euclidean')
            row_indices = leaves_list(row_linkage)
        
        if col_cluster:
            # Cluster cells (columns)
            col_linkage = linkage(heatmap_data, method=linkage_method, metric='euclidean')
            col_indices = leaves_list(col_linkage)
        
        # Reorder data based on clustering
        heatmap_data_reordered = heatmap_data[np.ix_(col_indices, row_indices)]
        feature_cols_reordered = [feature_cols[i] for i in row_indices]
        
        # Reorder annotation bar to match column clustering
        group_values = data_to_plot[group_col].values
        group_values_reordered = group_values[col_indices]
        
        # Create group color mapping
        unique_groups = sorted(data_to_plot[group_col].unique())
        # Use a vivid color palette that can handle many clusters
        cluster_colors_raw = _get_vivid_colors(len(unique_groups))
        # Convert to RGB (remove alpha channel and ensure proper format)
        cluster_color_map = {}
        for i, gid in enumerate(unique_groups):
            color = cluster_colors_raw[i]
            # Convert to RGB tuple (remove alpha if present)
            if len(color) == 4:
                rgb = tuple(color[:3])
            elif len(color) == 3:
                rgb = tuple(color)
            else:
                rgb = (color[0], color[1], color[2])
            cluster_color_map[gid] = rgb
        
        # Create reordered cell colors for annotation bar (match column clustering)
        # Convert to proper RGB array for imshow
        cell_colors_rgb = [cluster_color_map[val] for val in group_values_reordered]
        
        # Check if patient annotation is enabled
        # Determine which column to use for patient annotation
        # Priority: selected column, or source_file, batch_group, source_well
        patient_col = None
        if hasattr(self, 'patient_annotation_column') and self.patient_annotation_column:
            patient_col = self.patient_annotation_column
        else:
            # Default priority order
            for col in ['source_file', 'batch_group', 'source_well']:
                if col in data_to_plot.columns:
                    patient_col = col
                    break
        
        show_patient_annotation = (self.patient_annotation_enabled and
                                   patient_col is not None and patient_col in data_to_plot.columns)
        
        # Prepare patient annotation data if enabled
        patient_values_reordered = None
        patient_color_map = {}
        if show_patient_annotation:
            # Get patient annotation values from selected column and reorder to match column clustering
            patient_values = data_to_plot[patient_col].values
            patient_values_reordered = patient_values[col_indices]
            
            # Get unique patient values
            unique_patients = sorted([f for f in data_to_plot[patient_col].unique() if pd.notna(f)])
            
            # Create color mapping for patients (use different palette than clusters)
            patient_colors_raw = _get_patient_colors(len(unique_patients))
            for i, patient_file in enumerate(unique_patients):
                color = patient_colors_raw[i]
                if len(color) == 4:
                    rgb = tuple(color[:3])
                elif len(color) == 3:
                    rgb = tuple(color)
                else:
                    rgb = (color[0], color[1], color[2])
                patient_color_map[patient_file] = rgb
            
            # Create reordered patient colors for annotation bar
            patient_colors_rgb = [patient_color_map.get(val, (0.8, 0.8, 0.8)) for val in patient_values_reordered]
        
        # Create layout - adjust based on whether patient annotation is shown
        if show_patient_annotation:
            # Colorbar, patient annotation, cell annotation, heatmap
            gs = self.figure.add_gridspec(
                nrows=4, ncols=2, 
                height_ratios=[0.02, 0.04, 0.06, 0.88],  # Colorbar, patient annotation, cell annotation, heatmap
                width_ratios=[0.88, 0.12],  # Heatmap area, legend
                hspace=0.03, wspace=0.02,  # Space between elements
                left=0.15, right=0.98, top=0.88, bottom=0.12  # More top margin for colorbar label and ticks
            )
            heatmap_row = 3
            cell_annotation_row = 2
            patient_annotation_row = 1
        else:
            # Colorbar, annotation bar, heatmap
            gs = self.figure.add_gridspec(
                nrows=3, ncols=2, 
                height_ratios=[0.02, 0.06, 0.92],  # Colorbar, annotation bar, heatmap
                width_ratios=[0.88, 0.12],  # Heatmap area, legend
                hspace=0.03, wspace=0.02,  # Space between elements
                left=0.15, right=0.98, top=0.88, bottom=0.12  # More top margin for colorbar label and ticks
            )
            heatmap_row = 2
            cell_annotation_row = 1
            patient_annotation_row = None
        
        # Patient annotation bar (if enabled)
        if show_patient_annotation:
            ax_patient_annotation = self.figure.add_subplot(gs[patient_annotation_row, 0])
            patient_annotation_array = np.array(patient_colors_rgb).reshape(1, -1, 3)
            ax_patient_annotation.imshow(patient_annotation_array, aspect='auto', interpolation='nearest', 
                                         extent=[0, len(patient_colors_rgb), 0, 1])
            ax_patient_annotation.set_xlim(0, len(patient_colors_rgb))
            ax_patient_annotation.set_xticks([])
            ax_patient_annotation.set_yticks([])
            ax_patient_annotation.spines['top'].set_visible(False)
            ax_patient_annotation.spines['right'].set_visible(False)
            ax_patient_annotation.spines['bottom'].set_visible(False)
            ax_patient_annotation.spines['left'].set_visible(False)
        
        # Cell annotation bar below colorbar (or below patient annotation if enabled)
        ax_annotation = self.figure.add_subplot(gs[cell_annotation_row, 0])
        # Create a simple colored bar - convert to proper RGB array
        annotation_array = np.array(cell_colors_rgb).reshape(1, -1, 3)
        ax_annotation.imshow(annotation_array, aspect='auto', interpolation='nearest', extent=[0, len(cell_colors_rgb), 0, 1])
        ax_annotation.set_xlim(0, len(cell_colors_rgb))
        ax_annotation.set_xticks([])
        ax_annotation.set_yticks([])
        ax_annotation.spines['top'].set_visible(False)
        ax_annotation.spines['right'].set_visible(False)
        ax_annotation.spines['bottom'].set_visible(False)
        ax_annotation.spines['left'].set_visible(False)
        
        # Main heatmap
        ax_heatmap = self.figure.add_subplot(gs[heatmap_row, 0])
        
        # Get colormap
        colormap_name = self._get_colormap_name()
        
        # Create heatmap with reordered data
        # Ensure heatmap_data_reordered is numeric before percentile calculation
        print(f"[DEBUG PLOT] heatmap_data_reordered dtype before percentile: {heatmap_data_reordered.dtype}")
        if heatmap_data_reordered.dtype == bool:
            print(f"[DEBUG PLOT] WARNING: heatmap_data_reordered is boolean! Converting to float")
            heatmap_data_reordered = heatmap_data_reordered.astype(float)
        elif not np.issubdtype(heatmap_data_reordered.dtype, np.number):
            print(f"[DEBUG PLOT] WARNING: heatmap_data_reordered is not numeric (dtype: {heatmap_data_reordered.dtype})! Converting to float")
            heatmap_data_reordered = heatmap_data_reordered.astype(float)
        
        # Remove any NaN or Inf values before percentile calculation
        heatmap_data_reordered_clean = np.nan_to_num(heatmap_data_reordered, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"[DEBUG PLOT] Calculating percentiles on clean data, shape: {heatmap_data_reordered_clean.shape}, dtype: {heatmap_data_reordered_clean.dtype}")
        
        try:
            vmin_val = np.percentile(heatmap_data_reordered_clean, 2)
            vmax_val = np.percentile(heatmap_data_reordered_clean, 98)
            print(f"[DEBUG PLOT] Percentiles calculated: vmin={vmin_val}, vmax={vmax_val}")
        except Exception as e:
            print(f"[DEBUG PLOT] ERROR calculating percentiles: {str(e)}")
            print(f"[DEBUG PLOT] Data stats: min={np.nanmin(heatmap_data_reordered_clean)}, max={np.nanmax(heatmap_data_reordered_clean)}")
            # Fallback to min/max if percentile fails
            vmin_val = np.nanmin(heatmap_data_reordered_clean)
            vmax_val = np.nanmax(heatmap_data_reordered_clean)
        
        im = ax_heatmap.imshow(
            heatmap_data_reordered.T, 
            aspect='auto', 
            cmap=colormap_name, 
            interpolation='nearest',
            vmin=vmin_val,
            vmax=vmax_val
        )
        
        # Colorbar at top (horizontal)
        ax_cbar = self.figure.add_subplot(gs[0, 0])
        cbar = self.figure.colorbar(im, cax=ax_cbar, orientation='horizontal')
        # Move ticks and label to top of colorbar
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.tick_params(labelsize=8, top=True, labeltop=True, bottom=False, labelbottom=False)
        cbar.set_label('Normalized Feature Value', fontsize=8, labelpad=10)
        
        # Set feature labels on y-axis with proper spacing
        n_features = len(feature_cols_reordered)
        ax_heatmap.set_yticks(np.arange(n_features))
        # Use custom feature labels if available
        feature_labels_display = [self._get_feature_display_name(f) for f in feature_cols_reordered]
        ax_heatmap.set_yticklabels(feature_labels_display, fontsize=self.feature_tick_fontsize, rotation=0)
        ax_heatmap.set_ylabel('Features', fontsize=10, fontweight='bold')
        
        # Ensure all labels are visible
        ax_heatmap.tick_params(axis='y', which='major', labelsize=self.feature_tick_fontsize, pad=2)
        for label in ax_heatmap.get_yticklabels():
            label.set_visible(True)
        
        # Remove x-axis labels (cells)
        ax_heatmap.set_xticks([])
        ax_heatmap.set_xlabel('Cells', fontsize=10, fontweight='bold')
        
        # Set proper limits
        ax_heatmap.set_xlim(-0.5, heatmap_data_reordered.shape[0] - 0.5)
        ax_heatmap.set_ylim(-0.5, n_features - 0.5)
        
        # Remove spines for cleaner look
        ax_heatmap.spines['top'].set_visible(False)
        ax_heatmap.spines['right'].set_visible(False)
        ax_heatmap.spines['bottom'].set_visible(False)
        ax_heatmap.spines['left'].set_visible(False)
        
        # Legend on the right - adjust layout based on patient annotation
        # Check if legend should be shown
        show_legend = self.show_legend_checkbox.isChecked() if hasattr(self, 'show_legend_checkbox') else True
        
        if show_patient_annotation:
            # Create nested gridspec for two legends (patient on top, clusters below)
            legend_gs = gs[heatmap_row, 1].subgridspec(2, 1, hspace=0.0, height_ratios=[0.4, 0.6])
            
            # Patient legend on top
            ax_patient_legend = self.figure.add_subplot(legend_gs[0])
            ax_patient_legend.axis('off')
            if show_legend:
                patient_legend_elements = []
                for patient_file in sorted(patient_color_map.keys()):
                    color = patient_color_map[patient_file]
                    # Use custom patient label (helper function handles custom labels and defaults)
                    label = self._get_patient_display_name(patient_file)
                    patient_legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, label=label, edgecolor='black', linewidth=0.5))
                
                if patient_legend_elements:
                    ax_patient_legend.legend(handles=patient_legend_elements, loc='upper left', frameon=True, fontsize=8, 
                                            title=self.patient_legend_label, title_fontsize=9)
            
            # Cluster legend below
            ax_cluster_legend = self.figure.add_subplot(legend_gs[1])
            ax_cluster_legend.axis('off')
            if show_legend:
                legend_elements = []
                if source == 'Manual Gates':
                    for key in sorted(cluster_color_map.keys()):
                        color = cluster_color_map[key]
                        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, label=str(key), edgecolor='black', linewidth=0.5))
                else:
                    for key in sorted(cluster_color_map.keys()):
                        color = cluster_color_map[key]
                        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, label=self._get_cluster_display_name(key), edgecolor='black', linewidth=0.5))
                
                if legend_elements:
                    # Use multiple columns if there are more than 10 clusters
                    n_clusters = len(legend_elements)
                    ncol = max(1, (n_clusters + 9) // 10) if n_clusters > 10 else 1
                    ax_cluster_legend.legend(handles=legend_elements, loc='upper left', frameon=True, fontsize=8, 
                                            title='Clusters' if source == 'Clusters' else 'Groups', title_fontsize=9, ncol=ncol)
        else:
            # Single legend area
            ax_legend = self.figure.add_subplot(gs[heatmap_row, 1])
            ax_legend.axis('off')  # Hide axes for legend area
            
            if show_legend:
                # Add legend for groups/clusters - vertical layout
                # Use the same color mapping as annotation bar (sorted for consistency)
                legend_elements = []
                if source == 'Manual Gates':
                    for key in sorted(cluster_color_map.keys()):
                        color = cluster_color_map[key]
                        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, label=str(key), edgecolor='black', linewidth=0.5))
                else:
                    for key in sorted(cluster_color_map.keys()):
                        color = cluster_color_map[key]
                        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, label=self._get_cluster_display_name(key), edgecolor='black', linewidth=0.5))
                # Place legend vertically to the right of colorbar
                # Use multiple columns if there are more than 10 clusters
                n_clusters = len(legend_elements)
                ncol = max(1, (n_clusters + 9) // 10) if n_clusters > 10 else 1
                ax_legend.legend(handles=legend_elements, loc='center left', frameon=True, fontsize=8, 
                                 title='Clusters' if source == 'Clusters' else 'Groups', title_fontsize=9, ncol=ncol)
        
        self.canvas.draw()
    
    def _create_seaborn_heatmap(self):
        """Create heatmap using seaborn clustermap with color bars."""
        try:
            # Check if clustered_data exists
            if self.clustered_data is None:
                self._create_matplotlib_heatmap()
                return
            # Get selected scaling method
            scaling_text = "None (no scaling)"  # Default
            if hasattr(self, 'heatmap_scaling_combo'):
                scaling_text = self.heatmap_scaling_combo.currentText()
            
            # Map UI text to method string
            scaling_map = {
                "Z-score": "zscore",
                "MAD (Median Absolute Deviation)": "mad",
                "None (no scaling)": "none"
            }
            scaling_method = scaling_map.get(scaling_text, "none")
            
            # Use unscaled data if available, otherwise use clustered_data
            base_data = self.clustered_data_unscaled if self.clustered_data_unscaled is not None else self.clustered_data
            
            # Prepare data considering source/filter
            source = self.heatmap_source_combo.currentText() if hasattr(self, 'heatmap_source_combo') else 'Clusters'
            data_to_plot = base_data.copy()
            group_col = 'cluster'
            if source == 'Manual Gates' and 'manual_phenotype' in data_to_plot.columns:
                groups = self._get_manual_groups_series()
                if groups is not None:
                    data_to_plot['__group__'] = groups.values
                    group_col = '__group__'
                    if hasattr(self, 'heatmap_filter_selection') and self.heatmap_filter_selection:
                        data_to_plot = self._apply_heatmap_filter(data_to_plot, group_col)
                    data_to_plot = data_to_plot.sort_values(group_col)
            else:
                if hasattr(self, 'heatmap_filter_selection') and self.heatmap_filter_selection:
                    wanted_ids = set()
                    for cid in sorted(base_data['cluster'].unique()):
                        name = self._get_cluster_display_name(cid)
                        if name in self.heatmap_filter_selection or str(cid) in self.heatmap_filter_selection:
                            wanted_ids.add(cid)
                    if wanted_ids:
                        data_to_plot = data_to_plot[data_to_plot['cluster'].isin(sorted(wanted_ids))]
                data_to_plot = data_to_plot.sort_values('cluster')

            feature_cols = self._select_feature_columns(data_to_plot)
            
            # Apply selected scaling to feature data
            feature_data = data_to_plot[feature_cols].copy()
            feature_data_scaled = self._apply_scaling(feature_data, scaling_method)
            feature_data_scaled = feature_data_scaled.fillna(0)  # Handle any NaN from scaling
            
            heatmap_data = feature_data_scaled
            
            # Store original feature order for y-tick labels
            original_feature_order = list(feature_cols)
            
            # Create group color mapping
            unique_groups = sorted(data_to_plot[group_col].unique())
            # Use vivid colors instead of Set3
            cluster_colors_raw = _get_vivid_colors(len(unique_groups))
            # Convert to RGB tuples for seaborn (remove alpha channel)
            cluster_colors = [tuple(c[:3]) for c in cluster_colors_raw]
            cluster_color_map = {gid: cluster_colors[i] for i, gid in enumerate(unique_groups)}
            
            # Create color series for color bar
            cluster_colors_series = data_to_plot[group_col].map(cluster_color_map)
            
            # Determine clustering settings based on method
            clustering_type = self.clustering_type.currentText()
            is_leiden = clustering_type == "Leiden"
            is_louvain = clustering_type == "Louvain"
            is_hdbscan = clustering_type == "HDBSCAN"
            
            if is_leiden or is_louvain or is_hdbscan:
                # For Leiden, Louvain, and HDBSCAN clustering, disable dendrograms
                row_cluster = False
                col_cluster = False
                linkage_method = None
            else:
                # For hierarchical clustering, use dendrograms
                row_cluster = True
                col_cluster = (self.dendro_mode.currentText() == "Rows and columns")
                linkage_method = self.hierarchical_method.currentText()
            
            # Get canvas size to determine appropriate figure size
            canvas_width = self.canvas.width()
            canvas_height = self.canvas.height()
            # Convert pixels to inches (assuming 100 DPI)
            fig_width = max(8, canvas_width / 100)
            fig_height = max(6, canvas_height / 100)
            
            # Create clustermap with appropriate parameters
            colormap_name = self._get_colormap_name()
            g = sns.clustermap(
                heatmap_data.T,  # Transpose for features as rows, cells as columns
                cmap=colormap_name,
                row_cluster=row_cluster,
                col_cluster=col_cluster,
                method=linkage_method,
                metric='euclidean',
                cbar_kws={'label': 'Normalized Feature Value'},
                figsize=(fig_width, fig_height),  # Dynamic figure size based on canvas
                col_colors=cluster_colors_series  # This creates the color bar
            )
            
            # Labels – let seaborn manage tick order after clustering; just style
            g.ax_heatmap.set_xlabel('Cells')
            g.ax_heatmap.set_ylabel('Features')
            
            # Force all row tick labels to show (features)
            # Use reordered features if row clustering is enabled, otherwise use original order
            if row_cluster:
                # When row clustering is enabled, use the reordered feature names
                feature_labels = g.ax_heatmap.get_yticklabels()
                feature_names = [label.get_text() for label in feature_labels]
            else:
                # When row clustering is disabled, use original feature order
                feature_names = original_feature_order
            
            # Ensure all feature labels are displayed - disable automatic tick limiting
            g.ax_heatmap.set_yticks(range(len(feature_names)))
            # Use custom feature labels if available
            feature_labels_display = [self._get_feature_display_name(f) for f in feature_names]
            g.ax_heatmap.set_yticklabels(feature_labels_display, fontsize=self.feature_tick_fontsize, minor=False)
            # Prevent matplotlib from automatically hiding overlapping labels
            g.ax_heatmap.tick_params(axis='y', which='major', labelsize=self.feature_tick_fontsize)
            # Ensure y-axis limits show all features
            g.ax_heatmap.set_ylim(-0.5, len(feature_names) - 0.5)
            # Force all labels to be visible
            for label in g.ax_heatmap.get_yticklabels():
                label.set_visible(True)
            
            # Remove column tick labels (cells)
            g.ax_heatmap.set_xticks([])
            g.ax_heatmap.set_xticklabels([])
            
            # Add legend
            self._add_cluster_legend(g, cluster_color_map, source=source)
            
            # Replace the figure with the seaborn figure
            old_figure = self.figure
            self.figure = g.fig
            self.canvas.figure = self.figure
            
            # Use tight layout to maximize plot area
            # Avoid tight_layout on clustermap to prevent warnings
            
            # Close the old figure to free memory
            plt.close(old_figure)
            
            # Force canvas update
            self.canvas.draw()
            
        except Exception as e:
            # Fall back to matplotlib implementation
            self._create_matplotlib_heatmap()
    
    def _create_matplotlib_heatmap(self):
        """Fallback heatmap using matplotlib (original implementation)."""
        # Check if clustered_data exists
        if self.clustered_data is None:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "No clustered data available.\nPlease run clustering first.", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title("Heatmap")
            self.canvas.draw()
            return
        
        # Get selected scaling method
        scaling_text = "None (no scaling)"  # Default
        if hasattr(self, 'heatmap_scaling_combo'):
            scaling_text = self.heatmap_scaling_combo.currentText()
        
        # Map UI text to method string
        scaling_map = {
            "Z-score": "zscore",
            "MAD (Median Absolute Deviation)": "mad",
            "None (no scaling)": "none"
        }
        scaling_method = scaling_map.get(scaling_text, "none")
        
        # Use unscaled data if available, otherwise use clustered_data
        base_data = self.clustered_data_unscaled if self.clustered_data_unscaled is not None else self.clustered_data
        
        # Filter out dropped clusters (cluster 0)
        try:
            print(f"[DEBUG PLOT] About to filter base_data by cluster != 0")
            print(f"[DEBUG PLOT] base_data cluster dtype before filter: {base_data['cluster'].dtype}")
            print(f"[DEBUG PLOT] base_data cluster type: {type(base_data['cluster'])}")
            # Ensure cluster column is integer before comparison
            if base_data['cluster'].dtype == bool:
                print(f"[DEBUG PLOT] WARNING: cluster column is boolean! Converting to int")
                base_data['cluster'] = base_data['cluster'].astype(int)
            elif base_data['cluster'].dtype.name.startswith('object'):
                print(f"[DEBUG PLOT] WARNING: cluster column is object type! Converting to int")
                base_data['cluster'] = pd.to_numeric(base_data['cluster'], errors='coerce').fillna(0).astype(int)
            else:
                base_data['cluster'] = base_data['cluster'].astype(int)
            print(f"[DEBUG PLOT] base_data cluster dtype after conversion: {base_data['cluster'].dtype}")
            # Now do the comparison
            mask = base_data['cluster'] != 0
            print(f"[DEBUG PLOT] mask type: {type(mask)}, dtype: {mask.dtype if hasattr(mask, 'dtype') else 'N/A'}")
            base_data = base_data[mask].copy()
            print(f"[DEBUG PLOT] Filtered base_data shape: {base_data.shape}")
        except Exception as e:
            print(f"[DEBUG PLOT] ERROR filtering base_data by cluster: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
        # Determine source and prepare data ordering and optional grouping
        source = self.heatmap_source_combo.currentText() if hasattr(self, 'heatmap_source_combo') else 'Clusters'
        data_to_plot = base_data.copy()
        group_col = 'cluster'
        if source == 'Manual Gates' and 'manual_phenotype' in data_to_plot.columns:
            groups = self._get_manual_groups_series()
            if groups is not None:
                data_to_plot = data_to_plot.copy()
                data_to_plot['__group__'] = groups.values
                group_col = '__group__'
                # Apply filter by names if set
                if hasattr(self, 'heatmap_filter_selection') and self.heatmap_filter_selection:
                    data_to_plot = self._apply_heatmap_filter(data_to_plot, group_col)
                # Sort by group label
                data_to_plot = data_to_plot.sort_values(group_col)
            else:
                group_col = 'cluster'
        else:
            # Clusters source: optionally filter by selected clusters (by display name or id)
            if hasattr(self, 'heatmap_filter_selection') and self.heatmap_filter_selection:
                wanted_ids = set()
                for cid in sorted(base_data['cluster'].unique()):
                    name = self._get_cluster_display_name(cid)
                    if name in self.heatmap_filter_selection or str(cid) in self.heatmap_filter_selection:
                        wanted_ids.add(cid)
                if wanted_ids:
                    data_to_plot = data_to_plot[data_to_plot['cluster'].isin(sorted(wanted_ids))]
            data_to_plot = data_to_plot.sort_values('cluster')

        # Create subplots - simplified layout without cluster size bar
        gs = self.figure.add_gridspec(1, 1, hspace=0.1, wspace=0.1)
        
        # Main heatmap - use full figure area
        ax_heatmap = self.figure.add_subplot(gs[0])
        
        # Prepare feature columns before scaling
        feature_cols = self._select_feature_columns(data_to_plot)
        
        # Apply selected scaling to feature data
        feature_data = data_to_plot[feature_cols].copy()
        feature_data_scaled = self._apply_scaling(feature_data, scaling_method)
        feature_data_scaled = feature_data_scaled.fillna(0)  # Handle any NaN from scaling
        
        heatmap_data = feature_data_scaled.values

        # No dendrograms - just show the heatmap data as-is
        
        # Create heatmap with user-selected colormap
        colormap_name = self._get_colormap_name()
        im = ax_heatmap.imshow(heatmap_data.T, aspect='auto', cmap=colormap_name, interpolation='nearest')
        
        # Set labels and ticks
        ax_heatmap.set_xlabel('Cells')
        ax_heatmap.set_ylabel('Features')
        # Ensure all feature labels are displayed - disable automatic tick limiting
        ax_heatmap.set_yticks(np.arange(len(feature_cols)))
        # Use custom feature labels if available
        feature_labels_display = [self._get_feature_display_name(f) for f in feature_cols]
        ax_heatmap.set_yticklabels(feature_labels_display, fontsize=self.feature_tick_fontsize, rotation=0)
        # Prevent matplotlib from automatically hiding overlapping labels
        ax_heatmap.tick_params(axis='y', which='major', labelsize=self.feature_tick_fontsize)
        # Force all labels to be visible
        for label in ax_heatmap.get_yticklabels():
            label.set_visible(True)
        
        # Remove x-axis tick labels (cluster identity shown via color bar instead)
        ax_heatmap.set_xticks([])
        
        
        # Add group color bars along x-axis
        unique_groups = sorted(data_to_plot[group_col].unique())
        cluster_colors = _get_vivid_colors(len(unique_groups))
        cluster_color_map = {gid: cluster_colors[i] for i, gid in enumerate(unique_groups)}
        
        # Create color bar for each cell
        cell_colors = [cluster_color_map[val] for val in data_to_plot[group_col]]
        
        # Add color bar below the heatmap
        for i, color in enumerate(cell_colors):
            ax_heatmap.axvline(x=i, ymin=-0.05, ymax=0, color=color, linewidth=1, solid_capstyle='butt')
        
        # Adjust y-axis to make room for color bar
        ax_heatmap.set_ylim(-0.5, len(feature_cols) - 0.5)
        
        # Colorbar
        cbar = self.figure.colorbar(im, ax=ax_heatmap, shrink=0.8)
        cbar.set_label('Normalized Feature Value')
        
        # Row dendrogram (top-left)
        # No row dendrogram - just the heatmap
        
        # Cluster size bar removed - using color bars for cluster identity instead

        # Add legend for groups/clusters - horizontal at the top
        legend_elements = []
        if source == 'Manual Gates':
            for key, color in cluster_color_map.items():
                legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, label=str(key)))
        else:
            for key, color in cluster_color_map.items():
                legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, label=self._get_cluster_display_name(key)))
        # Place legend horizontally at the top of the figure
        self.figure.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.02),
                          ncol=min(len(legend_elements), 10), frameon=True, fontsize=8)
        
        # Adjust layout to account for legend at top - add extra top padding
        self.figure.tight_layout(pad=1.0, rect=[0, 0, 1, 0.95])
        
        self.canvas.draw()
    
    def _add_cluster_legend(self, g, cluster_color_map, source='Clusters'):
        """Add legend to seaborn clustermap using cluster names or manual group labels."""
        legend_elements = []
        for key, color in cluster_color_map.items():
            if source == 'Manual Gates':
                label = str(key)
            else:
                label = self._get_cluster_display_name(key)
            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, label=label))
        # Place legend horizontally at the top of the figure
        g.fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.02), 
                     ncol=min(len(legend_elements), 10), frameon=True, fontsize=8)
    
    def _run_umap(self):
        """Run UMAP dimensionality reduction analysis."""
        try:
            if not _HAVE_UMAP:
                QtWidgets.QMessageBox.warning(self, "UMAP Not Available", 
                    "UMAP is not installed. Please install umap-learn to use this feature.")
                return
            
            # Get feature selection from user
            available_cols = self._list_available_feature_columns(True)  # Include morphometric features
            from openimc.ui.dialogs.feature_selector_dialog import FeatureSelectorDialog
            selector = FeatureSelectorDialog(available_cols, self)
            # Pre-populate filter settings if available
            if self.filter_settings is not None:
                selector.set_filter_settings(self.filter_settings)
            if selector.exec_() != QtWidgets.QDialog.Accepted:
                return
            selected_columns = selector.get_selected_columns()
            
            if not selected_columns:
                QtWidgets.QMessageBox.warning(self, "No Features", "Please select at least one feature for UMAP analysis.")
                return
            
            # Get filter settings (use stored settings if available, otherwise get from dialog)
            filter_settings = self.filter_settings
            if filter_settings is None:
                filter_settings = selector.get_filter_settings()
            
            # Apply filters to feature dataframe
            filtered_df = self._apply_filters(self.feature_dataframe.copy(), filter_settings)
            if filtered_df.empty:
                QtWidgets.QMessageBox.warning(self, "No Data", "No cells remain after applying filters.")
                return
            
            # Prepare data for UMAP, align with clustered order if available
            if self.clustered_data is not None:
                # Use intersection of filtered data and clustered data indices
                ordered_index = self.clustered_data.index.intersection(filtered_df.index)
                data = filtered_df.loc[ordered_index, selected_columns].copy()
            else:
                data = filtered_df[selected_columns].copy()
            
            # Handle missing values and infinite values
            data = data.replace([np.inf, -np.inf], np.nan)
            data = data.fillna(data.median())
            
            if data.empty or data.shape[0] < 2:
                QtWidgets.QMessageBox.warning(self, "No Data", "No suitable data found for UMAP analysis.")
                return
            
            # Apply percentile censoring if enabled (before scaling)
            data = self._apply_percentile_censoring(data, filter_settings)
            
            # Ensure all columns are numeric (float64) to avoid boolean subtraction issues
            for col in data.columns:
                if data[col].dtype == bool:
                    data[col] = data[col].astype(int).astype(np.float64)
                elif not np.issubdtype(data[col].dtype, np.number):
                    try:
                        data[col] = pd.to_numeric(data[col], errors='coerce').astype(np.float64)
                    except (ValueError, TypeError):
                        data = data.drop(columns=[col])
            
            # Clear canvas before UMAP
            self.figure.clear()
            self.canvas.draw()
            
            # Allow user to choose scaling method
            scaling_options = ["None (no scaling)", "Z-score", "MAD (Median Absolute Deviation)"]
            # Default to clustering scaling method if available
            default_index = 0
            if (hasattr(self, 'clustering_scaling_method') and 
                self.clustering_scaling_method is not None and 
                self.clustering_scaling_method in scaling_options):
                default_index = scaling_options.index(self.clustering_scaling_method)
            
            scaling_method, ok = QtWidgets.QInputDialog.getItem(
                self,
                "UMAP Feature Scaling",
                "Select scaling method for features:",
                scaling_options,
                current=default_index,  # Default to clustering scaling method
                editable=False
            )
            if not ok:
                return
            
            # Map selection to method string
            scaling_map = {
                "None (no scaling)": "none",
                "Z-score": "zscore",
                "MAD (Median Absolute Deviation)": "mad"
            }
            selected_scaling = scaling_map[scaling_method]
            
            # Apply scaling
            data_scaled = self._apply_scaling(data, selected_scaling)
            
            # Handle any NaN values that might have been introduced
            data_scaled = data_scaled.fillna(0)
            
            if data_scaled.empty or data_scaled.shape[0] < 2:
                QtWidgets.QMessageBox.warning(self, "No Data", "No suitable data found for UMAP analysis after scaling.")
                return
            
            # Allow user to choose n_neighbors
            default_n = 15
            max_n = max(2, min(default_n, data_scaled.shape[0] - 1))
            # Get seed from UI to show in dialog
            seed = self.seed_spinbox.value()
            # Simple input dialog for n_neighbors with bounds, including seed info
            n_neighbors, ok = QtWidgets.QInputDialog.getInt(
                self,
                "UMAP n_neighbors",
                f"Set n_neighbors (2–{max(2, data_scaled.shape[0]-1)}):\n\nNote: Using random seed {seed} from clustering options above.",
                value=max_n,
                min=2,
                max=max(2, data_scaled.shape[0]-1)
            )
            if not ok:
                return
            # Perform UMAP with seed from UI
            reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=int(n_neighbors), min_dist=0.1)
            self.umap_embedding = reducer.fit_transform(data_scaled.values)
            # Persist for coloring
            self.umap_index = data.index.to_list()
            self.umap_selected_columns = list(selected_columns)
            self.umap_raw_data = data.copy()
            
            # Create UMAP plot
            self._create_umap_plot()
            
            # Force canvas refresh
            self.canvas.draw()
            
            # Populate color-by options
            self._populate_color_by_options()
            # Enable save button since a plot is shown
            self.save_plot_btn.setEnabled(True)
            self.save_output_btn.setEnabled(True)
            
            # Update statistical cluster combo if it exists
            if hasattr(self, 'stats_cluster_combo'):
                self._update_stats_cluster_combo()
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "UMAP Error", f"Error during UMAP analysis: {str(e)}")
    
    def _remake_umap(self):
        """Remake UMAP with new parameters. This allows users to easily regenerate UMAP."""
        # Simply call _run_umap which will prompt for new parameters
        self._run_umap()
    
    def _run_tsne(self):
        """Run t-SNE dimensionality reduction."""
        if not _HAVE_TSNE:
            QtWidgets.QMessageBox.warning(self, "t-SNE Not Available", "scikit-learn is required for t-SNE.")
            return
        
        if self.feature_dataframe is None or self.feature_dataframe.empty:
            QtWidgets.QMessageBox.warning(self, "No Data", "No feature data available for t-SNE.")
            return
        
        try:
            # Get feature selection from user (same as UMAP)
            available_cols = self._list_available_feature_columns(True)  # Include morphometric features
            from openimc.ui.dialogs.feature_selector_dialog import FeatureSelectorDialog
            selector = FeatureSelectorDialog(available_cols, self)
            # Pre-populate filter settings if available
            if self.filter_settings is not None:
                selector.set_filter_settings(self.filter_settings)
            if selector.exec_() != QtWidgets.QDialog.Accepted:
                return
            selected_columns = selector.get_selected_columns()
            
            if not selected_columns:
                QtWidgets.QMessageBox.warning(self, "No Features", "Please select at least one feature for t-SNE analysis.")
                return
            
            # Get filter settings (use stored settings if available, otherwise get from dialog)
            filter_settings = self.filter_settings
            if filter_settings is None:
                filter_settings = selector.get_filter_settings()
            
            # Apply filters to feature dataframe
            filtered_df = self._apply_filters(self.feature_dataframe.copy(), filter_settings)
            if filtered_df.empty:
                QtWidgets.QMessageBox.warning(self, "No Data", "No cells remain after applying filters.")
                return
            
            # Prepare data for t-SNE, align with clustered order if available
            if self.clustered_data is not None:
                # Use intersection of filtered data and clustered data indices
                ordered_index = self.clustered_data.index.intersection(filtered_df.index)
                data = filtered_df.loc[ordered_index, selected_columns].copy()
            else:
                data = filtered_df[selected_columns].copy()
            
            # Handle missing values and infinite values
            data = data.replace([np.inf, -np.inf], np.nan)
            data = data.fillna(data.median())
            
            if data.empty or data.shape[0] < 2:
                QtWidgets.QMessageBox.warning(self, "No Data", "No suitable data found for t-SNE analysis.")
                return
            
            # Apply percentile censoring if enabled (before scaling)
            data = self._apply_percentile_censoring(data, filter_settings)
            
            # Ensure all columns are numeric (float64) to avoid boolean subtraction issues
            for col in data.columns:
                if data[col].dtype == bool:
                    data[col] = data[col].astype(int).astype(np.float64)
                elif not np.issubdtype(data[col].dtype, np.number):
                    try:
                        data[col] = pd.to_numeric(data[col], errors='coerce').astype(np.float64)
                    except (ValueError, TypeError):
                        data = data.drop(columns=[col])
            
            # Clear canvas before t-SNE
            self.figure.clear()
            self.canvas.draw()
            
            # Allow user to choose scaling method
            scaling_options = ["None (no scaling)", "Z-score", "MAD (Median Absolute Deviation)"]
            # Default to clustering scaling method if available
            default_index = 0
            if (hasattr(self, 'clustering_scaling_method') and 
                self.clustering_scaling_method is not None and 
                self.clustering_scaling_method in scaling_options):
                default_index = scaling_options.index(self.clustering_scaling_method)
            
            scaling_method, ok = QtWidgets.QInputDialog.getItem(
                self,
                "t-SNE Feature Scaling",
                "Select scaling method for features:",
                scaling_options,
                current=default_index,  # Default to clustering scaling method
                editable=False
            )
            if not ok:
                return
            
            # Map selection to method string
            scaling_map = {
                "None (no scaling)": "none",
                "Z-score": "zscore",
                "MAD (Median Absolute Deviation)": "mad"
            }
            selected_scaling = scaling_map[scaling_method]
            
            # Apply scaling
            data_scaled = self._apply_scaling(data, selected_scaling)
            
            # Handle any NaN values that might have been introduced
            data_scaled = data_scaled.fillna(0)
            
            if data_scaled.empty or data_scaled.shape[0] < 2:
                QtWidgets.QMessageBox.warning(self, "No Data", "No suitable data found for t-SNE analysis after scaling.")
                return
            
            # Get seed from UI
            seed = self.seed_spinbox.value()
            
            # Simple input dialog for perplexity
            max_perplexity = min(30, data_scaled.shape[0] - 1)
            perplexity, ok = QtWidgets.QInputDialog.getInt(
                self,
                "t-SNE Perplexity",
                f"Set perplexity (5–{max_perplexity}):\n\nNote: Using random seed {seed} from clustering options above.",
                value=min(30, max_perplexity),
                min=5,
                max=max_perplexity
            )
            if not ok:
                return
            
            # Perform t-SNE
            self.tsne_embedding = TSNE(n_components=2, perplexity=perplexity, random_state=seed, max_iter=1000).fit_transform(data_scaled.values)
            
            # Persist for coloring
            self.tsne_index = data.index.to_list()
            self.tsne_selected_columns = list(selected_columns)
            self.tsne_raw_data = data.copy()
            
            # Create plot
            self._create_tsne_plot()
            
            # Populate color-by options
            self._populate_color_by_options()
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "t-SNE Error", f"Error during t-SNE analysis: {str(e)}")
    
    def _plot_tsne_single(self, ax, color_by, point_size, point_alpha, title=None):
        """Plot a single t-SNE subplot with specified coloring."""
        if color_by == 'Cluster' and self.clustered_data is not None and 'cluster' in self.clustered_data.columns:
            # Align cluster labels to t-SNE order
            if hasattr(self, 'tsne_index') and self.tsne_index is not None:
                cluster_labels_series = self.clustered_data['cluster']
                cluster_labels = cluster_labels_series.reindex(self.tsne_index).values
            else:
                cluster_labels = self.clustered_data['cluster'].values
            
            # Filter out dropped clusters (cluster 0)
            valid_mask = cluster_labels != 0
            cluster_labels = cluster_labels[valid_mask]
            tsne_embedding_filtered = self.tsne_embedding[valid_mask]
            
            unique_clusters = sorted(np.unique(cluster_labels))
            colors = _get_vivid_colors(len(unique_clusters))
            cluster_color_map = {cluster_id: colors[i] for i, cluster_id in enumerate(unique_clusters)}
            handles = []
            labels = []
            for cluster_id in unique_clusters:
                mask = cluster_labels == cluster_id
                sc = ax.scatter(tsne_embedding_filtered[mask, 0], tsne_embedding_filtered[mask, 1],
                                c=[cluster_color_map[cluster_id]],
                                alpha=point_alpha, s=point_size, edgecolors='none')
                # Create custom legend handle with fixed size (18)
                color = cluster_color_map[cluster_id]
                if len(color) == 4:
                    rgb = tuple(color[:3])
                elif len(color) == 3:
                    rgb = tuple(color)
                else:
                    rgb = (color[0], color[1], color[2])
                handle = Line2D([0], [0], marker='o', color='none', markerfacecolor=rgb,
                               markeredgecolor='none', markersize=6, alpha=point_alpha)
                handles.append(handle)
                labels.append(self._get_cluster_display_name(cluster_id))
            # Use multiple columns if there are more than 10 clusters
            n_clusters = len(handles)
            ncol = max(1, (n_clusters + 9) // 10) if n_clusters > 10 else 1
            ax.legend(handles, labels, loc='best', frameon=True, fontsize=8, ncol=ncol)
        elif color_by == 'Source File' and 'source_file' in self.feature_dataframe.columns:
            # Filter out dropped clusters (cluster 0) for source file coloring
            if hasattr(self, 'tsne_index') and self.tsne_index is not None:
                cluster_labels_series = self.clustered_data['cluster']
                cluster_labels = cluster_labels_series.reindex(self.tsne_index).values
            else:
                cluster_labels = self.clustered_data['cluster'].values
            valid_mask = cluster_labels != 0
            tsne_embedding_filtered = self.tsne_embedding[valid_mask]
            
            # Color by source file to visualize batch effects (using custom patient labels)
            if hasattr(self, 'tsne_index') and self.tsne_index is not None:
                source_file_series = self.feature_dataframe.loc[self.tsne_index, 'source_file']
                source_files = source_file_series.values[valid_mask]
            else:
                source_files = self.feature_dataframe['source_file'].values[valid_mask]
            unique_files = sorted([f for f in np.unique(source_files) if pd.notna(f)])
            if len(unique_files) > 0:
                # Use patient colors palette (different from cluster colors)
                patient_colors_raw = _get_patient_colors(len(unique_files))
                file_color_map = {file_name: patient_colors_raw[i] for i, file_name in enumerate(unique_files)}
                handles = []
                labels = []
                for file_name in unique_files:
                    mask = source_files == file_name
                    if np.any(mask):  # Only add if there are points for this file
                        sc = ax.scatter(tsne_embedding_filtered[mask, 0], tsne_embedding_filtered[mask, 1],
                                        c=[file_color_map[file_name]],
                                        alpha=point_alpha, s=point_size, edgecolors='none')
                        # Create custom legend handle with fixed size
                        color = file_color_map[file_name]
                        if len(color) == 4:
                            rgb = tuple(color[:3])
                        elif len(color) == 3:
                            rgb = tuple(color)
                        else:
                            rgb = (color[0], color[1], color[2])
                        handle = Line2D([0], [0], marker='o', color='none', markerfacecolor=rgb,
                                       markeredgecolor='none', markersize=6, alpha=point_alpha)
                        handles.append(handle)
                        # Use custom patient label if available
                        labels.append(self._get_patient_display_name(file_name))
                # Place legend inside axes to avoid clipping - ensure it's visible
                if handles and labels:
                    legend = ax.legend(handles, labels, loc='best', frameon=True, fontsize=8, title=self.patient_legend_label)
                    legend.set_visible(True)
            else:
                # Fallback if no source files
                ax.scatter(tsne_embedding_filtered[:, 0], tsne_embedding_filtered[:, 1], c='blue', alpha=point_alpha, s=point_size, edgecolors='none')
        elif color_by == 'Phenotype' and self.clustered_data is not None and 'cluster_phenotype' in self.clustered_data.columns:
            # Filter out dropped clusters (cluster 0) for phenotype coloring
            if hasattr(self, 'tsne_index') and self.tsne_index is not None:
                cluster_labels_series = self.clustered_data['cluster']
                cluster_labels = cluster_labels_series.reindex(self.tsne_index).values
            else:
                cluster_labels = self.clustered_data['cluster'].values
            valid_mask = cluster_labels != 0
            tsne_embedding_filtered = self.tsne_embedding[valid_mask]
            
            # Color by cluster phenotype
            if hasattr(self, 'tsne_index') and self.tsne_index is not None:
                phenotype_series = self.clustered_data['cluster_phenotype'].reindex(self.tsne_index)
                phenotypes = phenotype_series.fillna('Unassigned').values[valid_mask]
            else:
                phenotypes = self.clustered_data['cluster_phenotype'].fillna('Unassigned').values[valid_mask]
            unique_phenotypes = sorted([p for p in np.unique(phenotypes) if pd.notna(p)])
            colors = _get_vivid_colors(len(unique_phenotypes))
            phenotype_color_map = {p: colors[i] for i, p in enumerate(unique_phenotypes)}
            handles = []
            labels = []
            for phenotype in unique_phenotypes:
                mask = phenotypes == phenotype
                sc = ax.scatter(tsne_embedding_filtered[mask, 0], tsne_embedding_filtered[mask, 1],
                                c=[phenotype_color_map[phenotype]],
                                alpha=point_alpha, s=point_size, edgecolors='none')
                # Create custom legend handle with fixed size
                color = phenotype_color_map[phenotype]
                if len(color) == 4:
                    rgb = tuple(color[:3])
                elif len(color) == 3:
                    rgb = tuple(color)
                else:
                    rgb = (color[0], color[1], color[2])
                handle = Line2D([0], [0], marker='o', color='none', markerfacecolor=rgb,
                               markeredgecolor='none', markersize=6, alpha=point_alpha)
                handles.append(handle)
                labels.append(str(phenotype))
            ax.legend(handles, labels, loc='best', frameon=True, fontsize=8, title='Phenotype')
        elif color_by == 'Manual Phenotype' and 'manual_phenotype' in self.feature_dataframe.columns:
            # Filter out dropped clusters (cluster 0) for manual phenotype coloring
            if hasattr(self, 'tsne_index') and self.tsne_index is not None:
                cluster_labels_series = self.clustered_data['cluster']
                cluster_labels = cluster_labels_series.reindex(self.tsne_index).values
            else:
                cluster_labels = self.clustered_data['cluster'].values
            valid_mask = cluster_labels != 0
            tsne_embedding_filtered = self.tsne_embedding[valid_mask]
            
            # Color by manual phenotype
            if hasattr(self, 'tsne_index') and self.tsne_index is not None:
                manual_phenotype_series = self.feature_dataframe.loc[self.tsne_index, 'manual_phenotype']
                manual_phenotypes = manual_phenotype_series.fillna('Unassigned').values[valid_mask]
            else:
                manual_phenotypes = self.feature_dataframe['manual_phenotype'].fillna('Unassigned').values[valid_mask]
            unique_phenotypes = sorted([p for p in np.unique(manual_phenotypes) if pd.notna(p)])
            colors = _get_vivid_colors(len(unique_phenotypes))
            phenotype_color_map = {p: colors[i] for i, p in enumerate(unique_phenotypes)}
            handles = []
            labels = []
            for phenotype in unique_phenotypes:
                mask = manual_phenotypes == phenotype
                sc = ax.scatter(tsne_embedding_filtered[mask, 0], tsne_embedding_filtered[mask, 1],
                                c=[phenotype_color_map[phenotype]],
                                alpha=point_alpha, s=point_size, edgecolors='none')
                # Create custom legend handle with fixed size
                color = phenotype_color_map[phenotype]
                if len(color) == 4:
                    rgb = tuple(color[:3])
                elif len(color) == 3:
                    rgb = tuple(color)
                else:
                    rgb = (color[0], color[1], color[2])
                handle = Line2D([0], [0], marker='o', color='none', markerfacecolor=rgb,
                               markeredgecolor='none', markersize=6, alpha=point_alpha)
                handles.append(handle)
                labels.append(str(phenotype))
            ax.legend(handles, labels, loc='best', frameon=True, fontsize=8, title='Manual Phenotype')
        elif color_by in self.feature_dataframe.columns:
            # Handle metadata columns or other dataframe columns as categorical
            metadata_cols = self._get_metadata_columns(self.feature_dataframe)
            if color_by in metadata_cols or color_by == 'batch_group':
                # Filter out dropped clusters (cluster 0) for metadata column coloring
                if hasattr(self, 'tsne_index') and self.tsne_index is not None:
                    cluster_labels_series = self.clustered_data['cluster']
                    cluster_labels = cluster_labels_series.reindex(self.tsne_index).values
                else:
                    cluster_labels = self.clustered_data['cluster'].values
                valid_mask = cluster_labels != 0
                tsne_embedding_filtered = self.tsne_embedding[valid_mask]
                
                # Color by metadata column (categorical)
                if hasattr(self, 'tsne_index') and self.tsne_index is not None:
                    col_series = self.feature_dataframe.loc[self.tsne_index, color_by]
                    col_values = col_series.fillna('Unknown').values[valid_mask]
                else:
                    col_values = self.feature_dataframe[color_by].fillna('Unknown').values[valid_mask]
                unique_values = sorted([v for v in np.unique(col_values) if pd.notna(v)])
                if len(unique_values) > 0:
                    colors = _get_vivid_colors(len(unique_values))
                    value_color_map = {v: colors[i] for i, v in enumerate(unique_values)}
                    handles = []
                    labels = []
                    for value in unique_values:
                        mask = col_values == value
                        if np.any(mask):
                            sc = ax.scatter(tsne_embedding_filtered[mask, 0], tsne_embedding_filtered[mask, 1],
                                            c=[value_color_map[value]],
                                            alpha=point_alpha, s=point_size, edgecolors='none')
                            color = value_color_map[value]
                            if len(color) == 4:
                                rgb = tuple(color[:3])
                            elif len(color) == 3:
                                rgb = tuple(color)
                            else:
                                rgb = (color[0], color[1], color[2])
                            handle = Line2D([0], [0], marker='o', color='none', markerfacecolor=rgb,
                                           markeredgecolor='none', markersize=6, alpha=point_alpha)
                            handles.append(handle)
                            labels.append(str(value))
                    if handles and labels:
                        ax.legend(handles, labels, loc='best', frameon=True, fontsize=8, title=color_by)
        elif hasattr(self, 'tsne_raw_data') and color_by in getattr(self, 'tsne_selected_columns', []):
            # Filter out dropped clusters (cluster 0) for feature coloring
            if hasattr(self, 'tsne_index') and self.tsne_index is not None:
                cluster_labels_series = self.clustered_data['cluster']
                cluster_labels = cluster_labels_series.reindex(self.tsne_index).values
            else:
                cluster_labels = self.clustered_data['cluster'].values
            valid_mask = cluster_labels != 0
            tsne_embedding_filtered = self.tsne_embedding[valid_mask]
            
            # Continuous coloring by selected feature (aligned to t-SNE order)
            vals = self.tsne_raw_data[color_by].values[valid_mask]
            sc = ax.scatter(tsne_embedding_filtered[:, 0], tsne_embedding_filtered[:, 1], c=vals,
                            cmap='viridis', alpha=point_alpha, s=point_size, edgecolors='none')
            cbar = self.figure.colorbar(sc, ax=ax)
            cbar.set_label(color_by)
        else:
            # Filter out dropped clusters (cluster 0) for fallback
            if hasattr(self, 'tsne_index') and self.tsne_index is not None:
                cluster_labels_series = self.clustered_data['cluster']
                cluster_labels = cluster_labels_series.reindex(self.tsne_index).values
            else:
                cluster_labels = self.clustered_data['cluster'].values
            valid_mask = cluster_labels != 0
            tsne_embedding_filtered = self.tsne_embedding[valid_mask]
            
            # Fallback single color
            ax.scatter(tsne_embedding_filtered[:, 0], tsne_embedding_filtered[:, 1], c='blue', alpha=point_alpha, s=point_size, edgecolors='none')
        
        ax.set_xlabel('t-SNE 1', fontsize=10)
        ax.set_ylabel('t-SNE 2', fontsize=10)
        if title is None:
            title = f't-SNE: {color_by}'
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)

    def _create_tsne_plot(self):
        """Create t-SNE scatter plot(s) with faceted plotting support."""
        if self.tsne_embedding is None:
            return
        
        self.figure.clear()
        
        # Get selected color-by options (use stored data which contains actual column names)
        if hasattr(self, 'color_by_listwidget'):
            selected_items = []
            for item in self.color_by_listwidget.selectedItems():
                # Use stored data (actual column name) if available, otherwise use text
                actual_name = item.data(QtCore.Qt.UserRole)
                if actual_name:
                    selected_items.append(actual_name)
                else:
                    selected_items.append(item.text())
        else:
            # Fallback to combo box if list widget doesn't exist
            selected_items = [self.color_by_combo.currentText()] if hasattr(self, 'color_by_combo') else ['Cluster']
        
        # Ensure at least one selection
        if not selected_items:
            selected_items = ['Cluster']
        
        # Limit to max 3 plots (3 columns in single row)
        selected_items = selected_items[:3]
        
        # Get point size and alpha from controls
        point_size = self.point_size_spinbox.value() if hasattr(self, 'point_size_spinbox') else 18
        point_alpha = self.point_alpha_spinbox.value() if hasattr(self, 'point_alpha_spinbox') else 0.8
        
        n_plots = len(selected_items)
        
        if n_plots == 1:
            # Single plot - use full figure
            ax = self.figure.add_subplot(111)
            self._plot_tsne_single(ax, selected_items[0], point_size, point_alpha)
            self.figure.tight_layout(pad=1.0)
        else:
            # Multiple plots - create subplots in a single row with max 3 columns
            n_cols = n_plots
            n_rows = 1
            
            for idx, color_by in enumerate(selected_items):
                ax = self.figure.add_subplot(n_rows, n_cols, idx + 1)
                self._plot_tsne_single(ax, color_by, point_size, point_alpha)
            
            self.figure.tight_layout(pad=1.0)
        
        self.canvas.draw()
    
    def _show_heatmap(self):
        """Switch back to heatmap view."""
        if self.clustered_data is not None:
            # Sync heatmap scaling to clustering scaling if available
            if (hasattr(self, 'clustering_scaling_method') and 
                self.clustering_scaling_method is not None and 
                hasattr(self, 'heatmap_scaling_combo')):
                # Only update if the combo doesn't already match (to avoid unnecessary updates)
                if self.heatmap_scaling_combo.currentText() != self.clustering_scaling_method:
                    self.heatmap_scaling_combo.setCurrentText(self.clustering_scaling_method)
            self._create_heatmap()
        else:
            QtWidgets.QMessageBox.warning(self, "No Clustering", "Please run clustering first to view the heatmap.")

    def _on_view_changed(self, view: str):
        """Switch visualization based on selected view and manage dependencies."""
        # Clear canvas before switching views
        self.figure.clear()
        self.canvas.draw()
        
        self._update_viz_controls_visibility()
        if view == 'Heatmap':
            self._show_heatmap()
        elif view == 'UMAP':
            if getattr(self, 'umap_embedding', None) is None:
                self._run_umap()
            else:
                self._create_umap_plot()
        elif view == 't-SNE':
            if getattr(self, 'tsne_embedding', None) is None:
                self._run_tsne()
            else:
                self._create_tsne_plot()
        elif view == 'Stacked Bars':
            self._show_stacked_bars()
        elif view == 'Differential Expression':
            self._show_differential_expression()
        elif view == 'Boxplot/Violin Plot':
            self._show_boxplot_violin()
        
        # Force canvas refresh after view change
        self.canvas.draw()
        
        # Enable save if there is content
        self.save_plot_btn.setEnabled(True)
        self.save_output_btn.setEnabled(True)

    def _update_viz_controls_visibility(self):
        """Show/hide controls depending on selected view."""
        view = self.view_combo.currentText() if hasattr(self, 'view_combo') else 'Heatmap'
        # Color-by visible only for UMAP and t-SNE
        for i in range(self.color_by_combo.count()):
            pass
        if hasattr(self, 'color_by_label'):
            self.color_by_label.setVisible(view in ['UMAP', 't-SNE'])
        if hasattr(self, 'color_by_search'):
            self.color_by_search.setVisible(view in ['UMAP', 't-SNE'])
        if hasattr(self, 'color_by_listwidget'):
            self.color_by_listwidget.setVisible(view in ['UMAP', 't-SNE'])
        self.color_by_combo.setVisible(False)  # Keep hidden for backward compatibility
        # Point size and alpha visible only for UMAP and t-SNE
        if hasattr(self, 'point_size_label'):
            self.point_size_label.setVisible(view in ['UMAP', 't-SNE'])
            if hasattr(self, 'point_size_spinbox'):
                self.point_size_spinbox.setVisible(view in ['UMAP', 't-SNE'])
        if hasattr(self, 'point_alpha_label'):
            self.point_alpha_label.setVisible(view in ['UMAP', 't-SNE'])
            if hasattr(self, 'point_alpha_spinbox'):
                self.point_alpha_spinbox.setVisible(view in ['UMAP', 't-SNE'])
        # Show legend checkbox visible for all views that have legends
        if hasattr(self, 'show_legend_checkbox'):
            self.show_legend_checkbox.setVisible(view in ['UMAP', 't-SNE', 'Stacked Bars', 'Heatmap'])
        # Remake UMAP button visible only for UMAP
        if hasattr(self, 'remake_umap_btn'):
            self.remake_umap_btn.setVisible(view == 'UMAP')
        # Group-by visible only for Stacked Bars
        if hasattr(self, 'group_by_label'):
            self.group_by_label.setVisible(view == 'Stacked Bars')
        self.group_by_combo.setVisible(view == 'Stacked Bars')
        # View type, normalization, and filter controls visible only for Stacked Bars
        if hasattr(self, 'stacked_bars_view_type_label'):
            self.stacked_bars_view_type_label.setVisible(view == 'Stacked Bars')
        if hasattr(self, 'stacked_bars_view_type_combo'):
            self.stacked_bars_view_type_combo.setVisible(view == 'Stacked Bars')
        if hasattr(self, 'stacked_bars_filter_btn'):
            self.stacked_bars_filter_btn.setVisible(view == 'Stacked Bars')
        # Colormap visible only for Heatmap and Differential Expression; hidden for UMAP and Stacked Bars
        if hasattr(self, 'colormap_label'):
            self.colormap_label.setVisible(view in ['Heatmap', 'Differential Expression'])
        self.colormap_combo.setVisible(view in ['Heatmap', 'Differential Expression'])
        # Top N visible only for Differential Expression
        if hasattr(self, 'top_n_label'):
            self.top_n_label.setVisible(view == 'Differential Expression')
        self.top_n_spinbox.setVisible(view == 'Differential Expression')
        # Feature labels button visible for Differential Expression, Stacked Bars, and Boxplot/Violin Plot
        if hasattr(self, 'feature_labels_btn'):
            self.feature_labels_btn.setVisible(view in ['Differential Expression', 'Stacked Bars', 'Boxplot/Violin Plot'])
        # Marker selection and plot type visible only for Boxplot/Violin Plot
        is_boxplot_violin = view == 'Boxplot/Violin Plot'
        if hasattr(self, 'marker_select_label'):
            self.marker_select_label.setVisible(is_boxplot_violin)
        if hasattr(self, 'marker_select_btn'):
            self.marker_select_btn.setVisible(is_boxplot_violin)
        if hasattr(self, 'plot_type_label'):
            self.plot_type_label.setVisible(is_boxplot_violin)
        if hasattr(self, 'plot_type_combo'):
            self.plot_type_combo.setVisible(is_boxplot_violin)
        if hasattr(self, 'stats_test_checkbox'):
            self.stats_test_checkbox.setVisible(is_boxplot_violin)
        if hasattr(self, 'stats_mode_label'):
            self.stats_mode_label.setVisible(is_boxplot_violin)
        if hasattr(self, 'stats_mode_combo'):
            self.stats_mode_combo.setVisible(is_boxplot_violin)
        if hasattr(self, 'stats_cluster_label'):
            self.stats_cluster_label.setVisible(is_boxplot_violin)
        if hasattr(self, 'stats_cluster_combo'):
            self.stats_cluster_combo.setVisible(is_boxplot_violin)
        if hasattr(self, 'stats_export_btn'):
            self.stats_export_btn.setVisible(is_boxplot_violin)
        # Heatmap-only controls
        is_heatmap = view == 'Heatmap'
        if hasattr(self, 'heatmap_source_combo'):
            self.heatmap_source_combo.setVisible(is_heatmap)
        if hasattr(self, 'heatmap_source_label'):
            self.heatmap_source_label.setVisible(is_heatmap)
        if hasattr(self, 'heatmap_filter_btn'):
            self.heatmap_filter_btn.setVisible(is_heatmap)
        if hasattr(self, 'heatmap_scaling_combo'):
            self.heatmap_scaling_combo.setVisible(is_heatmap)
        if hasattr(self, 'heatmap_scaling_label'):
            self.heatmap_scaling_label.setVisible(is_heatmap)
        if hasattr(self, 'patient_annotation_checkbox'):
            self.patient_annotation_checkbox.setVisible(is_heatmap)
        # Configure plot button visible only for Heatmap
        if hasattr(self, 'configure_plot_btn'):
            self.configure_plot_btn.setVisible(is_heatmap)
    
    def _on_colormap_changed(self, _text: str):
        """Handle colormap selection change."""
        # Refresh the current view if it uses colormaps
        view = self.view_combo.currentText() if hasattr(self, 'view_combo') else 'Heatmap'
        if view in ['Heatmap', 'Differential Expression']:
            if view == 'Heatmap':
                self._show_heatmap()
            elif view == 'Differential Expression':
                self._show_differential_expression()
    
    def _on_group_by_changed(self, _text: str):
        """Handle group by selection change for stacked bars."""
        view = self.view_combo.currentText() if hasattr(self, 'view_combo') else 'Heatmap'
        if view == 'Stacked Bars':
            self._show_stacked_bars()
    
    def _on_stacked_bars_view_type_changed(self, _text: str):
        """Handle view type change for stacked bars (Fraction vs Total enumeration)."""
        view = self.view_combo.currentText() if hasattr(self, 'view_combo') else 'Heatmap'
        if view == 'Stacked Bars':
            self._show_stacked_bars()
    
    def _open_stacked_bars_filter_dialog(self):
        """Open a dialog to choose which clusters to show in stacked bars."""
        if self.clustered_data is None:
            QtWidgets.QMessageBox.warning(self, "No Clustering", "Run clustering first to filter stacked bars.")
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Select Clusters to Display")
        v = QtWidgets.QVBoxLayout(dlg)
        listw = QtWidgets.QListWidget()
        listw.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        # Build items from clusters (excluding cluster 0)
        options = sorted([c for c in self.clustered_data['cluster'].unique() if c != 0])
        items = [self._get_cluster_display_name(cid) for cid in options]
        for label in items:
            it = QtWidgets.QListWidgetItem(label)
            # If no filter selection exists, select all by default
            if not getattr(self, 'stacked_bars_filter_selection', None):
                it.setSelected(True)
            else:
                it.setSelected(label in self.stacked_bars_filter_selection)
            listw.addItem(it)
        v.addWidget(listw)
        # Action buttons
        btns = QtWidgets.QHBoxLayout()
        select_all_btn = QtWidgets.QPushButton("Select All")
        ok = QtWidgets.QPushButton("Apply")
        cancel = QtWidgets.QPushButton("Cancel")
        btns.addStretch()
        btns.addWidget(select_all_btn)
        btns.addWidget(ok)
        btns.addWidget(cancel)
        v.addLayout(btns)
        ok.clicked.connect(dlg.accept)
        cancel.clicked.connect(dlg.reject)
        def do_select_all():
            for i in range(listw.count()):
                listw.item(i).setSelected(True)
        select_all_btn.clicked.connect(do_select_all)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            selected_labels = [i.text() for i in listw.selectedItems()]
            if selected_labels:
                self.stacked_bars_filter_selection = set(selected_labels)
            else:
                # If nothing selected, show all (set to None)
                self.stacked_bars_filter_selection = None
            self._show_stacked_bars()
    
    def _on_top_n_changed(self, _value: int):
        """Handle top N markers selection change."""
        # Refresh the differential expression view
        view = self.view_combo.currentText() if hasattr(self, 'view_combo') else 'Heatmap'
        if view == 'Differential Expression':
            self._show_differential_expression()
    
    def _get_colormap_name(self):
        """Get the matplotlib colormap name from the combo box selection."""
        colormap_text = self.colormap_combo.currentText()
        # Extract the colormap name (part before the parenthesis)
        colormap_name = colormap_text.split(' (')[0]
        return colormap_name

    def _select_feature_columns(self, df: pd.DataFrame):
        """Return numeric feature columns to plot, excluding non-numeric/meta columns."""
        # Standard columns to exclude
        exclude_cols = { 'cluster', '__group__', 'cluster_phenotype', 'manual_phenotype' }
        
        # Get all metadata columns (including those added during batch correction)
        metadata_cols = set(self._get_metadata_columns(df))
        
        # Combine exclusions
        all_exclude_cols = exclude_cols | metadata_cols
        
        feature_cols = []
        for col in df.columns:
            if col in all_exclude_cols:
                continue
            try:
                # Check if numeric but exclude boolean columns (they cause issues with numpy operations)
                if pd.api.types.is_numeric_dtype(df[col]) and df[col].dtype != bool:
                    feature_cols.append(col)
            except Exception:
                continue
        return feature_cols

    def _on_heatmap_source_changed(self, _text: str):
        """Refresh heatmap when the source (Clusters vs Manual Gates) changes."""
        view = self.view_combo.currentText() if hasattr(self, 'view_combo') else 'Heatmap'
        if view == 'Heatmap':
            self._show_heatmap()
    
    def _on_heatmap_scaling_changed(self, _text: str):
        """Refresh heatmap when the scaling method changes."""
        view = self.view_combo.currentText() if hasattr(self, 'view_combo') else 'Heatmap'
        if view == 'Heatmap':
            self._show_heatmap()
    
    def _on_patient_annotation_changed(self, state: int):
        """Handle patient annotation checkbox state change."""
        # Update enabled flag
        self.patient_annotation_enabled = (state == 2)  # 2 = checked
        
        # Enable/disable the customize button (check if patient annotation column is available)
        has_patient_col = False
        if hasattr(self, 'patient_annotation_column') and self.patient_annotation_column:
            has_patient_col = self.patient_annotation_column in self.feature_dataframe.columns
        else:
            # Check default priority columns
            for col in ['source_file', 'batch_group', 'source_well']:
                if col in self.feature_dataframe.columns:
                    has_patient_col = True
                    break
            # Also check metadata columns
            if not has_patient_col:
                metadata_cols = self._get_metadata_columns(self.feature_dataframe)
                has_patient_col = len(metadata_cols) > 0
        
        # Refresh heatmap if it's the current view
        view = self.view_combo.currentText() if hasattr(self, 'view_combo') else 'Heatmap'
        if view == 'Heatmap':
            self._show_heatmap()

    def _get_cluster_display_name(self, cluster_id):
        """Return display label for a cluster id, using annotation if available."""
        if isinstance(self.cluster_annotation_map, dict) and cluster_id in self.cluster_annotation_map and self.cluster_annotation_map[cluster_id]:
            return self.cluster_annotation_map[cluster_id]
        return f"Cluster {cluster_id}"
    
    def _get_patient_display_name(self, source_file):
        """Return display label for a source file/patient, using custom annotation if available."""
        if pd.isna(source_file):
            return "Unknown"
        if isinstance(self.patient_annotation_map, dict) and source_file in self.patient_annotation_map and self.patient_annotation_map[source_file]:
            return self.patient_annotation_map[source_file]
        # For source_file column, use basename of file as default
        # For other columns (batch_group, source_well), use the value as-is
        import os
        # Check if it looks like a file path (contains path separators)
        source_str = str(source_file)
        if os.sep in source_str or '/' in source_str or '\\' in source_str:
            return os.path.basename(source_str)
        # Otherwise, return the value as-is (for batch_group, source_well, etc.)
        return source_str

    def _get_manual_groups_series(self):
        """Compute grouping series for manual gates. Single named phenotype -> name vs Other; otherwise names with Unassigned for blanks."""
        if self.clustered_data is None:
            return None
        if 'manual_phenotype' not in self.clustered_data.columns:
            return None
        series = self.clustered_data['manual_phenotype'].fillna('').astype(str)
        unique_named = sorted([s for s in series.unique() if s.strip() != ''])
        if len(unique_named) == 1:
            name = unique_named[0]
            return series.apply(lambda s: name if s == name else 'Other')
        return series.apply(lambda s: s if s.strip() != '' else 'Unassigned')

    def _apply_heatmap_filter(self, df: pd.DataFrame, group_col: str) -> pd.DataFrame:
        """Apply heatmap filter selection to the dataframe, if any selection present."""
        selected = getattr(self, 'heatmap_filter_selection', None)
        if not selected:
            return df
        mask = df[group_col].isin(list(selected))
        filtered = df.loc[mask]
        return filtered

    def _open_heatmap_filter_dialog(self):
        """Open a dialog to choose which clusters/phenotypes to show in heatmap."""
        if self.clustered_data is None:
            QtWidgets.QMessageBox.warning(self, "No Clustering", "Run clustering first to filter heatmap.")
            return
        source = self.heatmap_source_combo.currentText() if hasattr(self, 'heatmap_source_combo') else 'Clusters'
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Select groups to display")
        v = QtWidgets.QVBoxLayout(dlg)
        listw = QtWidgets.QListWidget()
        listw.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        # Build items
        items = []
        if source == 'Manual Gates' and 'manual_phenotype' in self.clustered_data.columns:
            groups = self._get_manual_groups_series()
            options = sorted(groups.unique()) if groups is not None else []
            items = options
        else:
            options = sorted(self.clustered_data['cluster'].unique())
            items = [self._get_cluster_display_name(cid) for cid in options]
        for label in items:
            it = QtWidgets.QListWidgetItem(label)
            it.setSelected(True if not getattr(self, 'heatmap_filter_selection', None) else (label in self.heatmap_filter_selection))
            listw.addItem(it)
        v.addWidget(listw)
        # Action buttons
        btns = QtWidgets.QHBoxLayout()
        select_all_btn = QtWidgets.QPushButton("Select All")
        ok = QtWidgets.QPushButton("Apply")
        cancel = QtWidgets.QPushButton("Cancel")
        btns.addStretch()
        btns.addWidget(select_all_btn)
        btns.addWidget(ok)
        btns.addWidget(cancel)
        v.addLayout(btns)
        ok.clicked.connect(dlg.accept)
        cancel.clicked.connect(dlg.reject)
        def do_select_all():
            for i in range(listw.count()):
                listw.item(i).setSelected(True)
        select_all_btn.clicked.connect(do_select_all)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.heatmap_filter_selection = set([i.text() for i in listw.selectedItems()])
            self._show_heatmap()
            # Also update UMAP if it's currently visible
            view = self.view_combo.currentText() if hasattr(self, 'view_combo') else ''
            if view == 'UMAP' and getattr(self, 'umap_embedding', None) is not None:
                self._create_umap_plot()
    
    def _plot_umap_single(self, ax, color_by, point_size, point_alpha, title=None, show_legend=True):
        """Plot a single UMAP subplot with specified coloring."""
        # DEBUG: Print at the start of the function
        print(f"[UMAP DEBUG] _plot_umap_single called with color_by='{color_by}'")
        print(f"[UMAP DEBUG] color_by type: {type(color_by)}")
        print(f"[UMAP DEBUG] hasattr(self, 'umap_raw_data'): {hasattr(self, 'umap_raw_data')}")
        print(f"[UMAP DEBUG] hasattr(self, 'umap_selected_columns'): {hasattr(self, 'umap_selected_columns')}")
        if hasattr(self, 'umap_selected_columns'):
            print(f"[UMAP DEBUG] umap_selected_columns: {self.umap_selected_columns}")
            print(f"[UMAP DEBUG] color_by in umap_selected_columns: {color_by in self.umap_selected_columns if self.umap_selected_columns else False}")
        if hasattr(self, 'umap_raw_data'):
            print(f"[UMAP DEBUG] umap_raw_data columns: {list(self.umap_raw_data.columns[:10])}... (showing first 10)")
            print(f"[UMAP DEBUG] color_by in umap_raw_data.columns: {color_by in self.umap_raw_data.columns}")
        
        # Compute filtered embedding once at the start (used for spread control)
        if hasattr(self, 'umap_index') and self.umap_index is not None:
            cluster_labels_series = self.clustered_data['cluster']
            cluster_labels = cluster_labels_series.reindex(self.umap_index).values
        else:
            cluster_labels = self.clustered_data['cluster'].values
        
        # Filter out cluster 0 (noise/unassigned)
        valid_mask = cluster_labels != 0
        
        # Apply heatmap filter if it exists (filter cells from UMAP that are filtered in heatmap)
        if hasattr(self, 'heatmap_filter_selection') and self.heatmap_filter_selection:
            # Convert display names to cluster IDs (same logic as heatmap)
            wanted_ids = set()
            for cid in sorted(self.clustered_data['cluster'].unique()):
                if pd.notna(cid) and cid != 0:
                    name = self._get_cluster_display_name(cid)
                    if name in self.heatmap_filter_selection or str(cid) in self.heatmap_filter_selection:
                        wanted_ids.add(int(cid))
            if wanted_ids:
                # Apply filter: only keep clusters that are in the heatmap filter
                heatmap_filter_mask = np.isin(cluster_labels, list(wanted_ids))
                valid_mask = valid_mask & heatmap_filter_mask
        
        # Filter out NaN clusters
        valid_mask = valid_mask & pd.notna(cluster_labels)
        
        umap_embedding_filtered = self.umap_embedding[valid_mask]
        
        print(f"[UMAP DEBUG] Checking conditions...")
        print(f"[UMAP DEBUG] color_by == 'Cluster': {color_by == 'Cluster'}")
        print(f"[UMAP DEBUG] self.clustered_data is not None: {self.clustered_data is not None}")
        if self.clustered_data is not None:
            print(f"[UMAP DEBUG] 'cluster' in self.clustered_data.columns: {'cluster' in self.clustered_data.columns}")
        
        if color_by == 'Cluster' and self.clustered_data is not None and 'cluster' in self.clustered_data.columns:
            # Use pre-computed filtered data
            cluster_labels_filtered = cluster_labels[valid_mask]
            
            # Filter out NaN clusters and ensure all are valid integers
            valid_cluster_mask = pd.notna(cluster_labels_filtered)
            cluster_labels_filtered = cluster_labels_filtered[valid_cluster_mask]
            umap_embedding_filtered = umap_embedding_filtered[valid_cluster_mask]
            
            unique_clusters = sorted([c for c in np.unique(cluster_labels_filtered) if pd.notna(c) and c != 0])
            if len(unique_clusters) == 0:
                # No valid clusters, just plot without legend
                ax.scatter(umap_embedding_filtered[:, 0], umap_embedding_filtered[:, 1], 
                           c='gray', alpha=point_alpha, s=point_size, edgecolors='none')
            else:
                colors = _get_vivid_colors(len(unique_clusters))
                cluster_color_map = {cluster_id: colors[i] for i, cluster_id in enumerate(unique_clusters)}
                handles = []
                labels = []
                for cluster_id in unique_clusters:
                    mask = cluster_labels_filtered == cluster_id
                    if np.any(mask):  # Only plot if there are points
                        sc = ax.scatter(umap_embedding_filtered[mask, 0], umap_embedding_filtered[mask, 1],
                                        c=[cluster_color_map[cluster_id]],
                                        alpha=point_alpha, s=point_size, edgecolors='none')
                        # Create custom legend handle with fixed size (18)
                        color = cluster_color_map[cluster_id]
                        if len(color) == 4:
                            rgb = tuple(color[:3])
                        elif len(color) == 3:
                            rgb = tuple(color)
                        else:
                            rgb = (color[0], color[1], color[2])
                        handle = Line2D([0], [0], marker='o', color='none', markerfacecolor=rgb,
                                       markeredgecolor='none', markersize=6, alpha=point_alpha)
                        handles.append(handle)
                        labels.append(self._get_cluster_display_name(cluster_id))
                # Place legend inside axes to avoid clipping (only if show_legend is True)
                if show_legend and handles and labels:
                    n_clusters = len(handles)
                    ncol = max(1, (n_clusters + 9) // 10) if n_clusters > 10 else 1
                    ax.legend(handles, labels, loc='best', frameon=True, fontsize=8, ncol=ncol)
        elif color_by == 'Source File' and 'source_file' in self.feature_dataframe.columns:
            # Use pre-computed filtered data
            # Color by source file to visualize batch effects (using custom patient labels)
            if hasattr(self, 'umap_index') and self.umap_index is not None:
                source_file_series = self.feature_dataframe.loc[self.umap_index, 'source_file']
                source_files = source_file_series.values[valid_mask]
            else:
                source_files = self.feature_dataframe['source_file'].values[valid_mask]
            unique_files = sorted([f for f in np.unique(source_files) if pd.notna(f)])
            if len(unique_files) > 0:
                # Use patient colors palette (different from cluster colors)
                patient_colors_raw = _get_patient_colors(len(unique_files))
                file_color_map = {file_name: patient_colors_raw[i] for i, file_name in enumerate(unique_files)}
                handles = []
                labels = []
                for file_name in unique_files:
                    mask = source_files == file_name
                    if np.any(mask):  # Only add if there are points for this file
                        sc = ax.scatter(umap_embedding_filtered[mask, 0], umap_embedding_filtered[mask, 1],
                                        c=[file_color_map[file_name]],
                                        alpha=point_alpha, s=point_size, edgecolors='none')
                        # Create custom legend handle with fixed size
                        color = file_color_map[file_name]
                        if len(color) == 4:
                            rgb = tuple(color[:3])
                        elif len(color) == 3:
                            rgb = tuple(color)
                        else:
                            rgb = (color[0], color[1], color[2])
                        handle = Line2D([0], [0], marker='o', color='none', markerfacecolor=rgb,
                                       markeredgecolor='none', markersize=6, alpha=point_alpha)
                        handles.append(handle)
                        # Use custom patient label if available
                        labels.append(self._get_patient_display_name(file_name))
                # Place legend inside axes to avoid clipping - ensure it's visible
                if show_legend and handles and labels:
                    legend = ax.legend(handles, labels, loc='best', frameon=True, fontsize=8, title=self.patient_legend_label)
                    legend.set_visible(True)
            else:
                # Fallback if no source files
                ax.scatter(umap_embedding_filtered[:, 0], umap_embedding_filtered[:, 1], c='blue', alpha=point_alpha, s=point_size, edgecolors='none')
        elif color_by == 'Phenotype' and self.clustered_data is not None and 'cluster_phenotype' in self.clustered_data.columns:
            # Use pre-computed filtered data
            # Color by cluster phenotype
            if hasattr(self, 'umap_index') and self.umap_index is not None:
                phenotype_series = self.clustered_data['cluster_phenotype'].reindex(self.umap_index)
                phenotypes = phenotype_series.fillna('Unassigned').values[valid_mask]
            else:
                phenotypes = self.clustered_data['cluster_phenotype'].fillna('Unassigned').values[valid_mask]
            unique_phenotypes = sorted([p for p in np.unique(phenotypes) if pd.notna(p)])
            colors = _get_vivid_colors(len(unique_phenotypes))
            phenotype_color_map = {p: colors[i] for i, p in enumerate(unique_phenotypes)}
            handles = []
            labels = []
            for phenotype in unique_phenotypes:
                mask = phenotypes == phenotype
                sc = ax.scatter(umap_embedding_filtered[mask, 0], umap_embedding_filtered[mask, 1],
                                c=[phenotype_color_map[phenotype]],
                                alpha=point_alpha, s=point_size, edgecolors='none')
                # Create custom legend handle with fixed size
                color = phenotype_color_map[phenotype]
                if len(color) == 4:
                    rgb = tuple(color[:3])
                elif len(color) == 3:
                    rgb = tuple(color)
                else:
                    rgb = (color[0], color[1], color[2])
                handle = Line2D([0], [0], marker='o', color='none', markerfacecolor=rgb,
                               markeredgecolor='none', markersize=6, alpha=point_alpha)
                handles.append(handle)
                labels.append(str(phenotype))
            if show_legend:
                ax.legend(handles, labels, loc='best', frameon=True, fontsize=8, title='Phenotype')
        elif color_by == 'Manual Phenotype' and 'manual_phenotype' in self.feature_dataframe.columns:
            # Use pre-computed filtered data
            # Color by manual phenotype
            if hasattr(self, 'umap_index') and self.umap_index is not None:
                manual_phenotype_series = self.feature_dataframe.loc[self.umap_index, 'manual_phenotype']
                manual_phenotypes = manual_phenotype_series.fillna('Unassigned').values[valid_mask]
            else:
                manual_phenotypes = self.feature_dataframe['manual_phenotype'].fillna('Unassigned').values[valid_mask]
            unique_phenotypes = sorted([p for p in np.unique(manual_phenotypes) if pd.notna(p)])
            colors = _get_vivid_colors(len(unique_phenotypes))
            phenotype_color_map = {p: colors[i] for i, p in enumerate(unique_phenotypes)}
            handles = []
            labels = []
            for phenotype in unique_phenotypes:
                mask = manual_phenotypes == phenotype
                sc = ax.scatter(umap_embedding_filtered[mask, 0], umap_embedding_filtered[mask, 1],
                                c=[phenotype_color_map[phenotype]],
                                alpha=point_alpha, s=point_size, edgecolors='none')
                # Create custom legend handle with fixed size
                color = phenotype_color_map[phenotype]
                if len(color) == 4:
                    rgb = tuple(color[:3])
                elif len(color) == 3:
                    rgb = tuple(color)
                else:
                    rgb = (color[0], color[1], color[2])
                handle = Line2D([0], [0], marker='o', color='none', markerfacecolor=rgb,
                               markeredgecolor='none', markersize=6, alpha=point_alpha)
                handles.append(handle)
                labels.append(str(phenotype))
            if show_legend:
                ax.legend(handles, labels, loc='best', frameon=True, fontsize=8, title='Manual Phenotype')
        elif hasattr(self, 'umap_raw_data') and color_by in getattr(self, 'umap_selected_columns', []):
            # Check all conditions before processing
            print(f"[UMAP DEBUG] Reached intensity feature check...")
            print(f"[UMAP DEBUG] hasattr(self, 'umap_raw_data'): {hasattr(self, 'umap_raw_data')}")
            print(f"[UMAP DEBUG] getattr(self, 'umap_selected_columns', []): {getattr(self, 'umap_selected_columns', [])}")
            print(f"[UMAP DEBUG] color_by in getattr(self, 'umap_selected_columns', []): {color_by in getattr(self, 'umap_selected_columns', [])}")
            print(f"[UMAP DEBUG] Combined condition: {hasattr(self, 'umap_raw_data') and color_by in getattr(self, 'umap_selected_columns', [])}")
            # DEBUG: Print diagnostic information
            print(f"[DEBUG] Coloring by intensity feature: {color_by}")
            print(f"[DEBUG] umap_raw_data exists: {hasattr(self, 'umap_raw_data')}")
            print(f"[DEBUG] umap_selected_columns: {getattr(self, 'umap_selected_columns', None)}")
            print(f"[DEBUG] color_by in umap_selected_columns: {color_by in getattr(self, 'umap_selected_columns', [])}")
            print(f"[DEBUG] umap_index exists: {hasattr(self, 'umap_index')}")
            if hasattr(self, 'umap_index'):
                print(f"[DEBUG] umap_index length: {len(self.umap_index) if self.umap_index else 0}")
            print(f"[DEBUG] umap_raw_data shape: {self.umap_raw_data.shape if hasattr(self, 'umap_raw_data') else 'N/A'}")
            print(f"[DEBUG] umap_raw_data columns: {list(self.umap_raw_data.columns) if hasattr(self, 'umap_raw_data') else 'N/A'}")
            print(f"[DEBUG] color_by in umap_raw_data.columns: {color_by in self.umap_raw_data.columns if hasattr(self, 'umap_raw_data') else False}")
            print(f"[DEBUG] valid_mask shape: {valid_mask.shape}")
            print(f"[DEBUG] valid_mask sum (valid points): {valid_mask.sum()}")
            
            # Use pre-computed filtered data
            # Align umap_raw_data with umap_index order
            if hasattr(self, 'umap_index') and self.umap_index is not None:
                # Align umap_raw_data with umap_index order
                print(f"[DEBUG] Using umap_index alignment")
                print(f"[DEBUG] umap_index type: {type(self.umap_index)}")
                print(f"[DEBUG] umap_index first few: {self.umap_index[:5] if len(self.umap_index) > 5 else self.umap_index}")
                print(f"[DEBUG] umap_raw_data index type: {type(self.umap_raw_data.index)}")
                print(f"[DEBUG] umap_raw_data index first few: {list(self.umap_raw_data.index[:5]) if len(self.umap_raw_data) > 5 else list(self.umap_raw_data.index)}")
                
                try:
                    feature_series = self.umap_raw_data.loc[self.umap_index, color_by]
                    vals = feature_series.values
                    print(f"[DEBUG] Successfully extracted feature_series, length: {len(feature_series)}")
                    print(f"[DEBUG] vals shape: {vals.shape}")
                    print(f"[DEBUG] vals dtype: {vals.dtype}")
                    print(f"[DEBUG] vals min/max: {np.nanmin(vals)}/{np.nanmax(vals)}")
                    print(f"[DEBUG] vals NaN count: {np.isnan(vals).sum()}")
                except Exception as e:
                    print(f"[DEBUG] ERROR extracting feature_series: {e}")
                    print(f"[DEBUG] Attempting direct column access...")
                    vals = self.umap_raw_data[color_by].values
                    print(f"[DEBUG] Direct access successful, vals length: {len(vals)}")
            else:
                print(f"[DEBUG] No umap_index, using direct column access")
                vals = self.umap_raw_data[color_by].values
                print(f"[DEBUG] vals shape: {vals.shape}")
                print(f"[DEBUG] vals dtype: {vals.dtype}")
            
            # Continuous coloring by selected feature (aligned to UMAP order)
            vals_filtered = vals[valid_mask]
            print(f"[DEBUG] vals_filtered shape: {vals_filtered.shape}")
            print(f"[DEBUG] vals_filtered min/max: {np.nanmin(vals_filtered)}/{np.nanmax(vals_filtered)}")
            print(f"[DEBUG] vals_filtered NaN count: {np.isnan(vals_filtered).sum()}")
            print(f"[DEBUG] vals_filtered non-NaN count: {(~np.isnan(vals_filtered)).sum()}")
            
            # Check for valid values
            if len(vals_filtered) == 0:
                print(f"[DEBUG] ERROR: vals_filtered is empty!")
                ax.scatter(umap_embedding_filtered[:, 0], umap_embedding_filtered[:, 1], c='blue', 
                          alpha=point_alpha, s=point_size, edgecolors='none')
                ax.text(0.5, 0.5, 'No valid values for coloring', transform=ax.transAxes, 
                       ha='center', va='center')
            elif np.all(np.isnan(vals_filtered)):
                print(f"[DEBUG] ERROR: All values are NaN!")
                ax.scatter(umap_embedding_filtered[:, 0], umap_embedding_filtered[:, 1], c='blue', 
                          alpha=point_alpha, s=point_size, edgecolors='none')
                ax.text(0.5, 0.5, 'All values are NaN', transform=ax.transAxes, 
                       ha='center', va='center')
            else:
                # Remove NaN values for plotting
                valid_vals_mask = ~np.isnan(vals_filtered)
                print(f"[DEBUG] valid_vals_mask sum: {valid_vals_mask.sum()}")
                if np.any(valid_vals_mask):
                    print(f"[DEBUG] Creating scatter plot with {valid_vals_mask.sum()} valid points")
                    print(f"[DEBUG] umap_embedding_filtered shape: {umap_embedding_filtered.shape}")
                    print(f"[DEBUG] umap_embedding_filtered[valid_vals_mask] shape: {umap_embedding_filtered[valid_vals_mask].shape}")
                    print(f"[DEBUG] vals_filtered[valid_vals_mask] shape: {vals_filtered[valid_vals_mask].shape}")
                    print(f"[DEBUG] vals_filtered[valid_vals_mask] range: [{np.min(vals_filtered[valid_vals_mask])}, {np.max(vals_filtered[valid_vals_mask])}]")
                    
                    try:
                        sc = ax.scatter(umap_embedding_filtered[valid_vals_mask, 0], 
                                      umap_embedding_filtered[valid_vals_mask, 1], 
                                      c=vals_filtered[valid_vals_mask],
                                      cmap='viridis', alpha=point_alpha, s=point_size, edgecolors='none')
                        cbar = self.figure.colorbar(sc, ax=ax)
                        cbar.set_label(color_by)
                        print(f"[DEBUG] Scatter plot created successfully")
                    except Exception as e:
                        print(f"[DEBUG] ERROR creating scatter plot: {e}")
                        import traceback
                        traceback.print_exc()
                        ax.scatter(umap_embedding_filtered[:, 0], umap_embedding_filtered[:, 1], c='blue', 
                                  alpha=point_alpha, s=point_size, edgecolors='none')
                        ax.text(0.5, 0.5, f'Error: {str(e)}', transform=ax.transAxes, 
                               ha='center', va='center', fontsize=8)
                else:
                    print(f"[DEBUG] ERROR: No valid (non-NaN) values after filtering!")
                    ax.scatter(umap_embedding_filtered[:, 0], umap_embedding_filtered[:, 1], c='blue', 
                              alpha=point_alpha, s=point_size, edgecolors='none')
                    ax.text(0.5, 0.5, 'No valid values for coloring', transform=ax.transAxes, 
                           ha='center', va='center')
        elif color_by in self.feature_dataframe.columns:
            # Handle metadata columns or other dataframe columns as categorical
            # (This comes after intensity feature check to avoid conflicts)
            metadata_cols = self._get_metadata_columns(self.feature_dataframe)
            if color_by in metadata_cols or color_by == 'batch_group':
                # Use pre-computed filtered data
                # Color by metadata column (categorical)
                if hasattr(self, 'umap_index') and self.umap_index is not None:
                    col_series = self.feature_dataframe.loc[self.umap_index, color_by]
                    col_values = col_series.fillna('Unknown').values[valid_mask]
                else:
                    col_values = self.feature_dataframe[color_by].fillna('Unknown').values[valid_mask]
                unique_values = sorted([v for v in np.unique(col_values) if pd.notna(v)])
                if len(unique_values) > 0:
                    colors = _get_vivid_colors(len(unique_values))
                    value_color_map = {v: colors[i] for i, v in enumerate(unique_values)}
                    handles = []
                    labels = []
                    for value in unique_values:
                        mask = col_values == value
                        if np.any(mask):
                            sc = ax.scatter(umap_embedding_filtered[mask, 0], umap_embedding_filtered[mask, 1],
                                            c=[value_color_map[value]],
                                            alpha=point_alpha, s=point_size, edgecolors='none')
                            color = value_color_map[value]
                            if len(color) == 4:
                                rgb = tuple(color[:3])
                            elif len(color) == 3:
                                rgb = tuple(color)
                            else:
                                rgb = (color[0], color[1], color[2])
                            handle = Line2D([0], [0], marker='o', color='none', markerfacecolor=rgb,
                                           markeredgecolor='none', markersize=6, alpha=point_alpha)
                            handles.append(handle)
                            labels.append(str(value))
                    if show_legend and handles and labels:
                        ax.legend(handles, labels, loc='best', frameon=True, fontsize=8, title=color_by)
        else:
            # Use pre-computed filtered data
            # Fallback single color
            print(f"[UMAP DEBUG] Falling through to else branch (fallback)")
            print(f"[UMAP DEBUG] color_by value: '{color_by}'")
            print(f"[UMAP DEBUG] All conditions checked, none matched. Using fallback blue scatter.")
            ax.scatter(umap_embedding_filtered[:, 0], umap_embedding_filtered[:, 1], c='blue', alpha=point_alpha, s=point_size, edgecolors='none')
        
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        if title is None:
            title = f'UMAP: {color_by}'
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    def _create_umap_plot(self):
        """Create UMAP scatter plot(s) with faceted plotting support."""
        if self.umap_embedding is None:
            return
        
        self.figure.clear()
        
        # Get selected color-by options (use stored data which contains actual column names)
        if hasattr(self, 'color_by_listwidget'):
            selected_items = []
            for item in self.color_by_listwidget.selectedItems():
                # Use stored data (actual column name) if available, otherwise use text
                actual_name = item.data(QtCore.Qt.UserRole)
                if actual_name:
                    selected_items.append(actual_name)
                else:
                    selected_items.append(item.text())
        else:
            # Fallback to combo box if list widget doesn't exist
            selected_items = [self.color_by_combo.currentText()] if hasattr(self, 'color_by_combo') else ['Cluster']
        
        # Ensure at least one selection
        if not selected_items:
            selected_items = ['Cluster']
        
        # Limit to max 3 plots (3 columns in single row)
        selected_items = selected_items[:3]
        
        # Get point size and alpha from controls
        point_size = self.point_size_spinbox.value() if hasattr(self, 'point_size_spinbox') else 18
        point_alpha = self.point_alpha_spinbox.value() if hasattr(self, 'point_alpha_spinbox') else 0.8
        show_legend = self.show_legend_checkbox.isChecked() if hasattr(self, 'show_legend_checkbox') else True
        
        n_plots = len(selected_items)
        
        if n_plots == 1:
            # Single plot - use full figure
            ax = self.figure.add_subplot(111)
            self._plot_umap_single(ax, selected_items[0], point_size, point_alpha, show_legend=show_legend)
            self.figure.tight_layout(pad=1.0)
        else:
            # Multiple plots - create subplots in a single row with max 3 columns
            n_cols = n_plots
            n_rows = 1
            
            for idx, color_by in enumerate(selected_items):
                ax = self.figure.add_subplot(n_rows, n_cols, idx + 1)
                self._plot_umap_single(ax, color_by, point_size, point_alpha, show_legend=show_legend)
            
            self.figure.tight_layout(pad=1.0)
        
        self.canvas.draw()

    def _populate_color_by_options(self):
        """Populate the color-by list widget with Cluster + used features."""
        if not hasattr(self, 'color_by_listwidget'):
            return
        # Get currently selected items (use stored data for actual column names)
        selected_items = []
        for item in self.color_by_listwidget.selectedItems():
            actual_name = item.data(QtCore.Qt.UserRole)
            if actual_name:
                selected_items.append(actual_name)
            else:
                selected_items.append(item.text())
        if not selected_items:
            selected_items = ['Cluster']  # Default selection
        
        self.color_by_listwidget.blockSignals(True)
        self.color_by_listwidget.clear()
        
        # Add all available options
        options = ['Cluster']
        # Add source_file if available
        if 'source_file' in self.feature_dataframe.columns:
            options.append('Source File')
        # Add batch_group if available
        if 'batch_group' in self.feature_dataframe.columns:
            options.append('Batch Group')
        # Add metadata columns for coloring
        metadata_cols = self._get_metadata_columns(self.feature_dataframe)
        for col in metadata_cols:
            if col not in options:
                options.append(col)
        # Add feature columns from UMAP or t-SNE (use display names for UI, but store actual column names)
        for col in getattr(self, 'umap_selected_columns', []) or []:
            # Check if column is already in options (either as string or as tuple)
            already_in = col in options or any(isinstance(opt, tuple) and opt[1] == col for opt in options)
            if not already_in:
                # Use display name for UI, but we'll store the actual column name
                display_name = self._get_feature_display_name(col)
                options.append((display_name, col))  # Store as (display, actual) tuple
        for col in getattr(self, 'tsne_selected_columns', []) or []:
            # Check if column is already in options (either as string or as tuple)
            already_in = col in options or any(isinstance(opt, tuple) and opt[1] == col for opt in options)
            if not already_in:
                display_name = self._get_feature_display_name(col)
                options.append((display_name, col))
        # Add phenotype if available
        if hasattr(self, 'clustered_data') and self.clustered_data is not None and 'cluster_phenotype' in self.clustered_data.columns:
            if 'Phenotype' not in options:
                options.append('Phenotype')
        # Add manual phenotype if available
        if 'manual_phenotype' in self.feature_dataframe.columns:
            if 'Manual Phenotype' not in options:
                options.append('Manual Phenotype')
        
        # Add items to list widget
        for option in options:
            if isinstance(option, tuple):
                # Feature column: (display_name, actual_column_name)
                display_name, actual_name = option
                item = QtWidgets.QListWidgetItem(display_name)
                item.setData(QtCore.Qt.UserRole, actual_name)  # Store actual column name
            else:
                # Standard option (Cluster, Source File, etc.)
                item = QtWidgets.QListWidgetItem(option)
                item.setData(QtCore.Qt.UserRole, option)  # Store same value
            self.color_by_listwidget.addItem(item)
            # Check if this item should be selected (compare display name or actual name)
            item_text = item.text()
            item_data = item.data(QtCore.Qt.UserRole)
            if item_text in selected_items or item_data in selected_items:
                item.setSelected(True)
        
        # Ensure at least "Cluster" is selected if nothing was selected
        if not self.color_by_listwidget.selectedItems():
            for i in range(self.color_by_listwidget.count()):
                item = self.color_by_listwidget.item(i)
                if item.text() == 'Cluster':
                    item.setSelected(True)
                    break
        
        self.color_by_listwidget.blockSignals(False)

    def _filter_color_by_options(self, search_text: str):
        """Filter the color-by list widget items based on search text."""
        if not hasattr(self, 'color_by_listwidget'):
            return
        
        search_text = search_text.lower()
        for i in range(self.color_by_listwidget.count()):
            item = self.color_by_listwidget.item(i)
            item_text = item.text().lower()
            # Show item if search text is empty or matches item text
            item.setHidden(bool(search_text) and search_text not in item_text)
    
    def _on_color_by_changed(self, _text: str = None):
        view = self.view_combo.currentText() if hasattr(self, 'view_combo') else ''
        if view == 'UMAP' and getattr(self, 'umap_embedding', None) is not None:
            self._create_umap_plot()
        elif view == 't-SNE' and getattr(self, 'tsne_embedding', None) is not None:
            self._create_tsne_plot()

    def _on_point_style_changed(self):
        """Update plot when point size or alpha changes."""
        view = self.view_combo.currentText() if hasattr(self, 'view_combo') else ''
        if view == 'UMAP' and getattr(self, 'umap_embedding', None) is not None:
            self._create_umap_plot()
        elif view == 't-SNE' and getattr(self, 'tsne_embedding', None) is not None:
            self._create_tsne_plot()
    
    def _on_legend_changed(self):
        """Handle legend visibility changes for all plot types."""
        view = self.view_combo.currentText() if hasattr(self, 'view_combo') else ''
        if view == 'UMAP' and getattr(self, 'umap_embedding', None) is not None:
            self._create_umap_plot()
        elif view == 't-SNE' and getattr(self, 'tsne_embedding', None) is not None:
            self._create_tsne_plot()
        elif view == 'Stacked Bars':
            self._show_stacked_bars()
        elif view == 'Heatmap':
            # Recreate heatmap with updated legend visibility
            self._create_heatmap()

    def _show_stacked_bars(self):
        """Show stacked bar plots of cluster frequencies per selected group (ROI/condition/slide)."""
        if self.clustered_data is None or 'cluster' not in self.clustered_data.columns:
            QtWidgets.QMessageBox.warning(self, "No Clustering", "Please run clustering first to view stacked bars.")
            return
        
        # Use patient_annotation_column if set and available, otherwise use group_by_combo
        group_col = None
        if hasattr(self, 'patient_annotation_column') and self.patient_annotation_column:
            # Check if patient_annotation_column exists in feature_dataframe
            if self.patient_annotation_column in self.feature_dataframe.columns:
                group_col = self.patient_annotation_column
                # Also update group_by_combo to match if it exists
                if hasattr(self, 'group_by_combo'):
                    for i in range(self.group_by_combo.count()):
                        if self.group_by_combo.itemText(i) == group_col:
                            self.group_by_combo.setCurrentIndex(i)
                            break
        
        # Fall back to group_by_combo if patient_annotation_column not set or not available
        if not group_col:
            group_col = self.group_by_combo.currentText() if hasattr(self, 'group_by_combo') and self.group_by_combo.count() > 0 else None
        
        if not group_col:
            QtWidgets.QMessageBox.warning(self, "No Grouping", "No valid grouping column is available.")
            return
        
        # Handle acquisition_id: merge with source_file to create unique identifier
        if group_col == 'acquisition_id' and 'source_file' in self.feature_dataframe.columns:
            # Create merged column if it doesn't exist
            merged_col = 'source_file_acquisition_id'
            if merged_col not in self.feature_dataframe.columns:
                self.feature_dataframe[merged_col] = (
                    self.feature_dataframe['source_file'].astype(str) + '_' + 
                    self.feature_dataframe['acquisition_id'].astype(str)
                )
            group_col = merged_col
        
        if group_col not in self.feature_dataframe.columns:
            QtWidgets.QMessageBox.warning(self, "No Grouping", "No valid grouping column is available.")
            return

        try:
            # Filter out dropped clusters (cluster 0)
            filtered_clustered_data = self.clustered_data[self.clustered_data['cluster'] != 0].copy()
            
            # Apply cluster filter if specified
            filter_selection = getattr(self, 'stacked_bars_filter_selection', None)
            if filter_selection:
                # Convert display names to cluster IDs
                cluster_id_map = {}
                for cid in filtered_clustered_data['cluster'].unique():
                    display_name = self._get_cluster_display_name(cid)
                    cluster_id_map[display_name] = cid
                # Get cluster IDs that match the selected display names
                selected_cluster_ids = [cluster_id_map[name] for name in filter_selection if name in cluster_id_map]
                if selected_cluster_ids:
                    filtered_clustered_data = filtered_clustered_data[filtered_clustered_data['cluster'].isin(selected_cluster_ids)]
            
            # Align metadata to clustered_data order
            meta_series = self.feature_dataframe.loc[filtered_clustered_data.index, group_col]
            clusters = filtered_clustered_data['cluster']
            
            # Apply custom patient labels if available and using patient annotation column
            if (hasattr(self, 'patient_annotation_column') and 
                self.patient_annotation_column == group_col and 
                hasattr(self, 'patient_annotation_map') and 
                self.patient_annotation_map):
                # Map values using custom labels
                meta_series = meta_series.map(lambda x: self.patient_annotation_map.get(x, x) if pd.notna(x) else x)
            
            # Build counts per group and cluster
            ct = pd.crosstab(meta_series, clusters).sort_index()
            
            # Get view type (Fraction or Total enumeration)
            view_type = self.stacked_bars_view_type_combo.currentText() if hasattr(self, 'stacked_bars_view_type_combo') else 'Fraction'
            
            if view_type == 'Fraction':
                # Convert to frequencies
                data_to_plot = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
                ylabel = 'Fraction of cells'
                ylim = (0, 1)
            else:
                # Use raw counts
                data_to_plot = ct.copy()
                ylabel = 'Number of cells'
                ylim = None

            # Prepare colors consistent with other views
            unique_clusters = sorted(clusters.unique())
            colors = _get_vivid_colors(len(unique_clusters))
            cluster_color_map = {cluster_id: colors[i] for i, cluster_id in enumerate(unique_clusters)}

            # Plot
            self.figure.clear()
            ax = self.figure.add_subplot(111)

            bottom = np.zeros(len(data_to_plot))
            x = np.arange(len(data_to_plot))
            for cluster_id in unique_clusters:
                vals = data_to_plot.get(cluster_id, pd.Series(0, index=data_to_plot.index)).values
                ax.bar(x, vals, bottom=bottom, color=cluster_color_map[cluster_id], label=self._get_cluster_display_name(cluster_id))
                bottom = bottom + vals

            ax.set_xticks(x)
            # Use custom legend label if available and using patient annotation column
            xlabel = group_col
            if (hasattr(self, 'patient_annotation_column') and 
                self.patient_annotation_column == group_col and 
                hasattr(self, 'patient_legend_label')):
                xlabel = self.patient_legend_label
            ax.set_xticklabels([str(i) for i in data_to_plot.index], rotation=45, ha='right')
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            ax.set_title(f'Cluster composition by {xlabel}')
            if ylim:
                ax.set_ylim(ylim)
            # Show legend only if checkbox is checked
            show_legend = self.show_legend_checkbox.isChecked() if hasattr(self, 'show_legend_checkbox') else True
            if show_legend:
                # Use multiple columns if there are many clusters to make legend more compact
                n_clusters = len(unique_clusters)
                # Calculate number of columns: use 2 columns if >10 clusters, 3 if >20, 4 if >30, etc.
                if n_clusters > 30:
                    ncol = 4
                elif n_clusters > 20:
                    ncol = 3
                elif n_clusters > 10:
                    ncol = 2
                else:
                    ncol = 1
                ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=ncol)
            self.figure.tight_layout()
            self.canvas.draw()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error creating stacked bars: {str(e)}")

    def _show_differential_expression(self):
        """Show differential expression heatmap showing top 5 markers per cluster."""
        if self.clustered_data is None or 'cluster' not in self.clustered_data.columns:
            QtWidgets.QMessageBox.warning(self, "No Clustering", "Please run clustering first to view differential expression.")
            return
        
        try:
            # Filter out dropped clusters (cluster 0)
            filtered_clustered_data = self.clustered_data[self.clustered_data['cluster'] != 0].copy()
            
            # Get numeric feature columns only (exclude cluster/phenotype/text metadata)
            feature_cols = self._select_feature_columns(filtered_clustered_data)
            
            if not feature_cols:
                QtWidgets.QMessageBox.warning(self, "No Features", "No features available for differential expression analysis.")
                return
            
            # Calculate mean expression per cluster for each feature
            cluster_means = filtered_clustered_data.groupby('cluster')[feature_cols].mean()
            
            # Calculate differential expression (z-score across clusters for each feature)
            # This shows which features are most variable across clusters
            feature_means = cluster_means.mean(axis=0)  # Mean across clusters
            feature_stds = cluster_means.std(axis=0)    # Std across clusters
            
            # Avoid division by zero
            feature_stds = feature_stds.replace(0, 1)
            
            # Z-score normalization: (value - mean) / std
            differential_scores = (cluster_means - feature_means) / feature_stds
            
            # Find top N markers FOR EACH cluster individually
            # Get the user-selected number of top markers
            top_n = self.top_n_spinbox.value()
            
            # For each cluster, find the top N features with highest z-scores
            cluster_top_features = {}
            top_features = []
            
            # Sort clusters for consistent ordering
            sorted_clusters = sorted(differential_scores.index)
            
            for cluster_id in sorted_clusters:
                # Get z-scores for this cluster
                cluster_scores = differential_scores.loc[cluster_id]
                # Sort by z-score (descending) and take top N
                top_n_for_cluster = cluster_scores.nlargest(top_n).index.tolist()
                cluster_top_features[cluster_id] = top_n_for_cluster
                # Add features for this cluster to the ordered list
                top_features.extend(top_n_for_cluster)
            
            if not top_features:
                QtWidgets.QMessageBox.warning(self, "No Features", "No features found for differential expression analysis.")
                return
            
            # Create heatmap data with all top features
            heatmap_data = differential_scores[top_features]
            
            # Create the plot
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            # Create heatmap with user-selected colormap
            colormap_name = self._get_colormap_name()
            im = ax.imshow(heatmap_data.T, cmap=colormap_name, aspect='auto', 
                          vmin=-3, vmax=3)  # Limit color scale to ±3 z-scores
            
            # Set labels
            ax.set_xticks(range(len(heatmap_data.index)))
            ax.set_xticklabels([self._get_cluster_display_name(i) for i in heatmap_data.index])
            ax.set_yticks(range(len(heatmap_data.columns)))
            # Use custom feature labels if available
            feature_labels_display = [self._get_feature_display_name(f) for f in heatmap_data.columns]
            ax.set_yticklabels(feature_labels_display, rotation=0)
            
            # Add colorbar
            cbar = self.figure.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Z-score (Differential Expression)', rotation=270, labelpad=20)
            
            # Add title and labels
            ax.set_title(f'Top {top_n} Differential Expression Markers per Cluster')
            ax.set_xlabel('Clusters')
            ax.set_ylabel('Features')
            
            # Add text annotations showing actual z-scores
            # Also highlight the top N markers for each cluster
            for i in range(len(heatmap_data.index)):
                cluster_id = heatmap_data.index[i]
                top_n_for_this_cluster = cluster_top_features[cluster_id]
                
                for j in range(len(heatmap_data.columns)):
                    feature_name = heatmap_data.columns[j]
                    value = heatmap_data.iloc[i, j]
                    
                    # Color text based on background
                    text_color = 'white' if abs(value) > 1.5 else 'black'
                    
                    # Make top N markers for this cluster more prominent
                    fontweight = 'bold'
                    fontsize = 9
                    if feature_name in top_n_for_this_cluster:
                        # Highlight top N markers with larger, bolder text
                        fontweight = 'bold'
                        fontsize = 10
                        # Add a subtle background highlight
                        ax.add_patch(plt.Rectangle((i-0.4, j-0.4), 0.8, 0.8, 
                                                 fill=False, edgecolor='black', 
                                                 linewidth=2, alpha=0.7))
                    
                    ax.text(i, j, f'{value:.2f}', ha='center', va='center', 
                           color=text_color, fontsize=fontsize, fontweight=fontweight)
            
            # Rotate x-axis labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Add explanation text below the colorbar
            explanation_text = (f"Black boxes highlight the top {top_n} markers for each cluster.\n"
                              "Z-scores show how much each cluster differs from the overall mean.")
            # Position text below the colorbar
            ax.text(1.02, -0.15, explanation_text, transform=ax.transAxes, 
                   fontsize=8, verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            self.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error creating differential expression heatmap: {str(e)}")

    def _open_marker_selection_dialog(self):
        """Open a dialog to select markers for boxplot/violin plot visualization."""
        if self.clustered_data is None:
            QtWidgets.QMessageBox.warning(self, "No Clustering", "Please run clustering first to select markers.")
            return
        
        # Get available marker columns
        marker_cols = self._select_feature_columns(self.clustered_data)
        
        if not marker_cols:
            QtWidgets.QMessageBox.warning(self, "No Markers", "No markers available for visualization.")
            return
        
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Select Markers")
        dlg.setMinimumSize(400, 500)
        
        layout = QtWidgets.QVBoxLayout(dlg)
        
        # Instructions
        instructions = QtWidgets.QLabel("Select markers to visualize (multiple selection allowed):")
        layout.addWidget(instructions)
        
        # List widget with multi-selection
        list_widget = QtWidgets.QListWidget()
        list_widget.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        
        # Add markers to list
        for marker in sorted(marker_cols):
            item = QtWidgets.QListWidgetItem(marker)
            # Pre-select if already in selected_markers
            if marker in self.selected_markers:
                item.setSelected(True)
            list_widget.addItem(item)
        
        layout.addWidget(list_widget)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        select_all_btn = QtWidgets.QPushButton("Select All")
        clear_all_btn = QtWidgets.QPushButton("Clear All")
        ok_btn = QtWidgets.QPushButton("OK")
        cancel_btn = QtWidgets.QPushButton("Cancel")
        
        def select_all():
            for i in range(list_widget.count()):
                list_widget.item(i).setSelected(True)
        
        def clear_all():
            for i in range(list_widget.count()):
                list_widget.item(i).setSelected(False)
        
        select_all_btn.clicked.connect(select_all)
        clear_all_btn.clicked.connect(clear_all)
        ok_btn.clicked.connect(dlg.accept)
        cancel_btn.clicked.connect(dlg.reject)
        
        button_layout.addWidget(select_all_btn)
        button_layout.addWidget(clear_all_btn)
        button_layout.addStretch()
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            # Get selected markers
            selected_items = list_widget.selectedItems()
            self.selected_markers = [item.text() for item in selected_items]
            
            # Refresh the plot if we're in boxplot/violin view
            if self.view_combo.currentText() == 'Boxplot/Violin Plot':
                self._show_boxplot_violin()

    def _on_plot_type_changed(self, _text: str):
        """Handle plot type change (boxplot vs violin plot)."""
        # Refresh the plot if we're in boxplot/violin view and have markers selected
        if self.view_combo.currentText() == 'Boxplot/Violin Plot':
            self._show_boxplot_violin()

    def _on_stats_test_changed(self, _state: int):
        """Handle statistical testing checkbox change."""
        # Update cluster combo visibility and enable export button
        if hasattr(self, 'stats_cluster_combo') and hasattr(self, 'stats_cluster_label'):
            is_enabled = self.stats_test_checkbox.isChecked()
            self.stats_cluster_combo.setEnabled(is_enabled)
            self.stats_cluster_label.setEnabled(is_enabled)
            if is_enabled:
                self._update_stats_cluster_combo()
        if hasattr(self, 'stats_export_btn'):
            self.stats_export_btn.setEnabled(self.stats_test_checkbox.isChecked() and 
                                           len(self.statistical_results) > 0)
        # Refresh the plot if we're in boxplot/violin view
        if self.view_combo.currentText() == 'Boxplot/Violin Plot':
            self._show_boxplot_violin()

    def _on_stats_mode_changed(self, _text: str):
        """Handle statistical test mode change."""
        # Update cluster combo visibility based on mode
        if hasattr(self, 'stats_mode_combo') and hasattr(self, 'stats_cluster_combo'):
            is_one_vs_others = self.stats_mode_combo.currentText() == "One vs Others"
            self.stats_cluster_combo.setVisible(is_one_vs_others)
            if hasattr(self, 'stats_cluster_label'):
                self.stats_cluster_label.setVisible(is_one_vs_others)
            if is_one_vs_others:
                self._update_stats_cluster_combo()
        # Refresh the plot if we're in boxplot/violin view
        if self.view_combo.currentText() == 'Boxplot/Violin Plot':
            self._show_boxplot_violin()

    def _on_stats_cluster_changed(self, _text: str):
        """Handle reference cluster selection change for one-vs-others mode."""
        # Refresh the plot if we're in boxplot/violin view
        if self.view_combo.currentText() == 'Boxplot/Violin Plot':
            self._show_boxplot_violin()

    def _update_stats_cluster_combo(self):
        """Update the cluster combo box with available clusters."""
        if not hasattr(self, 'stats_cluster_combo'):
            return
        if self.clustered_data is None or 'cluster' not in self.clustered_data.columns:
            return
        
        self.stats_cluster_combo.clear()
        # Filter out dropped clusters (cluster 0)
        unique_cluster_ids = sorted([cid for cid in self.clustered_data['cluster'].unique() if cid != 0])
        for cluster_id in unique_cluster_ids:
            cluster_name = self._get_cluster_display_name(cluster_id)
            self.stats_cluster_combo.addItem(cluster_name, cluster_id)

    def _bh_correction(self, p_values):
        """Apply Benjamini-Hochberg correction for multiple testing.
        
        Args:
            p_values: List or array of p-values
            
        Returns:
            List of adjusted p-values
        """
        p_values = np.array(p_values)
        n = len(p_values)
        if n == 0:
            return []
        
        # Sort p-values with their original indices
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        # Apply BH correction
        adjusted_p = np.zeros(n)
        for i in range(n-1, -1, -1):
            if i == n-1:
                adjusted_p[i] = min(sorted_p[i], 1.0)
            else:
                adjusted_p[i] = min(min(adjusted_p[i+1], sorted_p[i] * n / (i+1)), 1.0)
        
        # Restore original order
        result = np.zeros(n)
        result[sorted_indices] = adjusted_p
        return result.tolist()

    def _perform_pairwise_tests(self, data_dict, cluster_ids, mode='pairwise', reference_cluster=None):
        """Perform pairwise Mann-Whitney U tests.
        
        Args:
            data_dict: Dictionary mapping cluster_id to array of values
            cluster_ids: List of cluster IDs to test
            mode: 'pairwise' for all pairs, 'one_vs_others' for one vs all others
            reference_cluster: Cluster to compare against others (for one_vs_others mode)
            
        Returns:
            List of tuples: (cluster1, cluster2, p_value, adjusted_p_value)
        """
        results = []
        p_values = []
        pairs = []
        
        if mode == 'one_vs_others' and reference_cluster is not None:
            # One cluster vs all others
            if reference_cluster not in cluster_ids:
                return results
            
            data1 = data_dict[reference_cluster]
            if len(data1) < 2:
                return results
            
            for cluster2 in cluster_ids:
                if cluster2 == reference_cluster:
                    continue
                data2 = data_dict[cluster2]
                
                # Skip if insufficient data
                if len(data2) < 2:
                    continue
                
                try:
                    # Mann-Whitney U test (two-sided)
                    statistic, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
                    p_values.append(p_value)
                    pairs.append((reference_cluster, cluster2))
                except Exception:
                    # If test fails, skip this pair
                    continue
        else:
            # All pairwise comparisons
            for i, cluster1 in enumerate(cluster_ids):
                for cluster2 in cluster_ids[i+1:]:
                    data1 = data_dict[cluster1]
                    data2 = data_dict[cluster2]
                    
                    # Skip if insufficient data
                    if len(data1) < 2 or len(data2) < 2:
                        continue
                    
                    try:
                        # Mann-Whitney U test (two-sided)
                        statistic, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
                        p_values.append(p_value)
                        pairs.append((cluster1, cluster2))
                    except Exception:
                        # If test fails, skip this pair
                        continue
        
        # Apply BH correction
        if p_values:
            adjusted_p_values = self._bh_correction(p_values)
            
            # Build results list
            for (cluster1, cluster2), p_val, adj_p_val in zip(pairs, p_values, adjusted_p_values):
                results.append((cluster1, cluster2, p_val, adj_p_val))
        
        return results

    def _get_significance_stars(self, p_value):
        """Convert p-value to significance stars.
        
        Args:
            p_value: Adjusted p-value
            
        Returns:
            String with asterisks representing significance level
        """
        if p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        else:
            return 'ns'

    def _draw_significance_bar(self, ax, x1, x2, y, text, line_height=0.02, text_offset=0.03):
        """Draw a significance bar between two positions.
        
        Args:
            ax: Matplotlib axis
            x1, x2: X positions of the two groups
            y: Y position for the bar
            text: Text to display (e.g., '***')
            line_height: Height of the bar line
            text_offset: Offset for text above the bar (in data coordinates)
        """
        # Get y-axis range for proper scaling
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        
        # Draw horizontal line
        ax.plot([x1, x2], [y, y], 'k', linewidth=1)
        # Draw vertical lines at ends
        ax.plot([x1, x1], [y - line_height * y_range / 2, y], 'k', linewidth=1)
        ax.plot([x2, x2], [y - line_height * y_range / 2, y], 'k', linewidth=1)
        # Add text with proper spacing
        ax.text((x1 + x2) / 2, y + text_offset * y_range, text, ha='center', va='bottom', fontsize=9, fontweight='bold')

    def _show_boxplot_violin(self):
        """Show boxplot or violin plot of marker expressions by cluster."""
        if self.clustered_data is None or 'cluster' not in self.clustered_data.columns:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "Please run clustering first to view boxplot/violin plots", 
                    ha='center', va='center', transform=ax.transAxes, fontsize=14)
            self.canvas.draw()
            return
        
        if not self.selected_markers:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "Please select markers to visualize\n(Click 'Select Markers...' button)", 
                    ha='center', va='center', transform=ax.transAxes, fontsize=14)
            self.canvas.draw()
            return
        
        try:
            # Use raw (unscaled) data for boxplot/violin plots instead of z-scored data
            # Fall back to clustered_data if unscaled data is not available
            plot_data_source = self.clustered_data_unscaled if (
                hasattr(self, 'clustered_data_unscaled') and 
                self.clustered_data_unscaled is not None and 
                'cluster' in self.clustered_data_unscaled.columns
            ) else self.clustered_data
            
            # Filter out dropped clusters (cluster 0)
            plot_data_source = plot_data_source[plot_data_source['cluster'] != 0].copy()
            
            # Filter markers to only those available in the data
            available_markers = [m for m in self.selected_markers if m in plot_data_source.columns]
            
            if not available_markers:
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                ax.text(0.5, 0.5, "Selected markers not found in data", 
                        ha='center', va='center', transform=ax.transAxes, fontsize=14)
                self.canvas.draw()
                return
            
            self.figure.clear()
            
            # Prepare data for plotting
            # Store both original marker name (for filtering) and display name (for labels)
            marker_display_map = {marker: self._get_feature_display_name(marker) for marker in available_markers}
            plot_data = []
            for marker in available_markers:
                for cluster_id in sorted(plot_data_source['cluster'].unique()):
                    cluster_data = plot_data_source[
                        plot_data_source['cluster'] == cluster_id
                    ][marker].dropna()
                    
                    for value in cluster_data:
                        plot_data.append({
                            'Marker': marker,  # Keep original for filtering
                            'MarkerDisplay': marker_display_map[marker],  # Custom label for display
                            'Cluster': self._get_cluster_display_name(cluster_id),
                            'Value': value
                        })
            
            if not plot_data:
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                ax.text(0.5, 0.5, "No data available for selected markers", 
                        ha='center', va='center', transform=ax.transAxes, fontsize=14)
                self.canvas.draw()
                return
            
            df_plot = pd.DataFrame(plot_data)
            
            # Get cluster colors using vivid colormap (same as UMAP)
            unique_cluster_ids = sorted(self.clustered_data['cluster'].unique())
            unique_cluster_names = [self._get_cluster_display_name(cid) for cid in unique_cluster_ids]
            colors = _get_vivid_colors(len(unique_cluster_ids))
            cluster_color_map = {cid: colors[i] for i, cid in enumerate(unique_cluster_ids)}
            cluster_name_color_map = {name: colors[i] for i, name in enumerate(unique_cluster_names)}
            
            # Check if statistical testing is enabled
            perform_stats = self.stats_test_checkbox.isChecked() if hasattr(self, 'stats_test_checkbox') else False
            
            # Get statistical test mode and reference cluster
            test_mode = 'pairwise'
            reference_cluster = None
            if perform_stats and hasattr(self, 'stats_mode_combo'):
                mode_text = self.stats_mode_combo.currentText()
                if mode_text == "One vs Others":
                    test_mode = 'one_vs_others'
                    if hasattr(self, 'stats_cluster_combo') and self.stats_cluster_combo.count() > 0:
                        # Get the cluster name from combo and find corresponding cluster ID
                        ref_cluster_name = self.stats_cluster_combo.currentText()
                        for cid in unique_cluster_ids:
                            if self._get_cluster_display_name(cid) == ref_cluster_name:
                                reference_cluster = ref_cluster_name  # Use cluster name for consistency
                                break
            
            # Clear previous results
            self.statistical_results = {}
            
            # Determine plot type
            plot_type = self.plot_type_combo.currentText()
            use_violin = plot_type == 'Violin Plot'
            
            # Use seaborn if available, otherwise matplotlib
            if _HAVE_SEABORN and len(available_markers) > 1:
                # Faceted plot for multiple markers
                n_markers = len(available_markers)
                n_cols = min(2, n_markers)
                n_rows = (n_markers + n_cols - 1) // n_cols
                
                # Create faceted plot with shared x-axis but not shared y-axis
                self.figure.clear()
                axes = []
                first_ax_per_col = {}  # Store first axis for each column
                for idx, marker in enumerate(available_markers):
                    row = idx // n_cols
                    col = idx % n_cols
                    pos = row * n_cols + col + 1
                    
                    # Create subplot with appropriate sharing
                    if row == 0:
                        # First row - no sharing needed
                        ax = self.figure.add_subplot(n_rows, n_cols, pos)
                        first_ax_per_col[col] = ax
                    else:
                        # Share x-axis with first subplot in same column
                        share_ax = first_ax_per_col[col]
                        ax = self.figure.add_subplot(n_rows, n_cols, pos, sharex=share_ax)
                    
                    axes.append(ax)
                    marker_data = df_plot[df_plot['Marker'] == marker]
                    
                    # Get display name for this marker
                    marker_display = marker_display_map[marker]
                    
                    # Create color palette ordered by cluster names in the plot
                    cluster_order = sorted(marker_data['Cluster'].unique())
                    palette = [cluster_name_color_map.get(cluster, 'gray') for cluster in cluster_order]
                    
                    if use_violin:
                        sns.violinplot(data=marker_data, x='Cluster', y='Value', ax=ax, hue='Cluster', palette=palette, order=cluster_order, legend=False)
                    else:
                        sns.boxplot(data=marker_data, x='Cluster', y='Value', ax=ax, hue='Cluster', palette=palette, order=cluster_order, legend=False)
                    
                    # Add statistical tests if enabled
                    if perform_stats:
                        # Prepare data for statistical testing - explicitly use unscaled data
                        cluster_data_dict = {}
                        for cluster_name in cluster_order:
                            # Find corresponding cluster ID
                            cluster_id = None
                            for cid in unique_cluster_ids:
                                if self._get_cluster_display_name(cid) == cluster_name:
                                    cluster_id = cid
                                    break
                            if cluster_id is not None:
                                # Extract raw (unscaled) values directly from plot_data_source
                                cluster_data_dict[cluster_name] = plot_data_source[
                                    plot_data_source['cluster'] == cluster_id
                                ][marker].dropna().values
                        
                        # Perform statistical tests
                        test_results = self._perform_pairwise_tests(cluster_data_dict, cluster_order, 
                                                                     mode=test_mode, reference_cluster=reference_cluster)
                        
                        # Store results for export
                        self.statistical_results[marker] = test_results
                        
                        # Draw significance bars
                        if test_results:
                            # Get y-axis limits
                            y_min, y_max = ax.get_ylim()
                            y_range = y_max - y_min
                            bar_y_start = y_max + 0.05 * y_range
                            bar_spacing = 0.05 * y_range
                            
                            # Group bars by y position to avoid overlap
                            bar_groups = {}
                            for cluster1, cluster2, p_val, adj_p_val in test_results:
                                if adj_p_val < 0.05:  # Only show significant results
                                    x1 = cluster_order.index(cluster1)
                                    x2 = cluster_order.index(cluster2)
                                    stars = self._get_significance_stars(adj_p_val)
                                    
                                    # Find a suitable y position
                                    y_pos = bar_y_start
                                    key = (x1, x2)
                                    if key in bar_groups:
                                        y_pos = bar_groups[key] + bar_spacing
                                    else:
                                        # Check for overlap with existing bars
                                        for existing_key, existing_y in bar_groups.items():
                                            ex1, ex2 = existing_key
                                            # Check if bars overlap
                                            if not (x2 < ex1 or x1 > ex2):
                                                y_pos = max(y_pos, existing_y + bar_spacing)
                                    
                                    bar_groups[key] = y_pos
                                    # Seaborn uses 0-based positions for categorical data
                                    self._draw_significance_bar(ax, x1, x2, y_pos, stars)
                            
                            # Adjust y-axis limits to accommodate significance bars
                            if bar_groups:
                                max_bar_y = max(bar_groups.values()) if bar_groups else bar_y_start
                                ax.set_ylim(y_min, max_bar_y + 0.1 * y_range)
                        
                        # Add significance legend (only on first subplot and if stats are enabled)
                        if idx == 0 and perform_stats:
                            legend_text = "Significance (BH adjusted):\n* p<0.05, ** p<0.01, *** p<0.001"
                            ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, 
                                   fontsize=7, verticalalignment='top', horizontalalignment='left',
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
                    
                    ax.set_title(marker_display, fontsize=10)
                    if row == n_rows - 1:  # Only show xlabel on bottom row
                        ax.set_xlabel('Cluster', fontsize=9)
                    else:
                        ax.set_xlabel('')
                        # Hide x-axis tick labels for non-bottom rows (sharex handles this, but ensure it)
                        plt.setp(ax.get_xticklabels(), visible=False)
                    ax.set_ylabel('Expression Value', fontsize=9)
                    if row == n_rows - 1:  # Only rotate x-axis labels on bottom row
                        ax.tick_params(axis='x', rotation=45, labelsize=8)
                    else:
                        ax.tick_params(axis='x', labelsize=8)
                    ax.tick_params(axis='y', labelsize=8)
                
                self.figure.tight_layout()
                
            elif _HAVE_SEABORN and len(available_markers) == 1:
                # Single marker with seaborn
                marker = available_markers[0]
                marker_display = marker_display_map[marker]
                marker_data = df_plot[df_plot['Marker'] == marker]
                
                ax = self.figure.add_subplot(111)
                
                # Create color palette ordered by cluster names in the plot
                cluster_order = sorted(marker_data['Cluster'].unique())
                palette = [cluster_name_color_map.get(cluster, 'gray') for cluster in cluster_order]
                
                if use_violin:
                    sns.violinplot(data=marker_data, x='Cluster', y='Value', ax=ax, palette=palette, order=cluster_order)
                else:
                    sns.boxplot(data=marker_data, x='Cluster', y='Value', ax=ax, palette=palette, order=cluster_order)
                
                # Add statistical tests if enabled
                if perform_stats:
                    # Prepare data for statistical testing - explicitly use unscaled data
                    cluster_data_dict = {}
                    for cluster_name in cluster_order:
                        # Find corresponding cluster ID
                        cluster_id = None
                        for cid in unique_cluster_ids:
                            if self._get_cluster_display_name(cid) == cluster_name:
                                cluster_id = cid
                                break
                        if cluster_id is not None:
                            # Extract raw (unscaled) values directly from plot_data_source
                            cluster_data_dict[cluster_name] = plot_data_source[
                                plot_data_source['cluster'] == cluster_id
                            ][marker].dropna().values
                    
                    # Perform statistical tests
                    test_results = self._perform_pairwise_tests(cluster_data_dict, cluster_order,
                                                                 mode=test_mode, reference_cluster=reference_cluster)
                    
                    # Store results for export
                    self.statistical_results[marker] = test_results
                    
                    # Draw significance bars
                    if test_results:
                        # Get y-axis limits
                        y_min, y_max = ax.get_ylim()
                        y_range = y_max - y_min
                        bar_y_start = y_max + 0.05 * y_range
                        bar_spacing = 0.05 * y_range
                        
                        # Group bars by y position to avoid overlap
                        bar_groups = {}
                        for cluster1, cluster2, p_val, adj_p_val in test_results:
                            if adj_p_val < 0.05:  # Only show significant results
                                x1 = cluster_order.index(cluster1)
                                x2 = cluster_order.index(cluster2)
                                stars = self._get_significance_stars(adj_p_val)
                                
                                # Find a suitable y position
                                y_pos = bar_y_start
                                key = (x1, x2)
                                if key in bar_groups:
                                    y_pos = bar_groups[key] + bar_spacing
                                else:
                                    # Check for overlap with existing bars
                                    for existing_key, existing_y in bar_groups.items():
                                        ex1, ex2 = existing_key
                                        # Check if bars overlap
                                        if not (x2 < ex1 or x1 > ex2):
                                            y_pos = max(y_pos, existing_y + bar_spacing)
                                
                                    bar_groups[key] = y_pos
                                    # Matplotlib boxplot uses the positions we set (0-based)
                                    self._draw_significance_bar(ax, x1, x2, y_pos, stars)
                        
                        # Adjust y-axis limits to accommodate significance bars
                        if bar_groups:
                            max_bar_y = max(bar_groups.values()) if bar_groups else bar_y_start
                            ax.set_ylim(y_min, max_bar_y + 0.1 * y_range)
                    
                    # Add significance legend (if stats are enabled)
                    if perform_stats:
                        legend_text = "Significance (BH adjusted):\n* p<0.05, ** p<0.01, *** p<0.001"
                        ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, 
                               fontsize=7, verticalalignment='top', horizontalalignment='left',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
                
                ax.set_title(marker_display, fontsize=12)
                ax.set_xlabel('Cluster', fontsize=10)
                ax.set_ylabel('Expression Value', fontsize=10)
                ax.tick_params(axis='x', rotation=45, labelsize=9)
                self.figure.tight_layout()
                
            else:
                # Fallback to matplotlib if seaborn not available
                n_markers = len(available_markers)
                n_cols = min(2, n_markers)
                n_rows = (n_markers + n_cols - 1) // n_cols
                
                for idx, marker in enumerate(available_markers):
                    ax = self.figure.add_subplot(n_rows, n_cols, idx + 1)
                    marker_display = marker_display_map[marker]
                    marker_data = df_plot[df_plot['Marker'] == marker]
                    
                    # Group data by cluster
                    cluster_values = {}
                    cluster_order = sorted(marker_data['Cluster'].unique())
                    for cluster in cluster_order:
                        cluster_values[cluster] = marker_data[
                            marker_data['Cluster'] == cluster
                        ]['Value'].values
                    
                    # Get colors for clusters
                    cluster_colors = [cluster_name_color_map.get(cluster, 'gray') for cluster in cluster_order]
                    
                    # Create boxplot or violin-like plot
                    if use_violin:
                        # Simple violin plot approximation with KDE
                        positions = range(len(cluster_values))
                        cluster_names = list(cluster_values.keys())
                        
                        for i, (cluster, values) in enumerate(cluster_values.items()):
                            if len(values) > 0:
                                # Use kde for violin shape approximation
                                from scipy.stats import gaussian_kde
                                try:
                                    kde = gaussian_kde(values)
                                    y_range = np.linspace(values.min(), values.max(), 100)
                                    density = kde(y_range)
                                    # Normalize density for width
                                    density = density / density.max() * 0.3
                                    ax.fill_betweenx(y_range, i - density, i + density, 
                                                     alpha=0.6, color=cluster_colors[i])
                                except:
                                    # Fallback to histogram if kde fails
                                    parts = ax.violinplot([values], positions=[i], widths=0.6, showmeans=True)
                                    for pc in parts['bodies']:
                                        pc.set_facecolor(cluster_colors[i])
                    else:
                        # Boxplot
                        positions = range(len(cluster_values))
                        cluster_names = list(cluster_values.keys())
                        bp = ax.boxplot(list(cluster_values.values()), positions=positions, widths=0.6)
                        # Color the boxplot elements
                        for i, patch in enumerate(bp['boxes']):
                            patch.set_facecolor(cluster_colors[i])
                            patch.set_alpha(0.7)
                        for median in bp['medians']:
                            median.set_color('black')
                        for whisker in bp['whiskers']:
                            whisker.set_color('black')
                        for cap in bp['caps']:
                            cap.set_color('black')
                    
                    # Add statistical tests if enabled
                    if perform_stats:
                        # Prepare data for statistical testing - explicitly use unscaled data
                        cluster_data_dict = {}
                        for cluster_name in cluster_order:
                            # Find corresponding cluster ID
                            cluster_id = None
                            for cid in unique_cluster_ids:
                                if self._get_cluster_display_name(cid) == cluster_name:
                                    cluster_id = cid
                                    break
                            if cluster_id is not None:
                                # Extract raw (unscaled) values directly from plot_data_source
                                cluster_data_dict[cluster_name] = plot_data_source[
                                    plot_data_source['cluster'] == cluster_id
                                ][marker].dropna().values
                        
                        # Perform statistical tests
                        test_results = self._perform_pairwise_tests(cluster_data_dict, cluster_order,
                                                                     mode=test_mode, reference_cluster=reference_cluster)
                        
                        # Store results for export
                        self.statistical_results[marker] = test_results
                        
                        # Draw significance bars
                        if test_results:
                            # Get y-axis limits
                            y_min, y_max = ax.get_ylim()
                            y_range = y_max - y_min
                            bar_y_start = y_max + 0.05 * y_range
                            bar_spacing = 0.05 * y_range
                            
                            # Group bars by y position to avoid overlap
                            bar_groups = {}
                            for cluster1, cluster2, p_val, adj_p_val in test_results:
                                if adj_p_val < 0.05:  # Only show significant results
                                    x1 = cluster_order.index(cluster1)
                                    x2 = cluster_order.index(cluster2)
                                    stars = self._get_significance_stars(adj_p_val)
                                    
                                    # Find a suitable y position
                                    y_pos = bar_y_start
                                    key = (x1, x2)
                                    if key in bar_groups:
                                        y_pos = bar_groups[key] + bar_spacing
                                    else:
                                        # Check for overlap with existing bars
                                        for existing_key, existing_y in bar_groups.items():
                                            ex1, ex2 = existing_key
                                            # Check if bars overlap
                                            if not (x2 < ex1 or x1 > ex2):
                                                y_pos = max(y_pos, existing_y + bar_spacing)
                                    
                                    bar_groups[key] = y_pos
                                    # Seaborn uses 0-based positions for categorical data
                                    self._draw_significance_bar(ax, x1, x2, y_pos, stars)
                            
                            # Adjust y-axis limits to accommodate significance bars
                            if bar_groups:
                                max_bar_y = max(bar_groups.values()) if bar_groups else bar_y_start
                                ax.set_ylim(y_min, max_bar_y + 0.1 * y_range)
                        
                        # Add significance legend (only on first subplot and if stats are enabled)
                        if idx == 0 and perform_stats:
                            legend_text = "Significance (BH adjusted):\n* p<0.05, ** p<0.01, *** p<0.001"
                            ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, 
                                   fontsize=7, verticalalignment='top', horizontalalignment='left',
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
                    
                    ax.set_xticks(range(len(cluster_values)))
                    ax.set_xticklabels(cluster_names, rotation=45, ha='right')
                    ax.set_title(marker_display, fontsize=10)
                    ax.set_xlabel('Cluster', fontsize=9)
                    ax.set_ylabel('Expression Value', fontsize=9)
                
                self.figure.tight_layout()
            
            self.canvas.draw()
            
            # Enable export button if statistical results are available
            if hasattr(self, 'stats_export_btn'):
                self.stats_export_btn.setEnabled(perform_stats and len(self.statistical_results) > 0)
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error creating boxplot/violin plot: {str(e)}")
            import traceback
            traceback.print_exc()

    def _export_statistical_results(self):
        """Export statistical test results to CSV file."""
        if not self.statistical_results:
            QtWidgets.QMessageBox.warning(self, "No Results", "No statistical test results available to export.")
            return
        
        # Get default filename
        default = "statistical_test_results.csv"
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Statistical Test Results", default,
            "CSV Files (*.csv)"
        )
        if not file_path:
            return
        
        try:
            # Prepare data for export
            export_data = []
            for marker, results in self.statistical_results.items():
                for cluster1, cluster2, p_value, adj_p_value in results:
                    export_data.append({
                        'Marker': marker,
                        'Cluster_1': cluster1,
                        'Cluster_2': cluster2,
                        'P_value': p_value,
                        'Adjusted_P_value_BH': adj_p_value,
                        'Significant': 'Yes' if adj_p_value < 0.05 else 'No',
                        'Significance_level': self._get_significance_stars(adj_p_value)
                    })
            
            if not export_data:
                QtWidgets.QMessageBox.warning(self, "No Results", "No statistical test results to export.")
                return
            
            # Create DataFrame and save
            df_export = pd.DataFrame(export_data)
            df_export.to_csv(file_path, index=False)
            
            # Show success message
            n_tests = len(export_data)
            n_significant = sum(1 for r in export_data if r['Significant'] == 'Yes')
            summary = f"Exported {n_tests} statistical test results"
            if n_significant > 0:
                summary += f"\n{n_significant} significant comparisons (adjusted p < 0.05)"
            
            QtWidgets.QMessageBox.information(self, "Export Success", 
                                            f"Statistical test results saved to:\n{file_path}\n\n{summary}")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", f"Error exporting statistical results: {str(e)}")

    def _open_k_range_dialog(self):
        """Open dialog to search over k range and plot elbow/silhouette scores."""
        clustering_type = self.clustering_type.currentText()
        if clustering_type not in ["Hierarchical", "K-means"]:
            QtWidgets.QMessageBox.warning(self, "Invalid Method", 
                                         "K-range search is only available for Hierarchical and K-means clustering.")
            return
        
        # Check if we have prepared data
        if not hasattr(self, 'feature_dataframe') or self.feature_dataframe is None:
            QtWidgets.QMessageBox.warning(self, "No Data", "Please select features first.")
            return
        
        # Get feature columns
        feature_cols = self._select_feature_columns(self.feature_dataframe)
        if not feature_cols:
            QtWidgets.QMessageBox.warning(self, "No Features", "No numeric features available.")
            return
        
        # Create dialog
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Find Optimal K")
        dlg.setMinimumSize(600, 400)
        layout = QtWidgets.QVBoxLayout(dlg)
        
        # Input parameters
        params_layout = QtWidgets.QFormLayout()
        
        # K range
        k_min_spin = QtWidgets.QSpinBox()
        k_min_spin.setRange(2, 20)
        k_min_spin.setValue(2)
        k_max_spin = QtWidgets.QSpinBox()
        k_max_spin.setRange(2, 30)
        k_max_spin.setValue(10)
        k_range_layout = QtWidgets.QHBoxLayout()
        k_range_layout.addWidget(k_min_spin)
        k_range_layout.addWidget(QtWidgets.QLabel("to"))
        k_range_layout.addWidget(k_max_spin)
        params_layout.addRow("K range:", k_range_layout)
        
        # Linkage method (for hierarchical only)
        linkage_combo = None
        if clustering_type == "Hierarchical":
            linkage_combo = QtWidgets.QComboBox()
            linkage_combo.addItems(["ward", "complete", "average", "single"])
            linkage_combo.setCurrentText(self.hierarchical_method.currentText())
            params_layout.addRow("Linkage method:", linkage_combo)
        
        layout.addLayout(params_layout)
        
        # Progress bar
        progress = QtWidgets.QProgressBar()
        progress.setRange(0, 100)
        progress.setValue(0)
        layout.addWidget(progress)
        
        # Results label
        results_label = QtWidgets.QLabel("")
        layout.addWidget(results_label)
        
        # Plot area
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        fig = Figure(figsize=(10, 6))
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        run_btn = QtWidgets.QPushButton("Run Analysis")
        close_btn = QtWidgets.QPushButton("Close")
        button_layout.addWidget(run_btn)
        button_layout.addStretch()
        button_layout.addWidget(close_btn)
        layout.addLayout(button_layout)
        
        def run_analysis():
            from openimc.core import cluster
            
            k_min = k_min_spin.value()
            k_max = k_max_spin.value()
            if k_min >= k_max:
                QtWidgets.QMessageBox.warning(dlg, "Invalid Range", "K min must be less than K max.")
                return
            
            k_values = list(range(k_min, k_max + 1))
            progress.setRange(0, len(k_values))
            
            # Get scaling method (matching main clustering)
            scaling_method = self.clustering_scaling_combo.currentText()
            scaling_map = {
                "None (no scaling)": "none",
                "Z-score": "zscore",
                "MAD (Median Absolute Deviation)": "mad"
            }
            selected_scaling = scaling_map.get(scaling_method, "zscore")
            
            # Get full dataframe (core.cluster will handle column selection and scaling)
            full_data = self.feature_dataframe.copy()
            
            if full_data.empty:
                QtWidgets.QMessageBox.warning(dlg, "No Data", "No data available.")
                return
            
            seed = self.seed_spinbox.value()
            inertias = []
            silhouette_scores = []
            
            try:
                for idx, k in enumerate(k_values):
                    progress.setValue(idx + 1)
                    QtWidgets.QApplication.processEvents()
                    
                    # Use core.cluster for consistency with main clustering
                    if clustering_type == "K-means":
                        # Use core.cluster for K-means
                        clustered_df = cluster(
                            features_df=full_data,
                            method="kmeans",
                            columns=feature_cols,
                            scaling=selected_scaling,
                            output_path=None,
                            n_clusters=k,
                            seed=seed,
                            n_init=10
                        )
                        labels = clustered_df['cluster'].values - 1  # Convert back to 0-based for calculations
                        
                    else:  # Hierarchical
                        linkage_method = linkage_combo.currentText() if linkage_combo else "ward"
                        # Use core.cluster for hierarchical
                        clustered_df = cluster(
                            features_df=full_data,
                            method="hierarchical",
                            columns=feature_cols,
                            scaling=selected_scaling,
                            output_path=None,
                            n_clusters=k,
                            linkage=linkage_method,
                            seed=seed
                        )
                        labels = clustered_df['cluster'].values - 1  # Convert back to 0-based for calculations
                    
                    # Get scaled data for WCSS/inertia and silhouette calculations
                    # (core.cluster returns original dataframe, but clustering was done on scaled data)
                    # We need to match the exact data that was used for clustering
                    data_for_calc = full_data[feature_cols].copy()
                    
                    # Apply same preprocessing as core.cluster (handle missing/infinite, then scale)
                    data_for_calc = data_for_calc.replace([np.inf, -np.inf], np.nan)
                    data_for_calc = data_for_calc.fillna(data_for_calc.median(numeric_only=True))
                    data_scaled = self._apply_scaling(data_for_calc, selected_scaling)
                    data_scaled = data_scaled.replace([np.inf, -np.inf], np.nan)
                    # Drop rows/cols that would be dropped by core.cluster
                    data_scaled = data_scaled.dropna(axis=0, how='any').dropna(axis=1, how='any')
                    data_scaled = data_scaled.fillna(0)
                    
                    # Filter to only rows that have valid cluster labels (not NaN/0 from dropped rows)
                    # core.cluster maps labels back, so rows that were dropped will have cluster=0 or NaN
                    valid_cluster_mask = clustered_df['cluster'].notna() & (clustered_df['cluster'] > 0)
                    valid_indices = clustered_df.index[valid_cluster_mask]
                    
                    # Align data_scaled with valid indices (some rows may have been dropped)
                    data_scaled_valid = data_scaled.reindex(valid_indices).dropna()
                    labels_valid = clustered_df.loc[data_scaled_valid.index, 'cluster'].values - 1
                    
                    # Calculate WCSS/inertia from cluster labels and scaled data
                    wcss = 0
                    for cluster_id in np.unique(labels_valid):
                        if cluster_id < 0:  # Skip noise/unassigned
                            continue
                        cluster_mask = labels_valid == cluster_id
                        cluster_data = data_scaled_valid.values[cluster_mask]
                        if len(cluster_data) > 0:
                            centroid = cluster_data.mean(axis=0)
                            wcss += np.sum((cluster_data - centroid) ** 2)
                    inertias.append(wcss)
                    
                    # Calculate silhouette score (using scaled data and 0-based labels)
                    if not _HAVE_SKLEARN:
                        silhouette_scores.append(0)
                    else:
                        # Filter out noise points (label < 0) for silhouette calculation
                        valid_mask = labels_valid >= 0
                        if valid_mask.sum() > 1 and len(np.unique(labels_valid[valid_mask])) > 1:
                            data_for_silhouette = data_scaled_valid.values[valid_mask]
                            labels_for_silhouette = labels_valid[valid_mask]
                            sil_score = silhouette_score(data_for_silhouette, labels_for_silhouette)
                        else:
                            sil_score = 0
                        silhouette_scores.append(sil_score)
                
                # Find optimal k
                # Elbow: find point with maximum curvature (second derivative)
                if len(inertias) > 2:
                    # Calculate rate of change
                    deltas = np.diff(inertias)
                    deltas2 = np.diff(deltas)
                    if len(deltas2) > 0:
                        elbow_idx = np.argmax(np.abs(deltas2)) + 1
                        optimal_k_elbow = k_values[elbow_idx]
                    else:
                        optimal_k_elbow = k_values[np.argmin(inertias)]
                else:
                    optimal_k_elbow = k_values[0]
                
                # Silhouette: maximum score
                optimal_k_silhouette = k_values[np.argmax(silhouette_scores)]
                
                # Plot results
                fig.clear()
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)
                
                # Elbow plot
                ax1.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
                ax1.axvline(x=optimal_k_elbow, color='r', linestyle='--', alpha=0.7, label=f'Suggested (elbow): k={optimal_k_elbow}')
                ax1.set_xlabel('Number of clusters (k)', fontsize=10)
                ax1.set_ylabel('WCSS / Inertia', fontsize=10)
                ax1.set_title('Elbow Method', fontsize=12)
                ax1.grid(True, alpha=0.3)
                ax1.legend()
                
                # Silhouette plot
                ax2.plot(k_values, silhouette_scores, 'go-', linewidth=2, markersize=8)
                ax2.axvline(x=optimal_k_silhouette, color='r', linestyle='--', alpha=0.7, label=f'Suggested (silhouette): k={optimal_k_silhouette}')
                ax2.set_xlabel('Number of clusters (k)', fontsize=10)
                ax2.set_ylabel('Silhouette Score', fontsize=10)
                ax2.set_title('Silhouette Score', fontsize=12)
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                
                fig.tight_layout()
                canvas.draw()
                
                # Update results label
                results_text = (f"Optimal k (elbow method): {optimal_k_elbow}\n"
                              f"Optimal k (silhouette): {optimal_k_silhouette}\n"
                              f"Max silhouette score: {max(silhouette_scores):.3f}")
                results_label.setText(results_text)
                
                # Update n_clusters spinbox with optimal value
                if optimal_k_silhouette >= 2:
                    self.n_clusters.setValue(optimal_k_silhouette)
                
            except Exception as e:
                QtWidgets.QMessageBox.critical(dlg, "Error", f"Error during k-range analysis: {str(e)}")
                import traceback
                traceback.print_exc()
            finally:
                progress.setValue(len(k_values))
        
        run_btn.clicked.connect(run_analysis)
        close_btn.clicked.connect(dlg.accept)
        
        dlg.exec_()
    
    def _open_gating_dialog(self):
        """Open gating rules editor and apply on save."""
        # Allow selection among intensity features by default
        marker_cols = [col for col in self.feature_dataframe.columns
                       if any(col.endswith(suffix) for suffix in ['_mean', '_median', '_std', '_mad', '_p10', '_p90', '_integrated', '_frac_pos'])]
        dlg = GatingRulesDialog(self.gating_rules, marker_cols, self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.gating_rules = dlg.get_rules()
            self._apply_gating_rules()
            
            # Log gating operation
            logger = get_logger()
            acquisitions = []
            if self.feature_dataframe is not None and 'acquisition_id' in self.feature_dataframe.columns:
                acquisitions = list(self.feature_dataframe['acquisition_id'].unique())
            
            # Get source file name from parent if available
            source_file = None
            if self.parent() is not None and hasattr(self.parent(), 'current_path'):
                import os
                source_file = os.path.basename(self.parent().current_path) if self.parent().current_path else None
            
            logger.log_gating(
                gating_rules=self.gating_rules,
                acquisitions=acquisitions,
                notes=f"Applied {len(self.gating_rules)} gating rules",
                source_file=source_file
            )
            
            QtWidgets.QMessageBox.information(self, "Gating Applied", "Manual phenotypes assigned using gating rules.")
            # If user just applied manual gates, default heatmap source to Manual Gates for immediate view
            if hasattr(self, 'heatmap_source_combo'):
                self.heatmap_source_combo.setCurrentText('Manual Gates')

    def _apply_gating_rules(self):
        """Evaluate gating rules and create/update 'manual_phenotype' column on feature_dataframe."""
        if not self.gating_rules:
            return
        # Initialize column
        if 'manual_phenotype' not in self.feature_dataframe.columns:
            self.feature_dataframe['manual_phenotype'] = ''
        assigned = pd.Series(self.feature_dataframe['manual_phenotype'] != '', index=self.feature_dataframe.index)
        # Evaluate rules in order
        for rule in self.gating_rules:
            name = rule.get('name', '').strip()
            logic = rule.get('logic', 'AND').upper()
            conditions = rule.get('conditions', [])
            if not name or not conditions:
                continue
            masks = []
            for cond in conditions:
                col = cond.get('column')
                op = cond.get('op', '>')
                thr = cond.get('threshold', 0)
                if col not in self.feature_dataframe.columns:
                    continue
                series = self.feature_dataframe[col]
                if op == '>':
                    mask = series > thr
                elif op == '>=':
                    mask = series >= thr
                elif op == '<':
                    mask = series < thr
                elif op == '<=':
                    mask = series <= thr
                elif op == '==':
                    mask = series == thr
                elif op == '!=':
                    mask = series != thr
                else:
                    continue
                masks.append(mask.fillna(False))
            if not masks:
                continue
            if logic == 'OR':
                rule_mask = masks[0]
                for m in masks[1:]:
                    rule_mask = rule_mask | m
            else:
                rule_mask = masks[0]
                for m in masks[1:]:
                    rule_mask = rule_mask & m
            # Assign where not already assigned
            to_assign = rule_mask & (~assigned)
            self.feature_dataframe.loc[to_assign, 'manual_phenotype'] = name
            assigned = assigned | to_assign
        # If clustered_data exists, align and copy manual phenotype into it for plotting
        if self.clustered_data is not None:
            if 'manual_phenotype' not in self.clustered_data.columns:
                self.clustered_data['manual_phenotype'] = ''
            self.clustered_data.loc[:, 'manual_phenotype'] = self.feature_dataframe.loc[self.clustered_data.index, 'manual_phenotype'].values
        # Update color options and refresh plot
        self._populate_color_by_options()
        if getattr(self, 'umap_embedding', None) is not None:
            self._create_umap_plot()
        else:
            self._create_heatmap()

    def _save_gating_rules(self):
        """Save current gating rules to JSON."""
        import json
        if not self.gating_rules:
            QtWidgets.QMessageBox.information(self, "No Rules", "There are no gating rules to save.")
            return
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Gating Rules", "gating_rules.json", "JSON Files (*.json)"
        )
        if not file_path:
            return
        try:
            with open(file_path, 'w') as f:
                json.dump(self.gating_rules, f, indent=2)
            QtWidgets.QMessageBox.information(self, "Saved", f"Gating rules saved to: {file_path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Error", f"Error saving gating rules: {str(e)}")

    def _load_gating_rules(self):
        """Load gating rules from JSON and apply."""
        import json
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Gating Rules", "", "JSON Files (*.json)"
        )
        if not file_path:
            return
        try:
            with open(file_path, 'r') as f:
                rules = json.load(f)
            if isinstance(rules, list):
                self.gating_rules = rules
                self._apply_gating_rules()
                QtWidgets.QMessageBox.information(self, "Loaded", f"Loaded {len(self.gating_rules)} gating rules.")
            else:
                raise ValueError("JSON must be a list of rules")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load Error", f"Error loading gating rules: {str(e)}")

    def _open_annotation_dialog(self):
        """Open a dialog to annotate clusters with phenotype names. Includes save/load controls."""
        if self.clustered_data is None:
            QtWidgets.QMessageBox.warning(self, "No Clusters", "Please run clustering first.")
            return
        unique_clusters = sorted(self.clustered_data['cluster'].unique())
        # Build and show dialog
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Annotate Phenotypes")
        v = QtWidgets.QVBoxLayout(dlg)
        form = QtWidgets.QFormLayout()
        editors = {}
        for cid in unique_clusters:
            le = QtWidgets.QLineEdit()
            if cid in self.cluster_annotation_map:
                le.setText(self.cluster_annotation_map[cid])
            form.addRow(f"Cluster {cid}", le)
            editors[cid] = le
        v.addLayout(form)
        # (Load/Save removed)
        # LLM assist row
        llm_row = QtWidgets.QHBoxLayout()
        llm_btn = QtWidgets.QPushButton("Suggest phenotypes with LLM…")
        llm_btn.setToolTip("Requires OpenAI API key. Uses per-cluster marker statistics.")
        llm_row.addWidget(llm_btn)
        llm_row.addStretch()
        v.addLayout(llm_row)
        btns = QtWidgets.QHBoxLayout()
        ok = QtWidgets.QPushButton("Apply")
        cancel = QtWidgets.QPushButton("Cancel")
        ok.clicked.connect(dlg.accept)
        cancel.clicked.connect(dlg.reject)
        btns.addStretch()
        btns.addWidget(ok)
        btns.addWidget(cancel)
        v.addLayout(btns)

        # (Load/Save handlers removed)
        def open_llm_dialog():
            def apply_names(display_name_map, backend_name_map):
                # Set display names in the UI
                for cid, name in display_name_map.items():
                    if cid in editors and isinstance(name, str):
                        editors[cid].setText(name)
                # Store backend names for CSV export
                self.cluster_backend_names.update(backend_name_map)
            d = PhenotypeSuggestionDialog(self, unique_clusters, apply_names, self.llm_phenotype_cache, self.normalization_config)
            d.exec_()
        llm_btn.clicked.connect(open_llm_dialog)

        # Make the dialog wider for better usability
        dlg.resize(500, dlg.sizeHint().height())

        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            # Save mapping from editors
            self.cluster_annotation_map = {
                cid: editors[cid].text().strip() for cid in unique_clusters if editors[cid].text().strip()
            }
            self._apply_cluster_annotations()
            
            # Log annotation operation
            logger = get_logger()
            acquisitions = []
            if self.clustered_data is not None and 'acquisition_id' in self.clustered_data.columns:
                acquisitions = list(self.clustered_data['acquisition_id'].unique())
            
            # Get source file name from parent if available
            source_file = None
            if self.parent() is not None and hasattr(self.parent(), 'current_path'):
                import os
                source_file = os.path.basename(self.parent().current_path) if self.parent().current_path else None
            
            logger.log_class_annotation(
                annotation_map=self.cluster_annotation_map,
                method="manual",
                acquisitions=acquisitions,
                notes=f"Annotated {len(self.cluster_annotation_map)} clusters",
                source_file=source_file
            )
            
            QtWidgets.QMessageBox.information(self, "Annotations Applied", "Cluster annotations have been applied.")

    def _open_merge_clusters_dialog(self):
        """Open a dialog to merge two clusters."""
        if self.clustered_data is None:
            QtWidgets.QMessageBox.warning(self, "No Clusters", "Please run clustering first.")
            return
        
        unique_clusters = sorted([c for c in self.clustered_data['cluster'].unique() if c != 0])  # Exclude noise cluster
        if len(unique_clusters) < 2:
            QtWidgets.QMessageBox.warning(self, "Not Enough Clusters", "At least two clusters (excluding noise) are required to merge.")
            return
        
        # Build and show dialog
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Merge Clusters")
        dlg.setMinimumWidth(400)
        v = QtWidgets.QVBoxLayout(dlg)
        
        # Instructions
        instructions = QtWidgets.QLabel("Select two clusters to merge. The second cluster will be merged into the first.")
        instructions.setWordWrap(True)
        v.addWidget(instructions)
        
        # First cluster selector
        form = QtWidgets.QFormLayout()
        cluster_options = [self._get_cluster_display_name(cid) for cid in unique_clusters]
        merge_cluster1_combo = QtWidgets.QComboBox()
        merge_cluster1_combo.addItems(cluster_options)
        form.addRow("First cluster (target):", merge_cluster1_combo)
        
        # Second cluster selector
        merge_cluster2_combo = QtWidgets.QComboBox()
        merge_cluster2_combo.addItems(cluster_options)
        form.addRow("Second cluster (to merge):", merge_cluster2_combo)
        v.addLayout(form)
        
        # Buttons
        btns = QtWidgets.QHBoxLayout()
        reset_btn = QtWidgets.QPushButton("Reset All Merges")
        reset_btn.setToolTip("Reset all cluster merges to original assignments")
        reset_btn.clicked.connect(lambda: self._reset_cluster_merges_from_dialog(dlg))
        ok = QtWidgets.QPushButton("Merge")
        cancel = QtWidgets.QPushButton("Cancel")
        ok.clicked.connect(dlg.accept)
        cancel.clicked.connect(dlg.reject)
        btns.addWidget(reset_btn)
        btns.addStretch()
        btns.addWidget(ok)
        btns.addWidget(cancel)
        v.addLayout(btns)
        
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            # Get selected cluster IDs
            idx1 = merge_cluster1_combo.currentIndex()
            idx2 = merge_cluster2_combo.currentIndex()
            
            if idx1 == idx2:
                QtWidgets.QMessageBox.warning(self, "Invalid Selection", "Please select two different clusters.")
                return
            
            cluster1_id = unique_clusters[idx1]
            cluster2_id = unique_clusters[idx2]
            
            # Perform merge
            self._merge_clusters(cluster1_id, cluster2_id)
            
            # Refresh plots
            self._refresh_all_plots()
            
            QtWidgets.QMessageBox.information(self, "Clusters Merged", 
                f"Cluster {cluster2_id} has been merged into cluster {cluster1_id}.")
    
    def _merge_clusters(self, target_cluster_id, source_cluster_id):
        """Merge source_cluster_id into target_cluster_id."""
        if self.clustered_data is None:
            return
        
        # Update cluster labels in clustered_data
        mask = self.clustered_data['cluster'] == source_cluster_id
        self.clustered_data.loc[mask, 'cluster'] = target_cluster_id
        
        # Update cluster labels in clustered_data_unscaled if it exists
        if self.clustered_data_unscaled is not None and 'cluster' in self.clustered_data_unscaled.columns:
            mask_unscaled = self.clustered_data_unscaled['cluster'] == source_cluster_id
            self.clustered_data_unscaled.loc[mask_unscaled, 'cluster'] = target_cluster_id
        
        # Update feature_dataframe if cluster column exists
        if 'cluster' in self.feature_dataframe.columns:
            mask_feature = self.feature_dataframe['cluster'] == source_cluster_id
            self.feature_dataframe.loc[mask_feature, 'cluster'] = target_cluster_id
        
        # Update cluster_annotation_map: merge annotations if both have them
        if source_cluster_id in self.cluster_annotation_map:
            source_annotation = self.cluster_annotation_map[source_cluster_id]
            if target_cluster_id not in self.cluster_annotation_map:
                # If target doesn't have annotation, use source's annotation
                self.cluster_annotation_map[target_cluster_id] = source_annotation
            # Remove source cluster from annotation map
            del self.cluster_annotation_map[source_cluster_id]
        
        # Update cluster_backend_names similarly
        if source_cluster_id in self.cluster_backend_names:
            source_backend_name = self.cluster_backend_names[source_cluster_id]
            if target_cluster_id not in self.cluster_backend_names:
                self.cluster_backend_names[target_cluster_id] = source_backend_name
            del self.cluster_backend_names[source_cluster_id]
        
        # Reapply cluster annotations to update cluster_phenotype column
        if self.cluster_annotation_map:
            self._apply_cluster_annotations()
    
    def _reset_cluster_merges(self):
        """Reset all cluster merges by restoring original cluster assignments."""
        if self.original_cluster_assignments is None or self.clustered_data is None:
            return
        
        # Restore original cluster assignments
        # Ensure indices match (they should since original_cluster_assignments was copied from clustered_data)
        if self.original_cluster_assignments.index.equals(self.clustered_data.index):
            self.clustered_data['cluster'] = self.original_cluster_assignments
        else:
            # If indices don't match, align them
            self.clustered_data['cluster'] = self.original_cluster_assignments.reindex(
                self.clustered_data.index, fill_value=0
            )
        
        # Update clustered_data_unscaled if it exists
        if self.clustered_data_unscaled is not None and 'cluster' in self.clustered_data_unscaled.columns:
            # clustered_data_unscaled should have the same index as clustered_data
            if self.clustered_data_unscaled.index.equals(self.clustered_data.index):
                self.clustered_data_unscaled['cluster'] = self.clustered_data['cluster']
            else:
                # If indices don't match, align them
                self.clustered_data_unscaled['cluster'] = self.clustered_data['cluster'].reindex(
                    self.clustered_data_unscaled.index, fill_value=0
                )
        
        # Update feature_dataframe if cluster column exists
        if 'cluster' in self.feature_dataframe.columns:
            # Restore original assignments for clustered cells
            matching_indices = self.clustered_data.index.intersection(self.feature_dataframe.index)
            if len(matching_indices) > 0:
                self.feature_dataframe.loc[matching_indices, 'cluster'] = self.clustered_data.loc[matching_indices, 'cluster']
        
        # Refresh plots
        self._refresh_all_plots()
    
    def _reset_cluster_merges_from_dialog(self, dialog):
        """Reset cluster merges from the merge dialog and close it."""
        self._reset_cluster_merges()
        dialog.accept()
        QtWidgets.QMessageBox.information(self, "Merges Reset", 
            "All cluster merges have been reset to original assignments.")
    
    def _refresh_all_plots(self):
        """Refresh all plots to reflect cluster changes."""
        current_view = self.view_combo.currentText() if hasattr(self, 'view_combo') else 'Heatmap'
        
        # Update stats cluster combo if it exists
        if hasattr(self, 'stats_cluster_combo'):
            self._update_stats_cluster_combo()
        
        # Refresh the current view
        self._on_view_changed(current_view)

    def _open_patient_annotation_dialog(self):
        """Open a dialog to customize patient/source file labels."""
        # Determine which column to use for patient annotation
        patient_col = None
        if hasattr(self, 'patient_annotation_column') and self.patient_annotation_column:
            patient_col = self.patient_annotation_column
        else:
            # Default priority order: standard columns first, then metadata
            for col in ['source_file', 'batch_group', 'source_well']:
                if col in self.feature_dataframe.columns:
                    patient_col = col
                    break
            
            # If no standard column, check metadata
            if patient_col is None:
                metadata_cols = self._get_metadata_columns(self.feature_dataframe)
                if metadata_cols:
                    # Prefer columns that might be batch identifiers
                    priority_metadata = [col for col in metadata_cols 
                                       if any(keyword in col.lower() for keyword in ['pid', 'patient', 'batch', 'sample', 'subject'])]
                    if priority_metadata:
                        patient_col = priority_metadata[0]
                    else:
                        patient_col = metadata_cols[0]
        
        if not patient_col or patient_col not in self.feature_dataframe.columns:
            QtWidgets.QMessageBox.warning(self, "No Patient Annotation Data", 
                                          "No patient annotation column (source_file, batch_group, or source_well) is available in the data.")
            return
        
        # Get unique values from the selected column
        unique_values = sorted([f for f in self.feature_dataframe[patient_col].unique() if pd.notna(f)])
        if not unique_values:
            QtWidgets.QMessageBox.warning(self, "No Values", 
                                          f"No values found in {patient_col} column.")
            return
        
        # Build and show dialog
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Customize Patient/Source File Labels")
        v = QtWidgets.QVBoxLayout(dlg)
        
        # Add instruction label
        instruction = QtWidgets.QLabel(f"Customize labels for each value in {patient_col}. Leave blank to use original value.")
        instruction.setWordWrap(True)
        v.addWidget(instruction)
        
        form = QtWidgets.QFormLayout()
        editors = {}
        for value in unique_values:
            le = QtWidgets.QLineEdit()
            # Use custom label if available, otherwise use the value
            if value in self.patient_annotation_map:
                le.setText(self.patient_annotation_map[value])
            else:
                # Use basename if it's a file path, otherwise use value as-is
                if os.sep in str(value) or '/' in str(value) or '\\' in str(value):
                    default_label = os.path.basename(str(value))
                else:
                    default_label = str(value)
                le.setText(default_label)
            # Display label for the form row
            if os.sep in str(value) or '/' in str(value) or '\\' in str(value):
                display_name = os.path.basename(str(value))
            else:
                display_name = str(value)
            form.addRow(f"{patient_col}:\n{display_name}", le)
            editors[value] = le
        v.addLayout(form)
        
        btns = QtWidgets.QHBoxLayout()
        ok = QtWidgets.QPushButton("Apply")
        cancel = QtWidgets.QPushButton("Cancel")
        ok.clicked.connect(dlg.accept)
        cancel.clicked.connect(dlg.reject)
        btns.addStretch()
        btns.addWidget(ok)
        btns.addWidget(cancel)
        v.addLayout(btns)

        # Make the dialog wider for better usability
        dlg.resize(600, dlg.sizeHint().height())

        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            # Save mapping from editors
            self.patient_annotation_map = {
                value: editors[value].text().strip() 
                for value in unique_values 
                if editors[value].text().strip()
            }
            # Refresh current view if patient labels are used
            view = self.view_combo.currentText() if hasattr(self, 'view_combo') else 'Heatmap'
            if view == 'Heatmap' and hasattr(self, 'patient_annotation_checkbox') and self.patient_annotation_checkbox.isChecked():
                self._show_heatmap()
            elif view == 'UMAP' and self.umap_embedding is not None:
                # Refresh UMAP plot to update patient labels in legend
                self._create_umap_plot()
            elif view == 't-SNE' and self.tsne_embedding is not None:
                # Refresh t-SNE plot to update patient labels in legend
                self._create_tsne_plot()
            QtWidgets.QMessageBox.information(self, "Labels Applied", "Patient labels have been applied.")

    def _open_plot_config_dialog(self):
        """Open the plot configuration dialog."""
        dlg = PlotConfigDialog(self, parent=self)
        dlg.exec_()

    def _open_feature_labels_dialog(self):
        """Open a dialog to customize feature labels (friendly names for y-axis ticks)."""
        if self.clustered_data is None:
            QtWidgets.QMessageBox.warning(self, "No Clustering", "Run clustering first to customize feature labels.")
            return
        
        # Get feature columns from clustered data
        feature_cols = [col for col in self.clustered_data.columns 
                       if col not in ['cluster', 'cluster_phenotype', 'cell_id', 'acquisition_id', 
                                     'source_file', 'source_well', 'manual_phenotype'] and 
                       not col.startswith('centroid_')]
        
        if not feature_cols:
            QtWidgets.QMessageBox.warning(self, "No Features", "No feature columns found in the data.")
            return
        
        # Build and show dialog
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Customize Feature Labels")
        dlg.resize(700, 600)
        v = QtWidgets.QVBoxLayout(dlg)
        
        # Add instruction label
        instruction = QtWidgets.QLabel("Set custom display names for features (e.g., 'Vimentin_mean' -> 'Mean Vimentin'). Leave blank to use original name.")
        instruction.setWordWrap(True)
        v.addWidget(instruction)
        
        # Create scroll area for many features
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(scroll_content)
        
        editors = {}
        for feature_name in sorted(feature_cols):
            le = QtWidgets.QLineEdit()
            # Use custom label if available, otherwise use original name
            if feature_name in self.feature_label_map:
                le.setText(self.feature_label_map[feature_name])
            else:
                le.setText(feature_name)
            form.addRow(feature_name, le)
            editors[feature_name] = le
        
        scroll.setWidget(scroll_content)
        v.addWidget(scroll)
        
        btns = QtWidgets.QHBoxLayout()
        ok = QtWidgets.QPushButton("Apply")
        cancel = QtWidgets.QPushButton("Cancel")
        ok.clicked.connect(dlg.accept)
        cancel.clicked.connect(dlg.reject)
        btns.addStretch()
        btns.addWidget(ok)
        btns.addWidget(cancel)
        v.addLayout(btns)

        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            # Save mapping from editors (only non-empty custom labels)
            self.feature_label_map = {
                feature_name: editors[feature_name].text().strip() 
                for feature_name in feature_cols 
                if editors[feature_name].text().strip() and editors[feature_name].text().strip() != feature_name
            }
            # Refresh current view to apply new labels
            view = self.view_combo.currentText() if hasattr(self, 'view_combo') else 'Heatmap'
            if view == 'Heatmap':
                self._show_heatmap()
            elif view == 'Differential Expression':
                self._show_differential_expression()
            elif view == 'Stacked Bars':
                self._show_stacked_bars()
            elif view == 'Boxplot/Violin Plot':
                self._show_boxplot_violin()
            QtWidgets.QMessageBox.information(self, "Labels Applied", "Feature labels have been applied.")

    def _get_feature_display_name(self, feature_name: str) -> str:
        """Return display label for a feature, using custom label if available."""
        if feature_name in self.feature_label_map:
            return self.feature_label_map[feature_name]
        return feature_name
    
    def _get_metadata_columns(self, df) -> List[str]:
        """Identify metadata columns (non-feature, non-standard columns) in the dataframe."""
        if df is None or df.empty:
            return []
        
        # Standard metadata columns to exclude
        exclude_cols = {
            'label', 'cell_id', 'acquisition_id', 'acquisition_name', 'acquisition_label',
            'well', 'cluster', 'source_file', 'source_well', 'source_file_acquisition_id',
            'centroid_x', 'centroid_y', 'batch_group', 'cluster_phenotype', 'cluster_id'
        }
        
        # Identify feature columns (intensity and morphology)
        feature_cols = set()
        for col in df.columns:
            if col in exclude_cols:
                continue
            # Check if it's a feature column (has intensity suffix or is morphology)
            if any(col.endswith(suffix) for suffix in ['_mean', '_median', '_std', '_mad', '_p10', '_p90', '_integrated', '_frac_pos']):
                feature_cols.add(col)
            elif col in ['area_um2', 'perimeter_um', 'equivalent_diameter_um', 'eccentricity',
                        'solidity', 'extent', 'circularity', 'major_axis_len_um', 'minor_axis_len_um',
                        'aspect_ratio', 'bbox_area_um2', 'touches_border', 'touches_edge', 'holes_count']:
                feature_cols.add(col)
        
        # Metadata columns are everything else
        metadata_cols = [col for col in df.columns 
                        if col not in exclude_cols and col not in feature_cols]
        
        return sorted(metadata_cols)

    def _apply_cluster_annotations(self):
        """Apply current annotation map to clustered_data and feature_dataframe as 'cluster_phenotype'."""
        if not self.clustered_data is None and self.cluster_annotation_map:
            # Use backend names for CSV export if available, otherwise use display names
            export_name_map = self.cluster_backend_names if self.cluster_backend_names else self.cluster_annotation_map
            
            # Map on clustered_data using backend names for CSV export
            self.clustered_data['cluster_phenotype'] = self.clustered_data['cluster'].map(export_name_map).fillna('')
            # Write back to feature_dataframe aligned by index
            aligned = self.feature_dataframe.reindex(self.clustered_data.index)
            if 'cluster_phenotype' not in self.feature_dataframe.columns:
                self.feature_dataframe['cluster_phenotype'] = ''
            self.feature_dataframe.loc[self.clustered_data.index, 'cluster_phenotype'] = self.clustered_data['cluster_phenotype'].values
            # Update color-by options
            self._populate_color_by_options()
            # Redraw the currently selected view
            current_view = self.view_combo.currentText() if hasattr(self, 'view_combo') else 'Heatmap'
            if current_view == 'UMAP':
                self._create_umap_plot()
            elif current_view == 'Heatmap':
                self._create_heatmap()
            elif current_view == 'Stacked Bars':
                self._show_stacked_bars()
            elif current_view == 'Differential Expression':
                self._show_differential_expression()

    # Top-level save/load removed; handled inside annotation dialog
    
    def _explore_clusters(self):
        """Open cluster explorer window."""
        if self.clustered_data is None:
            return
        
        # Get cluster info
        cluster_info = []
        for cluster_id in sorted(self.clustered_data['cluster'].unique()):
            cluster_cells = self.clustered_data[self.clustered_data['cluster'] == cluster_id]
            cluster_info.append({
                'cluster_id': cluster_id,
                'size': len(cluster_cells),
                'cells': cluster_cells.index.tolist()
            })
        
        # Open explorer dialog
        explorer = ClusterExplorerDialog(cluster_info, self.feature_dataframe, self.parent(), label_provider=self)
        explorer.exec_()
    
    def _save_current_plot(self):
        """Save whatever plot is currently shown in the canvas."""
        if self.figure is None:
            return
        default = "plot.png"
        view = self.view_combo.currentText() if hasattr(self, 'view_combo') else 'Heatmap'
        if view == 'UMAP':
            default = 'umap_plot.png'
        elif view == 'Heatmap':
            default = 'cell_clustering_heatmap.png'
        elif view == 'Stacked Bars':
            default = 'stacked_bars.png'
        elif view == 'Differential Expression':
            default = 'differential_expression_heatmap.png'
        
        if save_figure_with_options(self.figure, default, self):
            QtWidgets.QMessageBox.information(self, "Success", "Plot saved successfully")

    def _save_clustering_output(self):
        """Save clustering output as CSV with all features and labels."""
        if self.clustered_data is None:
            QtWidgets.QMessageBox.warning(self, "No Data", "No clustering data available to save.")
            return
        
        # Get default filename
        default = "clustering_output.csv"
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Clustering Output", default,
            "CSV Files (*.csv)"
        )
        if not file_path:
            return
        
        try:
            # Start with the original feature dataframe (all features)
            output_df = self.feature_dataframe.copy()
            
            # Add cluster labels
            if self.clustered_data is not None and 'cluster' in self.clustered_data.columns:
                # Align cluster data with feature dataframe by index
                cluster_series = self.clustered_data['cluster'].reindex(output_df.index)
                output_df['cluster'] = cluster_series
            
            # Add cluster phenotype annotations if available
            if self.clustered_data is not None and 'cluster_phenotype' in self.clustered_data.columns:
                phenotype_series = self.clustered_data['cluster_phenotype'].reindex(output_df.index)
                output_df['cluster_phenotype'] = phenotype_series
            
            # Add manual phenotype annotations if available
            if self.clustered_data is not None and 'manual_phenotype' in self.clustered_data.columns:
                manual_series = self.clustered_data['manual_phenotype'].reindex(output_df.index)
                output_df['manual_phenotype'] = manual_series
            
            # Save to CSV
            output_df.to_csv(file_path, index=True)
            
            # Show success message with summary
            total_cells = len(output_df)
            n_clusters = len(output_df['cluster'].unique()) if 'cluster' in output_df.columns else 0
            n_annotated = len(output_df[output_df['cluster_phenotype'].notna() & (output_df['cluster_phenotype'] != '')]) if 'cluster_phenotype' in output_df.columns else 0
            n_manual = len(output_df[output_df['manual_phenotype'].notna() & (output_df['manual_phenotype'] != '')]) if 'manual_phenotype' in output_df.columns else 0
            
            summary = f"Saved {total_cells} cells with {n_clusters} clusters"
            if n_annotated > 0:
                summary += f", {n_annotated} with cluster annotations"
            if n_manual > 0:
                summary += f", {n_manual} with manual annotations"
            
            QtWidgets.QMessageBox.information(self, "Success", f"Clustering output saved to: {file_path}\n\n{summary}")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Error", f"Error saving clustering output: {str(e)}")

# --------------------------
# Cluster Explorer Dialog
# --------------------------
class ClusterExplorerDialog(QtWidgets.QDialog):
    def __init__(self, cluster_info, feature_dataframe, parent=None, label_provider=None):
        super().__init__(parent)
        self.setWindowTitle("Cluster Explorer")
        self.setModal(True)
        
        # Set size to 90% of parent window if available
        if parent is not None:
            parent_size = parent.size()
            dialog_width = int(parent_size.width() * 0.9)
            dialog_height = int(parent_size.height() * 0.9)
            self.resize(dialog_width, dialog_height)
        
        self.setMinimumSize(1000, 700)
        self.cluster_info = cluster_info
        self.feature_dataframe = feature_dataframe
        self.current_cluster = None
        self.cell_images = []
        self._label_provider = label_provider
        
        # Cache for global min/max values per channel (for RGB normalization)
        # Format: {channel_name: (min_value, max_value)}
        self._global_channel_minmax_cache = {}
        self._cache_initialized = False
        
        # RGB channel brightness/scaling factors (default 1.0 = no change)
        self.rgb_r_scale = 1.0
        self.rgb_g_scale = 1.0
        self.rgb_b_scale = 1.0
        
        self._create_ui()
        
    def _create_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # Title
        title_label = QtWidgets.QLabel("Cluster Explorer")
        title_label.setStyleSheet("QLabel { font-weight: bold; font-size: 12pt; }")
        layout.addWidget(title_label)
        
        # Controls
        controls_layout = QtWidgets.QHBoxLayout()
        
        # Cluster selection
        controls_layout.addWidget(QtWidgets.QLabel("Select Cluster:"))
        self.cluster_combo = QtWidgets.QComboBox()
        for info in self.cluster_info:
            label = self._get_cluster_label(info['cluster_id'])
            self.cluster_combo.addItem(f"{label} ({info['size']} cells)", info)
        self.cluster_combo.currentIndexChanged.connect(self._on_cluster_changed)
        controls_layout.addWidget(self.cluster_combo)
        
        # Channel selection
        controls_layout.addWidget(QtWidgets.QLabel("Channel:"))
        self.channel_combo = QtWidgets.QComboBox()
        controls_layout.addWidget(self.channel_combo)
        
        # RGB mode checkbox
        self.rgb_checkbox = QtWidgets.QCheckBox("RGB Mode")
        self.rgb_checkbox.setToolTip("Show RGB composite instead of single channel")
        self.rgb_checkbox.toggled.connect(self._on_rgb_mode_toggled)
        controls_layout.addWidget(self.rgb_checkbox)
        
        # RGB channel selection and brightness controls (initially hidden)
        self.rgb_channels_widget = QtWidgets.QWidget()
        rgb_main_layout = QtWidgets.QVBoxLayout(self.rgb_channels_widget)
        rgb_main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Channel selection row
        self.rgb_channels_layout = QtWidgets.QHBoxLayout()
        self.rgb_channels_layout.addWidget(QtWidgets.QLabel("R:"))
        self.rgb_r_combo = QtWidgets.QComboBox()
        self.rgb_channels_layout.addWidget(self.rgb_r_combo)
        
        self.rgb_channels_layout.addWidget(QtWidgets.QLabel("G:"))
        self.rgb_g_combo = QtWidgets.QComboBox()
        self.rgb_channels_layout.addWidget(self.rgb_g_combo)
        
        self.rgb_channels_layout.addWidget(QtWidgets.QLabel("B:"))
        self.rgb_b_combo = QtWidgets.QComboBox()
        self.rgb_channels_layout.addWidget(self.rgb_b_combo)
        rgb_main_layout.addLayout(self.rgb_channels_layout)
        
        # Brightness/scaling controls row
        self.rgb_brightness_layout = QtWidgets.QHBoxLayout()
        self.rgb_brightness_layout.addWidget(QtWidgets.QLabel("Brightness:"))
        
        # R channel brightness
        r_brightness_layout = QtWidgets.QHBoxLayout()
        r_brightness_layout.addWidget(QtWidgets.QLabel("R:"))
        self.rgb_r_scale_spinbox = QtWidgets.QDoubleSpinBox()
        self.rgb_r_scale_spinbox.setRange(0.0, 5.0)
        self.rgb_r_scale_spinbox.setSingleStep(0.1)
        self.rgb_r_scale_spinbox.setValue(1.0)
        self.rgb_r_scale_spinbox.setDecimals(1)
        self.rgb_r_scale_spinbox.setToolTip("Brightness scaling for red channel (1.0 = normal, >1.0 = brighter, <1.0 = dimmer)")
        self.rgb_r_scale_spinbox.valueChanged.connect(self._on_rgb_scale_changed)
        r_brightness_layout.addWidget(self.rgb_r_scale_spinbox)
        self.rgb_brightness_layout.addLayout(r_brightness_layout)
        
        # G channel brightness
        g_brightness_layout = QtWidgets.QHBoxLayout()
        g_brightness_layout.addWidget(QtWidgets.QLabel("G:"))
        self.rgb_g_scale_spinbox = QtWidgets.QDoubleSpinBox()
        self.rgb_g_scale_spinbox.setRange(0.0, 5.0)
        self.rgb_g_scale_spinbox.setSingleStep(0.1)
        self.rgb_g_scale_spinbox.setValue(1.0)
        self.rgb_g_scale_spinbox.setDecimals(1)
        self.rgb_g_scale_spinbox.setToolTip("Brightness scaling for green channel (1.0 = normal, >1.0 = brighter, <1.0 = dimmer)")
        self.rgb_g_scale_spinbox.valueChanged.connect(self._on_rgb_scale_changed)
        g_brightness_layout.addWidget(self.rgb_g_scale_spinbox)
        self.rgb_brightness_layout.addLayout(g_brightness_layout)
        
        # B channel brightness
        b_brightness_layout = QtWidgets.QHBoxLayout()
        b_brightness_layout.addWidget(QtWidgets.QLabel("B:"))
        self.rgb_b_scale_spinbox = QtWidgets.QDoubleSpinBox()
        self.rgb_b_scale_spinbox.setRange(0.0, 5.0)
        self.rgb_b_scale_spinbox.setSingleStep(0.1)
        self.rgb_b_scale_spinbox.setValue(1.0)
        self.rgb_b_scale_spinbox.setDecimals(1)
        self.rgb_b_scale_spinbox.setToolTip("Brightness scaling for blue channel (1.0 = normal, >1.0 = brighter, <1.0 = dimmer)")
        self.rgb_b_scale_spinbox.valueChanged.connect(self._on_rgb_scale_changed)
        b_brightness_layout.addWidget(self.rgb_b_scale_spinbox)
        self.rgb_brightness_layout.addLayout(b_brightness_layout)
        
        rgb_main_layout.addLayout(self.rgb_brightness_layout)
        
        self.rgb_channels_widget.hide()
        controls_layout.addWidget(self.rgb_channels_widget)
        
        # Load images button
        self.load_btn = QtWidgets.QPushButton("Load Cell Images")
        self.load_btn.clicked.connect(self._load_cell_images)
        controls_layout.addWidget(self.load_btn)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Suggested markers label (will be populated when cluster is selected)
        self.suggested_markers_label = QtWidgets.QLabel("")
        self.suggested_markers_label.setWordWrap(True)
        self.suggested_markers_label.setStyleSheet("QLabel { color: #0066cc; font-style: italic; padding: 5px; }")
        layout.addWidget(self.suggested_markers_label)
        
        # Image grid
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumHeight(400)
        layout.addWidget(self.scroll_area)
        
        # Status
        self.status_label = QtWidgets.QLabel("Select a cluster and channel to view cell images")
        layout.addWidget(self.status_label)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()
        
        self.export_btn = QtWidgets.QPushButton("Export to HDF5")
        self.export_btn.setToolTip("Export cell images, features, channels, and masks to HDF5 file")
        self.export_btn.clicked.connect(self._export_to_hdf5)
        button_layout.addWidget(self.export_btn)
        
        self.close_btn = QtWidgets.QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
        # Initialize
        self._populate_channels()
        self._on_cluster_changed()
        
        # Initialize global min/max cache for RGB normalization (in background)
        self._initialize_global_minmax_cache()
    
    def _populate_channels(self):
        """Populate channel combo box with available channels."""
        # Get channels from feature dataframe columns
        # Look for any intensity feature suffix to identify channels
        intensity_suffixes = ['_mean', '_std', '_p10', '_p90', '_integrated', '_frac_pos', '_median', '_mad']
        channels = set()
        
        for col in self.feature_dataframe.columns:
            for suffix in intensity_suffixes:
                if col.endswith(suffix):
                    channel = col[:-len(suffix)]  # Remove the suffix to get channel name
                    channels.add(channel)
                    break  # Found a match, no need to check other suffixes
        
        # Add channels (RGB mode is controlled by checkbox)
        self.channel_combo.addItems(sorted(channels))
        
        # Populate RGB channel combos
        for combo in [self.rgb_r_combo, self.rgb_g_combo, self.rgb_b_combo]:
            combo.addItems(sorted(channels))
    
    def _on_cluster_changed(self):
        """Handle cluster selection change."""
        self.current_cluster = self.cluster_combo.currentData()
        if self.current_cluster:
            label = self._get_cluster_label(self.current_cluster['cluster_id'])
            self.status_label.setText(f"Selected {label} with {self.current_cluster['size']} cells")
            
            # Calculate and suggest markers for this cluster
            self._suggest_markers_for_cluster(self.current_cluster['cluster_id'])
    
    def _suggest_markers_for_cluster(self, cluster_id):
        """Calculate and suggest markers that are highly expressed or low/negative in this cluster."""
        try:
            # Get clustered data from parent dialog (label_provider)
            if self._label_provider is None or not hasattr(self._label_provider, 'clustered_data'):
                self.suggested_markers_label.setText("")
                return
            
            clustered_data = self._label_provider.clustered_data
            if clustered_data is None or 'cluster' not in clustered_data.columns:
                self.suggested_markers_label.setText("")
                return
            
            # Get intensity feature columns only (exclude morphology features)
            # Intensity features typically end with _mean, _median, _integrated, etc.
            intensity_suffixes = ['_mean', '_median', '_integrated', '_std', '_p10', '_p90', '_frac_pos', '_mad']
            all_cols = clustered_data.columns
            intensity_cols = [col for col in all_cols 
                             if any(col.endswith(suffix) for suffix in intensity_suffixes)]
            
            # Also exclude metadata columns
            metadata_cols = ['cluster', 'cluster_phenotype', 'manual_phenotype', 'acquisition_id', 
                           'acquisition_label', 'source_file', 'source_well', 'well', 'cell_id']
            intensity_cols = [col for col in intensity_cols if col not in metadata_cols]
            
            if not intensity_cols:
                self.suggested_markers_label.setText("")
                return
            
            # Calculate mean expression per cluster for intensity features
            cluster_means = clustered_data.groupby('cluster')[intensity_cols].mean()
            
            # Calculate z-scores across clusters for each feature
            feature_means = cluster_means.mean(axis=0)  # Mean across clusters
            feature_stds = cluster_means.std(axis=0)    # Std across clusters
            feature_stds = feature_stds.replace(0, 1)   # Avoid division by zero
            
            # Z-score: (cluster_mean - overall_mean) / overall_std
            cluster_z_scores = (cluster_means.loc[cluster_id] - feature_means) / feature_stds
            
            # Sort by z-score to find highly expressed (high z) and low/negative (low z)
            sorted_markers = cluster_z_scores.sort_values(ascending=False)
            
            # Get top 3 highly expressed markers (z-score > 1.0)
            high_markers = sorted_markers[sorted_markers > 1.0].head(3)
            # Get top 3 low/negative markers (z-score < -1.0)
            low_markers = sorted_markers[sorted_markers < -1.0].tail(3).sort_values(ascending=True)
            
            # Extract channel names from feature names (remove suffix)
            def extract_channel_name(feature_name):
                for suffix in intensity_suffixes:
                    if feature_name.endswith(suffix):
                        return feature_name[:-len(suffix)]
                return feature_name
            
            # Build suggestion text
            suggestions = []
            if len(high_markers) > 0:
                high_channels = [extract_channel_name(m) for m in high_markers.index]
                suggestions.append(f"High: {', '.join(high_channels)}")
            
            if len(low_markers) > 0:
                low_channels = [extract_channel_name(m) for m in low_markers.index]
                suggestions.append(f"Low: {', '.join(low_channels)}")
            
            if suggestions:
                suggestion_text = "Suggested markers: " + " | ".join(suggestions)
                self.suggested_markers_label.setText(suggestion_text)
                
                # Also update channel combo to highlight suggested channels
                # Select the first highly expressed marker if available
                if len(high_markers) > 0:
                    suggested_channel = extract_channel_name(high_markers.index[0])
                    # Find and select this channel in the combo box
                    index = self.channel_combo.findText(suggested_channel)
                    if index >= 0:
                        self.channel_combo.setCurrentIndex(index)
            else:
                self.suggested_markers_label.setText("No strong marker patterns detected")
                
        except Exception as e:
            print(f"[ClusterExplorer] Error suggesting markers: {e}")
            import traceback
            traceback.print_exc()
            self.suggested_markers_label.setText("")
    
    def _on_rgb_mode_toggled(self):
        """Handle RGB mode checkbox toggle."""
        if self.rgb_checkbox.isChecked():
            self.rgb_channels_widget.show()
            self.channel_combo.setEnabled(False)
            # Ensure cache is initialized when RGB mode is enabled
            if not self._cache_initialized:
                self._initialize_global_minmax_cache()
        else:
            self.rgb_channels_widget.hide()
            self.channel_combo.setEnabled(True)
    
    def _on_rgb_scale_changed(self):
        """Handle RGB brightness scaling changes."""
        self.rgb_r_scale = self.rgb_r_scale_spinbox.value()
        self.rgb_g_scale = self.rgb_g_scale_spinbox.value()
        self.rgb_b_scale = self.rgb_b_scale_spinbox.value()
        
        # If images are already loaded, reload them with new scaling
        if self.cell_images and self.rgb_checkbox.isChecked():
            # Reload images with new scaling
            self._load_cell_images()
    
    def _initialize_global_minmax_cache(self):
        """Initialize the global min/max cache for all channels across all acquisitions.
        
        This is a slow operation that scans all images, so it's done once and cached.
        """
        if self._cache_initialized:
            return
        
        print("[ClusterExplorer] Initializing global min/max cache for RGB normalization...")
        self.status_label.setText("Computing global intensity ranges (this may take a moment)...")
        QtWidgets.QApplication.processEvents()  # Update UI
        
        try:
            parent_window = self.parent()
            if parent_window is None:
                return
            
            # Get all unique channels from feature dataframe
            intensity_suffixes = ['_mean', '_std', '_p10', '_p90', '_integrated', '_frac_pos', '_median', '_mad']
            channels = set()
            for col in self.feature_dataframe.columns:
                for suffix in intensity_suffixes:
                    if col.endswith(suffix):
                        channel = col[:-len(suffix)]
                        channels.add(channel)
                        break
            
            if not channels:
                print("[ClusterExplorer] No channels found for cache initialization")
                self._cache_initialized = True
                return
            
            # Get all unique acquisition IDs from feature dataframe
            if 'acquisition_id' not in self.feature_dataframe.columns:
                print("[ClusterExplorer] No acquisition_id column found")
                self._cache_initialized = True
                return
            
            unique_acq_ids = self.feature_dataframe['acquisition_id'].unique()
            
            # Initialize min/max for each channel
            channel_mins = {ch: float('inf') for ch in channels}
            channel_maxs = {ch: float('-inf') for ch in channels}
            
            # Iterate through all acquisitions and channels to find global min/max
            total_acqs = len(unique_acq_ids)
            processed = 0
            
            for acq_id in unique_acq_ids:
                processed += 1
                if processed % 10 == 0:
                    progress = f"Processing {processed}/{total_acqs} acquisitions..."
                    self.status_label.setText(progress)
                    QtWidgets.QApplication.processEvents()
                
                # Get loader for this acquisition
                loader = parent_window._get_loader_for_acquisition(acq_id)
                if loader is None:
                    continue
                
                # Get original acquisition ID
                original_acq_id = parent_window._get_original_acq_id(acq_id)
                
                # Check which channels are available for this acquisition
                try:
                    available_channels = loader.get_channels(original_acq_id)
                except Exception:
                    continue
                
                # For each channel, load the image and update min/max
                for channel in channels:
                    if channel not in available_channels:
                        continue
                    
                    try:
                        img = loader.get_image(original_acq_id, channel)
                        if img is not None and img.size > 0:
                            img_min = float(np.min(img))
                            img_max = float(np.max(img))
                            
                            if img_min < channel_mins[channel]:
                                channel_mins[channel] = img_min
                            if img_max > channel_maxs[channel]:
                                channel_maxs[channel] = img_max
                    except Exception as e:
                        # Skip channels that can't be loaded
                        continue
            
            # Store in cache
            for channel in channels:
                if channel_mins[channel] != float('inf') and channel_maxs[channel] != float('-inf'):
                    self._global_channel_minmax_cache[channel] = (channel_mins[channel], channel_maxs[channel])
            
            self._cache_initialized = True
            print(f"[ClusterExplorer] Cache initialized for {len(self._global_channel_minmax_cache)} channels")
            self.status_label.setText("Ready to load cell images")
            
        except Exception as e:
            print(f"[ClusterExplorer] Error initializing cache: {e}")
            import traceback
            traceback.print_exc()
            self._cache_initialized = True  # Mark as initialized to avoid retrying
            self.status_label.setText("Cache initialization failed, using per-image normalization")
    
    def _load_cell_images(self):
        """Load and display cell images for the selected cluster."""
        print(f"[ClusterExplorer] _load_cell_images called")
        
        if not self.current_cluster:
            print(f"[ClusterExplorer] ERROR: No current cluster selected")
            return
        
        print(f"[ClusterExplorer] Current cluster: {self.current_cluster['cluster_id']}, size: {self.current_cluster['size']}")
        
        channel = self.channel_combo.currentText()
        if not channel:
            QtWidgets.QMessageBox.warning(self, "No Channel", "Please select a channel.")
            return
        
        print(f"[ClusterExplorer] Selected channel: {channel}")
        
        try:
            # Get parent window to access loader and segmentation masks
            parent_window = self.parent()
            print(f"[ClusterExplorer] Parent window type: {type(parent_window)}")
            
            if not hasattr(parent_window, 'segmentation_masks'):
                print(f"[ClusterExplorer] ERROR: Parent window has no 'segmentation_masks' attribute")
                QtWidgets.QMessageBox.warning(self, "No Data", "Cannot access image data. Please ensure segmentation masks are loaded.")
                return
            
            print(f"[ClusterExplorer] segmentation_masks keys: {list(parent_window.segmentation_masks.keys())[:5]}... (showing first 5)")
            print(f"[ClusterExplorer] Total segmentation masks: {len(parent_window.segmentation_masks)}")
            
            # Check if we have any loaders available (single loader or multiple loaders)
            has_single_loader = hasattr(parent_window, 'loader') and parent_window.loader is not None
            has_multi_loaders = hasattr(parent_window, 'mcd_loaders') and len(parent_window.mcd_loaders) > 0
            has_loader = has_single_loader or has_multi_loaders
            
            print(f"[ClusterExplorer] Loader check - single: {has_single_loader}, multi: {has_multi_loaders}, total: {has_loader}")
            if has_multi_loaders:
                print(f"[ClusterExplorer] mcd_loaders count: {len(parent_window.mcd_loaders)}")
            
            if not has_loader:
                print(f"[ClusterExplorer] ERROR: No loaders available")
                QtWidgets.QMessageBox.warning(self, "No Data", "Cannot access image data. Please ensure data files are loaded.")
                return
            
            # Clear previous images
            self.cell_images = []
            
            # Create image grid
            grid_widget = QtWidgets.QWidget()
            grid_layout = QtWidgets.QGridLayout(grid_widget)
            
            # Get cell data for this cluster
            cluster_cells = self.current_cluster['cells']
            print(f"[ClusterExplorer] Cluster cells list length: {len(cluster_cells)}")
            print(f"[ClusterExplorer] First few cell indices: {cluster_cells[:5] if len(cluster_cells) > 0 else 'N/A'}")
            
            # Use all cells from the cluster (support multiple files)
            filtered_cells = cluster_cells
            
            if not filtered_cells:
                print(f"[ClusterExplorer] ERROR: No cells in filtered_cells")
                QtWidgets.QMessageBox.information(
                    self,
                    "No Cells Available",
                    "No cells available in this cluster."
                )
                return
            
            # Limit to first 10 cells for performance
            max_cells = min(10, len(filtered_cells))
            crop_size = 30  # 30x30 pixel crop
            print(f"[ClusterExplorer] Processing up to {max_cells} cells")
            
            cells_processed = 0
            cells_skipped_no_index = 0
            cells_skipped_no_mask = 0
            cells_skipped_no_cell_mask = 0
            cells_skipped_no_loader = 0
            cells_loaded = 0
            
            for i, cell_idx in enumerate(filtered_cells[:max_cells]):
                if i >= max_cells:
                    break
                
                cells_processed += 1
                print(f"[ClusterExplorer] Processing cell {i+1}/{max_cells}: cell_idx={cell_idx}")
                
                # Get cell data
                # cell_idx is an index label, not an integer position, so use loc instead of iloc
                if cell_idx not in self.feature_dataframe.index:
                    print(f"[ClusterExplorer] WARNING: Cell index {cell_idx} not found in feature dataframe index")
                    print(f"[ClusterExplorer]   Feature dataframe index type: {type(self.feature_dataframe.index)}")
                    print(f"[ClusterExplorer]   Feature dataframe index length: {len(self.feature_dataframe.index)}")
                    print(f"[ClusterExplorer]   First few indices: {list(self.feature_dataframe.index[:5])}")
                    cells_skipped_no_index += 1
                    continue
                
                cell_data = self.feature_dataframe.loc[cell_idx]
                acq_id = cell_data['acquisition_id']
                cell_id = int(cell_data['cell_id'])
                print(f"[ClusterExplorer]   Cell data retrieved - acq_id: {acq_id}, cell_id: {cell_id}")
                
                # Convert old acquisition ID format to new unique format (for multi-file support)
                # Old format: "slide_0_acq_1", New format: "slide_0_acq_1__file_e149256f"
                unique_acq_id = acq_id
                if acq_id not in parent_window.segmentation_masks:
                    # Try to find the unique acquisition ID that starts with the old ID + "__file_"
                    matching_keys = [key for key in parent_window.segmentation_masks.keys() 
                                   if key.startswith(acq_id + "__file_")]
                    if matching_keys:
                        # If multiple matches, we need to determine which file this cell came from
                        # Check if we can determine from source_file column or use the first match
                        if len(matching_keys) == 1:
                            unique_acq_id = matching_keys[0]
                            print(f"[ClusterExplorer]   Mapped old acq_id '{acq_id}' to unique acq_id '{unique_acq_id}'")
                        else:
                            # Multiple files have the same acquisition ID - need to determine which file
                            # Try to use source_file from cell_data if available
                            source_file = cell_data.get('source_file', None)
                            if source_file:
                                # Find the unique_acq_id that corresponds to this source_file
                                # We need to check which file path matches
                                for key in matching_keys:
                                    # Get the file path for this unique_acq_id
                                    if hasattr(parent_window, 'acq_to_file') and key in parent_window.acq_to_file:
                                        file_path = parent_window.acq_to_file[key]
                                        if source_file in file_path or file_path.endswith(source_file):
                                            unique_acq_id = key
                                            print(f"[ClusterExplorer]   Mapped old acq_id '{acq_id}' to unique acq_id '{unique_acq_id}' using source_file '{source_file}'")
                                            break
                                else:
                                    # Fallback: use first match
                                    unique_acq_id = matching_keys[0]
                                    print(f"[ClusterExplorer]   Multiple matches found, using first: '{unique_acq_id}'")
                            else:
                                # Fallback: use first match
                                unique_acq_id = matching_keys[0]
                                print(f"[ClusterExplorer]   Multiple matches found, using first: '{unique_acq_id}'")
                    else:
                        print(f"[ClusterExplorer]   WARNING: acq_id {acq_id} not in segmentation_masks and no matching unique ID found")
                        print(f"[ClusterExplorer]   Available acq_ids in masks: {list(parent_window.segmentation_masks.keys())[:10]}")
                        cells_skipped_no_mask += 1
                        continue
                
                # Get mask and image using the unique acquisition ID
                if unique_acq_id not in parent_window.segmentation_masks:
                    print(f"[ClusterExplorer]   WARNING: unique_acq_id {unique_acq_id} not in segmentation_masks")
                    cells_skipped_no_mask += 1
                    continue
                
                print(f"[ClusterExplorer]   Found mask for unique_acq_id: {unique_acq_id}")
                mask = parent_window.segmentation_masks[unique_acq_id]
                print(f"[ClusterExplorer]   Mask shape: {mask.shape}, dtype: {mask.dtype}")
                
                # Create cell mask
                cell_mask = (mask == cell_id).astype(np.uint8)
                cell_mask_pixels = np.sum(cell_mask)
                print(f"[ClusterExplorer]   Cell mask pixels: {cell_mask_pixels}")
                
                if not np.any(cell_mask):
                    print(f"[ClusterExplorer]   WARNING: No pixels found for cell_id {cell_id} in mask")
                    print(f"[ClusterExplorer]   Unique cell_ids in mask: {np.unique(mask)[:20]}... (showing first 20)")
                    cells_skipped_no_cell_mask += 1
                    continue
                
                # Cell mask found, proceed with image loading
                print(f"[ClusterExplorer]   Cell mask found, proceeding to load image")
                try:
                    # Calculate cell center using regionprops
                    props = regionprops(cell_mask)
                    if not props:
                        print(f"[ClusterExplorer]   WARNING: No regionprops found for cell_mask")
                        continue
                    
                    center_y, center_x = props[0].centroid
                    center_y, center_x = int(center_y), int(center_x)
                    print(f"[ClusterExplorer]   Cell center: ({center_y}, {center_x})")
                    
                    # Define crop boundaries
                    half_crop = crop_size // 2
                    y_start = max(0, center_y - half_crop)
                    y_end = min(mask.shape[0], center_y + half_crop)
                    x_start = max(0, center_x - half_crop)
                    x_end = min(mask.shape[1], center_x + half_crop)
                    print(f"[ClusterExplorer]   Crop boundaries: y=[{y_start}:{y_end}], x=[{x_start}:{x_end}]")
                    
                    # Crop the cell mask
                    cropped_mask = cell_mask[y_start:y_end, x_start:x_end]
                    
                    # Get the correct loader for this acquisition (use unique_acq_id)
                    loader = parent_window._get_loader_for_acquisition(unique_acq_id)
                    if loader is None:
                        print(f"[ClusterExplorer]   WARNING: No loader found for acquisition {unique_acq_id}, skipping cell {cell_id}")
                        cells_skipped_no_loader += 1
                        continue
                    
                    print(f"[ClusterExplorer]   Loader found for unique_acq_id: {unique_acq_id}")
                    
                    # Get original acquisition ID (needed for multi-file support)
                    original_acq_id = parent_window._get_original_acq_id(unique_acq_id)
                    print(f"[ClusterExplorer]   Original acq_id: {original_acq_id}")
                    
                    if self.rgb_checkbox.isChecked():
                        print(f"[ClusterExplorer]   Loading RGB image")
                        # Load RGB composite using user-selected channels (use unique_acq_id)
                        rgb_img = self._load_rgb_image(parent_window, unique_acq_id, loader, original_acq_id)
                        if rgb_img is not None:
                            print(f"[ClusterExplorer]   RGB image loaded, shape: {rgb_img.shape}")
                            # Crop RGB image
                            cropped_rgb = rgb_img[y_start:y_end, x_start:x_end]
                            # Apply mask
                            for c in range(3):
                                cropped_rgb[:, :, c] *= cropped_mask
                            
                            # Create image widget with source file and well name
                            source_file = cell_data.get('source_file', 'Unknown')
                            well_name = cell_data.get('well', cell_data.get('source_well', 'Unknown'))
                            # Clean up source_file to just the filename without path
                            if source_file and source_file != 'Unknown':
                                source_file = os.path.basename(str(source_file))
                            label = f"{source_file} - {well_name}" if well_name and well_name != 'Unknown' else source_file
                            img_widget = self._create_image_widget(cropped_rgb, label, is_rgb=True)
                            grid_layout.addWidget(img_widget, i // 4, i % 4)
                            
                            self.cell_images.append({
                                'cell_id': cell_id,
                                'acquisition_id': unique_acq_id,
                                'image': cropped_rgb
                            })
                            cells_loaded += 1
                            print(f"[ClusterExplorer]   Successfully loaded RGB image for cell {cell_id}")
                        else:
                            print(f"[ClusterExplorer]   WARNING: Failed to load RGB image")
                    else:
                        print(f"[ClusterExplorer]   Loading single channel image: {channel}")
                        # Load single channel using the correct loader and original acquisition ID
                        try:
                            channel_img = loader.get_image(original_acq_id, channel)
                            print(f"[ClusterExplorer]   Channel image loaded, shape: {channel_img.shape}")
                            # Crop channel image
                            cropped_channel = channel_img[y_start:y_end, x_start:x_end]
                            # Apply mask
                            cropped_channel *= cropped_mask
                            
                            # Create image widget with source file and well name
                            source_file = cell_data.get('source_file', 'Unknown')
                            well_name = cell_data.get('well', cell_data.get('source_well', 'Unknown'))
                            # Clean up source_file to just the filename without path
                            if source_file and source_file != 'Unknown':
                                source_file = os.path.basename(str(source_file))
                            label = f"{source_file} - {well_name}" if well_name and well_name != 'Unknown' else source_file
                            img_widget = self._create_image_widget(cropped_channel, label, is_rgb=False)
                            grid_layout.addWidget(img_widget, i // 4, i % 4)
                            
                            self.cell_images.append({
                                'cell_id': cell_id,
                                'acquisition_id': unique_acq_id,
                                'image': cropped_channel
                            })
                            cells_loaded += 1
                            print(f"[ClusterExplorer]   Successfully loaded channel image for cell {cell_id}")
                        except Exception as img_error:
                            print(f"[ClusterExplorer]   ERROR loading channel image: {img_error}")
                            import traceback
                            traceback.print_exc()
                            continue
                
                except Exception as e:
                    print(f"[ClusterExplorer]   ERROR loading image for cell {cell_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Print summary statistics
            print(f"[ClusterExplorer] ===== SUMMARY =====")
            print(f"[ClusterExplorer] Cells processed: {cells_processed}")
            print(f"[ClusterExplorer] Cells loaded successfully: {cells_loaded}")
            print(f"[ClusterExplorer] Cells skipped - no index: {cells_skipped_no_index}")
            print(f"[ClusterExplorer] Cells skipped - no mask: {cells_skipped_no_mask}")
            print(f"[ClusterExplorer] Cells skipped - no cell mask: {cells_skipped_no_cell_mask}")
            print(f"[ClusterExplorer] Cells skipped - no loader: {cells_skipped_no_loader}")
            print(f"[ClusterExplorer] Total images loaded: {len(self.cell_images)}")
            print(f"[ClusterExplorer] ===================")
            
            self.scroll_area.setWidget(grid_widget)
            self.status_label.setText(f"Loaded {len(self.cell_images)} cell images for Cluster {self.current_cluster['cluster_id']}")
            
        except Exception as e:
            print(f"[ClusterExplorer] ===== FATAL ERROR =====")
            print(f"[ClusterExplorer] Exception type: {type(e).__name__}")
            print(f"[ClusterExplorer] Exception message: {str(e)}")
            import traceback
            traceback.print_exc()
            print(f"[ClusterExplorer] =======================")
            QtWidgets.QMessageBox.critical(self, "Error", f"Error loading cell images: {str(e)}")
    
    def _normalize_channel_global(self, channel_img, channel_name, scale_factor=1.0):
        """Normalize a channel image using global min/max values.
        
        If global min/max are available in cache, use those. Otherwise, fall back
        to per-image normalization. Optionally apply a brightness scaling factor.
        
        Args:
            channel_img: The channel image array
            channel_name: Name of the channel
            scale_factor: Brightness scaling factor (default 1.0 = no change)
            
        Returns:
            Normalized image array (0-1 range, scaled by scale_factor and clipped)
        """
        if channel_name in self._global_channel_minmax_cache:
            global_min, global_max = self._global_channel_minmax_cache[channel_name]
            # Normalize using global min/max
            if global_max > global_min:
                normalized = (channel_img - global_min) / (global_max - global_min + 1e-8)
                # Apply brightness scaling
                normalized = normalized * scale_factor
                # Clip to [0, 1] range in case some pixels exceed bounds
                normalized = np.clip(normalized, 0, 1)
                return normalized
            else:
                # Fallback if min == max
                return np.zeros_like(channel_img)
        else:
            # Fallback to per-image normalization if cache not available
            img_min = channel_img.min()
            img_max = channel_img.max()
            if img_max > img_min:
                normalized = (channel_img - img_min) / (img_max - img_min + 1e-8)
                # Apply brightness scaling
                normalized = normalized * scale_factor
                # Clip to [0, 1] range
                normalized = np.clip(normalized, 0, 1)
                return normalized
            else:
                return np.zeros_like(channel_img)
    
    def _load_rgb_image(self, parent_window, acq_id, loader=None, original_acq_id=None):
        """Load RGB composite image for an acquisition using user-selected channels."""
        try:
            # Get loader and original acquisition ID if not provided
            if loader is None:
                loader = parent_window._get_loader_for_acquisition(acq_id)
            if loader is None:
                return None
            if original_acq_id is None:
                original_acq_id = parent_window._get_original_acq_id(acq_id)
            
            # Use user-selected channels if RGB mode is enabled
            if self.rgb_checkbox.isChecked():
                r_channel = self.rgb_r_combo.currentText()
                g_channel = self.rgb_g_combo.currentText()
                b_channel = self.rgb_b_combo.currentText()
                
                if r_channel and g_channel and b_channel:
                    # Load the three user-selected channels using the correct loader and original acquisition ID
                    ch1 = loader.get_image(original_acq_id, r_channel)
                    ch2 = loader.get_image(original_acq_id, g_channel)
                    ch3 = loader.get_image(original_acq_id, b_channel)
                    
                    # Normalize each channel to 0-1 range using GLOBAL min/max (not per-image)
                    # Apply user-defined brightness scaling for each channel
                    ch1_norm = self._normalize_channel_global(ch1, r_channel, self.rgb_r_scale)
                    ch2_norm = self._normalize_channel_global(ch2, g_channel, self.rgb_g_scale)
                    ch3_norm = self._normalize_channel_global(ch3, b_channel, self.rgb_b_scale)
                    
                    # Create RGB composite
                    rgb_img = np.stack([ch1_norm, ch2_norm, ch3_norm], axis=-1)
                    return rgb_img
            else:
                # Fallback to automatic channel detection
                channels = set()
                intensity_suffixes = ['_mean', '_std', '_p10', '_p90', '_integrated', '_frac_pos', '_median', '_mad']
                for col in self.feature_dataframe.columns:
                    for suffix in intensity_suffixes:
                        if col.endswith(suffix):
                            channel = col[:-len(suffix)]  # Remove the suffix to get channel name
                            channels.add(channel)
                            break  # Found a match, no need to check other suffixes
                
                # Try to find RGB channels (common naming patterns)
                rgb_channels = []
                for pattern in ['DAPI', 'FITC', 'TRITC', 'Cy5', 'Hoechst', 'GFP', 'RFP', 'mCherry']:
                    for channel in channels:
                        if pattern.lower() in channel.lower():
                            rgb_channels.append(channel)
                            break
                
                # If we don't have 3 channels, just use the first 3 available
                if len(rgb_channels) < 3:
                    rgb_channels = list(channels)[:3]
                
                if len(rgb_channels) >= 3:
                    # Load the three channels
                    ch1 = parent_window.loader.get_image(acq_id, rgb_channels[0])
                    ch2 = parent_window.loader.get_image(acq_id, rgb_channels[1])
                    ch3 = parent_window.loader.get_image(acq_id, rgb_channels[2])
                    
                    # Normalize each channel to 0-1 range using GLOBAL min/max (not per-image)
                    # Apply user-defined brightness scaling for each channel
                    ch1_norm = self._normalize_channel_global(ch1, rgb_channels[0], self.rgb_r_scale)
                    ch2_norm = self._normalize_channel_global(ch2, rgb_channels[1], self.rgb_g_scale)
                    ch3_norm = self._normalize_channel_global(ch3, rgb_channels[2], self.rgb_b_scale)
                    
                    # Create RGB composite
                    rgb_img = np.stack([ch1_norm, ch2_norm, ch3_norm], axis=-1)
                    return rgb_img
            
        except Exception as e:
            print(f"Error loading RGB image: {e}")
        
        return None
    
    def _export_to_hdf5(self):
        """Export cell images, features, channels, and masks to HDF5 file."""
        if not self.current_cluster:
            QtWidgets.QMessageBox.warning(self, "No Cluster", "Please select a cluster first.")
            return
        
        # Get crop size from user
        crop_size, ok = QtWidgets.QInputDialog.getInt(
            self,
            "Crop Size",
            "Enter crop size (square, in pixels):",
            value=30,
            min=10,
            max=200,
            step=5
        )
        
        if not ok:
            return
        
        # Get output file path
        default = f"cluster_{self.current_cluster['cluster_id']}_export.h5"
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export to HDF5",
            default,
            "HDF5 Files (*.h5 *.hdf5)"
        )
        
        if not file_path:
            return
        
        try:
            import h5py
        except ImportError:
            QtWidgets.QMessageBox.critical(
                self,
                "Missing Dependency",
                "h5py is required for HDF5 export. Please install it: pip install h5py"
            )
            return
        
        # Show progress
        progress_dlg = QtWidgets.QProgressDialog("Exporting to HDF5...", "Cancel", 0, 100, self)
        progress_dlg.setWindowModality(QtCore.Qt.WindowModal)
        progress_dlg.setValue(0)
        QtWidgets.QApplication.processEvents()
        
        try:
            parent_window = self.parent()
            if parent_window is None:
                QtWidgets.QMessageBox.warning(self, "Error", "Cannot access parent window.")
                return
            
            if not hasattr(parent_window, 'segmentation_masks'):
                QtWidgets.QMessageBox.warning(self, "No Masks", "Segmentation masks are required for export.")
                return
            
            # Get all cells from current cluster
            cluster_cells = self.current_cluster['cells']
            total_cells = len(cluster_cells)
            
            if total_cells == 0:
                QtWidgets.QMessageBox.warning(self, "No Cells", "No cells in selected cluster.")
                return
            
            # Get all available channels from feature dataframe
            intensity_suffixes = ['_mean', '_std', '_p10', '_p90', '_integrated', '_frac_pos', '_median', '_mad']
            channels = set()
            for col in self.feature_dataframe.columns:
                for suffix in intensity_suffixes:
                    if col.endswith(suffix):
                        channel = col[:-len(suffix)]
                        channels.add(channel)
                        break
            
            channels = sorted(list(channels))
            
            if not channels:
                QtWidgets.QMessageBox.warning(self, "No Channels", "No channels found in feature dataframe.")
                return
            
            # Group cells by acquisition
            cells_by_acq = {}
            for cell_idx in cluster_cells:
                if cell_idx not in self.feature_dataframe.index:
                    continue
                
                cell_data = self.feature_dataframe.loc[cell_idx]
                acq_id = cell_data['acquisition_id']
                
                # Convert old acquisition ID to new unique format
                unique_acq_id = acq_id
                if acq_id not in parent_window.segmentation_masks:
                    matching_keys = [key for key in parent_window.segmentation_masks.keys() 
                                   if key.startswith(acq_id + "__file_")]
                    if matching_keys:
                        if len(matching_keys) == 1:
                            unique_acq_id = matching_keys[0]
                        else:
                            source_file = cell_data.get('source_file', None)
                            if source_file:
                                for key in matching_keys:
                                    if hasattr(parent_window, 'acq_to_file') and key in parent_window.acq_to_file:
                                        file_path_check = parent_window.acq_to_file[key]
                                        if source_file in file_path_check or file_path_check.endswith(source_file):
                                            unique_acq_id = key
                                            break
                                else:
                                    unique_acq_id = matching_keys[0]
                            else:
                                unique_acq_id = matching_keys[0]
                    else:
                        continue
                
                if unique_acq_id not in parent_window.segmentation_masks:
                    continue
                
                if unique_acq_id not in cells_by_acq:
                    cells_by_acq[unique_acq_id] = []
                cells_by_acq[unique_acq_id].append(cell_idx)
            
            if not cells_by_acq:
                QtWidgets.QMessageBox.warning(self, "No Valid Cells", "No valid cells could be found for export.")
                return
            
            # Prepare data for multiprocessing
            # We need to pass serializable data to workers
            acq_tasks = []
            for unique_acq_id, cell_indices in cells_by_acq.items():
                # Get loader info
                loader = parent_window._get_loader_for_acquisition(unique_acq_id)
                if loader is None:
                    continue
                
                original_acq_id = parent_window._get_original_acq_id(unique_acq_id)
                mask = parent_window.segmentation_masks[unique_acq_id]
                
                # Get file path for loader
                file_path_for_loader = None
                if hasattr(parent_window, 'acq_to_file') and unique_acq_id in parent_window.acq_to_file:
                    file_path_for_loader = parent_window.acq_to_file[unique_acq_id]
                
                # Get loader type - check if it's MCD or OME-TIFF
                loader_type = 'mcd'
                if hasattr(parent_window, 'mcd_loaders') and file_path_for_loader and file_path_for_loader in parent_window.mcd_loaders:
                    loader_type = 'mcd'
                elif file_path_for_loader and os.path.isdir(file_path_for_loader):
                    loader_type = 'ometiff'
                elif hasattr(parent_window, 'loader') and parent_window.loader is not None:
                    # Single file case - try to determine from file extension
                    if file_path_for_loader and file_path_for_loader.endswith(('.mcd', '.mcdx')):
                        loader_type = 'mcd'
                    else:
                        loader_type = 'ometiff'
                
                # Prepare cell data for this acquisition
                cell_data_list = []
                for cell_idx in cell_indices:
                    cell_data = self.feature_dataframe.loc[cell_idx].to_dict()
                    cell_data['_cell_idx'] = cell_idx
                    cell_data_list.append(cell_data)
                
                acq_tasks.append((
                    unique_acq_id,
                    original_acq_id,
                    file_path_for_loader,
                    loader_type,
                    cell_data_list,
                    channels,
                    crop_size,
                    mask
                ))
            
            # Determine number of workers (max_cores - 2)
            max_workers = max(1, mp.cpu_count() - 2)
            total_acqs = len(acq_tasks)
            
            progress_dlg.setMaximum(total_acqs)
            progress_dlg.setLabelText(f"Processing {total_acqs} acquisitions with {max_workers} workers...")
            QtWidgets.QApplication.processEvents()
            
            # Process acquisitions in parallel
            images_list = []
            masks_list = []
            features_list = []
            valid_cell_indices = []
            
            with mp.Pool(processes=max_workers) as pool:
                futures = []
                for task in acq_tasks:
                    future = pool.apply_async(_process_acquisition_export, task)
                    futures.append(future)
                
                # Collect results as they complete
                completed = 0
                for future in futures:
                    if progress_dlg.wasCanceled():
                        pool.terminate()
                        pool.join()
                        return
                    
                    try:
                        result = future.get(timeout=600)  # 10 minute timeout per acquisition
                        if result:
                            acq_images, acq_masks, acq_features, acq_indices = result
                            images_list.extend(acq_images)
                            masks_list.extend(acq_masks)
                            features_list.extend(acq_features)
                            valid_cell_indices.extend(acq_indices)
                    except Exception as e:
                        print(f"[ClusterExplorer] Error processing acquisition: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                    
                    completed += 1
                    progress_dlg.setValue(completed)
                    progress_dlg.setLabelText(f"Processed {completed}/{total_acqs} acquisitions ({len(images_list)} cells)...")
                    QtWidgets.QApplication.processEvents()
            
            if not images_list:
                QtWidgets.QMessageBox.warning(self, "No Valid Cells", "No valid cells could be processed for export.")
                return
            
            progress_dlg.setLabelText("Saving to HDF5 file...")
            QtWidgets.QApplication.processEvents()
            
            # Convert to numpy arrays
            images_array = np.array(images_list)  # Shape: (N, H, W, C)
            masks_array = np.array(masks_list)    # Shape: (N, H, W)
            
            # Create features dataframe for valid cells
            features_df = pd.DataFrame(features_list)
            features_df.index = valid_cell_indices
            
            # Save to HDF5
            with h5py.File(file_path, 'w') as f:
                # Save images
                f.create_dataset('images', data=images_array, compression='gzip', compression_opts=4)
                
                # Save masks
                f.create_dataset('masks', data=masks_array, compression='gzip', compression_opts=4)
                
                # Save channels as list (convert to numpy array of strings)
                channels_array = np.array([ch.encode('utf-8') for ch in channels], dtype='S')
                f.create_dataset('channels', data=channels_array)
                
                # Save features as structured array
                # Convert dataframe to numpy structured array
                features_rec = features_df.to_records(index=False)
                f.create_dataset('features', data=features_rec, compression='gzip', compression_opts=4)
                
                # Store column names as attribute
                f['features'].attrs['columns'] = [col.encode('utf-8') for col in features_df.columns]
                f['features'].attrs['index'] = [str(idx).encode('utf-8') for idx in features_df.index]
            
            progress_dlg.setValue(total_acqs)
            
            QtWidgets.QMessageBox.information(
                self,
                "Export Complete",
                f"Successfully exported {len(images_list)} cells to:\n{file_path}\n\n"
                f"Shape: {images_array.shape}\n"
                f"Channels: {len(channels)}\n"
                f"Crop size: {crop_size}x{crop_size}"
            )
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Export Error",
                f"Error exporting to HDF5: {str(e)}"
            )
            import traceback
            traceback.print_exc()
        finally:
            progress_dlg.close()
    
    def _get_acquisition_label(self, acq_id):
        """Get a user-friendly label for an acquisition ID."""
        # Try to find the acquisition in the parent window
        parent_window = self.parent()
        if hasattr(parent_window, 'acquisitions'):
            for acq in parent_window.acquisitions:
                if acq.id == acq_id:
                    return acq.name
        return acq_id

    def _get_cluster_label(self, cluster_id):
        """Get annotated cluster name from parent dialog if available."""
        provider = self._label_provider or self.parent()
        if provider is not None and hasattr(provider, '_get_cluster_display_name'):
            try:
                return provider._get_cluster_display_name(cluster_id)
            except Exception:
                pass
        return f"Cluster {cluster_id}"
    
    def _create_image_widget(self, image, title, is_rgb=False):
        """Create a widget to display a cell image."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        
        # Create matplotlib figure
        fig = Figure(figsize=(2, 2))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        # Display image
        if is_rgb:
            ax.imshow(image)
        else:
            ax.imshow(image, cmap='gray')
        ax.set_title(title, fontsize=8)
        ax.axis('off')
        
        layout.addWidget(canvas)
        return widget


def _process_acquisition_export(unique_acq_id, original_acq_id, file_path, loader_type, cell_data_list, channels, crop_size, mask):
    """Worker function to process all cells for one acquisition.
    
    This function runs in a separate process and processes all cells for a single acquisition.
    Returns: (images_list, masks_list, features_list, cell_indices_list)
    """
    try:
        # Import here to avoid issues with multiprocessing
        from openimc.data.mcd_loader import MCDLoader
        from openimc.data.ometiff_loader import OMETIFFLoader
        from skimage.measure import regionprops
        
        # Create loader
        if loader_type == 'mcd' and file_path and os.path.isfile(file_path):
            loader = MCDLoader()
            loader.open(file_path)
        elif file_path and os.path.isdir(file_path):
            # OME-TIFF directory
            loader = OMETIFFLoader(channel_format='CHW')
            loader.open(file_path)
        else:
            # For single file or if we can't determine, we'll need to handle differently
            # This is a fallback - in practice, we should have file_path
            return None
        
        images_list = []
        masks_list = []
        features_list = []
        cell_indices_list = []
        
        # Process all cells for this acquisition
        for cell_data_dict in cell_data_list:
            cell_idx = cell_data_dict.pop('_cell_idx')
            cell_id = int(cell_data_dict['cell_id'])
            
            # Get cell mask
            cell_mask = (mask == cell_id).astype(np.uint8)
            
            if not np.any(cell_mask):
                continue
            
            # Get cell center
            props = regionprops(cell_mask)
            if not props:
                continue
            
            center_y, center_x = props[0].centroid
            center_y, center_x = int(center_y), int(center_x)
            
            # Define crop boundaries
            half_crop = crop_size // 2
            y_start = max(0, center_y - half_crop)
            y_end = min(mask.shape[0], center_y + half_crop)
            x_start = max(0, center_x - half_crop)
            x_end = min(mask.shape[1], center_x + half_crop)
            
            # Load all channels for this cell
            try:
                available_channels = loader.get_channels(original_acq_id)
                cell_image_channels = []
                
                for channel in channels:
                    if channel not in available_channels:
                        # Fill with zeros if channel not available
                        cell_image_channels.append(np.zeros((y_end - y_start, x_end - x_start), dtype=np.float32))
                    else:
                        try:
                            channel_img = loader.get_image(original_acq_id, channel)
                            cropped_channel = channel_img[y_start:y_end, x_start:x_end]
                            cell_image_channels.append(cropped_channel.astype(np.float32))
                        except Exception:
                            # Fill with zeros if loading fails
                            cell_image_channels.append(np.zeros((y_end - y_start, x_end - x_start), dtype=np.float32))
                
                # Stack channels: shape will be (H, W, C)
                cell_image = np.stack(cell_image_channels, axis=-1)
                
                # Crop mask to same size and convert to binary (0-1)
                cropped_mask = cell_mask[y_start:y_end, x_start:x_end].astype(np.float32)
                # Ensure binary (0 or 1)
                cropped_mask = (cropped_mask > 0).astype(np.float32)
                
                # Ensure same size (pad if necessary)
                if cell_image.shape[:2] != (crop_size, crop_size):
                    # Pad to crop_size
                    pad_y = crop_size - cell_image.shape[0]
                    pad_x = crop_size - cell_image.shape[1]
                    cell_image = np.pad(cell_image, ((0, pad_y), (0, pad_x), (0, 0)), mode='constant', constant_values=0)
                    cropped_mask = np.pad(cropped_mask, ((0, pad_y), (0, pad_x)), mode='constant', constant_values=0)
                
                images_list.append(cell_image)
                masks_list.append(cropped_mask)
                features_list.append(cell_data_dict)
                cell_indices_list.append(cell_idx)
                
            except Exception as e:
                print(f"[ClusterExplorer] Error processing cell {cell_id} in acquisition {unique_acq_id}: {e}")
                continue
        
        return (images_list, masks_list, features_list, cell_indices_list)
        
    except Exception as e:
        print(f"[ClusterExplorer] Error processing acquisition {unique_acq_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


class GatingRulesDialog(QtWidgets.QDialog):
    def __init__(self, rules, available_columns, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manual Gating Rules")
        self.setModal(True)
        
        # Set size to 90% of parent window if available
        if parent is not None:
            parent_size = parent.size()
            dialog_width = int(parent_size.width() * 0.9)
            dialog_height = int(parent_size.height() * 0.9)
            self.resize(dialog_width, dialog_height)
        
        self.setMinimumSize(700, 500)
        self._available_columns = list(sorted(set(available_columns)))
        self._rules = [r.copy() for r in (rules or [])]
        self._create_ui()

    def _create_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        # Existing rules list
        self.rules_list = QtWidgets.QListWidget()
        self._refresh_rules_list()
        layout.addWidget(self.rules_list)

        # Buttons + Save/Load
        btns = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("Add Rule")
        edit_btn = QtWidgets.QPushButton("Edit")
        del_btn = QtWidgets.QPushButton("Delete")
        load_btn = QtWidgets.QPushButton("Load…")
        save_btn = QtWidgets.QPushButton("Save…")
        btns.addWidget(add_btn)
        btns.addWidget(edit_btn)
        btns.addWidget(del_btn)
        btns.addSpacing(20)
        btns.addWidget(load_btn)
        btns.addWidget(save_btn)
        btns.addStretch()
        layout.addLayout(btns)

        # OK/Cancel
        ok_cancel = QtWidgets.QHBoxLayout()
        ok = QtWidgets.QPushButton("Apply")
        cancel = QtWidgets.QPushButton("Cancel")
        ok.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        ok_cancel.addStretch()
        ok_cancel.addWidget(ok)
        ok_cancel.addWidget(cancel)
        layout.addLayout(ok_cancel)

        # Wire actions
        add_btn.clicked.connect(self._on_add)
        edit_btn.clicked.connect(self._on_edit)
        del_btn.clicked.connect(self._on_delete)
        def do_load():
            from PyQt5 import QtWidgets, QtCore as _QtW
            import json
            path, _ = _QtW.QFileDialog.getOpenFileName(self, "Load Gating Rules", "", "JSON Files (*.json)")
            if not path:
                return
            try:
                with open(path, 'r') as f:
                    rules = json.load(f)
                if isinstance(rules, list):
                    self._rules = rules
                    self._refresh_rules_list()
                else:
                    raise ValueError("JSON must be a list of rules")
            except Exception as e:
                _QtW.QMessageBox.critical(self, "Load Error", f"Error loading gating rules: {str(e)}")
        def do_save():
            from PyQt5 import QtWidgets, QtCore as _QtW
            import json
            path, _ = _QtW.QFileDialog.getSaveFileName(self, "Save Gating Rules", "gating_rules.json", "JSON Files (*.json)")
            if not path:
                return
            try:
                with open(path, 'w') as f:
                    json.dump(self._rules, f, indent=2)
                _QtW.QMessageBox.information(self, "Saved", f"Gating rules saved to: {path}")
            except Exception as e:
                _QtW.QMessageBox.critical(self, "Save Error", f"Error saving gating rules: {str(e)}")
        load_btn.clicked.connect(do_load)
        save_btn.clicked.connect(do_save)

    def _refresh_rules_list(self):
        self.rules_list.clear()
        for r in self._rules:
            name = r.get('name', '(unnamed)')
            logic = r.get('logic', 'AND')
            conds = r.get('conditions', [])
            desc_parts = [f"{c.get('column')} {c.get('op')} {c.get('threshold')}" for c in conds]
            item = QtWidgets.QListWidgetItem(f"{name}  [{logic}]  ::  " + " AND ".join(desc_parts))
            self.rules_list.addItem(item)

    def _on_add(self):
        rule = self._edit_rule_dialog()
        if rule:
            self._rules.append(rule)
            self._refresh_rules_list()

    def _on_edit(self):
        row = self.rules_list.currentRow()
        if row < 0 or row >= len(self._rules):
            return
        rule = self._edit_rule_dialog(self._rules[row])
        if rule:
            self._rules[row] = rule
            self._refresh_rules_list()

    def _on_delete(self):
        row = self.rules_list.currentRow()
        if row < 0 or row >= len(self._rules):
            return
        del self._rules[row]
        self._refresh_rules_list()

    def _edit_rule_dialog(self, existing=None):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Edit Rule")
        v = QtWidgets.QVBoxLayout(dlg)

        # Name
        name_edit = QtWidgets.QLineEdit()
        if existing and existing.get('name'):
            name_edit.setText(existing['name'])
        form = QtWidgets.QFormLayout()
        form.addRow("Phenotype name:", name_edit)
        v.addLayout(form)

        # Logic
        logic_combo = QtWidgets.QComboBox()
        logic_combo.addItems(["AND", "OR"])
        if existing and existing.get('logic'):
            idx = logic_combo.findText(existing['logic'].upper())
            if idx >= 0:
                logic_combo.setCurrentIndex(idx)
        v.addWidget(QtWidgets.QLabel("Combine conditions with:"))
        v.addWidget(logic_combo)

        # Conditions table
        table = QtWidgets.QTableWidget(0, 3)
        table.setHorizontalHeaderLabels(["Feature", "Operator", "Threshold"])
        table.horizontalHeader().setStretchLastSection(True)
        v.addWidget(table)

        # Row buttons
        row_btns = QtWidgets.QHBoxLayout()
        add_row = QtWidgets.QPushButton("Add Condition")
        del_row = QtWidgets.QPushButton("Delete Condition")
        row_btns.addWidget(add_row)
        row_btns.addWidget(del_row)
        row_btns.addStretch()
        v.addLayout(row_btns)

        def add_condition_row(cond=None):
            r = table.rowCount()
            table.insertRow(r)
            # Feature combo
            feat = QtWidgets.QComboBox()
            feat.addItems(self._available_columns)
            if cond and cond.get('column') in self._available_columns:
                feat.setCurrentText(cond['column'])
            table.setCellWidget(r, 0, feat)
            # Operator combo
            op = QtWidgets.QComboBox()
            op.addItems(['>', '>=', '<', '<=', '==', '!='])
            if cond and cond.get('op'):
                idx = op.findText(cond['op'])
                if idx >= 0:
                    op.setCurrentIndex(idx)
            table.setCellWidget(r, 1, op)
            # Threshold edit
            thr = QtWidgets.QDoubleSpinBox()
            thr.setRange(-1e12, 1e12)
            thr.setDecimals(6)
            thr.setSingleStep(0.1)
            if cond and cond.get('threshold') is not None:
                try:
                    thr.setValue(float(cond['threshold']))
                except Exception:
                    pass
            table.setCellWidget(r, 2, thr)

        # Seed from existing
        if existing and existing.get('conditions'):
            for cond in existing['conditions']:
                add_condition_row(cond)
        else:
            add_condition_row()

        add_row.clicked.connect(lambda: add_condition_row())
        def delete_selected_rows():
            rows = sorted({i.row() for i in table.selectedIndexes()}, reverse=True)
            for r in rows:
                table.removeRow(r)
        del_row.clicked.connect(delete_selected_rows)

        # OK/Cancel
        okc = QtWidgets.QHBoxLayout()
        ok = QtWidgets.QPushButton("OK")
        cancel = QtWidgets.QPushButton("Cancel")
        ok.clicked.connect(dlg.accept)
        cancel.clicked.connect(dlg.reject)
        okc.addStretch()
        okc.addWidget(ok)
        okc.addWidget(cancel)
        v.addLayout(okc)

        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            # Build rule
            rule = {
                'name': name_edit.text().strip(),
                'logic': logic_combo.currentText(),
                'conditions': []
            }
            for r in range(table.rowCount()):
                feat = table.cellWidget(r, 0).currentText()
                op = table.cellWidget(r, 1).currentText()
                thr = table.cellWidget(r, 2).value()
                rule['conditions'].append({'column': feat, 'op': op, 'threshold': float(thr)})
            if rule['name'] and rule['conditions']:
                return rule
            return None
        return None

    def get_rules(self):
        return [r.copy() for r in self._rules]


class PhenotypeSuggestionDialog(QtWidgets.QDialog):
    def __init__(self, parent_dialog: 'CellClusteringDialog', cluster_ids, apply_callback, cache_dict=None, normalization_config=None, parent=None):
        super().__init__(parent or parent_dialog)
        self.setWindowTitle("Suggest Phenotypes with LLM (Based on Markers Used in Clustering)")
        self.setModal(True)
        self._parent_dialog = parent_dialog
        self._cluster_ids = list(cluster_ids)
        self._apply_callback = apply_callback
        self._cache_dict = cache_dict or {}
        self.normalization_config = normalization_config
        self._create_ui()
        # Resize dialog to 75% of the parent window size for better usability
        try:
            base_widget = parent_dialog if parent_dialog is not None else self.parent()
            if base_widget is not None:
                base_size = base_widget.size()
                w = max(600, int(base_size.width() * 0.75))
                h = max(400, int(base_size.height() * 0.75))
            else:
                # Fallback to primary screen available geometry
                screen = QtWidgets.QApplication.primaryScreen()
                geo = screen.availableGeometry() if screen is not None else QtCore.QRect(0, 0, 1200, 800)
                w = int(geo.width() * 0.75)
                h = int(geo.height() * 0.75)
            self.resize(w, h)
        except Exception:
            pass

    def _create_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()

        self.api_key_edit = QtWidgets.QLineEdit()
        self.api_key_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.api_key_edit.setPlaceholderText("sk-... OpenAI API Key")
        form.addRow("OpenAI API Key:", self.api_key_edit)

        self.context_edit = QtWidgets.QLineEdit()
        self.context_edit.setPlaceholderText("e.g., human colorectal cancer (optional)")
        form.addRow("Cohort/tissue context:", self.context_edit)

        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems(["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4.1", "gpt-5.1"])
        self.model_combo.setCurrentText("gpt-5.1")  # Set gpt-5.1 as default
        form.addRow("Model:", self.model_combo)

        # Reasoning level dropdown (only visible for gpt-5.1)
        self.reasoning_combo = QtWidgets.QComboBox()
        self.reasoning_combo.addItems(["none", "low", "medium", "high"])
        self.reasoning_combo.setCurrentText("none")  # Set none as default
        # Create a container widget that includes both label and combo for proper hiding
        # This ensures both the label and dropdown are hidden/shown together
        self._reasoning_container = QtWidgets.QWidget()
        reasoning_container_layout = QtWidgets.QHBoxLayout(self._reasoning_container)
        reasoning_container_layout.setContentsMargins(0, 0, 0, 0)
        reasoning_label = QtWidgets.QLabel("Reasoning level:")
        reasoning_container_layout.addWidget(reasoning_label)
        reasoning_container_layout.addWidget(self.reasoning_combo)
        reasoning_container_layout.addStretch()
        # Add as a row with empty label since we include the label in the container
        form.addRow("", self._reasoning_container)
        self._reasoning_container.hide()  # Hidden by default

        # System prompt selection (fine vs broad cell types)
        self.system_prompt_combo = QtWidgets.QComboBox()
        self.system_prompt_combo.addItems(["Fine cell types (detailed)", "Broad cell types (Myeloid, Tumor, Stroma, etc.)"])
        form.addRow("System prompt:", self.system_prompt_combo)

        # Feature mode selection for markers used in LLM prompt
        self.feature_mode_combo = QtWidgets.QComboBox()
        self.feature_mode_combo.addItems(["Markers only", "Morphometrics only", "Both"])
        form.addRow("Feature mode:", self.feature_mode_combo)

        # Per-type K controls
        self.k_int_spin = QtWidgets.QSpinBox()
        self.k_int_spin.setRange(1, 30)
        self.k_int_spin.setValue(5)
        self.k_morpho_spin = QtWidgets.QSpinBox()
        self.k_morpho_spin.setRange(1, 30)
        self.k_morpho_spin.setValue(5)

        # Container widgets for visibility toggling
        self._k_int_row = QtWidgets.QWidget()
        kint_layout = QtWidgets.QHBoxLayout(self._k_int_row)
        kint_layout.setContentsMargins(0,0,0,0)
        kint_layout.addWidget(self.k_int_spin)
        form.addRow("Top-K intensity:", self._k_int_row)

        self._k_morpho_row = QtWidgets.QWidget()
        kmorph_layout = QtWidgets.QHBoxLayout(self._k_morpho_row)
        kmorph_layout.setContentsMargins(0,0,0,0)
        kmorph_layout.addWidget(self.k_morpho_spin)
        form.addRow("Top-K morphometric:", self._k_morpho_row)

        # Default visibility
        def _update_feature_mode():
            mode = self.feature_mode_combo.currentText()
            if mode == "Markers only":
                self._k_int_row.show()
                self._k_morpho_row.hide()
            elif mode == "Morphometrics only":
                self._k_int_row.hide()
                self._k_morpho_row.show()
            else:
                self._k_int_row.show()
                self._k_morpho_row.show()
        self.feature_mode_combo.currentTextChanged.connect(lambda _t: _update_feature_mode())
        # Default to Both
        self.feature_mode_combo.setCurrentText("Both")
        _update_feature_mode()

        # Update reasoning level visibility when model changes
        def _update_reasoning_visibility():
            model = self.model_combo.currentText()
            if model == "gpt-5.1":
                self._reasoning_container.show()
            else:
                self._reasoning_container.hide()
        self.model_combo.currentTextChanged.connect(lambda _t: _update_reasoning_visibility())
        _update_reasoning_visibility()  # Set initial state

        layout.addLayout(form)

        btns = QtWidgets.QHBoxLayout()
        self.run_btn = QtWidgets.QPushButton("Run Suggestion")
        self.run_btn.clicked.connect(self._run)
        self.apply_btn = QtWidgets.QPushButton("Apply Names")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self._apply)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btns.addStretch()
        btns.addWidget(self.run_btn)
        btns.addWidget(self.apply_btn)
        btns.addWidget(close_btn)
        layout.addLayout(btns)

        # Progress bar for long-running suggestions
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        layout.addWidget(self.progress)

        # Area to render per-cluster choices after suggestions arrive (scrollable for many clusters)
        self.choices_widget = QtWidgets.QWidget()
        self.choices_layout = QtWidgets.QVBoxLayout(self.choices_widget)
        self.choices_layout.setContentsMargins(8, 8, 8, 8)
        self.choices_layout.setSpacing(8)
        self.choices_scroll = QtWidgets.QScrollArea()
        self.choices_scroll.setWidgetResizable(True)
        self.choices_scroll.setWidget(self.choices_widget)
        layout.addWidget(self.choices_scroll)

        # Holds QButtonGroup per cluster for selection
        self._cluster_choice_groups = {}
        
        # Check for cached results and display them immediately
        self._check_and_display_cached_results()

        self._suggestions = {}  # cluster_id -> parsed json

    def closeEvent(self, event):
        """Handle dialog closing to preserve cache."""
        # Ensure the current suggestions are cached for future use
        if self._cache_dict is not None and self._suggestions:
            self._cache_dict.update(self._suggestions)
        event.accept()

    def _reset_progress_bar(self):
        """Reset the progress bar to its default state."""
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("")
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Run Suggestion")

    def _check_and_display_cached_results(self):
        """Check if we have cached results for the current cluster set and display them."""
        if not self._cache_dict:
            return
            
        # Check if we have cached results for all current clusters
        cached_results = {}
        for cid in self._cluster_ids:
            # Convert cluster ID to int for consistent comparison
            cid_int = int(cid)
            # Check both the original cid and converted int version
            if cid in self._cache_dict:
                cached_results[cid] = self._cache_dict[cid]
            elif cid_int in self._cache_dict:
                cached_results[cid] = self._cache_dict[cid_int]
        
        # If we have cached results for all clusters, display them
        if cached_results and len(cached_results) == len(self._cluster_ids):
            # Store the cached results in _suggestions so they can be applied
            self._suggestions = cached_results.copy()
            self._render_choices(cached_results)
            self.apply_btn.setEnabled(True)
            # Disable the run button since we already have results
            self.run_btn.setEnabled(False)
            self.run_btn.setText("Results Cached - Re-run to refresh")
        else:
            pass

    def _apply(self):
        display_name_map = {}
        backend_name_map = {}
        # Prefer user-selected guess when available
        for cid, obj in self._suggestions.items():
            # Ensure cid is an integer for consistent handling
            cid_int = int(cid)
            try:
                selected_idx = None
                grp = self._cluster_choice_groups.get(cid_int)
                if grp is not None:
                    id_ = grp.checkedId()
                    if id_ != -1:
                        selected_idx = id_
                guesses = obj.get('phenotype_guesses') or []
                chosen = None
                if selected_idx is not None and 0 <= selected_idx < len(guesses):
                    chosen = guesses[selected_idx]
                elif guesses:
                    chosen = guesses[0]
                if chosen:
                    name = str(chosen.get('name', '')).strip()
                    if name:
                        # Store human-readable name for display
                        display_name_map[cid_int] = name
                        
                        # Create normalized name for backend CSV
                        norm = name.replace(' ', '_')
                        if norm.lower() == 't_cell':
                            norm = 'T_cell'
                        if 'macrophage' in norm.lower():
                            norm = 'Myeloid_Macrophage'
                        backend_name_map[cid_int] = norm
            except Exception:
                continue
        if display_name_map:
            self._apply_callback(display_name_map, backend_name_map)
            # Ensure the current suggestions are cached for future use
            if self._cache_dict is not None and self._suggestions:
                self._cache_dict.update(self._suggestions)
            QtWidgets.QMessageBox.information(self, "Applied", f"Applied {len(display_name_map)} suggested names.")

    def _debug_validate_payload(self, payload: dict) -> bool:
        try:
            ok = True
            # Basic keys
            if not isinstance(payload, dict):
                return False
            if 'input' not in payload:
                ok = False
            if 'model' not in payload:
                ok = False
            # Input structure
            msgs = payload.get('input')
            if not isinstance(msgs, list) or len(msgs) < 2:
                ok = False
            else:
                # Check roles and content types
                roles = [m.get('role') for m in msgs if isinstance(m, dict)]
                if roles[:2] != ['system', 'user']:
                    pass
                for m in msgs:
                    content = m.get('content') if isinstance(m, dict) else None
                    if not isinstance(content, list):
                        ok = False
                        break
                    for block in content:
                        if not isinstance(block, dict) or block.get('type') != 'input_text' or 'text' not in block:
                            ok = False
                            break
            # Ensure user context JSON parses back
            try:
                user_blocks = msgs[1]['content'] if isinstance(msgs, list) and len(msgs) > 1 else []
                user_texts = [b.get('text') for b in user_blocks if isinstance(b, dict) and b.get('type') == 'input_text']
                if user_texts:
                    json.loads(user_texts[0])
            except Exception as e:
                ok = False
            return ok
        except Exception as e:
            return False

    def _run(self):
        # Disable the run button to prevent multiple clicks
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Processing...")
        
        # Show immediate feedback that processing has started
        self.progress.setRange(0, 0)  # Indeterminate progress
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("Starting LLM analysis...")
        QtWidgets.QApplication.processEvents()  # Force UI update
        
        try:
            api_key = self.api_key_edit.text().strip()
            if not api_key:
                self._reset_progress_bar()
                QtWidgets.QMessageBox.warning(self, "API Key Required", "Please enter an OpenAI API key.")
                return
            if self._parent_dialog.clustered_data is None:
                self._reset_progress_bar()
                QtWidgets.QMessageBox.warning(self, "No Clusters", "Run clustering first.")
                return
            
            mode = self.feature_mode_combo.currentText()
            k_int = self.k_int_spin.value()
            k_morpho = self.k_morpho_spin.value()
            # Use k_int as the K parameter for consistency
            self.progress.setFormat("Computing cluster statistics...")
            QtWidgets.QApplication.processEvents()
            stats_per_cluster = self._compute_stats(self._parent_dialog.clustered_data, K=k_int, mode=mode, k_int=k_int, k_morpho=k_morpho)
            results = {}
            total = max(1, len(self._cluster_ids))
            self.progress.setRange(0, total)
            self.progress.setValue(0)
            self.progress.setFormat("Processing clusters with LLM...")
            QtWidgets.QApplication.processEvents()
            for idx, cid in enumerate(self._cluster_ids, start=1):
                context_str = self.context_edit.text().strip() or "IMC panel of single cells"
                payload = self._build_prompt_payload(cid, stats_per_cluster.get(cid, {}), k_int, context_str)
                payload['model'] = self.model_combo.currentText()
                # Validate payload before sending
                self._debug_validate_payload(payload)
                suggestion = self._call_openai(api_key, payload)
                # Validate JSON
                obj = self._validate_json(suggestion, cid)
                if obj is None:
                    # One retry with repair instruction
                    suggestion = self._call_openai(api_key, payload, repair=True)
                    obj = self._validate_json(suggestion, cid)
                if obj is not None:
                    # Store with consistent integer keys
                    cid_int = int(cid)
                    results[cid_int] = obj
                    self._suggestions[cid_int] = obj
                # Update progress
                self.progress.setValue(idx)
                QtWidgets.QApplication.processEvents()
            if results:
                # Cache the results
                if self._cache_dict is not None:
                    self._cache_dict.update(results)
                self._render_choices(results)
                self.apply_btn.setEnabled(True)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "LLM Error", f"Error suggesting phenotypes: {str(e)}")
        finally:
            # Reset progress bar and button state
            self._reset_progress_bar()

    def _numeric_feature_columns(self, df: pd.DataFrame):
        exclude = {'cluster', 'cluster_phenotype', 'manual_phenotype'}
        cols = []
        for c in df.columns:
            if c in exclude:
                continue
            try:
                if pd.api.types.is_numeric_dtype(df[c]):
                    cols.append(c)
            except Exception:
                continue
        return cols

    def _base_marker_name(self, feature_name: str) -> str:
        # Strip trailing suffix like _mean/_median/etc.
        if '_' in feature_name:
            return feature_name.rsplit('_', 1)[0]
        return feature_name

    def _synonymize(self, name: str) -> str:
        m = name
        repl = {
            'KRT8/18': 'CK8/18',
            'EPCAM': 'EpCAM',
            'KRT8': 'CK8',
            'KRT18': 'CK18',
        }
        k = m.upper()
        for src, dst in repl.items():
            if src == k:
                return dst
        return name

    def _compute_stats(self, clustered_df: pd.DataFrame, K: int=None, mode: str="Both", k_int: int=5, k_morpho: int=5):
        # Use K as default for k_int and k_morpho if not provided
        if K is not None:
            k_int = K
            k_morpho = K
        from math import log2
        try:
            from sklearn.metrics import roc_auc_score
        except Exception:
            roc_auc_score = None
        eps = 1e-6
        cols = self._numeric_feature_columns(clustered_df)
        # Split into intensity vs morphometric by suffix
        intensity_suffixes = ['_mean', '_median', '_std', '_mad', '_p10', '_p90', '_integrated', '_frac_pos']
        intensity_cols = [c for c in cols if any(c.endswith(s) for s in intensity_suffixes)]
        morpho_cols = [c for c in cols if c not in intensity_cols]
        
        # Filter out DNA markers (positive in all cells) and ICSK membrane markers
        exclude_tokens = ['DNA1', 'DNA2', 'DNA_', 'IR191', 'IR193', 'ICSK']
        excluded_markers_int = [col for col in intensity_cols if any(tok in col.upper() for tok in exclude_tokens)]
        intensity_cols = [col for col in intensity_cols if col not in excluded_markers_int]
        # For morphometrics keep all (no DNA markers)

        # Choose working columns based on mode
        if mode == 'Markers only':
            work_cols = intensity_cols
        elif mode == 'Morphometrics only':
            work_cols = morpho_cols
        else:
            work_cols = intensity_cols + morpho_cols
        
        # Prepare per-cluster means
        cluster_ids = sorted(clustered_df['cluster'].unique())
        means = clustered_df.groupby('cluster')[work_cols].mean()
        stats = {}
        # Precompute across-cluster ranges per feature (based on means per cluster)
        across_range = (means.max(axis=0) - means.min(axis=0))
        for cid in cluster_ids:
            this_mean = means.loc[cid]
            rest_mean = means.drop(index=cid).mean()
            # z across clusters per feature
            col_means = means.mean(axis=0)
            col_stds = means.std(axis=0).replace(0, np.nan)
            z = (this_mean - col_means) / col_stds
            # Within-cluster distribution stats
            in_cluster = clustered_df[clustered_df['cluster'] == cid]
            in_min = in_cluster[work_cols].min(axis=0)
            in_max = in_cluster[work_cols].max(axis=0)
            in_mean = in_cluster[work_cols].mean(axis=0)
            in_median = in_cluster[work_cols].median(axis=0)
            # logFC with robust clipping to avoid log of non-positive values
            ratio = (this_mean + eps) / (rest_mean + eps)
            # Replace inf/-inf with NaN and non-positive ratios with NaN
            ratio = ratio.replace([np.inf, -np.inf], np.nan)
            ratio = ratio.where(ratio > 0, np.nan)
            with np.errstate(divide='ignore', invalid='ignore'):
                logfc = np.log2(ratio)
            # AUROC
            auroc = pd.Series(index=work_cols, dtype=float)
            if roc_auc_score is not None:
                labels = (clustered_df['cluster'] == cid).astype(int).values
                for f in work_cols:
                    vals = clustered_df[f].values
                    try:
                        if labels.sum() > 0 and labels.sum() < len(labels):
                            auroc[f] = roc_auc_score(labels, vals)
                        else:
                            auroc[f] = np.nan
                    except Exception:
                        auroc[f] = np.nan
            else:
                auroc[:] = np.nan
            # pct_pos at threshold tau (0 by default on normalized scale)
            tau = 0.0
            out_cluster = clustered_df[clustered_df['cluster'] != cid]
            pct_pos_in = (in_cluster[work_cols] > tau).sum(axis=0) / max(1, len(in_cluster))
            pct_pos_out = (out_cluster[work_cols] > tau).sum(axis=0) / max(1, len(out_cluster))

            # Ranking: by z-score only (descending)
            ranked = z.sort_values(ascending=False).index.tolist()
            # Select counts based on mode
            if mode == 'Both':
                # Split selection across intensity and morpho
                ranked_int = [f for f in ranked if f in intensity_cols][:k_int]
                ranked_morpho = [f for f in ranked if f in morpho_cols][:k_morpho]
                selected_up = ranked_int + ranked_morpho
            else:
                k = k_int if mode == 'Markers only' else k_morpho
                selected_up = ranked[:k]
            top_up = []
            for f in selected_up:
                base = self._synonymize(self._base_marker_name(f))
                top_up.append({
                    'marker': base,
                    'auroc': None if pd.isna(auroc[f]) else float(auroc[f]),
                    'logFC': None if pd.isna(logfc[f]) else float(logfc[f]),
                    'z': None if pd.isna(z[f]) else float(z[f]),
                    'mean': None if pd.isna(this_mean[f]) else float(this_mean[f]),
                    'pct_pos': None if pd.isna(pct_pos_in[f]) else float(pct_pos_in[f]),
                    'within_min': None if pd.isna(in_min[f]) else float(in_min[f]),
                    'within_mean': None if pd.isna(in_mean[f]) else float(in_mean[f]),
                    'within_median': None if pd.isna(in_median[f]) else float(in_median[f]),
                    'within_max': None if pd.isna(in_max[f]) else float(in_max[f]),
                    'range_across_clusters': None if pd.isna(across_range[f]) else float(across_range[f])
                })
            # Down markers: lowest z-scores — take bottom K as requested
            down_ranked = z.sort_values(ascending=True).index.tolist()
            if mode == 'Both':
                down_int = [f for f in down_ranked if f in intensity_cols][:k_int]
                down_morpho = [f for f in down_ranked if f in morpho_cols][:k_morpho]
                selected_down = down_int + down_morpho
            else:
                k = k_int if mode == 'Markers only' else k_morpho
                selected_down = down_ranked[:k]
            top_down = []
            for f in selected_down:
                base = self._synonymize(self._base_marker_name(f))
                top_down.append({
                    'marker': base,
                    'auroc': None if pd.isna(auroc[f]) else float(auroc[f]),
                    'logFC': None if pd.isna(logfc[f]) else float(logfc[f]),
                    'z': None if pd.isna(z[f]) else float(z[f]),
                    'within_min': None if pd.isna(in_min[f]) else float(in_min[f]),
                    'within_mean': None if pd.isna(in_mean[f]) else float(in_mean[f]),
                    'within_median': None if pd.isna(in_median[f]) else float(in_median[f]),
                    'within_max': None if pd.isna(in_max[f]) else float(in_max[f]),
                    'range_across_clusters': None if pd.isna(across_range[f]) else float(across_range[f])
                })
            stats[cid] = {
                'top_up': top_up,
                'top_down': top_down,
            }
        return stats

    def _render_choices(self, results: dict):
        # Clear previous choices
        for i in reversed(range(self.choices_layout.count())):
            item = self.choices_layout.takeAt(i)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._cluster_choice_groups = {}

        # Build UI for each cluster
        for cid in sorted(results.keys(), key=lambda x: int(str(x)) if str(x).isdigit() else str(x)):
            obj = results[cid]
            guesses = obj.get('phenotype_guesses') or []

            group_box = QtWidgets.QGroupBox(f"Cluster {cid} – Select phenotype")
            v = QtWidgets.QVBoxLayout(group_box)
            btn_group = QtWidgets.QButtonGroup(group_box)
            btn_group.setExclusive(True)

            # Create a radio per guess and show rationale with confidence
            for idx, g in enumerate(guesses):
                name = str(g.get('name', '')).strip() or 'Unknown'
                confidence = g.get('confidence')
                if confidence is not None:
                    confidence_str = f"{confidence:.1f}%" if isinstance(confidence, (int, float)) else str(confidence)
                    display_name = f"{name} ({confidence_str})"
                else:
                    display_name = name
                rb = QtWidgets.QRadioButton(display_name)
                btn_group.addButton(rb, idx)
                if idx == 0:
                    rb.setChecked(True)
                v.addWidget(rb)
                rationale = str(g.get('rationale', '')).strip()
                if rationale:
                    rationale_lbl = QtWidgets.QLabel(rationale)
                    rationale_lbl.setWordWrap(True)
                    rationale_lbl.setStyleSheet("color: #555;")
                    v.addWidget(rationale_lbl)

            # If no guesses, indicate
            if not guesses:
                v.addWidget(QtWidgets.QLabel("No plausible types returned."))

            self._cluster_choice_groups[int(cid)] = btn_group
            self.choices_layout.addWidget(group_box)

        self.choices_layout.addStretch(1)

    def _build_prompt_payload(self, cid, stat_obj, K: int, context_str: str):
        # Select system prompt based on user choice
        prompt_type = self.system_prompt_combo.currentText()
        if "Broad" in prompt_type:
            # Broad cell types system prompt
            system_prompt = (
                "You are assisting with IMC cell type suggestions. Use only the provided marker statistics. "
                "Classify cells into broad categories: Myeloid (macrophages, monocytes, dendritic cells, neutrophils, etc.), "
                "Tumor (cancer cells), Stroma (fibroblasts, endothelial cells, etc.), Lymphoid (T cells, B cells, NK cells, etc.), "
                "or other broad categories as appropriate. Give exactly 3 plausible phenotypes per cluster. "
                "When you have high confidence (typically when confidence is above 50%), you may specify more specific cell types within the broad categories. "
                "For example, within Lymphoid you can specify 'T cells', 'B cells', or 'NK cells' if the marker profile strongly supports it. "
                "Within Myeloid, you can specify 'Macrophages', 'Dendritic cells', 'Neutrophils', etc. if confident. "
                "For lower confidence predictions, use the broad category names (e.g., 'Lymphoid', 'Myeloid', 'Stroma', 'Tumor'). "
                "Rank the 3 phenotypes from most likely to least likely, and provide a confidence percentage for each phenotype. "
                "The three confidence percentages must sum to exactly 100%. "
                "Consider the range of values across and within clusters for each marker to help determine if the marker is truly unique to the cluster. "
                "Consider if the z-score and mean value is true expression of the marker or if it is due to noise. "
                "Try to give varied phenotypes, rather than the same phenotype with different names. "
                "Focus on different marker combinations to avoid giving the same phenotype with different names. "
                "If uncertain, return \"Unknown\". Output valid JSON exactly matching the given schema. Do not invent markers. "
                "Return valid JSON only and no prose/explanations."
            )
        else:
            # Fine cell types system prompt (default)
            system_prompt = (
                "You are assisting with IMC cell type suggestions. Use only the provided marker statistics. "
                "Prefer canonical immune/epithelial/stromal names and give exactly 3 plausible phenotypes per cluster. "
                "Rank the 3 phenotypes from most likely to least likely, and provide a confidence percentage for each phenotype. "
                "The three confidence percentages must sum to exactly 100%. "
                "Consider the range of values across and within clusters for each marker to help determine if the marker is truly unique to the cluster. "
                "Consider if the z-score and mean value is true expression of the marker or if it is due to noise. "
                "Try to give varied phenotypes, rather than the same phenotype with different names. "
                "Focus on different marker combinations to avoid giving the same phenotype with different names. "
                "If uncertain, return \"Unknown\". Output valid JSON exactly matching the given schema. Do not invent markers. "
                "Return valid JSON only and no prose/explanations."
            )
        schema = {
            "cluster_id": str(cid),
            "phenotype_guesses": [ { "name": "", "rationale": "", "confidence": 0 } ],
            "key_markers_positive": [],
            "key_markers_negative": [],
            "notes": ""
        }
        # Determine if arcsinh transformation was used during feature extraction
        arcsinh_used = (self.normalization_config is not None and 
                       self.normalization_config.get('method') == 'arcsinh')
        
        # Set semantics based on whether arcsinh transformation was applied
        if arcsinh_used:
            semantics = 'intensities are arcsinh-transformed; higher = more expression'
        else:
            semantics = 'intensities are raw values; higher = more expression'
        
        user_context = {
            'context': context_str,
            'semantics': semantics,
            'cluster_id': str(cid),
            'top_up': stat_obj.get('top_up', []),
            'top_down': stat_obj.get('top_down', []),
        }
        # Build Responses API input structure
        input_msgs = [
            {
                "role": "system",
                "content": [
                    {"type": "input_text", "text": system_prompt + " Schema: {\\n  \\\"cluster_id\\\": \\\"string\\\",\\n  \\\"phenotype_guesses\\\": [\\n    { \\\"name\\\": \\\"string\\\", \\\"rationale\\\": \\\"string\\\", \\\"confidence\\\": number }\\n  ],\\n  \\\"key_markers_positive\\\": [\\\"string\\\"],\\n  \\\"key_markers_negative\\\": [\\\"string\\\"],\\n  \\\"notes\\\": \\\"string\\\"\\n}"}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": json.dumps(user_context)}
                ]
            }
        ]
        return {
            'model': 'gpt-5',
            'temperature': 0.1,
            'max_tokens': 2000,
            'input': input_msgs
        }

    def _call_openai(self, api_key: str, payload: dict, repair: bool=False) -> str:
        # Use OpenAI official SDK per developer guide
        from openai import OpenAI
        
        client = OpenAI(api_key=api_key, timeout=30.0)
        data = payload.copy()
        # Append repair instruction as another system content block if needed
        input_payload = data.get('input')
        if repair and isinstance(input_payload, list):
            input_payload = input_payload + [{"role": "system", "content": [{"type": "input_text", "text": "Return valid JSON only, no prose."}]}]
        # Debug to console: request meta
        try:
            pass
        except Exception:
            pass
        # Responses API call
        try:
            model_name = data.get('model', 'gpt-5')
            create_kwargs = {
                'model': model_name,
                'max_output_tokens': data.get('max_tokens', 2000),
                'input': input_payload
            }
            # Only add reasoning parameter for gpt-5.1
            if model_name == 'gpt-5.1':
                reasoning_level = self.reasoning_combo.currentText()
                if reasoning_level != 'none':
                    create_kwargs['reasoning'] = {'effort': reasoning_level}
            resp = client.responses.create(**create_kwargs)
        except Exception as e:
            error_msg = str(e)
            
            # Provide more specific error information
            if "connection" in error_msg.lower() or "timeout" in error_msg.lower():
                raise Exception(f"Connection error: {error_msg}. Please check your internet connection and try again.")
            elif "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
                raise Exception(f"Authentication error: {error_msg}. Please check your API key.")
            elif "rate_limit" in error_msg.lower():
                raise Exception(f"Rate limit exceeded: {error_msg}. Please wait a moment and try again.")
            elif "model" in error_msg.lower():
                raise Exception(f"Model error: {error_msg}. Please try a different model.")
            else:
                raise Exception(f"OpenAI API error: {error_msg}")
        # SDK v1.42+ provides output_text for assembled content
        content = getattr(resp, 'output_text', None)
        if not content:
            try:
                # Fallback: extract text from output items
                pieces = []
                for item in getattr(resp, 'output', []) or []:
                    for block in item.get('content', []) or []:
                        if block.get('type') in ('output_text', 'summary_text'):
                            pieces.append(block.get('text', ''))
                content = "\n".join([p for p in pieces if p]) or "{}"
            except Exception:
                content = "{}"
        # If response looks empty/minimal, dump a truncated raw view to console for debugging
        try:
            pass
        except Exception:
            pass
        return content

    def _validate_json(self, s: str, cid) -> dict:
        try:
            obj = json.loads(s)
            # Basic schema checks
            if str(obj.get('cluster_id', '')) != str(cid):
                obj['cluster_id'] = str(cid)
            if not isinstance(obj.get('phenotype_guesses', []), list):
                obj['phenotype_guesses'] = []
            # Ensure confidence values are present and validate they sum to 100%
            guesses = obj.get('phenotype_guesses', [])
            if guesses:
                confidences = []
                for g in guesses:
                    if 'confidence' not in g:
                        g['confidence'] = None
                    else:
                        conf = g.get('confidence')
                        if conf is not None:
                            try:
                                confidences.append(float(conf))
                            except (ValueError, TypeError):
                                confidences.append(None)
                # If all confidences are provided, normalize to sum to 100
                if confidences and all(c is not None for c in confidences):
                    total = sum(confidences)
                    if total > 0 and abs(total - 100.0) > 0.1:  # Allow small floating point differences
                        # Normalize to sum to 100
                        for i, g in enumerate(guesses):
                            if i < len(confidences) and confidences[i] is not None:
                                g['confidence'] = round((confidences[i] / total) * 100.0, 1)
            if not isinstance(obj.get('key_markers_positive', []), list):
                obj['key_markers_positive'] = []
            if not isinstance(obj.get('key_markers_negative', []), list):
                obj['key_markers_negative'] = []
            if 'notes' not in obj:
                obj['notes'] = ""
            return obj
        except Exception:
            return None
