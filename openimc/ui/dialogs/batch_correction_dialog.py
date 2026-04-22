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
Batch Correction Dialog for OpenIMC

This module provides batch correction capabilities using Combat or Harmony
to correct for batch effects in feature data from multiple files.
"""

from typing import Optional, Dict, List
import os
import importlib.util
from datetime import datetime
import pandas as pd
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

from openimc.processing.batch_correction import (
    get_feature_columns_from_dataframe,
    validate_batch_correction_inputs
)
from openimc.core import batch_correction
from openimc.ui.dialogs.progress_dialog import run_blocking_task_with_progress
from openimc.ui.dialogs.custom_grouping_dialog import CustomGroupingDialog
from openimc.utils.logger import get_logger

def _optional_dependency_available(module_name: str) -> bool:
    """Check for an optional dependency without importing it."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


_HAVE_COMBAT = _optional_dependency_available("combat.pycombat")
_HAVE_HARMONY = _optional_dependency_available("harmonypy")



class BatchCorrectionDialog(QtWidgets.QDialog):
    """Dialog for batch correction of feature data."""
    
    def __init__(self, feature_dataframe: Optional[pd.DataFrame] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Correction")
        self.setModal(True)
        
        # Set dialog size to match parent window height
        if parent:
            parent_size = parent.size()
            dialog_width = int(parent_size.width() * 0.8)
            dialog_height = parent_size.height()  # Same height as main window
            self.resize(dialog_width, dialog_height)
        else:
            self.resize(900, 700)
        
        self.feature_dataframe = feature_dataframe
        self.corrected_dataframe: Optional[pd.DataFrame] = None
        self.custom_grouping: Optional[Dict[str, str]] = None  # Maps acquisition_id -> group_name
        
        # Store metadata files: {file_path: {'filename_column': str, 'dataframe': pd.DataFrame}}
        self.metadata_files: Dict[str, Dict[str, any]] = {}
        
        self._create_ui()
        self._update_ui_state()
    
    def _create_ui(self):
        """Create the user interface."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        
        # Create scroll area
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        
        scroll_content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(scroll_content)
        content_layout.setContentsMargins(5, 5, 5, 5)
        content_layout.setSpacing(8)
        
        # Information section
        info_group = QtWidgets.QGroupBox("Information")
        info_layout = QtWidgets.QVBoxLayout(info_group)
        info_layout.setContentsMargins(8, 8, 8, 8)
        info_layout.setSpacing(4)
        
        info_label = QtWidgets.QLabel(
            "Batch correction removes technical variation (batch effects) between different files or batches.\n"
            "This is useful when combining features from multiple .mcd files or uploaded feature files.\n\n"
            "You can load additional feature files extracted by this app, or use the currently loaded features.\n\n"
            "Note: All features will be preserved in their original state in the CSV if they are not batch corrected."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("QLabel { color: #666; font-size: 9pt; }")
        info_layout.addWidget(info_label)
        
        content_layout.addWidget(info_group)
        
        # Data source section
        source_group = QtWidgets.QGroupBox("Data Source")
        source_layout = QtWidgets.QVBoxLayout(source_group)
        source_layout.setContentsMargins(8, 8, 8, 8)
        source_layout.setSpacing(4)
        
        # Radio buttons for data source
        self.use_current_radio = QtWidgets.QRadioButton("Use currently loaded features")
        self.use_current_radio.setChecked(True)
        self.use_current_radio.toggled.connect(self._on_source_changed)
        source_layout.addWidget(self.use_current_radio)
        
        self.load_files_radio = QtWidgets.QRadioButton("Load feature files (CSV)")
        self.load_files_radio.toggled.connect(self._on_source_changed)
        source_layout.addWidget(self.load_files_radio)
        
        # File list for loaded files (hidden by default)
        self.file_list_widget = QtWidgets.QWidget()
        file_list_layout = QtWidgets.QVBoxLayout(self.file_list_widget)
        file_list_layout.setContentsMargins(20, 4, 0, 4)
        
        file_list_label = QtWidgets.QLabel("Loaded feature files:")
        file_list_layout.addWidget(file_list_label)
        
        self.file_list = QtWidgets.QListWidget()
        self.file_list.setMaximumHeight(100)
        self.file_list.setEnabled(False)
        file_list_layout.addWidget(self.file_list)
        
        file_buttons_layout = QtWidgets.QHBoxLayout()
        self.add_file_btn = QtWidgets.QPushButton("Add File...")
        self.add_file_btn.setEnabled(False)
        self.add_file_btn.clicked.connect(self._add_feature_file)
        file_buttons_layout.addWidget(self.add_file_btn)
        
        self.remove_file_btn = QtWidgets.QPushButton("Remove Selected")
        self.remove_file_btn.setEnabled(False)
        self.remove_file_btn.clicked.connect(self._remove_feature_file)
        file_buttons_layout.addWidget(self.remove_file_btn)
        file_buttons_layout.addStretch()
        file_list_layout.addLayout(file_buttons_layout)
        
        self.file_list_widget.setVisible(False)  # Hidden by default
        source_layout.addWidget(self.file_list_widget)
        
        content_layout.addWidget(source_group)
        
        # Metadata upload section
        metadata_group = QtWidgets.QGroupBox("Additional Metadata (Optional)")
        metadata_layout = QtWidgets.QVBoxLayout(metadata_group)
        metadata_layout.setContentsMargins(8, 8, 8, 8)
        metadata_layout.setSpacing(4)

        metadata_header_layout = QtWidgets.QHBoxLayout()
        self.metadata_summary_label = QtWidgets.QLabel("No metadata files configured.")
        self.metadata_summary_label.setStyleSheet("QLabel { color: #666; font-size: 8pt; }")
        metadata_header_layout.addWidget(self.metadata_summary_label, 1)
        self.metadata_toggle_btn = QtWidgets.QPushButton("Show Details")
        self.metadata_toggle_btn.setCheckable(True)
        self.metadata_toggle_btn.toggled.connect(self._set_metadata_expanded)
        metadata_header_layout.addWidget(self.metadata_toggle_btn)
        metadata_layout.addLayout(metadata_header_layout)

        self.metadata_content = QtWidgets.QWidget()
        metadata_content_layout = QtWidgets.QVBoxLayout(self.metadata_content)
        metadata_content_layout.setContentsMargins(0, 0, 0, 0)
        metadata_content_layout.setSpacing(4)

        metadata_info = QtWidgets.QLabel(
            "Attach optional metadata CSVs by matching a filename column to the features "
            "`source_file` field, which stores the source data file basename "
            "(for MCD workflows, usually the source .mcd file)."
        )
        metadata_info.setWordWrap(True)
        metadata_info.setStyleSheet("QLabel { color: #666; font-size: 8pt; }")
        metadata_content_layout.addWidget(metadata_info)

        self.metadata_list = QtWidgets.QListWidget()
        self.metadata_list.setMaximumHeight(110)
        self.metadata_list.setToolTip("Double-click a file to configure which metadata column matches source_file.")
        self.metadata_list.itemDoubleClicked.connect(self._configure_metadata_file)
        self.metadata_list.itemSelectionChanged.connect(self._update_metadata_controls)
        metadata_content_layout.addWidget(self.metadata_list)

        metadata_buttons_layout = QtWidgets.QHBoxLayout()
        self.add_metadata_btn = QtWidgets.QPushButton("Add Metadata...")
        self.add_metadata_btn.clicked.connect(self._add_metadata_file)
        metadata_buttons_layout.addWidget(self.add_metadata_btn)

        self.configure_metadata_btn = QtWidgets.QPushButton("Configure Selected...")
        self.configure_metadata_btn.clicked.connect(self._configure_metadata_file)
        metadata_buttons_layout.addWidget(self.configure_metadata_btn)

        self.remove_metadata_btn = QtWidgets.QPushButton("Remove Selected")
        self.remove_metadata_btn.clicked.connect(self._remove_metadata_file)
        metadata_buttons_layout.addWidget(self.remove_metadata_btn)
        metadata_buttons_layout.addStretch()
        metadata_content_layout.addLayout(metadata_buttons_layout)

        metadata_layout.addWidget(self.metadata_content)
        
        content_layout.addWidget(metadata_group)
        
        # Batch correction method section
        method_group = QtWidgets.QGroupBox("Batch Correction Method")
        method_layout = QtWidgets.QVBoxLayout(method_group)
        method_layout.setContentsMargins(8, 8, 8, 8)
        method_layout.setSpacing(4)
        
        method_label = QtWidgets.QLabel("Select batch correction method:")
        method_layout.addWidget(method_label)
        
        self.method_combo = QtWidgets.QComboBox()
        # Add methods in priority order: Harmony, Combat
        if _HAVE_HARMONY:
            self.method_combo.addItem("Harmony")
        if _HAVE_COMBAT:
            self.method_combo.addItem("Combat")
        
        # Set default to Harmony if available, otherwise Combat
        if _HAVE_HARMONY:
            self.method_combo.setCurrentText("Harmony")
        elif _HAVE_COMBAT:
            self.method_combo.setCurrentText("Combat")
        
        if self.method_combo.count() == 0:
            self.method_combo.addItem("No methods available")
            self.method_combo.setEnabled(False)
            no_methods_label = QtWidgets.QLabel(
                "Please install batch correction libraries:\n"
                "  - Combat: pip install combat\n"
                "  - Harmony: pip install harmonypy"
            )
            no_methods_label.setStyleSheet("QLabel { color: #d9534f; font-size: 9pt; }")
            method_layout.addWidget(no_methods_label)
        
        method_layout.addWidget(self.method_combo)
        self.method_combo.currentTextChanged.connect(self._on_method_changed)
        
        # PCA variance threshold for Harmony (hidden by default, shown only for Harmony)
        self.pca_variance_layout = QtWidgets.QHBoxLayout()
        self.pca_variance_layout.addWidget(QtWidgets.QLabel("PCA variance to retain:"))
        self.pca_variance_spin = QtWidgets.QDoubleSpinBox()
        self.pca_variance_spin.setRange(0.1, 1.0)
        self.pca_variance_spin.setSingleStep(0.05)
        self.pca_variance_spin.setValue(0.9)
        self.pca_variance_spin.setDecimals(2)
        self.pca_variance_spin.setToolTip(
            "Proportion of variance to retain in PCA before applying Harmony.\n"
            "Higher values retain more information but may be slower. Default: 90%"
        )
        self.pca_variance_spin.valueChanged.connect(self._update_pca_variance_suffix)
        # Set initial suffix
        self._update_pca_variance_suffix(0.9)
        self.pca_variance_layout.addWidget(self.pca_variance_spin)
        self.pca_variance_layout.addStretch()
        self.pca_variance_widget = QtWidgets.QWidget()
        self.pca_variance_widget.setLayout(self.pca_variance_layout)
        self.pca_variance_widget.setVisible(False)  # Hidden by default
        method_layout.addWidget(self.pca_variance_widget)
        
        # Batch variable selection
        batch_var_layout = QtWidgets.QVBoxLayout()
        batch_var_label_layout = QtWidgets.QHBoxLayout()
        batch_var_label_layout.addWidget(QtWidgets.QLabel("Batch variable:"))
        self.batch_var_combo = QtWidgets.QComboBox()
        self.batch_var_combo.addItems(["source_file", "acquisition_id", "Custom grouping"])
        self.batch_var_combo.setToolTip(
            "Variable to use for batch identification.\n"
            "'source_file' groups by the source data file basename "
            "(for MCD workflows, the source .mcd file), 'acquisition_id' groups by acquisition.\n"
            "'Custom grouping' allows you to create custom groups using source_well (recommended) or acquisition_id."
        )
        self.batch_var_combo.currentTextChanged.connect(self._on_batch_var_changed)
        batch_var_label_layout.addWidget(self.batch_var_combo)
        batch_var_label_layout.addStretch()
        batch_var_layout.addLayout(batch_var_label_layout)
        
        # Custom grouping button (hidden by default)
        self.custom_grouping_btn = QtWidgets.QPushButton("Configure Custom Groups...")
        self.custom_grouping_btn.clicked.connect(self._open_custom_grouping_dialog)
        self.custom_grouping_btn.setVisible(False)
        self.custom_grouping_status = QtWidgets.QLabel("")
        self.custom_grouping_status.setStyleSheet("QLabel { color: #666; font-size: 8pt; }")
        self.custom_grouping_status.setVisible(False)
        batch_var_layout.addWidget(self.custom_grouping_btn)
        batch_var_layout.addWidget(self.custom_grouping_status)
        
        method_layout.addLayout(batch_var_layout)
        
        content_layout.addWidget(method_group)
        
        # Feature selection section
        feature_group = QtWidgets.QGroupBox("Features to Correct")
        feature_layout = QtWidgets.QVBoxLayout(feature_group)
        feature_layout.setContentsMargins(8, 8, 8, 8)
        feature_layout.setSpacing(4)
        
        feature_info = QtWidgets.QLabel(
            "Select which features to apply batch correction to.\n"
            "Batch correction is typically applied to intensity features (marker expression),\n"
            "not morphological features (cell size, shape, etc.), as batch effects primarily\n"
            "affect staining and signal intensity rather than cell morphology.\n"
            "Non-feature columns (label, cell_id, acquisition_id, etc.) will be preserved."
        )
        feature_info.setWordWrap(True)
        feature_info.setStyleSheet("QLabel { color: #666; font-size: 8pt; }")
        feature_layout.addWidget(feature_info)
        
        # Feature filter section
        filter_layout = QtWidgets.QHBoxLayout()
        filter_layout.addWidget(QtWidgets.QLabel("Filter features:"))
        
        self.filter_mean_chk = QtWidgets.QCheckBox("_mean")
        self.filter_mean_chk.setChecked(True)
        self.filter_mean_chk.toggled.connect(self._on_filter_changed)
        filter_layout.addWidget(self.filter_mean_chk)
        
        self.filter_median_chk = QtWidgets.QCheckBox("_median")
        self.filter_median_chk.setChecked(True)
        self.filter_median_chk.toggled.connect(self._on_filter_changed)
        filter_layout.addWidget(self.filter_median_chk)
        
        self.filter_other_chk = QtWidgets.QCheckBox("Other features")
        self.filter_other_chk.setChecked(False)
        self.filter_other_chk.toggled.connect(self._on_filter_changed)
        filter_layout.addWidget(self.filter_other_chk)
        
        filter_layout.addStretch()
        feature_layout.addLayout(filter_layout)
        
        self.feature_list = QtWidgets.QListWidget()
        self.feature_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.feature_list.setMaximumHeight(150)
        feature_layout.addWidget(self.feature_list)
        
        feature_buttons_layout = QtWidgets.QHBoxLayout()
        
        self.select_all_features_btn = QtWidgets.QPushButton("Select All")
        self.select_all_features_btn.clicked.connect(self._select_all_features)
        feature_buttons_layout.addWidget(self.select_all_features_btn)
        
        self.deselect_all_features_btn = QtWidgets.QPushButton("Deselect All")
        self.deselect_all_features_btn.clicked.connect(self._deselect_all_features)
        feature_buttons_layout.addWidget(self.deselect_all_features_btn)
        feature_buttons_layout.addStretch()
        feature_layout.addLayout(feature_buttons_layout)
        
        content_layout.addWidget(feature_group)
        
        # Output section
        output_group = QtWidgets.QGroupBox("Output")
        output_layout = QtWidgets.QVBoxLayout(output_group)
        output_layout.setContentsMargins(8, 8, 8, 8)
        output_layout.setSpacing(4)
        
        save_layout = QtWidgets.QHBoxLayout()
        self.save_output_chk = QtWidgets.QCheckBox("Save corrected features to CSV")
        self.save_output_chk.setChecked(False)
        save_layout.addWidget(self.save_output_chk)
        save_layout.addStretch()
        output_layout.addLayout(save_layout)
        
        output_path_layout = QtWidgets.QHBoxLayout()
        output_path_layout.addWidget(QtWidgets.QLabel("Output path:"))
        self.output_path_edit = QtWidgets.QLineEdit()
        self.output_path_edit.setPlaceholderText("Select output file...")
        self.output_path_edit.setReadOnly(True)
        self.output_path_edit.setEnabled(False)
        self.output_path_btn = QtWidgets.QPushButton("Browse...")
        self.output_path_btn.setEnabled(False)
        self.output_path_btn.clicked.connect(self._select_output_path)
        output_path_layout.addWidget(self.output_path_edit)
        output_path_layout.addWidget(self.output_path_btn)
        output_layout.addLayout(output_path_layout)
        
        content_layout.addWidget(output_group)
        
        self.scroll_area.setWidget(scroll_content)
        layout.addWidget(self.scroll_area)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()
        self.correct_btn = QtWidgets.QPushButton("Apply Batch Correction")
        self.correct_btn.clicked.connect(self._apply_batch_correction)
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.correct_btn)
        layout.addLayout(button_layout)
        
        # Store loaded files
        self.loaded_files: List[str] = []
    
    def _on_batch_var_changed(self, text: str):
        """Handle batch variable selection change."""
        is_custom = (text == "Custom grouping")
        self.custom_grouping_btn.setVisible(is_custom)
        self.custom_grouping_status.setVisible(is_custom)
        if not is_custom:
            self.custom_grouping = None
            self.custom_grouping_status.setText("")
    
    def _update_batch_var_options(self):
        """Update batch variable combo box to include metadata columns."""
        # Store current selection
        current_selection = self.batch_var_combo.currentText() if self.batch_var_combo.count() > 0 else None
        
        # Get combined dataframe to see what columns are available
        # Use a flag to prevent recursion during metadata merge
        try:
            combined_df = self._get_combined_dataframe()
        except Exception:
            # If there's an error, just use standard options
            combined_df = None
        
        # Clear and rebuild options
        self.batch_var_combo.clear()
        
        # Always include standard options
        standard_options = ["source_file", "acquisition_id", "Custom grouping"]
        self.batch_var_combo.addItems(standard_options)
        
        # Add metadata columns if available
        if combined_df is not None and not combined_df.empty:
            # Identify metadata columns (non-feature, non-standard columns)
            exclude_cols = {
                'label', 'cell_id', 'acquisition_id', 'acquisition_name', 'acquisition_label',
                'well', 'cluster', 'source_file', 'source_well', 'source_file_acquisition_id',
                'centroid_x', 'centroid_y', 'batch_group'
            }

            # Infer feature columns directly from the feature table.
            feature_cols = set(get_feature_columns_from_dataframe(combined_df, include_custom_numeric=False))

            # Metadata columns are everything else
            metadata_cols = [col for col in combined_df.columns 
                           if col not in exclude_cols and col not in feature_cols]
            
            # Add metadata columns to combo
            if metadata_cols:
                # Sort for consistency
                metadata_cols = sorted(metadata_cols)
                # Add separator if we have metadata
                self.batch_var_combo.insertSeparator(self.batch_var_combo.count())
                self.batch_var_combo.addItems(metadata_cols)
        
        # Restore selection if it still exists
        if current_selection and self.batch_var_combo.findText(current_selection) >= 0:
            self.batch_var_combo.setCurrentText(current_selection)
        elif self.batch_var_combo.count() > 0:
            # Default to first option
            self.batch_var_combo.setCurrentIndex(0)
    
    def _open_custom_grouping_dialog(self):
        """Open the custom grouping dialog."""
        # Get combined dataframe to pass to dialog
        combined_df = self._get_combined_dataframe()
        if combined_df is None or combined_df.empty:
            QtWidgets.QMessageBox.warning(
                self,
                "No Data",
                "No feature data available. Please load features first."
            )
            return
        
        # Create and show dialog (pass existing grouping if any)
        dialog = CustomGroupingDialog(combined_df, self.custom_grouping, self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.custom_grouping = dialog.get_grouping()
            # Update status label
            if self.custom_grouping:
                num_groups = len(set(self.custom_grouping.values()))
                num_acqs = len(self.custom_grouping)
                self.custom_grouping_status.setText(
                    f"Custom grouping configured: {num_acqs} acquisition(s) in {num_groups} group(s)"
                )
            else:
                self.custom_grouping_status.setText("No custom grouping configured")
    
    def _on_source_changed(self):
        """Handle data source radio button change."""
        use_current = self.use_current_radio.isChecked()
        # Show/hide file list widget based on selection
        self.file_list_widget.setVisible(not use_current)
        self.file_list.setEnabled(not use_current)
        self.add_file_btn.setEnabled(not use_current)
        self.remove_file_btn.setEnabled(not use_current)
        
        if use_current:
            self._populate_features()
        else:
            # Clear feature list until files are loaded
            self.feature_list.clear()
    
    def _on_method_changed(self, method_name: str):
        """Handle batch correction method change."""
        # Show PCA variance control only for Harmony
        self.pca_variance_widget.setVisible(method_name == "Harmony")
    
    def _update_pca_variance_suffix(self, value: float):
        """Update the suffix display for PCA variance spinbox."""
        percentage = int(value * 100)
        self.pca_variance_spin.setSuffix(f" ({percentage}%)")
    
    def _add_feature_file(self):
        """Add a feature file to the list."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Feature CSV File",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            # Validate that it's a valid feature file
            try:
                df = pd.read_csv(file_path)
                # Check for required columns
                if 'cell_id' not in df.columns:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Invalid File",
                        f"The file does not appear to be a valid feature file.\n"
                        f"Missing required column: 'cell_id'"
                    )
                    return
                
                # Add to list if not already present
                if file_path not in self.loaded_files:
                    self.loaded_files.append(file_path)
                    self.file_list.addItem(os.path.basename(file_path))
                    self._populate_features()
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to load file:\n{str(e)}"
                )
    
    def _remove_feature_file(self):
        """Remove selected feature file from the list."""
        current_item = self.file_list.currentItem()
        if current_item:
            index = self.file_list.row(current_item)
            file_path = self.loaded_files[index]
            self.loaded_files.pop(index)
            self.file_list.takeItem(index)
            self._populate_features()
    
    def _add_metadata_file(self):
        """Add a metadata file to the list."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Metadata CSV File",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            try:
                df = pd.read_csv(file_path)
                if df.empty:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Empty File",
                        "The metadata file is empty."
                    )
                    return
                
                # If file not already added, add it
                if file_path not in self.metadata_files:
                    # Auto-detect filename column (common names)
                    filename_col = None
                    for col in df.columns:
                        col_lower = col.lower()
                        if any(keyword in col_lower for keyword in ['filename', 'file_name', 'file', 'name']):
                            if 'full' in col_lower or 'stack' in col_lower or 'ome' in col_lower or 'tiff' in col_lower:
                                filename_col = col
                                break
                            elif filename_col is None:
                                filename_col = col
                    
                    # If no auto-detection, prompt user
                    if filename_col is None:
                        filename_col = self._select_filename_column(df, file_path)
                        if filename_col is None:
                            return  # User cancelled
                    
                    self.metadata_files[file_path] = {
                        'filename_column': filename_col,
                        'dataframe': df
                    }
                    self._update_metadata_list()
                    if not self.metadata_toggle_btn.isChecked():
                        self.metadata_toggle_btn.setChecked(True)
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to load metadata file:\n{str(e)}"
                )
    
    def _select_filename_column(self, df: pd.DataFrame, file_path: str) -> Optional[str]:
        """Dialog to select which column contains the filename."""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Select Filename Column")
        dialog.setModal(True)
        dialog.resize(400, 200)
        
        layout = QtWidgets.QVBoxLayout(dialog)
        
        info_label = QtWidgets.QLabel(
            f"Select the column in '{os.path.basename(file_path)}' that contains the source data filename "
            f"(this will be matched against the 'source_file' column in features, which is typically the "
            f"source .mcd filename basename for MCD workflows):"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        combo = QtWidgets.QComboBox()
        combo.addItems(df.columns.tolist())
        layout.addWidget(combo)
        
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()
        ok_btn = QtWidgets.QPushButton("OK")
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(ok_btn)
        layout.addLayout(button_layout)
        
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            return combo.currentText()
        return None
    
    def _configure_metadata_file(self):
        """Configure the filename column for a metadata file."""
        current_item = self.metadata_list.currentItem()
        if not current_item:
            return
        
        # Find the file path for this item
        file_path = None
        for path, info in self.metadata_files.items():
            display_text = self._format_metadata_item(path, info['filename_column'])
            if current_item.text() == display_text:
                file_path = path
                break
        
        if file_path is None:
            return
        
        df = self.metadata_files[file_path]['dataframe']
        filename_col = self._select_filename_column(df, file_path)
        if filename_col is not None:
            self.metadata_files[file_path]['filename_column'] = filename_col
            self._update_metadata_list()
    
    def _format_metadata_item(self, file_path: str, filename_column: str) -> str:
        """Format metadata file item for display."""
        basename = os.path.basename(file_path)
        return f"{basename} (filename: {filename_column})"
    
    def _update_metadata_list(self):
        """Update the metadata file list display."""
        self.metadata_list.clear()
        for file_path, info in self.metadata_files.items():
            item_text = self._format_metadata_item(file_path, info['filename_column'])
            self.metadata_list.addItem(item_text)
        self._update_metadata_controls()
        # Update batch variable options to include metadata columns
        self._update_batch_var_options()

    def _set_metadata_expanded(self, expanded: bool):
        """Expand or collapse the optional metadata controls."""
        self.metadata_content.setVisible(expanded)
        self.metadata_toggle_btn.setText("Hide Details" if expanded else "Show Details")

    def _update_metadata_controls(self):
        """Refresh metadata summary text and button enabled state."""
        file_count = len(self.metadata_files)
        if file_count == 0:
            self.metadata_summary_label.setText("No metadata files configured.")
            if self.metadata_toggle_btn.isChecked():
                self.metadata_toggle_btn.blockSignals(True)
                self.metadata_toggle_btn.setChecked(False)
                self.metadata_toggle_btn.blockSignals(False)
            self._set_metadata_expanded(False)
        else:
            plural = "file" if file_count == 1 else "files"
            self.metadata_summary_label.setText(
                f"{file_count} metadata {plural} configured for source_file matching."
            )
        has_selection = self.metadata_list.currentItem() is not None
        self.configure_metadata_btn.setEnabled(has_selection)
        self.remove_metadata_btn.setEnabled(has_selection)
    
    def _remove_metadata_file(self):
        """Remove selected metadata file from the list."""
        current_item = self.metadata_list.currentItem()
        if current_item:
            # Find the file path for this item
            file_path = None
            for path, info in self.metadata_files.items():
                display_text = self._format_metadata_item(path, info['filename_column'])
                if current_item.text() == display_text:
                    file_path = path
                    break
            
            if file_path:
                del self.metadata_files[file_path]
                self._update_metadata_list()
    
    def _on_filter_changed(self):
        """Handle filter checkbox changes - repopulate feature list."""
        self._populate_features()
    
    def _populate_features(self):
        """Populate the feature list based on current data source and filter settings."""
        self.feature_list.clear()
        
        # Get combined dataframe
        combined_df = self._get_combined_dataframe()
        if combined_df is None or combined_df.empty:
            return
        
        # Get filter settings
        show_mean = self.filter_mean_chk.isChecked() if hasattr(self, 'filter_mean_chk') else True
        show_median = self.filter_median_chk.isChecked() if hasattr(self, 'filter_median_chk') else True
        show_other = self.filter_other_chk.isChecked() if hasattr(self, 'filter_other_chk') else False
        
        # Infer features directly from the table schema (not image channels).
        feature_cols = get_feature_columns_from_dataframe(combined_df)
        
        # Separate intensity and morphology features
        # Morphology features (based on feature_selector_dialog.py)
        morpho_names = {
            'area_um2', 'perimeter_um', 'equivalent_diameter_um', 'eccentricity',
            'solidity', 'extent', 'circularity', 'major_axis_len_um', 'minor_axis_len_um',
            'aspect_ratio', 'bbox_area_um2', 'touches_border', 'touches_edge', 'holes_count',
            'centroid_x', 'centroid_y'
        }
        
        # Intensity features identified by suffixes
        intensity_suffixes = ('_mean', '_median', '_std', '_mad', '_p10', '_p90', '_integrated', '_frac_pos')
        
        mean_features = []
        median_features = []
        other_intensity_features = []
        morphology_features = []
        other_features = []
        
        for col in sorted(feature_cols):
            if col in morpho_names:
                morphology_features.append(col)
            elif col.endswith('_mean'):
                mean_features.append(col)
            elif col.endswith('_median'):
                median_features.append(col)
            elif any(col.endswith(suffix) for suffix in intensity_suffixes):
                other_intensity_features.append(col)
            else:
                other_features.append(col)
        
        # Add features based on filter settings
        # Mean features
        if show_mean:
            for col in mean_features:
                item = QtWidgets.QListWidgetItem(col)
                self.feature_list.addItem(item)
                item.setSelected(True)  # Auto-select mean features
        
        # Median features
        if show_median:
            for col in median_features:
                item = QtWidgets.QListWidgetItem(col)
                self.feature_list.addItem(item)
                item.setSelected(True)  # Auto-select median features
        
        # Other intensity features (if showing other)
        if show_other:
            for col in other_intensity_features:
                item = QtWidgets.QListWidgetItem(col)
                self.feature_list.addItem(item)
                item.setSelected(True)  # Default: select intensity features
            
            # Add morphology features (not selected by default)
            for col in morphology_features:
                item = QtWidgets.QListWidgetItem(col)
                self.feature_list.addItem(item)
                item.setSelected(False)  # Default: don't select morphology features
            
            # Add other features (not selected by default)
            for col in other_features:
                item = QtWidgets.QListWidgetItem(col)
                self.feature_list.addItem(item)
                item.setSelected(False)  # Default: don't select other features
        
        # Explicitly select all mean and median features to ensure they are selected
        for i in range(self.feature_list.count()):
            item = self.feature_list.item(i)
            col_name = item.text()
            if col_name.endswith('_mean') or col_name.endswith('_median'):
                item.setSelected(True)
        
        # Update batch variable options to include metadata columns
        self._update_batch_var_options()
    
    def _get_combined_dataframe(self) -> Optional[pd.DataFrame]:
        """Get the combined dataframe from current source or loaded files, with metadata merged."""
        # Get base dataframe
        if self.use_current_radio.isChecked():
            combined = self.feature_dataframe.copy() if self.feature_dataframe is not None else None
        else:
            if not self.loaded_files:
                return None
            
            # Load and combine all files
            dfs = []
            for file_path in self.loaded_files:
                try:
                    df = pd.read_csv(file_path)
                    # Ensure source_file column exists
                    if 'source_file' not in df.columns:
                        df['source_file'] = os.path.basename(file_path)
                    dfs.append(df)
                except Exception as e:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Load Error",
                        f"Failed to load {os.path.basename(file_path)}:\n{str(e)}"
                    )
                    continue
            
            if not dfs:
                return None
            
            # Combine dataframes
            combined = pd.concat(dfs, ignore_index=True)
        
        if combined is None or combined.empty:
            return None
        
        # Merge metadata if any metadata files are loaded
        if self.metadata_files:
            combined = self._merge_metadata(combined)
        
        return combined
    
    def _merge_metadata(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Merge metadata from all loaded metadata files into the features dataframe.
        
        Metadata is merged based on filename matching:
        - Features have a 'source_file' column (basename of the source data file;
          for MCD workflows this is typically the .mcd filename)
        - Metadata files have a user-specified filename column
        - Matching is done by comparing these columns (case-insensitive, ignoring extensions)
        - Columns with the same name in multiple metadata files are recognized as the same column
          and merged together (last value wins if there are conflicts for the same filename)
        """
        if features_df.empty or 'source_file' not in features_df.columns:
            return features_df
        
        result_df = features_df.copy()
        
        # Normalize source_file for matching (lowercase, remove common extensions)
        def normalize_filename(filename):
            if pd.isna(filename):
                return None
            filename = str(filename).lower().strip()
            # Remove common extensions
            for ext in ['.ome.tif', '.ome.tiff', '.tif', '.tiff', '.mcd', '.mcdx']:
                if filename.endswith(ext):
                    filename = filename[:-len(ext)]
            return filename
        
        result_df['_match_key'] = result_df['source_file'].apply(normalize_filename)
        
        # First, combine all metadata files into a single dataframe
        # This allows columns with the same name to be recognized as the same column
        combined_metadata_list = []
        
        for file_path, metadata_info in self.metadata_files.items():
            metadata_df = metadata_info['dataframe'].copy()
            filename_col = metadata_info['filename_column']
            
            if filename_col not in metadata_df.columns:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Metadata Error",
                    f"Filename column '{filename_col}' not found in {os.path.basename(file_path)}"
                )
                continue
            
            # Normalize metadata filename column
            metadata_df['_match_key'] = metadata_df[filename_col].apply(normalize_filename)
            
            # Get columns to merge (exclude the filename column and match key)
            cols_to_merge = [col for col in metadata_df.columns 
                           if col not in [filename_col, '_match_key']]
            
            if not cols_to_merge:
                continue
            
            # Select only the match key and columns to merge
            metadata_subset = metadata_df[['_match_key'] + cols_to_merge].copy()
            combined_metadata_list.append(metadata_subset)
        
        # If we have metadata to merge, combine all metadata files first
        if combined_metadata_list:
            # Combine all metadata dataframes
            # Columns with the same name will be recognized as the same column
            # Strategy: concatenate all metadata, then group by match_key and take last non-null value for each column
            all_metadata = pd.concat(combined_metadata_list, ignore_index=True)
            
            # Group by match_key and aggregate columns
            # For each column, take the last non-null value (this handles cases where same filename appears in multiple files)
            def take_last_nonnull(series):
                # Remove nulls and take the last value, or last value if all are null
                nonnull_values = series.dropna()
                if len(nonnull_values) > 0:
                    return nonnull_values.iloc[-1]
                return series.iloc[-1] if len(series) > 0 else None
            
            # Aggregate by match_key, using last non-null value for each column
            combined_metadata = all_metadata.groupby('_match_key').agg(take_last_nonnull).reset_index()
            
            # Now merge the combined metadata into the features dataframe
            result_df = result_df.merge(
                combined_metadata,
                on='_match_key',
                how='left'
            )
        
        # Remove temporary match key column
        if '_match_key' in result_df.columns:
            result_df = result_df.drop(columns=['_match_key'])
        
        return result_df
    
    def _select_all_features(self):
        """Select all features in the list."""
        for i in range(self.feature_list.count()):
            self.feature_list.item(i).setSelected(True)
    
    def _deselect_all_features(self):
        """Deselect all features in the list."""
        for i in range(self.feature_list.count()):
            self.feature_list.item(i).setSelected(False)
    
    def _auto_generate_output_path(self):
        """Auto-generate output file path based on input data."""
        if not self.save_output_chk.isChecked():
            return
        
        # Get method name
        method = self.method_combo.currentText().lower() if self.method_combo.count() > 0 else "batch"
        
        # Get base directory from first loaded file or use current directory
        base_dir = ""
        if self.loaded_files:
            base_dir = os.path.dirname(self.loaded_files[0])
        elif hasattr(self, 'parent') and self.parent() and hasattr(self.parent(), 'current_file_path'):
            # Try to get directory from parent's current file
            try:
                base_dir = os.path.dirname(self.parent().current_file_path)
            except:
                pass
        
        if not base_dir:
            base_dir = os.getcwd()
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"features_batch_corrected_{method}_{timestamp}.csv"
        file_path = os.path.join(base_dir, filename)
        
        self.output_path_edit.setText(file_path)
    
    def _select_output_path(self):
        """Select output file path."""
        # Start with auto-generated path if available
        initial_path = self.output_path_edit.text() if self.output_path_edit.text() else ""
        if not initial_path:
            self._auto_generate_output_path()
            initial_path = self.output_path_edit.text()
        
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Corrected Features",
            initial_path,
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            if not file_path.endswith('.csv'):
                file_path += '.csv'
            self.output_path_edit.setText(file_path)
    
    def _update_ui_state(self):
        """Update UI state based on available methods and data."""
        # Enable/disable save output controls
        def _on_save_output_toggled(checked):
            self.output_path_edit.setEnabled(checked)
            self.output_path_btn.setEnabled(checked)
            if checked and not self.output_path_edit.text():
                # Auto-generate filename if not set
                self._auto_generate_output_path()
        
        self.save_output_chk.toggled.connect(_on_save_output_toggled)
        
        # Set initial PCA variance widget visibility based on selected method
        if hasattr(self, 'method_combo'):
            current_method = self.method_combo.currentText()
            self._on_method_changed(current_method)
        
        # Populate features if we have current data
        if self.feature_dataframe is not None and not self.feature_dataframe.empty:
            self._populate_features()
        
        # Update batch variable options
        self._update_batch_var_options()
        self._update_metadata_controls()
    
    def get_corrected_dataframe(self) -> Optional[pd.DataFrame]:
        """Get the batch-corrected dataframe."""
        return self.corrected_dataframe
    
    def get_combined_dataframe(self) -> Optional[pd.DataFrame]:
        """Get the combined dataframe (original or from loaded files) before correction."""
        return self._get_combined_dataframe()
    
    def get_output_path(self) -> Optional[str]:
        """Get the output file path if saving."""
        if self.save_output_chk.isChecked():
            path = self.output_path_edit.text().strip()
            return path if path else None
        return None
    
    def _apply_batch_correction(self):
        """Apply batch correction to the data."""
        # Get combined dataframe
        combined_df = self._get_combined_dataframe()
        if combined_df is None or combined_df.empty:
            QtWidgets.QMessageBox.warning(
                self,
                "No Data",
                "No feature data available. Please load features first."
            )
            return
        
        # Get selected features
        selected_features = []
        for i in range(self.feature_list.count()):
            item = self.feature_list.item(i)
            if item.isSelected():
                selected_features.append(item.text())

        # Guard against stale UI selections by re-validating against the table columns.
        available_features = set(get_feature_columns_from_dataframe(combined_df))
        selected_features = [f for f in selected_features if f in available_features]
        
        if not selected_features:
            QtWidgets.QMessageBox.warning(
                self,
                "No Features Selected",
                "Please select at least one valid feature column to correct."
            )
            return
        
        # Get batch variable
        batch_var = self.batch_var_combo.currentText()
        
        # Handle custom grouping
        temp_batch_var = None
        if batch_var == "Custom grouping":
            if not self.custom_grouping:
                QtWidgets.QMessageBox.warning(
                    self,
                    "No Custom Grouping",
                    "Please configure custom groups first by clicking 'Configure Custom Groups...'"
                )
                return
            
            # Create batch variable column based on custom grouping
            # Use a user-friendly column name that will be preserved in the output
            temp_batch_var = "batch_group"
            combined_df = combined_df.copy()
            
            # Determine which column to use for grouping (source_well preferred, acquisition_id as fallback)
            grouping_column = None
            if 'source_well' in combined_df.columns:
                grouping_column = 'source_well'
            elif 'acquisition_id' in combined_df.columns:
                grouping_column = 'acquisition_id'
            
            if not grouping_column:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Missing Column",
                    "Custom grouping requires 'source_well' or 'acquisition_id' column in the feature dataframe."
                )
                return
            
            # Map source_well or acquisition_id to group name
            def get_group(acq_id):
                return self.custom_grouping.get(str(acq_id), "__unassigned__")
            
            combined_df[temp_batch_var] = combined_df[grouping_column].apply(get_group)
            
            # Check that we have at least 2 groups (excluding unassigned)
            unique_groups = set(combined_df[temp_batch_var].unique())
            assigned_groups = unique_groups - {"__unassigned__"}
            
            if len(assigned_groups) < 2:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Insufficient Groups",
                    f"Custom grouping resulted in only {len(assigned_groups)} assigned group(s). "
                    "At least 2 groups are required for batch correction."
                )
                return
            
            # Warn if there are unassigned acquisitions
            unassigned_count = (combined_df[temp_batch_var] == "__unassigned__").sum()
            if unassigned_count > 0:
                reply = QtWidgets.QMessageBox.warning(
                    self,
                    "Unassigned Acquisitions",
                    f"{unassigned_count} acquisition(s) are not assigned to any group. "
                    "They will be treated as a separate batch. Continue?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.No
                )
                if reply == QtWidgets.QMessageBox.No:
                    return
            
            # Use the temporary batch variable
            batch_var = temp_batch_var
        
        # Validate inputs
        try:
            validate_batch_correction_inputs(combined_df, batch_var, selected_features)
        except ValueError as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Validation Error",
                f"Invalid inputs for batch correction:\n{str(e)}"
            )
            return
        
        # Get method
        method = self.method_combo.currentText()
        
        try:
            # Map method name to core function format
            method_lower = method.lower()  # "Combat" -> "combat", "Harmony" -> "harmony"
            
            # Get Harmony-specific parameters if needed
            pca_variance = 0.9  # Default
            if method == "Harmony" and hasattr(self, 'pca_variance_spin'):
                pca_variance = self.pca_variance_spin.value()

            def _batch_correction_task():
                # Use core batch_correction function.
                # Note: output_path is None here since we handle saving separately below.
                return batch_correction(
                    features_df=combined_df,
                    method=method_lower,
                    batch_var=batch_var,
                    features=selected_features,
                    output_path=None,
                    covariates=None,
                    # Harmony parameters (only used if method == "harmony")
                    n_clusters=30,
                    sigma=0.1,
                    theta=2.0,
                    lambda_reg=1.0,
                    max_iter=20,
                    pca_variance=pca_variance,
                )

            self.corrected_dataframe = run_blocking_task_with_progress(
                parent=self,
                window_title="Batch Correction In Progress",
                initial_message=f"Applying {method} batch correction",
                detail_text=(
                    "Computing correction and rebuilding corrected features.\n"
                    "Large datasets may take several minutes."
                ),
                task=_batch_correction_task,
            )
        except ImportError as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Import Error",
                f"Required package not installed:\n{str(e)}\n\nPlease install the required package and try again."
            )
            return
        
        try:
            # Validate output path if saving
            output_path = None
            if self.save_output_chk.isChecked():
                output_path = self.output_path_edit.text().strip()
                if not output_path:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "No Output Path",
                        "Please specify an output path to save the corrected features."
                    )
                    return
            
            # Log batch correction
            logger = get_logger()
            
            # Collect source files from the combined dataframe
            source_files = set()
            if 'source_file' in combined_df.columns:
                source_files = set(combined_df['source_file'].dropna().unique())
            
            source_file_str = None
            if source_files:
                if len(source_files) == 1:
                    source_file_str = list(source_files)[0]
                else:
                    sorted_files = sorted(source_files)
                    if len(sorted_files) <= 3:
                        source_file_str = ", ".join(sorted_files)
                    else:
                        source_file_str = ", ".join(sorted_files[:3]) + f" and {len(sorted_files) - 3} more"
            
            # Collect acquisition IDs
            acquisitions = []
            if 'acquisition_id' in combined_df.columns:
                acquisitions = list(combined_df['acquisition_id'].dropna().unique())
            
            # Prepare parameters
            # Get number of batches (use the actual batch variable column, not the display name)
            actual_batch_var = temp_batch_var if temp_batch_var else batch_var
            n_batches = len(combined_df[actual_batch_var].unique()) if actual_batch_var in combined_df.columns else 0
            
            params = {
                "method": method,
                "batch_variable": batch_var if not temp_batch_var else "custom_grouping",
                "n_features": len(selected_features),
                "n_cells": len(combined_df),
                "n_batches": n_batches
            }
            
            # Add method-specific parameters
            if method == "Harmony" and hasattr(self, 'pca_variance_spin'):
                params["pca_variance"] = self.pca_variance_spin.value()
            
            if temp_batch_var:
                params["custom_grouping"] = True
                params["n_groups"] = len(set(combined_df[actual_batch_var].unique()))
            
            logger._write_entry(
                entry_type="batch_correction",
                operation=method.lower(),
                parameters=params,
                acquisitions=acquisitions,
                output_path=output_path,
                notes=f"Batch correction applied to {len(selected_features)} features across {len(combined_df)} cells",
                source_file=source_file_str
            )
            
            # Note: The batch_group column is preserved in the corrected dataframe
            # for use in visualization (e.g., coloring by batch in clustering/spatial analysis)
            
            # Success - accept dialog
            self.accept()
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Batch Correction Error",
                f"Batch correction failed:\n{str(e)}"
            )
            import traceback
            traceback.print_exc()
