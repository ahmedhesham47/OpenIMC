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

from typing import List, Optional, Tuple, Dict
import os
from pathlib import Path
import numpy as np

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.cm as cm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.optimize import curve_fit

from openimc.data.mcd_loader import AcquisitionInfo, MCDLoader
from openimc.data.ometiff_loader import OMETIFFLoader


class DeconvolutionDialog(QtWidgets.QDialog):
    def __init__(self, acquisitions: List[AcquisitionInfo], current_acq_id: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("High Resolution IMC Deconvolution")
        self.setModal(True)
        
        # Set dialog size
        if parent:
            parent_size = parent.size()
            dialog_width = int(parent_size.width() * 0.8)
            dialog_height = parent_size.height()  # Same height as main window
            self.resize(dialog_width, dialog_height)
        else:
            self.resize(800, 700)
        
        self.setMinimumSize(700, 600)
        self.acquisitions = acquisitions
        self.current_acq_id = current_acq_id
        self.output_directory = ""
        self.parent_window = parent
        
        # Experimental Design data
        self.roi_passes = []  # List of (acq_id, pass_number) tuples
        self.intensity_data = {}  # Dict mapping pass_number to intensity value
        self.fitted_curve = None
        self.energy_fit_params = {}  # Dict mapping energy to (I_max, k, x0) fit parameters
        
        # Kernel generation data
        self.generated_passes = None
        self.generated_contributions = None
        self.generated_psf_kernel = None
        
        # Deconvolution kernel arrays (loaded from file or generated)
        self.deconv_passes = None
        self.deconv_contributions = None
        self.deconv_psf_kernel = None
        
        # Create UI
        self._create_ui()
        
    def _create_ui(self):
        # Main layout for the dialog
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)
        
        # Create tab widget
        self.tab_widget = QtWidgets.QTabWidget()
        
        # Create Experimental Design tab
        self.experimental_design_tab = self._create_experimental_design_tab()
        self.tab_widget.addTab(self.experimental_design_tab, "Experimental Design")
        
        # Create Apply High Resolution Deconvolution tab
        self.deconvolution_tab = self._create_deconvolution_tab()
        self.tab_widget.addTab(self.deconvolution_tab, "Apply High Resolution Deconvolution")
        
        main_layout.addWidget(self.tab_widget, 1)
        
        # Buttons (outside tabs)
        button_layout = QtWidgets.QHBoxLayout()
        self.cancel_btn = QtWidgets.QPushButton("Close")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_btn)
        main_layout.addLayout(button_layout)
    
    def _create_experimental_design_tab(self):
        """Create the Experimental Design tab."""
        tab = QtWidgets.QWidget()
        
        # Create nested tab widget
        self.exp_design_tabs = QtWidgets.QTabWidget()
        
        # Create content layout for ROI selection tab
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        
        # Information note
        info_group = QtWidgets.QGroupBox("Information and Guidelines")
        info_layout = QtWidgets.QVBoxLayout(info_group)
        info_label = QtWidgets.QLabel(
            "This tool helps determine optimal ablation energy by modeling intensity decay across multiple passes. "
            "Select ROIs representing different passes of the same region, and analyze how intensity changes. "
            "Each pixel is 1μm², and intensities are summed across all pixels in each ROI.\n\n"
            "<b>Optimal Energy Criteria:</b><br>"
            "• Intensity should be retained for <b>at least 4 consecutive passes</b><br>"
            "• Intensity should be retained for <b>at least 7 total passes</b><br><br>"
            "<b>Determining x0 for Deconvolution:</b><br>"
            "The x0 parameter should be set to the <b>highest-numbered pass</b> for which signal is still retained "
            "(i.e., where intensity has not significantly decayed)."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("QLabel { color: #0066cc; }")
        info_layout.addWidget(info_label)
        layout.addWidget(info_group)
        
        # Channel selection
        channel_group = QtWidgets.QGroupBox("Channel Selection")
        channel_layout = QtWidgets.QVBoxLayout(channel_group)
        
        channel_combo_layout = QtWidgets.QHBoxLayout()
        channel_combo_layout.addWidget(QtWidgets.QLabel("Channel:"))
        self.channel_combo = QtWidgets.QComboBox()
        self._populate_channels()
        channel_combo_layout.addWidget(self.channel_combo, 1)
        channel_layout.addLayout(channel_combo_layout)
        
        layout.addWidget(channel_group)
        
        # ROI selection and ordering
        roi_group = QtWidgets.QGroupBox("ROI Selection and Pass Ordering")
        roi_layout = QtWidgets.QVBoxLayout(roi_group)
        
        # Instructions
        instructions = QtWidgets.QLabel(
            "Select ROIs and assign pass numbers. Each ROI should represent a different pass of the same region."
        )
        instructions.setWordWrap(True)
        roi_layout.addWidget(instructions)
        
        # ROI list with pass numbers
        roi_list_layout = QtWidgets.QHBoxLayout()
        
        # Available ROIs
        available_group = QtWidgets.QGroupBox("Available ROIs")
        available_layout = QtWidgets.QVBoxLayout(available_group)
        
        # Search box
        search_layout = QtWidgets.QHBoxLayout()
        search_label = QtWidgets.QLabel("Search:")
        search_layout.addWidget(search_label)
        self.roi_search_edit = QtWidgets.QLineEdit()
        self.roi_search_edit.setPlaceholderText("Type to filter ROIs...")
        self.roi_search_edit.textChanged.connect(self._filter_available_rois)
        search_layout.addWidget(self.roi_search_edit)
        available_layout.addLayout(search_layout)
        
        self.available_roi_list = QtWidgets.QListWidget()
        self.available_roi_list.setMinimumHeight(180)
        self.available_roi_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self._populate_available_rois()
        available_layout.addWidget(self.available_roi_list)
        
        # Buttons for adding/removing
        roi_btn_layout = QtWidgets.QVBoxLayout()
        self.add_roi_btn = QtWidgets.QPushButton("Add →")
        self.add_roi_btn.clicked.connect(self._add_roi_to_passes)
        self.add_roi_btn.setEnabled(False)
        roi_btn_layout.addWidget(self.add_roi_btn)
        
        self.remove_roi_btn = QtWidgets.QPushButton("← Remove")
        self.remove_roi_btn.clicked.connect(self._remove_roi_from_passes)
        self.remove_roi_btn.setEnabled(False)
        roi_btn_layout.addWidget(self.remove_roi_btn)
        roi_btn_layout.addStretch()
        
        # Selected ROIs with pass numbers
        selected_group = QtWidgets.QGroupBox("Selected ROIs (Pass Order)")
        selected_layout = QtWidgets.QVBoxLayout(selected_group)
        
        # Table for ROI passes
        self.roi_pass_table = QtWidgets.QTableWidget()
        self.roi_pass_table.setColumnCount(4)
        self.roi_pass_table.setHorizontalHeaderLabels(["Pass #", "ROI Name", "Energy", "Actions"])
        self.roi_pass_table.horizontalHeader().setStretchLastSection(False)
        # Set column widths: Pass # narrow, ROI Name very wide, Energy medium, Actions slightly wider
        self.roi_pass_table.setColumnWidth(0, 60)  # Pass #
        self.roi_pass_table.setColumnWidth(1, 600)  # ROI Name - even wider
        self.roi_pass_table.setColumnWidth(2, 100)  # Energy
        # Calculate width for "×" button (slightly wider)
        temp_btn = QtWidgets.QPushButton("×")
        font_metrics = temp_btn.fontMetrics()
        char_width = font_metrics.width("×") + 18  # Add more padding
        self.roi_pass_table.setColumnWidth(3, char_width)  # Actions - slightly wider
        self.roi_pass_table.setMaximumHeight(200)
        self.roi_pass_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.roi_pass_table.itemSelectionChanged.connect(self._on_roi_table_selection_changed)
        selected_layout.addWidget(self.roi_pass_table)
        
        # Buttons for reordering
        reorder_btn_layout = QtWidgets.QHBoxLayout()
        self.move_up_btn = QtWidgets.QPushButton("↑ Move Up")
        self.move_up_btn.clicked.connect(self._move_roi_up)
        self.move_up_btn.setEnabled(False)
        self.move_down_btn = QtWidgets.QPushButton("↓ Move Down")
        self.move_down_btn.clicked.connect(self._move_roi_down)
        self.move_down_btn.setEnabled(False)
        reorder_btn_layout.addWidget(self.move_up_btn)
        reorder_btn_layout.addWidget(self.move_down_btn)
        reorder_btn_layout.addStretch()
        selected_layout.addLayout(reorder_btn_layout)
        
        roi_list_layout.addWidget(available_group, 1)
        roi_list_layout.addLayout(roi_btn_layout)
        roi_list_layout.addWidget(selected_group, 1)
        roi_layout.addLayout(roi_list_layout)
        
        layout.addWidget(roi_group)
        
        # Analysis button
        analysis_btn_layout = QtWidgets.QHBoxLayout()
        self.analyze_btn = QtWidgets.QPushButton("Analyze Intensity Decay")
        self.analyze_btn.clicked.connect(self._analyze_intensity_decay)
        self.analyze_btn.setEnabled(False)
        analysis_btn_layout.addWidget(self.analyze_btn)
        analysis_btn_layout.addStretch()
        layout.addLayout(analysis_btn_layout)
        
        # Create ROI selection tab content
        roi_selection_tab = QtWidgets.QWidget()
        roi_selection_layout = QtWidgets.QVBoxLayout(roi_selection_tab)
        roi_selection_layout.setContentsMargins(0, 0, 0, 0)
        # Move all widgets from layout to roi_selection_layout
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                roi_selection_layout.addWidget(item.widget())
            elif item.layout():
                roi_selection_layout.addLayout(item.layout())
        
        # Create plot tab
        plot_tab = self._create_plot_tab()
        
        # Create kernel generation tab
        kernel_gen_tab = self._create_kernel_generation_tab()
        
        # Add tabs to nested tab widget
        self.exp_design_tabs.addTab(roi_selection_tab, "ROI Selection")
        self.exp_design_tabs.addTab(plot_tab, "Experimental Design Plot")
        self.exp_design_tabs.addTab(kernel_gen_tab, "Generate Kernel Arrays")
        
        # Disable plot and kernel generation tabs initially
        self.exp_design_tabs.setTabEnabled(1, False)
        self.exp_design_tabs.setTabEnabled(2, False)
        
        # Main layout for experimental design tab
        main_layout = QtWidgets.QVBoxLayout(tab)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.addWidget(self.exp_design_tabs)
        
        # Connect signals
        self.available_roi_list.itemSelectionChanged.connect(self._on_available_roi_selection_changed)
        
        return tab
    
    def _create_plot_tab(self):
        """Create the plot tab for Experimental Design."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        
        # Plot widget (much larger)
        self.plot_figure = Figure(figsize=(12, 8))
        self.plot_canvas = FigureCanvas(self.plot_figure)
        self.plot_canvas.setMinimumHeight(600)
        layout.addWidget(self.plot_canvas, 1)  # Give it stretch factor
        
        # Fit parameters display (tiny, compact box below plot)
        fit_params_layout = QtWidgets.QHBoxLayout()
        fit_params_layout.addWidget(QtWidgets.QLabel("Fit:"))
        self.fit_params_label = QtWidgets.QLabel("")
        self.fit_params_label.setWordWrap(False)
        self.fit_params_label.setStyleSheet("QLabel { background-color: #fff8dc; padding: 2px 5px; border: 1px solid #ccc; font-size: 9pt; }")
        self.fit_params_label.setVisible(False)
        fit_params_layout.addWidget(self.fit_params_label, 1)
        fit_params_layout.addStretch()
        layout.addLayout(fit_params_layout)
        
        # Export button
        export_btn_layout = QtWidgets.QHBoxLayout()
        self.export_plot_btn = QtWidgets.QPushButton("Export Plot...")
        self.export_plot_btn.clicked.connect(self._export_plot)
        self.export_plot_btn.setEnabled(False)
        export_btn_layout.addWidget(self.export_plot_btn)
        export_btn_layout.addStretch()
        layout.addLayout(export_btn_layout)
        
        return tab
    
    def _create_kernel_generation_tab(self):
        """Create the kernel generation tab for generating passes and contributions arrays."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        
        # Information note
        info_group = QtWidgets.QGroupBox("Information")
        info_layout = QtWidgets.QVBoxLayout(info_group)
        info_label = QtWidgets.QLabel(
            "Generate passes and contributions arrays for High Resolution IMC deconvolution. "
            "First, complete ROI selection and 'Analyze Intensity Decay' in the previous tabs. "
            "Then select an energy from your analysis to automatically populate the sigmoidal loss "
            "function parameters. Configure instrument geometry and generate custom arrays that match "
            "your experimental setup. These arrays can be saved and loaded for use in deconvolution.\n\n"
            "<b>Note:</b> The generated arrays will replace the default hard-coded arrays when used in deconvolution."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("QLabel { color: #0066cc; }")
        info_layout.addWidget(info_label)
        layout.addWidget(info_group)
        
        # Energy selection
        energy_group = QtWidgets.QGroupBox("Energy Selection")
        energy_layout = QtWidgets.QFormLayout(energy_group)
        
        self.energy_combo = QtWidgets.QComboBox()
        self.energy_combo.setEnabled(False)
        self.energy_combo.currentIndexChanged.connect(self._on_energy_selected)
        energy_layout.addRow("Energy from Analysis:", self.energy_combo)
        
        energy_info = QtWidgets.QLabel(
            "Select an energy from your intensity decay analysis. The sigmoidal parameters will be "
            "automatically filled from the fitted curve."
        )
        energy_info.setWordWrap(True)
        energy_info.setStyleSheet("QLabel { color: #666; font-size: 9pt; }")
        energy_layout.addRow("", energy_info)
        
        layout.addWidget(energy_group)
        
        # Sigmoidal loss function parameters
        sigmoid_group = QtWidgets.QGroupBox("Sigmoidal Loss Function Parameters")
        sigmoid_layout = QtWidgets.QFormLayout(sigmoid_group)
        
        self.midpoint_spin = QtWidgets.QDoubleSpinBox()
        self.midpoint_spin.setRange(0.1, 20.0)
        self.midpoint_spin.setValue(4.0)
        self.midpoint_spin.setDecimals(2)
        self.midpoint_spin.setSingleStep(0.1)
        sigmoid_layout.addRow("Midpoint:", self.midpoint_spin)
        
        self.slope_spin = QtWidgets.QDoubleSpinBox()
        self.slope_spin.setRange(0.1, 10.0)
        self.slope_spin.setValue(1.0)
        self.slope_spin.setDecimals(2)
        self.slope_spin.setSingleStep(0.1)
        sigmoid_layout.addRow("Slope:", self.slope_spin)
        
        self.min_fraction_spin = QtWidgets.QDoubleSpinBox()
        self.min_fraction_spin.setRange(0.0, 1.0)
        self.min_fraction_spin.setValue(0.0)
        self.min_fraction_spin.setDecimals(3)
        self.min_fraction_spin.setSingleStep(0.01)
        sigmoid_layout.addRow("Min Fraction:", self.min_fraction_spin)
        
        self.max_fraction_spin = QtWidgets.QDoubleSpinBox()
        self.max_fraction_spin.setRange(0.0, 1.0)
        self.max_fraction_spin.setValue(1.0)
        self.max_fraction_spin.setDecimals(3)
        self.max_fraction_spin.setSingleStep(0.01)
        sigmoid_layout.addRow("Max Fraction:", self.max_fraction_spin)
        
        layout.addWidget(sigmoid_group)
        
        # Instrument geometry parameters
        geometry_group = QtWidgets.QGroupBox("Instrument Geometry Parameters")
        geometry_layout = QtWidgets.QFormLayout(geometry_group)
        
        self.step_size_spin = QtWidgets.QDoubleSpinBox()
        self.step_size_spin.setRange(0.1, 2.0)
        self.step_size_spin.setValue(0.333)
        self.step_size_spin.setDecimals(3)
        self.step_size_spin.setSingleStep(0.001)
        geometry_layout.addRow("Step Size (μm):", self.step_size_spin)
        
        self.pixel_size_spin = QtWidgets.QDoubleSpinBox()
        self.pixel_size_spin.setRange(0.1, 10.0)
        self.pixel_size_spin.setValue(1.0)
        self.pixel_size_spin.setDecimals(2)
        self.pixel_size_spin.setSingleStep(0.1)
        geometry_layout.addRow("Pixel Size (μm):", self.pixel_size_spin)
        
        self.spot_diameter_spin = QtWidgets.QDoubleSpinBox()
        self.spot_diameter_spin.setRange(0.1, 5.0)
        self.spot_diameter_spin.setValue(1.0)
        self.spot_diameter_spin.setDecimals(2)
        self.spot_diameter_spin.setSingleStep(0.1)
        geometry_layout.addRow("Spot Diameter (μm):", self.spot_diameter_spin)
        
        layout.addWidget(geometry_group)
        
        # Generate button
        generate_btn_layout = QtWidgets.QHBoxLayout()
        self.generate_kernel_btn = QtWidgets.QPushButton("Generate Arrays")
        self.generate_kernel_btn.clicked.connect(self._generate_kernel_arrays)
        generate_btn_layout.addWidget(self.generate_kernel_btn)
        generate_btn_layout.addStretch()
        layout.addLayout(generate_btn_layout)
        
        # Status label
        self.kernel_status_label = QtWidgets.QLabel("")
        self.kernel_status_label.setWordWrap(True)
        self.kernel_status_label.setStyleSheet("QLabel { color: #666; font-style: italic; }")
        layout.addWidget(self.kernel_status_label)
        
        # Save button
        file_btn_layout = QtWidgets.QHBoxLayout()
        self.save_kernel_btn = QtWidgets.QPushButton("Save Arrays...")
        self.save_kernel_btn.clicked.connect(self._save_kernel_arrays)
        self.save_kernel_btn.setEnabled(False)
        file_btn_layout.addWidget(self.save_kernel_btn)
        file_btn_layout.addStretch()
        layout.addLayout(file_btn_layout)
        
        layout.addStretch()
        
        return tab
    
    def _update_energy_dropdown(self):
        """Update the energy dropdown with available energies from analysis."""
        self.energy_combo.clear()
        
        if not self.energy_fit_params:
            self.energy_combo.setEnabled(False)
            return
        
        # Add energies sorted by value
        energies = sorted(self.energy_fit_params.keys())
        for energy in energies:
            self.energy_combo.addItem(f"{energy:.2f}", energy)
        
        self.energy_combo.setEnabled(True)
        
        # Select first energy and populate parameters
        if self.energy_combo.count() > 0:
            self.energy_combo.setCurrentIndex(0)
            self._on_energy_selected()
    
    def _on_energy_selected(self):
        """Handle energy selection - auto-populate sigmoidal parameters."""
        if self.energy_combo.currentIndex() < 0:
            return
        
        # Get energy value from combo box (PyQt5 uses itemData)
        energy = self.energy_combo.itemData(self.energy_combo.currentIndex())
        if energy is None or energy not in self.energy_fit_params:
            return
        
        # Get fit parameters for this energy
        fit_params = self.energy_fit_params[energy]
        I_max = fit_params['I_max']
        k = fit_params['k']
        x0 = fit_params['x0']
        
        # Convert fitted parameters to sigmoidal loss function parameters
        # The fitted curve: I(x) = I_max / (1 + exp(k * (x - x0)))
        # The loss function: fraction = 1.0 / (1.0 + exp(slope * (x - midpoint)))
        # So: midpoint = x0, slope = k
        # For min_fraction and max_fraction, we normalize based on the intensity range
        # max_fraction = 1.0 (full signal), min_fraction can be set based on data or default to 0.0
        
        # Set parameters
        self.midpoint_spin.setValue(float(x0))
        self.slope_spin.setValue(float(k))
        
        # Set min_fraction to 0.0 (no signal remaining after many passes)
        # and max_fraction to 1.0 (full signal at first pass)
        self.min_fraction_spin.setValue(0.0)
        self.max_fraction_spin.setValue(1.0)
    
    def _create_deconvolution_tab(self):
        """Create the Apply High Resolution Deconvolution tab (existing functionality)."""
        tab = QtWidgets.QWidget()
        
        # Main layout
        main_layout = QtWidgets.QVBoxLayout(tab)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)
        
        # Create scroll area
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Create widget to hold the scrollable content
        scroll_content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(scroll_content)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(6)
        
        # Information note
        info_group = QtWidgets.QGroupBox("Information")
        info_layout = QtWidgets.QVBoxLayout(info_group)
        info_label = QtWidgets.QLabel(
            "This deconvolution method is optimized for high resolution IMC images with step sizes of 333 nm and 500 nm. "
            "The deconvolution uses Richardson-Lucy deconvolution with a circular kernel optimized for IMC data.\n\n"
            "Works with both MCD files and OME-TIFF directories."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("QLabel { color: #0066cc; font-style: italic; }")
        info_layout.addWidget(info_label)
        layout.addWidget(info_group)
        
        # Acquisition selection
        acq_group = QtWidgets.QGroupBox("Acquisition Selection")
        acq_layout = QtWidgets.QVBoxLayout(acq_group)
        
        self.single_roi_radio = QtWidgets.QRadioButton("Single ROI (Current Acquisition)")
        self.whole_slide_radio = QtWidgets.QRadioButton("Whole Slide (All Acquisitions)")
        self.single_roi_radio.setChecked(True)
        
        acq_layout.addWidget(self.single_roi_radio)
        acq_layout.addWidget(self.whole_slide_radio)
        
        # Current acquisition info
        self.acq_info_label = QtWidgets.QLabel("")
        acq_layout.addWidget(self.acq_info_label)
        
        layout.addWidget(acq_group)
        
        # Output directory selection
        dir_group = QtWidgets.QGroupBox("Output Directory")
        dir_layout = QtWidgets.QVBoxLayout(dir_group)
        
        dir_row = QtWidgets.QHBoxLayout()
        self.dir_label = QtWidgets.QLabel("No directory selected")
        self.dir_label.setStyleSheet("QLabel { color: #666; }")
        dir_row.addWidget(self.dir_label)
        dir_row.addStretch()
        
        self.browse_btn = QtWidgets.QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._browse_directory)
        dir_row.addWidget(self.browse_btn)
        
        dir_layout.addLayout(dir_row)
        layout.addWidget(dir_group)
        
        # Deconvolution parameters
        params_group = QtWidgets.QGroupBox("Deconvolution Parameters")
        params_layout = QtWidgets.QVBoxLayout(params_group)
        
        # x0 parameter (hidden when kernel is loaded)
        self.x0_layout = QtWidgets.QHBoxLayout()
        self.x0_layout.addWidget(QtWidgets.QLabel("x0 parameter:"))
        self.x0_spin = QtWidgets.QDoubleSpinBox()
        self.x0_spin.setRange(1.0, 20.0)
        self.x0_spin.setValue(7.0)
        self.x0_spin.setDecimals(1)
        self.x0_spin.setSingleStep(0.5)
        self.x0_layout.addWidget(self.x0_spin)
        self.x0_layout.addStretch()
        params_layout.addLayout(self.x0_layout)
        
        # Iterations parameter
        iter_layout = QtWidgets.QHBoxLayout()
        iter_layout.addWidget(QtWidgets.QLabel("Iterations:"))
        self.iterations_spin = QtWidgets.QSpinBox()
        self.iterations_spin.setRange(1, 20)
        self.iterations_spin.setValue(4)
        iter_layout.addWidget(self.iterations_spin)
        iter_layout.addStretch()
        params_layout.addLayout(iter_layout)
        
        # Separator
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.HLine)
        separator.setFrameShadow(QtWidgets.QFrame.Sunken)
        params_layout.addWidget(separator)
        
        # Kernel arrays (passes and contributions) - moved into parameters section
        kernel_info = QtWidgets.QLabel(
            "Load custom kernel arrays generated from the Experimental Design tab. "
            "If a PSF kernel is loaded, x0 parameter will be hidden (it's baked into the kernel)."
        )
        kernel_info.setWordWrap(True)
        kernel_info.setStyleSheet("QLabel { color: #666; font-size: 9pt; }")
        params_layout.addWidget(kernel_info)
        
        kernel_file_layout = QtWidgets.QHBoxLayout()
        self.kernel_file_label = QtWidgets.QLabel("No kernel arrays loaded (using defaults)")
        self.kernel_file_label.setStyleSheet("QLabel { color: #666; }")
        kernel_file_layout.addWidget(self.kernel_file_label)
        kernel_file_layout.addStretch()
        
        self.load_kernel_file_btn = QtWidgets.QPushButton("Load Kernel...")
        self.load_kernel_file_btn.clicked.connect(self._load_kernel_arrays_for_deconv)
        kernel_file_layout.addWidget(self.load_kernel_file_btn)
        
        self.clear_kernel_btn = QtWidgets.QPushButton("Clear")
        self.clear_kernel_btn.clicked.connect(self._clear_kernel_arrays)
        self.clear_kernel_btn.setEnabled(False)
        kernel_file_layout.addWidget(self.clear_kernel_btn)
        
        params_layout.addLayout(kernel_file_layout)
        
        layout.addWidget(params_group)
        
        # Output format
        format_group = QtWidgets.QGroupBox("Output Format")
        format_layout = QtWidgets.QVBoxLayout(format_group)
        
        self.float_radio = QtWidgets.QRadioButton("Float (32-bit, preferred)")
        self.uint16_radio = QtWidgets.QRadioButton("16-bit unsigned integer")
        self.float_radio.setChecked(True)
        
        format_layout.addWidget(self.float_radio)
        format_layout.addWidget(self.uint16_radio)
        
        layout.addWidget(format_group)
        
        # Set scroll content widget
        scroll_area.setWidget(scroll_content)
        
        # Add scroll area to main layout
        main_layout.addWidget(scroll_area, 1)
        
        # Buttons (outside scroll area)
        button_layout = QtWidgets.QHBoxLayout()
        self.deconvolve_btn = QtWidgets.QPushButton("Deconvolve")
        self.deconvolve_btn.setEnabled(False)  # Disabled until directory is selected
        self.deconvolve_btn.clicked.connect(self._on_deconvolve_clicked)
        self.cancel_deconv_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_deconv_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(self.deconvolve_btn)
        button_layout.addWidget(self.cancel_deconv_btn)
        main_layout.addLayout(button_layout)
        
        # Connect signals
        self.single_roi_radio.toggled.connect(self._on_acq_type_changed)
        self.whole_slide_radio.toggled.connect(self._on_acq_type_changed)
        
        # Initialize the display
        self._on_acq_type_changed()
        
        return tab
        
    def _browse_directory(self):
        """Browse for output directory."""
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Directory for Deconvolved OME-TIFF Files", ""
        )
        if directory:
            self.output_directory = directory
            self.dir_label.setText(directory)
            self.dir_label.setStyleSheet("QLabel { color: black; }")
            self.deconvolve_btn.setEnabled(True)
    
    def _on_acq_type_changed(self):
        """Update UI when acquisition type changes."""
        if self.single_roi_radio.isChecked():
            if self.current_acq_id:
                # Find current acquisition info
                current_acq = next((acq for acq in self.acquisitions if acq.id == self.current_acq_id), None)
                if current_acq:
                    info_text = f"Will deconvolve: {current_acq.name}\n"
                    info_text += f"Channels: {len(current_acq.channels)}\n"
                    if current_acq.well:
                        info_text += f"Well: {current_acq.well}\n"
                    # Show source file if available (for multiple files)
                    if current_acq.source_file:
                        source_name = os.path.basename(current_acq.source_file)
                        info_text += f"Source: {source_name}"
                    self.acq_info_label.setText(info_text)
                else:
                    self.acq_info_label.setText("Will deconvolve only the currently selected acquisition.")
            else:
                self.acq_info_label.setText("Will deconvolve only the currently selected acquisition.")
        else:
            # Show more detailed information about what will be deconvolved
            total_channels = sum(len(acq.channels) for acq in self.acquisitions)
            
            # Count files if multiple files are loaded
            source_files = set()
            for acq in self.acquisitions:
                if acq.source_file:
                    source_files.add(acq.source_file)
            
            info_text = f"Will deconvolve all {len(self.acquisitions)} acquisition(s)"
            if len(source_files) > 1:
                info_text += f" from {len(source_files)} file(s)"
            info_text += ".\n"
            info_text += f"Total channels: {total_channels}\n"
            info_text += f"Acquisitions: {', '.join([acq.name for acq in self.acquisitions[:3]])}"
            if len(self.acquisitions) > 3:
                info_text += f" and {len(self.acquisitions) - 3} more..."
            self.acq_info_label.setText(info_text)
    
    def get_acq_type(self):
        """Get the selected acquisition type."""
        return "single" if self.single_roi_radio.isChecked() else "whole"
    
    def get_output_directory(self):
        """Get the selected output directory."""
        return self.output_directory
    
    def get_x0(self):
        """Get the x0 parameter."""
        return self.x0_spin.value()
    
    def get_iterations(self):
        """Get the iterations parameter."""
        return self.iterations_spin.value()
    
    def get_output_format(self):
        """Get the output format: 'float' or 'uint16'."""
        return "float" if self.float_radio.isChecked() else "uint16"
    
    # Experimental Design tab methods
    def _populate_channels(self):
        """Populate the channel combo box with available channels."""
        self.channel_combo.clear()
        
        if not self.acquisitions:
            return
        
        # Get channels from first acquisition
        first_acq = self.acquisitions[0]
        channels = first_acq.channels if hasattr(first_acq, 'channels') else []
        
        if not channels:
            return
        
        self.channel_combo.addItems(channels)
        
        # Try to find and select DNA channel (case-insensitive)
        dna_channel = None
        for i, ch in enumerate(channels):
            if 'DNA' in ch.upper():
                dna_channel = i
                break
        
        if dna_channel is not None:
            self.channel_combo.setCurrentIndex(dna_channel)
        elif channels:
            self.channel_combo.setCurrentIndex(0)
    
    def _populate_available_rois(self):
        """Populate the available ROI list."""
        self.available_roi_list.clear()
        
        if not self.acquisitions:
            return
        
        for acq in self.acquisitions:
            # Build display name
            display_name = acq.name
            if hasattr(acq, 'well') and acq.well:
                display_name = f"{acq.well} - {acq.name}"
            if hasattr(acq, 'source_file') and acq.source_file:
                display_name += f" ({os.path.basename(acq.source_file)})"
            
            item = QtWidgets.QListWidgetItem(display_name)
            item.setData(QtCore.Qt.UserRole, acq.id)
            self.available_roi_list.addItem(item)
    
    def _on_available_roi_selection_changed(self):
        """Handle selection change in available ROI list."""
        selected_items = self.available_roi_list.selectedItems()
        has_selection = len(selected_items) > 0
        
        # Check if any selected ROI is already in the pass table
        if has_selection:
            for selected_item in selected_items:
                acq_id = selected_item.data(QtCore.Qt.UserRole)
                # Check if this ROI is already in the table
                for row in range(self.roi_pass_table.rowCount()):
                    if self.roi_pass_table.item(row, 1).data(QtCore.Qt.UserRole) == acq_id:
                        has_selection = False
                        break
                if not has_selection:
                    break
        self.add_roi_btn.setEnabled(has_selection)
    
    def _filter_available_rois(self, text):
        """Filter available ROI list based on search text."""
        search_text = text.lower()
        for i in range(self.available_roi_list.count()):
            item = self.available_roi_list.item(i)
            item_text = item.text().lower()
            item.setHidden(search_text not in item_text)
    
    def _on_roi_table_selection_changed(self):
        """Handle selection change in ROI pass table."""
        has_selection = len(self.roi_pass_table.selectedItems()) > 0
        self.remove_roi_btn.setEnabled(has_selection)
        
        if has_selection:
            selected_row = self.roi_pass_table.currentRow()
            self.move_up_btn.setEnabled(selected_row > 0)
            self.move_down_btn.setEnabled(selected_row < self.roi_pass_table.rowCount() - 1)
        else:
            self.move_up_btn.setEnabled(False)
            self.move_down_btn.setEnabled(False)
        
        # Update analyze button
        self.analyze_btn.setEnabled(self.roi_pass_table.rowCount() >= 2)
    
    def _get_energy_from_row(self, row):
        """Get energy value from a table row."""
        energy_widget = self.roi_pass_table.cellWidget(row, 2)
        if energy_widget and isinstance(energy_widget, QtWidgets.QDoubleSpinBox):
            return energy_widget.value()
        return 0.0
    
    def _parse_roi_name(self, roi_name):
        """Parse pass number and energy from ROI name.
        
        Examples:
            "pass_1_-10db" -> pass_num=1, energy=-10
            "20250319_JWAB_TMA_consecutive_SHRIMP_Classic_0dB_pass_1_8_008" -> pass_num=1, energy=0
        
        Args:
            roi_name: The ROI name string to parse
            
        Returns:
            Tuple of (pass_num, energy) where both can be None if not parseable
        """
        import re
        
        pass_num = None
        energy = None
        
        # Try to parse pass number: look for "pass_" followed by a number
        # Pattern: pass_ followed by optional negative sign, digits, optional decimal point and more digits
        pass_pattern = r'pass[_\s]+(-?\d+\.?\d*)'
        pass_match = re.search(pass_pattern, roi_name, re.IGNORECASE)
        if pass_match:
            try:
                pass_num = float(pass_match.group(1))
            except (ValueError, IndexError):
                pass_num = None
        
        # Try to parse energy: look for number followed by "db" or "dB"
        # Pattern: optional negative sign, digits, optional decimal point and more digits, followed by "db" or "dB"
        energy_pattern = r'(-?\d+\.?\d*)\s*[dD][bB]'
        energy_match = re.search(energy_pattern, roi_name)
        if energy_match:
            try:
                energy = float(energy_match.group(1))
            except (ValueError, IndexError):
                energy = None
        
        return pass_num, energy
    
    def _add_roi_to_passes(self):
        """Add selected ROIs to the pass table."""
        selected_items = self.available_roi_list.selectedItems()
        if not selected_items:
            return
        
        added_count = 0
        skipped_count = 0
        
        for selected_item in selected_items:
            acq_id = selected_item.data(QtCore.Qt.UserRole)
            display_name = selected_item.text()
            
            # Check if already added
            already_added = False
            for row in range(self.roi_pass_table.rowCount()):
                if self.roi_pass_table.item(row, 1).data(QtCore.Qt.UserRole) == acq_id:
                    already_added = True
                    skipped_count += 1
                    break
            
            if already_added:
                continue
            
            # Add to table
            row = self.roi_pass_table.rowCount()
            self.roi_pass_table.insertRow(row)
            
            # Try to parse pass number and energy from ROI name
            parsed_pass, parsed_energy = self._parse_roi_name(display_name)
            
            # Pass number (editable, defaults to parsed value or row number)
            if parsed_pass is not None:
                default_pass = str(parsed_pass)
            else:
                default_pass = str(row + 1)
            pass_item = QtWidgets.QTableWidgetItem(default_pass)
            pass_item.setTextAlignment(Qt.AlignCenter)
            pass_item.setFlags(pass_item.flags() | Qt.ItemIsEditable)  # Make editable
            self.roi_pass_table.setItem(row, 0, pass_item)
            
            # ROI name
            roi_item = QtWidgets.QTableWidgetItem(display_name)
            roi_item.setData(QtCore.Qt.UserRole, acq_id)
            self.roi_pass_table.setItem(row, 1, roi_item)
            
            # Energy input (numeric, can be negative)
            energy_spin = QtWidgets.QDoubleSpinBox()
            energy_spin.setRange(-999999.0, 999999.0)
            energy_spin.setDecimals(2)
            # Use parsed energy if available, otherwise default to 0.0
            if parsed_energy is not None:
                energy_spin.setValue(parsed_energy)
            else:
                energy_spin.setValue(0.0)
            energy_spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)  # No up/down buttons for cleaner look
            self.roi_pass_table.setCellWidget(row, 2, energy_spin)
            
            # Remove button (slightly wider)
            remove_btn = QtWidgets.QPushButton("×")
            # Get the width needed for the "×" character (slightly wider)
            font_metrics = remove_btn.fontMetrics()
            char_width = font_metrics.width("×") + 18  # Add more padding
            remove_btn.setFixedWidth(char_width)
            remove_btn.clicked.connect(lambda checked, r=row: self._remove_roi_row(r))
            self.roi_pass_table.setCellWidget(row, 3, remove_btn)
            
            added_count += 1
        
        # Update pass numbers
        self._update_pass_numbers()
        
        # Clear selection in available list
        self.available_roi_list.clearSelection()
        
        # Update analyze button
        self.analyze_btn.setEnabled(self.roi_pass_table.rowCount() >= 2)
        
        # Show message if some were skipped
        if skipped_count > 0 and added_count > 0:
            QtWidgets.QMessageBox.information(self, "ROIs Added", 
                f"Added {added_count} ROI(s). {skipped_count} ROI(s) were already in the list.")
        elif skipped_count > 0:
            QtWidgets.QMessageBox.warning(self, "Already Added", 
                f"All selected ROI(s) are already in the pass list.")
    
    def _remove_roi_from_passes(self):
        """Remove selected ROI from the pass table."""
        selected_rows = set()
        for item in self.roi_pass_table.selectedItems():
            selected_rows.add(item.row())
        
        # Remove rows in reverse order to maintain indices
        for row in sorted(selected_rows, reverse=True):
            self.roi_pass_table.removeRow(row)
        
        # Update pass numbers
        self._update_pass_numbers()
        
        # Update analyze button
        self.analyze_btn.setEnabled(self.roi_pass_table.rowCount() >= 2)
    
    def _remove_roi_row(self, row):
        """Remove a specific row from the ROI pass table."""
        self.roi_pass_table.removeRow(row)
        self._update_pass_numbers()
        self.analyze_btn.setEnabled(self.roi_pass_table.rowCount() >= 2)
    
    def _update_pass_numbers(self):
        """Update pass numbers in the table (only if not manually edited)."""
        # Only update if user hasn't manually edited - check if current value matches expected
        for row in range(self.roi_pass_table.rowCount()):
            pass_item = self.roi_pass_table.item(row, 0)
            if pass_item:
                current_value = pass_item.text()
                expected_value = str(row + 1)
                # Only auto-update if it still matches the expected value
                # This allows users to manually edit and keep their edits
                if current_value == expected_value:
                    pass_item.setText(expected_value)
                # If user has edited it, keep their value
    
    def _move_roi_up(self):
        """Move selected ROI up in the pass order."""
        current_row = self.roi_pass_table.currentRow()
        if current_row <= 0:
            return
        
        # Swap rows (pass numbers stay as user edited them)
        self._swap_table_rows(current_row, current_row - 1)
        # Don't auto-update pass numbers - let user keep their edits
        self.roi_pass_table.selectRow(current_row - 1)
        self._on_roi_table_selection_changed()
    
    def _move_roi_down(self):
        """Move selected ROI down in the pass order."""
        current_row = self.roi_pass_table.currentRow()
        if current_row >= self.roi_pass_table.rowCount() - 1:
            return
        
        # Swap rows (pass numbers stay as user edited them)
        self._swap_table_rows(current_row, current_row + 1)
        # Don't auto-update pass numbers - let user keep their edits
        self.roi_pass_table.selectRow(current_row + 1)
        self._on_roi_table_selection_changed()
    
    def _swap_table_rows(self, row1, row2):
        """Swap two rows in the table."""
        # Get all items and widgets
        items1 = []
        items2 = []
        for col in range(self.roi_pass_table.columnCount()):
            items1.append(self.roi_pass_table.takeItem(row1, col))
            items2.append(self.roi_pass_table.takeItem(row2, col))
        
        widget1_energy = self.roi_pass_table.cellWidget(row1, 2)
        widget2_energy = self.roi_pass_table.cellWidget(row2, 2)
        widget1_action = self.roi_pass_table.cellWidget(row1, 3)
        widget2_action = self.roi_pass_table.cellWidget(row2, 3)
        
        # Swap items
        for col in range(self.roi_pass_table.columnCount()):
            if items2[col]:
                self.roi_pass_table.setItem(row1, col, items2[col])
            if items1[col]:
                self.roi_pass_table.setItem(row2, col, items1[col])
        
        # Swap energy widgets
        if widget1_energy:
            self.roi_pass_table.setCellWidget(row2, 2, widget1_energy)
        if widget2_energy:
            self.roi_pass_table.setCellWidget(row1, 2, widget2_energy)
        
        # Swap action widgets
        if widget1_action:
            widget1_action.clicked.disconnect()
            widget1_action.clicked.connect(lambda checked, r=row2: self._remove_roi_row(r))
            self.roi_pass_table.setCellWidget(row2, 3, widget1_action)
        if widget2_action:
            widget2_action.clicked.disconnect()
            widget2_action.clicked.connect(lambda checked, r=row1: self._remove_roi_row(r))
            self.roi_pass_table.setCellWidget(row1, 3, widget2_action)
    
    def _analyze_intensity_decay(self):
        """Analyze intensity decay across passes."""
        if self.roi_pass_table.rowCount() < 2:
            QtWidgets.QMessageBox.warning(self, "Insufficient Data", "Please select at least 2 ROIs for analysis.")
            return
        
        channel = self.channel_combo.currentText()
        if not channel:
            QtWidgets.QMessageBox.warning(self, "No Channel Selected", "Please select a channel for analysis.")
            return
        
        # Get ROI passes with energy values
        roi_passes = []
        for row in range(self.roi_pass_table.rowCount()):
            # Allow float pass numbers for better overlay control
            try:
                pass_num = float(self.roi_pass_table.item(row, 0).text())
            except (ValueError, AttributeError):
                QtWidgets.QMessageBox.warning(self, "Invalid Pass Number", 
                    f"Row {row + 1} has an invalid pass number. Please enter a numeric value.")
                return
            acq_id = self.roi_pass_table.item(row, 1).data(QtCore.Qt.UserRole)
            energy = self._get_energy_from_row(row)
            roi_passes.append((acq_id, pass_num, energy))
        
        # Group by energy
        energy_groups = {}
        for acq_id, pass_num, energy in roi_passes:
            if energy not in energy_groups:
                energy_groups[energy] = []
            energy_groups[energy].append((acq_id, pass_num))
        
        # Calculate intensities for each energy group
        energy_data = {}  # Dict mapping energy to (pass_numbers, intensities)
        
        try:
            for energy, group_rois in energy_groups.items():
                intensities = []
                pass_numbers = []
                
                for acq_id, pass_num in group_rois:
                    # Get loader for this acquisition
                    if not self.parent_window:
                        QtWidgets.QMessageBox.critical(self, "Error", "Cannot access parent window.")
                        return
                    
                    loader = self.parent_window._get_loader_for_acquisition(acq_id)
                    if not loader:
                        QtWidgets.QMessageBox.warning(self, "Error", f"Cannot get loader for ROI: {acq_id}")
                        continue
                    
                    # Get original acquisition ID
                    original_acq_id = self.parent_window._get_original_acq_id(acq_id)
                    
                    # Get channel image
                    try:
                        channel_img = loader.get_image(original_acq_id, channel)
                    except Exception as e:
                        QtWidgets.QMessageBox.warning(self, "Error", f"Failed to load channel '{channel}' for ROI: {str(e)}")
                        continue
                    
                    # Sum all pixel values (each pixel is 1um²)
                    intensity = np.sum(channel_img)
                    intensities.append(intensity)
                    pass_numbers.append(pass_num)
                
                if len(intensities) >= 2:
                    energy_data[energy] = (pass_numbers, intensities)
            
            if not energy_data:
                QtWidgets.QMessageBox.warning(self, "Insufficient Data", "Could not calculate intensities for at least 2 passes for any energy group.")
                return
            
            # Store data (for backward compatibility, store first energy's data)
            if energy_data:
                first_energy = list(energy_data.keys())[0]
                pass_nums, ints = energy_data[first_energy]
                self.intensity_data = dict(zip(pass_nums, ints))
            
            # Plot intensity decay with separate curves for each energy
            self._plot_intensity_decay(energy_data, channel)
            
            # Enable plot tab, kernel generation tab, and export button
            self.exp_design_tabs.setTabEnabled(1, True)
            self.exp_design_tabs.setTabEnabled(2, True)  # Enable Generate Kernel Arrays tab
            self.exp_design_tabs.setCurrentIndex(1)  # Switch to plot tab
            self.export_plot_btn.setEnabled(True)
            
            # Update energy dropdown in kernel generation tab
            self._update_energy_dropdown()
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _inverse_sigmoidal_decay(self, x, I_max, k, x0):
        """Inverse sigmoidal decay function: I(x) = I_max / (1 + exp(k * (x - x0)))."""
        return I_max / (1 + np.exp(k * (x - x0)))
    
    def _inverse_sigmoidal_decay_derivative(self, x, I_max, k, x0):
        """Derivative of inverse sigmoidal decay function: dI/dx = -I_max * k * exp(k*(x-x0)) / (1 + exp(k*(x-x0)))^2."""
        exp_term = np.exp(k * (x - x0))
        return -I_max * k * exp_term / ((1 + exp_term) ** 2)
    
    def _plot_intensity_decay(self, energy_data, channel):
        """Plot intensity decay with fitted curves for each energy value.
        
        Args:
            energy_data: Dict mapping energy (float) to (pass_numbers, intensities) tuples
            channel: Channel name for the plot title
        """
        self.plot_figure.clear()
        
        # Create two subplots side by side
        ax1 = self.plot_figure.add_subplot(121)  # Left plot: intensity decay
        ax2 = self.plot_figure.add_subplot(122)  # Right plot: derivative
        
        # Get color map for different energies
        energies = sorted(energy_data.keys())
        n_energies = len(energies)
        colors = cm.tab10(np.linspace(0, 1, max(10, n_energies)))[:n_energies]
        
        # Collect all unique pass numbers across all energies for x-axis ticks
        all_pass_numbers = set()
        for pass_nums, _ in energy_data.values():
            all_pass_numbers.update(pass_nums)
        all_pass_numbers = sorted(all_pass_numbers)
        
        # Store fit parameters for all energies
        all_fit_params = []
        fit_text_parts = []
        
        # Clear previous fit parameters
        self.energy_fit_params = {}
        
        # Plot each energy group
        for idx, energy in enumerate(energies):
            pass_nums, ints = energy_data[energy]
            
            # Sort by pass number
            sorted_data = sorted(zip(pass_nums, ints))
            pass_nums = [p for p, _ in sorted_data]
            ints = [i for _, i in sorted_data]
            
            color = colors[idx]
            
            # Plot data points
            ax1.scatter(pass_nums, ints, color=color, s=100, zorder=3, alpha=0.7, 
                       label=f'Energy={energy:.2f} (data)')
            
            # Plot connecting lines
            ax1.plot(pass_nums, ints, '--', color=color, alpha=0.3, linewidth=1)
            
            # Fit curve if we have enough points
            if len(pass_nums) >= 3:
                try:
                    # Initial guess: I_max is first intensity, k and x0 estimated
                    I_max_guess = ints[0]
                    k_guess = 0.5  # Positive for decay
                    x0_guess = np.mean(pass_nums)
                    
                    # Fit the curve
                    popt, _ = curve_fit(
                        self._inverse_sigmoidal_decay,
                        pass_nums,
                        ints,
                        p0=[I_max_guess, k_guess, x0_guess],
                        maxfev=5000
                    )
                    
                    I_max_fit, k_fit, x0_fit = popt
                    
                    # Store fit parameters for this energy
                    self.energy_fit_params[energy] = {
                        'I_max': I_max_fit,
                        'k': k_fit,
                        'x0': x0_fit
                    }
                    
                    # Generate smooth curve for plotting
                    x_smooth = np.linspace(min(pass_nums), max(pass_nums), 100)
                    y_smooth = self._inverse_sigmoidal_decay(x_smooth, I_max_fit, k_fit, x0_fit)
                    
                    # Plot fitted curve
                    ax1.plot(x_smooth, y_smooth, '-', color=color, linewidth=2, 
                            label=f'Energy={energy:.2f} (fit)', zorder=2)
                    
                    # Calculate fit quality
                    y_predicted = self._inverse_sigmoidal_decay(np.array(pass_nums), I_max_fit, k_fit, x0_fit)
                    residuals = np.array(ints) - y_predicted
                    sum_abs_residuals = np.sum(np.abs(residuals))
                    mean_abs_residual = np.mean(np.abs(residuals))
                    
                    # Store fit parameters
                    all_fit_params.append((energy, I_max_fit, k_fit, x0_fit, sum_abs_residuals, mean_abs_residual))
                    
                    # Plot derivative on right subplot
                    y_derivative = self._inverse_sigmoidal_decay_derivative(x_smooth, I_max_fit, k_fit, x0_fit)
                    ax2.plot(x_smooth, y_derivative, '-', color=color, linewidth=2, 
                            label=f'Energy={energy:.2f}', zorder=2)
                    
                except Exception as e:
                    print(f"Curve fitting failed for energy {energy}: {e}")
                    fit_text_parts.append(f"Energy={energy:.2f}: Fit failed")
            else:
                fit_text_parts.append(f"Energy={energy:.2f}: Need ≥3 points")
        
        # Build fit parameters text
        if all_fit_params:
            fit_text_parts = []
            for energy, I_max, k, x0, sum_res, mean_res in all_fit_params:
                fit_text_parts.append(
                    f"E={energy:.2f}: I_max={I_max:.2e}, k={k:.3f}, x0={x0:.2f}, "
                    f"Sum|res|={sum_res:.2e}"
                )
            fit_text = " | ".join(fit_text_parts)
        else:
            fit_text = " | ".join(fit_text_parts) if fit_text_parts else "No fits available"
        
        self.fit_params_label.setText(fit_text)
        self.fit_params_label.setVisible(True)
        
        # Add zero line to derivative plot
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_xlabel('Pass Number', fontsize=12)
        ax2.set_ylabel('dI/dx (Rate of Change)', fontsize=12)
        ax2.set_title('Derivative of Fitted Curves', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)
        
        # Set labels and title for main plot
        ax1.set_xlabel('Pass Number', fontsize=12)
        ax1.set_ylabel('Total Intensity (sum of all pixels)', fontsize=12)
        ax1.set_title(f'Intensity Decay Analysis - Channel: {channel}', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=8)
        
        # Set x-axis ticks to show all unique pass numbers
        if all_pass_numbers:
            ax1.set_xticks(all_pass_numbers)
            ax2.set_xticks(all_pass_numbers)
        
        self.plot_figure.tight_layout()
        self.plot_canvas.draw()
    
    def _export_plot(self):
        """Export the plot to a file."""
        if not self.plot_figure.axes:
            QtWidgets.QMessageBox.warning(self, "No Plot", "Please run analysis first.")
            return
        
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Plot",
            "",
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*)"
        )
        
        if file_path:
            try:
                self.plot_figure.savefig(file_path, dpi=300, bbox_inches='tight')
                QtWidgets.QMessageBox.information(self, "Success", f"Plot exported to:\n{file_path}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to export plot: {str(e)}")
    
    def _generate_kernel_arrays(self):
        """Generate passes and contributions arrays based on current parameters."""
        from openimc.processing.hrimc_kernel_generator import (
            compute_hrimc_passes_contributions,
            example_inverse_sigmoid_loss
        )
        
        try:
            # Get parameters
            midpoint = self.midpoint_spin.value()
            slope = self.slope_spin.value()
            min_fraction = self.min_fraction_spin.value()
            max_fraction = self.max_fraction_spin.value()
            step_size = self.step_size_spin.value()
            pixel_size = self.pixel_size_spin.value()
            spot_diameter = self.spot_diameter_spin.value()
            
            # Create loss function
            def loss_fn(passes):
                return example_inverse_sigmoid_loss(
                    passes,
                    midpoint=midpoint,
                    slope=slope,
                    min_fraction=min_fraction,
                    max_fraction=max_fraction
                )
            
            # Generate arrays using Shapely geometry
            passes, contributions, psf_kernel = compute_hrimc_passes_contributions(
                step_size_um=step_size,
                loss_fn=loss_fn,
                pixel_size_um=pixel_size,
                spot_diameter_um=spot_diameter,
                n_subpixels=15,  # Default subpixel discretization
                circle_resolution=64  # Default circle resolution
            )
            
            # Store generated arrays and PSF kernel
            self.generated_passes = passes
            self.generated_contributions = contributions
            self.generated_psf_kernel = psf_kernel
            
            # Update status
            status_text = f"Generated arrays successfully!\n"
            status_text += f"Passes shape: {passes.shape}, Contributions shape: {contributions.shape}\n"
            status_text += f"3×3 PSF kernel:\n{psf_kernel}"
            self.kernel_status_label.setText(status_text)
            self.kernel_status_label.setStyleSheet("QLabel { color: #006600; font-style: normal; }")
            
            # Enable save button
            self.save_kernel_btn.setEnabled(True)
            
            QtWidgets.QMessageBox.information(
                self,
                "Success",
                f"Arrays generated successfully!\n\n"
                f"Passes: {passes.shape[0]} elements\n"
                f"Contributions: {contributions.shape[0]} elements\n"
                f"Sum of contributions: {contributions.sum():.6f}"
            )
            
        except Exception as e:
            error_msg = f"Failed to generate arrays: {str(e)}"
            self.kernel_status_label.setText(error_msg)
            self.kernel_status_label.setStyleSheet("QLabel { color: #cc0000; font-style: normal; }")
            QtWidgets.QMessageBox.critical(self, "Error", error_msg)
            import traceback
            traceback.print_exc()
    
    def _save_kernel_arrays(self):
        """Save generated passes and contributions arrays to a file."""
        if self.generated_passes is None or self.generated_contributions is None:
            QtWidgets.QMessageBox.warning(self, "No Arrays", "Please generate arrays first.")
            return
        
        from openimc.processing.hrimc_kernel_generator import save_passes_contributions
        
        # Generate default filename based on selected energy
        default_filename = "energy_kernels.npz"
        if self.energy_combo.currentIndex() >= 0:
            energy = self.energy_combo.itemData(self.energy_combo.currentIndex())
            if energy is not None:
                # Format energy value (remove negative sign, replace decimal point)
                energy_str = f"{energy:.2f}".replace("-", "neg").replace(".", "_")
                default_filename = f"energy_{energy_str}_kernels.npz"
        
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Kernel Arrays",
            default_filename,
            "NumPy Files (*.npz);;All Files (*)"
        )
        
        if file_path:
            try:
                # Create metadata
                metadata = {
                    'midpoint': self.midpoint_spin.value(),
                    'slope': self.slope_spin.value(),
                    'min_fraction': self.min_fraction_spin.value(),
                    'max_fraction': self.max_fraction_spin.value(),
                    'step_size_um': self.step_size_spin.value(),
                    'pixel_size_um': self.pixel_size_spin.value(),
                    'spot_diameter_um': self.spot_diameter_spin.value()
                }
                
                save_passes_contributions(
                    self.generated_passes,
                    self.generated_contributions,
                    file_path,
                    metadata=metadata,
                    psf_kernel=self.generated_psf_kernel
                )
                
                QtWidgets.QMessageBox.information(
                    self,
                    "Success",
                    f"Arrays saved to:\n{file_path}"
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save arrays: {str(e)}")
    
    def _load_kernel_arrays(self):
        """Load passes and contributions arrays from a file."""
        from openimc.processing.hrimc_kernel_generator import load_passes_contributions
        
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Kernel Arrays",
            "",
            "NumPy Files (*.npz);;All Files (*)"
        )
        
        if file_path:
            try:
                passes, contributions, metadata, psf_kernel = load_passes_contributions(file_path)
                
                # Store loaded arrays and PSF kernel
                self.generated_passes = passes
                self.generated_contributions = contributions
                self.generated_psf_kernel = psf_kernel
                
                # Update parameters if metadata is available
                if metadata:
                    if 'midpoint' in metadata:
                        self.midpoint_spin.setValue(float(metadata['midpoint']))
                    if 'slope' in metadata:
                        self.slope_spin.setValue(float(metadata['slope']))
                    if 'min_fraction' in metadata:
                        self.min_fraction_spin.setValue(float(metadata['min_fraction']))
                    if 'max_fraction' in metadata:
                        self.max_fraction_spin.setValue(float(metadata['max_fraction']))
                    if 'step_size_um' in metadata:
                        self.step_size_spin.setValue(float(metadata['step_size_um']))
                    if 'pixel_size_um' in metadata:
                        self.pixel_size_spin.setValue(float(metadata['pixel_size_um']))
                    if 'spot_diameter_um' in metadata:
                        self.spot_diameter_spin.setValue(float(metadata['spot_diameter_um']))
                
                # Update status
                status_text = f"Loaded arrays from: {file_path}\n"
                status_text += f"Passes shape: {passes.shape}, Contributions shape: {contributions.shape}\n"
                status_text += f"Sum of contributions: {contributions.sum():.6f}"
                if psf_kernel is not None:
                    status_text += f"\nPSF kernel: {psf_kernel.shape} (available)"
                self.kernel_status_label.setText(status_text)
                self.kernel_status_label.setStyleSheet("QLabel { color: #006600; font-style: normal; }")
                
                # Enable save button
                self.save_kernel_btn.setEnabled(True)
                
                msg = f"Arrays loaded successfully!\n\n"
                msg += f"Passes: {passes.shape[0]} elements\n"
                msg += f"Contributions: {contributions.shape[0]} elements"
                if psf_kernel is not None:
                    msg += f"\nPSF kernel: {psf_kernel.shape} (will be saved)"
                QtWidgets.QMessageBox.information(self, "Success", msg)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load arrays: {str(e)}")
                import traceback
                traceback.print_exc()
    
    def _on_deconvolve_clicked(self):
        """Handle deconvolve button click - accept dialog to trigger deconvolution."""
        if not self.output_directory:
            QtWidgets.QMessageBox.warning(self, "No Output Directory", "Please select an output directory first.")
            return
        self.accept()
    
    def get_passes_contributions(self):
        """Get the generated/loaded passes and contributions arrays."""
        return self.generated_passes, self.generated_contributions
    
    def _load_kernel_arrays_for_deconv(self):
        """Load kernel arrays for use in deconvolution."""
        from openimc.processing.hrimc_kernel_generator import load_passes_contributions
        
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Kernel Arrays for Deconvolution",
            "",
            "NumPy Files (*.npz);;All Files (*)"
        )
        
        if file_path:
            try:
                passes, contributions, metadata, psf_kernel = load_passes_contributions(file_path)
                
                # Store loaded arrays and PSF kernel
                self.deconv_passes = passes
                self.deconv_contributions = contributions
                self.deconv_psf_kernel = psf_kernel
                
                # Update UI
                if psf_kernel is not None:
                    self.kernel_file_label.setText(f"Loaded: {os.path.basename(file_path)} (PSF kernel available)")
                    # Hide x0 parameter when PSF kernel is loaded (x0 is baked into the kernel)
                    self.x0_layout.itemAt(0).widget().hide()  # Hide label
                    self.x0_spin.hide()
                else:
                    self.kernel_file_label.setText(f"Loaded: {os.path.basename(file_path)}")
                self.kernel_file_label.setStyleSheet("QLabel { color: #006600; }")
                self.clear_kernel_btn.setEnabled(True)
                
                msg = f"Kernel arrays loaded successfully!\n\n"
                msg += f"Passes: {passes.shape[0]} elements\n"
                msg += f"Contributions: {contributions.shape[0]} elements\n"
                if psf_kernel is not None:
                    msg += f"PSF kernel: {psf_kernel.shape} (will be used for direct kernel override)\n"
                    msg += f"\nNote: x0 parameter is hidden since it's baked into the PSF kernel."
                msg += f"\nThese arrays will be used for deconvolution."
                QtWidgets.QMessageBox.information(self, "Success", msg)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load arrays: {str(e)}")
                import traceback
                traceback.print_exc()
    
    def _clear_kernel_arrays(self):
        """Clear loaded kernel arrays and use defaults."""
        self.deconv_passes = None
        self.deconv_contributions = None
        self.deconv_psf_kernel = None
        self.kernel_file_label.setText("No kernel arrays loaded (using defaults)")
        self.kernel_file_label.setStyleSheet("QLabel { color: #666; }")
        self.clear_kernel_btn.setEnabled(False)
        # Show x0 parameter again when kernel is cleared
        self.x0_layout.itemAt(0).widget().show()  # Show label
        self.x0_spin.show()
    
    def get_deconv_passes_contributions(self):
        """Get the passes, contributions arrays, and PSF kernel for deconvolution."""
        return self.deconv_passes, self.deconv_contributions, self.deconv_psf_kernel

