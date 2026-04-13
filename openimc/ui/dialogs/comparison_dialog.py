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

from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

from openimc.data.mcd_loader import AcquisitionInfo, MCDLoader  # noqa: F401
from openimc.ui.mpl_canvas import MplCanvas
from openimc.ui.utils import arcsinh_normalize


class DynamicComparisonDialog(QtWidgets.QDialog):
    def __init__(self, acqs: List[AcquisitionInfo], loader: MCDLoader, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dynamic Comparison Mode")

        screen = QtWidgets.QApplication.desktop().screenGeometry()
        self.resize(int(screen.width() * 0.8), int(screen.height() * 0.8))

        self.acqs = acqs
        self.loader = loader
        self.parent_window = parent
        self.selected_acquisitions: List[str] = []

        # Cache full ROI stacks so RGB rendering can slice channels without re-reading.
        self.image_stack_cache: Dict[str, np.ndarray] = {}
        self._prefetch_limit = 5
        self._range_cache: Dict[Tuple[str, Optional[str], bool, float, Tuple[str, ...]], Tuple[float, float]] = {}

        self.last_selected_channel: Optional[str] = None
        self.previous_scaling_channel: Optional[str] = None
        self._loaded_scaling_image_id: Optional[str] = None

        self.channel_linked_scaling: Dict[str, Dict[str, float]] = {}
        self.channel_per_image_scaling: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.image_arcsinh_state: Dict[str, Dict[str, object]] = {}

        self.channel_color_assignments: Dict[str, str] = {}
        self.channel_enabled_state: Dict[str, bool] = {}
        self.available_colors_rgb = ["Red", "Green", "Blue"]

        self._display_update_depth = 0
        self._pending_display_update = False
        self._canvas_slots: List[Dict[str, object]] = []

        self.setMinimumSize(1000, 700)
        self._create_ui()

    def _get_loader_for_acquisition(self, acq_id: str):
        """Get the appropriate loader for a given acquisition ID."""
        if self.parent_window and hasattr(self.parent_window, "_get_loader_for_acquisition"):
            loader = self.parent_window._get_loader_for_acquisition(acq_id)
            if loader is not None:
                return loader
        return self.loader

    def _get_original_acq_id(self, acq_id: str) -> str:
        """Get the original acquisition ID from a unique ID."""
        if self.parent_window and hasattr(self.parent_window, "_get_original_acq_id"):
            return self.parent_window._get_original_acq_id(acq_id)
        return acq_id

    @contextmanager
    def _batch_display_updates(self):
        self._display_update_depth += 1
        try:
            yield
        finally:
            self._display_update_depth = max(0, self._display_update_depth - 1)
            if self._display_update_depth == 0 and self._pending_display_update:
                self._pending_display_update = False
                self._update_display()

    def _request_display_update(self):
        if self._display_update_depth > 0:
            self._pending_display_update = True
            return
        self._update_display()

    def _create_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        splitter = QtWidgets.QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        self.control_scroll = QtWidgets.QScrollArea()
        self.control_scroll.setWidgetResizable(True)
        self.control_scroll.setMinimumWidth(360)
        self.control_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.control_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        control_panel = QtWidgets.QWidget()
        control_panel.setMinimumWidth(360)
        control_layout = QtWidgets.QVBoxLayout(control_panel)
        self.control_scroll.setWidget(control_panel)
        splitter.addWidget(self.control_scroll)

        acq_group = QtWidgets.QGroupBox("Acquisitions")
        acq_layout = QtWidgets.QVBoxLayout(acq_group)

        acq_layout.addWidget(QtWidgets.QLabel("Available:"))
        self.acq_search = QtWidgets.QLineEdit()
        self.acq_search.setPlaceholderText("Search acquisitions...")
        self.acq_search.textChanged.connect(self._filter_acquisitions)
        acq_layout.addWidget(self.acq_search)

        self.available_acq_list = QtWidgets.QListWidget()
        self.available_acq_list.setMaximumHeight(120)
        for ai in self.acqs:
            import os

            file_name = os.path.basename(ai.source_file) if getattr(ai, "source_file", None) else "Unknown"
            label = ai.well if ai.well else ai.name
            item = QtWidgets.QListWidgetItem(f"{label} [{file_name}]")
            item.setData(Qt.UserRole, ai.id)
            self.available_acq_list.addItem(item)
        acq_layout.addWidget(self.available_acq_list)

        acq_buttons = QtWidgets.QHBoxLayout()
        self.add_acq_btn = QtWidgets.QPushButton("Add →")
        acq_buttons.addWidget(self.add_acq_btn)
        acq_layout.addLayout(acq_buttons)

        acq_layout.addWidget(QtWidgets.QLabel("Selected:"))
        self.acq_list = QtWidgets.QListWidget()
        self.acq_list.setMaximumHeight(120)
        acq_layout.addWidget(self.acq_list)

        self.remove_acq_btn = QtWidgets.QPushButton("← Remove")
        acq_layout.addWidget(self.remove_acq_btn)
        control_layout.addWidget(acq_group)

        channel_group = QtWidgets.QGroupBox("Channel")
        channel_layout = QtWidgets.QVBoxLayout(channel_group)

        self.channel_combo = QtWidgets.QComboBox()
        channel_layout.addWidget(QtWidgets.QLabel("Marker channel:"))
        channel_layout.addWidget(self.channel_combo)

        self.rgb_mode_chk = QtWidgets.QCheckBox("RGB Mode")
        self.rgb_mode_chk.toggled.connect(self._on_rgb_mode_toggled)
        channel_layout.addWidget(self.rgb_mode_chk)

        self.rgb_frame = QtWidgets.QFrame()
        self.rgb_frame.setFrameStyle(QtWidgets.QFrame.Box)
        rgb_layout = QtWidgets.QVBoxLayout(self.rgb_frame)

        self.channel_color_search = QtWidgets.QLineEdit()
        self.channel_color_search.setPlaceholderText("Search RGB channels...")
        self.channel_color_search.textChanged.connect(self._filter_rgb_channels)
        rgb_layout.addWidget(self.channel_color_search)

        self.channel_color_table = QtWidgets.QTableWidget()
        self.channel_color_table.setColumnCount(2)
        self.channel_color_table.setHorizontalHeaderLabels(["Channel", "Color"])
        self.channel_color_table.horizontalHeader().setStretchLastSection(True)
        self.channel_color_table.verticalHeader().setVisible(False)
        self.channel_color_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.channel_color_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.channel_color_table.setMinimumHeight(170)
        self.channel_color_table.setMaximumHeight(320)
        self.channel_color_table.itemChanged.connect(self._on_rgb_channel_item_changed)
        rgb_layout.addWidget(self.channel_color_table)

        self.rgb_frame.setVisible(False)
        channel_layout.addWidget(self.rgb_frame)
        control_layout.addWidget(channel_group)

        options_group = QtWidgets.QGroupBox("Display Options")
        options_layout = QtWidgets.QVBoxLayout(options_group)

        self.link_chk = QtWidgets.QCheckBox("Linked scaling (shared min/max)")
        self.link_chk.setChecked(True)
        self.grayscale_chk = QtWidgets.QCheckBox("Grayscale mode")
        self.custom_scaling_chk = QtWidgets.QCheckBox("Custom scaling")
        self.overlay_chk = QtWidgets.QCheckBox("Show segmentation overlay (when available)")

        options_layout.addWidget(self.link_chk)
        options_layout.addWidget(self.grayscale_chk)
        options_layout.addWidget(self.custom_scaling_chk)
        options_layout.addWidget(self.overlay_chk)

        self.scaling_frame = QtWidgets.QFrame()
        self.scaling_frame.setFrameStyle(QtWidgets.QFrame.Box)
        scaling_layout = QtWidgets.QVBoxLayout(self.scaling_frame)
        scaling_layout.addWidget(QtWidgets.QLabel("Custom Intensity Range:"))

        self.scaling_channel_row = QtWidgets.QWidget()
        scaling_channel_layout = QtWidgets.QHBoxLayout(self.scaling_channel_row)
        scaling_channel_layout.setContentsMargins(0, 0, 0, 0)
        scaling_channel_layout.addWidget(QtWidgets.QLabel("Channel:"))
        self.scaling_channel_combo = QtWidgets.QComboBox()
        scaling_channel_layout.addWidget(self.scaling_channel_combo)
        scaling_channel_layout.addStretch()
        scaling_layout.addWidget(self.scaling_channel_row)

        self.image_selection_row = QtWidgets.QWidget()
        image_selection_layout = QtWidgets.QHBoxLayout(self.image_selection_row)
        image_selection_layout.setContentsMargins(0, 0, 0, 0)
        image_selection_layout.addWidget(QtWidgets.QLabel("Image:"))
        self.image_combo = QtWidgets.QComboBox()
        image_selection_layout.addWidget(self.image_combo)
        image_selection_layout.addStretch()
        scaling_layout.addWidget(self.image_selection_row)

        minmax_layout = QtWidgets.QHBoxLayout()
        minmax_layout.addWidget(QtWidgets.QLabel("Min:"))
        self.min_spinbox = QtWidgets.QDoubleSpinBox()
        self.min_spinbox.setRange(-999999, 999999)
        self.min_spinbox.setDecimals(3)
        self.min_spinbox.setValue(0.0)
        minmax_layout.addWidget(self.min_spinbox)
        minmax_layout.addWidget(QtWidgets.QLabel("Max:"))
        self.max_spinbox = QtWidgets.QDoubleSpinBox()
        self.max_spinbox.setRange(-999999, 999999)
        self.max_spinbox.setDecimals(3)
        self.max_spinbox.setValue(100.0)
        minmax_layout.addWidget(self.max_spinbox)
        scaling_layout.addLayout(minmax_layout)

        norm_layout = QtWidgets.QHBoxLayout()
        self.arcsinh_chk = QtWidgets.QCheckBox("Arcsinh")
        norm_layout.addWidget(self.arcsinh_chk)
        norm_layout.addWidget(QtWidgets.QLabel("cofactor:"))
        self.arcsinh_cofactor = QtWidgets.QDoubleSpinBox()
        self.arcsinh_cofactor.setRange(0.01, 1000.0)
        self.arcsinh_cofactor.setDecimals(2)
        self.arcsinh_cofactor.setSingleStep(0.25)
        self.arcsinh_cofactor.setValue(1.0)
        norm_layout.addWidget(self.arcsinh_cofactor)
        norm_layout.addStretch()
        scaling_layout.addLayout(norm_layout)

        button_layout = QtWidgets.QHBoxLayout()
        self.default_range_btn = QtWidgets.QPushButton("Original Range")
        button_layout.addWidget(self.default_range_btn)
        button_layout.addStretch()
        scaling_layout.addLayout(button_layout)

        self.scaling_frame.setVisible(False)
        options_layout.addWidget(self.scaling_frame)
        control_layout.addWidget(options_group)
        control_layout.addStretch(1)

        self.image_scroll = QtWidgets.QScrollArea()
        self.image_widget = QtWidgets.QWidget()
        self.image_layout = QtWidgets.QGridLayout(self.image_widget)
        self.image_layout.setContentsMargins(8, 8, 8, 8)
        self.image_layout.setSpacing(8)
        self.image_scroll.setWidget(self.image_widget)
        self.image_scroll.setWidgetResizable(True)
        splitter.addWidget(self.image_scroll)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, max(700, self.width() - 420)])

        self.add_acq_btn.clicked.connect(self._add_acquisition)
        self.remove_acq_btn.clicked.connect(self._remove_acquisition)
        self.channel_combo.currentTextChanged.connect(self._on_single_channel_changed)
        self.link_chk.toggled.connect(self._on_link_contrast_toggled)
        self.grayscale_chk.toggled.connect(lambda _checked: self._request_display_update())
        self.custom_scaling_chk.toggled.connect(self._on_comparison_scaling_toggled)
        self.overlay_chk.toggled.connect(lambda _checked: self._request_display_update())
        self.scaling_channel_combo.currentTextChanged.connect(self._on_scaling_channel_changed)
        self.image_combo.currentIndexChanged.connect(self._on_image_selection_changed)
        self.min_spinbox.valueChanged.connect(lambda _value: self._apply_comparison_scaling())
        self.max_spinbox.valueChanged.connect(lambda _value: self._apply_comparison_scaling())
        self.arcsinh_chk.toggled.connect(self._on_arcsinh_toggled)
        self.arcsinh_cofactor.valueChanged.connect(self._on_arcsinh_cofactor_changed)
        self.default_range_btn.clicked.connect(self._comparison_default_range)

        self._sync_channel_controls()
        self._update_scaling_widget_states()

    def closeEvent(self, event):
        self._clear_display()
        super().closeEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_canvas_geometry()

    def __del__(self):
        try:
            self._clear_display()
        except Exception:
            pass

    def _get_acquisition_info(self, acq_id: str) -> Optional[AcquisitionInfo]:
        return next((ai for ai in self.acqs if ai.id == acq_id), None)

    def _get_available_channels(self) -> List[str]:
        if not self.selected_acquisitions:
            return []

        acquisition_channels = []
        for acq_id in self.selected_acquisitions:
            acq_info = self._get_acquisition_info(acq_id)
            if acq_info is None:
                continue
            acquisition_channels.append(list(acq_info.channels))

        if not acquisition_channels:
            return []

        common_channels = set(acquisition_channels[0])
        for channels in acquisition_channels[1:]:
            common_channels &= set(channels)

        return [channel for channel in acquisition_channels[0] if channel in common_channels]

    def _get_acquisition_subtitle(self, acq_id: str) -> str:
        acq_info = self._get_acquisition_info(acq_id)
        if not acq_info:
            return "Unknown"

        import os

        file_name = os.path.basename(acq_info.source_file) if getattr(acq_info, "source_file", None) else "Unknown"
        label = acq_info.well if acq_info.well else acq_info.name
        return f"{label} [{file_name}]"

    def _filter_acquisitions(self):
        text = self.acq_search.text().lower()
        for i in range(self.available_acq_list.count()):
            item = self.available_acq_list.item(i)
            item.setHidden(text not in item.text().lower())

    def _filter_rgb_channels(self):
        text = self.channel_color_search.text().lower()
        for row in range(self.channel_color_table.rowCount()):
            item = self.channel_color_table.item(row, 0)
            if item is not None:
                self.channel_color_table.setRowHidden(row, text not in item.text().lower())

    def _populate_channel_combo(self, channels: List[str], preferred_channel: Optional[str]):
        self.channel_combo.blockSignals(True)
        self.channel_combo.clear()
        self.channel_combo.addItems(channels)

        final_channel: Optional[str] = None
        if preferred_channel and preferred_channel in channels:
            final_channel = preferred_channel
        elif channels:
            final_channel = channels[0]

        if final_channel:
            self.channel_combo.setCurrentIndex(channels.index(final_channel))
            self.last_selected_channel = final_channel
        else:
            self.last_selected_channel = None
        self.channel_combo.blockSignals(False)

    def _populate_rgb_table(self, channels: List[str]):
        prev_assignments = self.channel_color_assignments.copy()
        prev_enabled = self.channel_enabled_state.copy()

        self.channel_color_table.blockSignals(True)
        self.channel_color_table.setRowCount(0)
        self.channel_color_assignments = {}
        self.channel_enabled_state = {}

        for row, channel in enumerate(channels):
            self.channel_color_table.insertRow(row)

            channel_item = QtWidgets.QTableWidgetItem(channel)
            channel_item.setFlags(
                (channel_item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                & ~Qt.ItemIsEditable
            )
            channel_item.setCheckState(Qt.Checked if prev_enabled.get(channel, False) else Qt.Unchecked)
            self.channel_color_table.setItem(row, 0, channel_item)

            color_combo = QtWidgets.QComboBox()
            color_combo.addItems(self.available_colors_rgb)
            initial_color = prev_assignments.get(channel, self.available_colors_rgb[row % len(self.available_colors_rgb)])
            if initial_color not in self.available_colors_rgb:
                initial_color = self.available_colors_rgb[0]
            color_combo.blockSignals(True)
            color_combo.setCurrentText(initial_color)
            color_combo.blockSignals(False)
            color_combo.currentTextChanged.connect(
                lambda text, channel_name=channel: self._on_rgb_color_changed(channel_name, text)
            )
            self.channel_color_table.setCellWidget(row, 1, color_combo)

            self.channel_enabled_state[channel] = channel_item.checkState() == Qt.Checked
            self.channel_color_assignments[channel] = initial_color

        self.channel_color_table.blockSignals(False)

    def _sync_channel_controls(self):
        channels = self._get_available_channels()
        preferred_channel = self.last_selected_channel or self.channel_combo.currentText()
        preferred_scaling_channel = self.scaling_channel_combo.currentText() or self.previous_scaling_channel
        preferred_image = self.image_combo.currentData()

        with self._batch_display_updates():
            self._populate_channel_combo(channels, preferred_channel)
            self._populate_rgb_table(channels)
            self._refresh_scaling_channel_options(preferred_scaling_channel)
            self._update_image_combo(preferred_image)
            self._load_scaling_controls()

        self._request_display_update()

    def _on_single_channel_changed(self, channel: str):
        if channel:
            self.last_selected_channel = channel
        with self._batch_display_updates():
            self._refresh_scaling_channel_options(self.scaling_channel_combo.currentText())
            self._update_image_combo(self.image_combo.currentData())
            self._load_scaling_controls()
        self._request_display_update()

    def _on_rgb_mode_toggled(self):
        is_rgb = self.rgb_mode_chk.isChecked()
        self.rgb_frame.setVisible(is_rgb)
        self.channel_combo.setEnabled(not is_rgb)

        with self._batch_display_updates():
            self._refresh_scaling_channel_options(self.scaling_channel_combo.currentText())
            self._update_image_combo(self.image_combo.currentData())
            self._load_scaling_controls()

        self._request_display_update()

    def _on_rgb_channel_item_changed(self, item: QtWidgets.QTableWidgetItem):
        if item.column() != 0:
            return

        channel = item.text()
        self.channel_enabled_state[channel] = item.checkState() == Qt.Checked

        with self._batch_display_updates():
            self._refresh_scaling_channel_options(self.scaling_channel_combo.currentText())
            self._load_scaling_controls()

        self._request_display_update()

    def _on_rgb_color_changed(self, channel_name: str, color_name: str):
        self.channel_color_assignments[channel_name] = color_name
        self._request_display_update()

    def _checked_rgb_channels(self) -> List[str]:
        channels: List[str] = []
        for row in range(self.channel_color_table.rowCount()):
            item = self.channel_color_table.item(row, 0)
            if item is not None and item.checkState() == Qt.Checked:
                channels.append(item.text())
        return channels

    def _checked_rgb_channels_by_color(self) -> Dict[str, List[str]]:
        by_color = {color: [] for color in self.available_colors_rgb}
        for row in range(self.channel_color_table.rowCount()):
            item = self.channel_color_table.item(row, 0)
            combo = self.channel_color_table.cellWidget(row, 1)
            if item is None or combo is None or item.checkState() != Qt.Checked:
                continue
            color = combo.currentText()
            if color in by_color:
                by_color[color].append(item.text())
        return by_color

    def _current_scaling_channels(self) -> List[str]:
        if self.rgb_mode_chk.isChecked():
            return self._checked_rgb_channels()
        channel = self.channel_combo.currentText()
        return [channel] if channel else []

    def _refresh_scaling_channel_options(self, preferred_channel: Optional[str] = None):
        channels = self._current_scaling_channels()
        seen = set()
        ordered_channels = []
        for channel in channels:
            if channel not in seen:
                ordered_channels.append(channel)
                seen.add(channel)

        current_channel = preferred_channel or self.scaling_channel_combo.currentText() or self.previous_scaling_channel
        self.scaling_channel_combo.blockSignals(True)
        self.scaling_channel_combo.clear()
        self.scaling_channel_combo.addItems(ordered_channels)

        final_channel: Optional[str] = None
        if current_channel and current_channel in ordered_channels:
            final_channel = current_channel
        elif ordered_channels:
            final_channel = ordered_channels[0]

        if final_channel:
            self.scaling_channel_combo.setCurrentIndex(ordered_channels.index(final_channel))
            self.previous_scaling_channel = final_channel
        else:
            self.previous_scaling_channel = None
        self.scaling_channel_combo.blockSignals(False)

    def _update_image_combo(self, preferred_image: Optional[str] = None):
        self.image_combo.blockSignals(True)
        self.image_combo.clear()

        for acq_id in self.selected_acquisitions:
            self.image_combo.addItem(self._get_acquisition_subtitle(acq_id), acq_id)

        final_image: Optional[str] = None
        if preferred_image and preferred_image in self.selected_acquisitions:
            final_image = preferred_image
        elif self.selected_acquisitions:
            final_image = self.selected_acquisitions[0]

        if final_image:
            self.image_combo.setCurrentIndex(self.selected_acquisitions.index(final_image))
        self.image_combo.blockSignals(False)

    def _update_scaling_widget_states(self):
        custom_scaling = self.custom_scaling_chk.isChecked()
        linked = self.link_chk.isChecked()
        arcsinh_enabled = custom_scaling and not linked
        spinboxes_enabled = custom_scaling and (linked or not self.arcsinh_chk.isChecked())

        self.scaling_frame.setVisible(custom_scaling)
        self.scaling_channel_row.setVisible(custom_scaling)
        self.image_selection_row.setVisible(custom_scaling and not linked)
        self.image_combo.setEnabled(custom_scaling and not linked)
        self.default_range_btn.setEnabled(custom_scaling)
        self.min_spinbox.setEnabled(spinboxes_enabled)
        self.max_spinbox.setEnabled(spinboxes_enabled)
        self.arcsinh_chk.setEnabled(arcsinh_enabled)
        self.arcsinh_cofactor.setEnabled(arcsinh_enabled and self.arcsinh_chk.isChecked())

    def _clear_range_cache(self):
        self._range_cache.clear()

    def _clear_cache(self):
        self.image_stack_cache.clear()
        self._clear_range_cache()

    def _prune_stack_cache(self):
        allowed = set(self.selected_acquisitions)
        for acq_id in list(self.image_stack_cache.keys()):
            if acq_id not in allowed:
                del self.image_stack_cache[acq_id]
        self._clear_range_cache()

    def _get_cached_stack(self, acq_id: str) -> np.ndarray:
        if acq_id in self.image_stack_cache:
            return self.image_stack_cache[acq_id]

        loader = self._get_loader_for_acquisition(acq_id)
        original_acq_id = self._get_original_acq_id(acq_id)
        if loader is None:
            raise RuntimeError(f"No loader available for acquisition {acq_id}.")

        stack = loader.get_all_channels(original_acq_id)
        if stack.ndim == 2:
            stack = stack[..., np.newaxis]
        self.image_stack_cache[acq_id] = stack
        return stack

    def _get_channel_image(self, acq_id: str, channel: str) -> np.ndarray:
        stack = self._get_cached_stack(acq_id)
        acq_info = self._get_acquisition_info(acq_id)
        if acq_info is None or channel not in acq_info.channels:
            raise ValueError(f"Channel '{channel}' not found in acquisition {acq_id}.")
        channel_index = acq_info.channels.index(channel)
        return stack[..., channel_index]

    def _start_prefetch_selected(self):
        self._prune_stack_cache()
        for acq_id in self.selected_acquisitions[: self._prefetch_limit]:
            try:
                self._get_cached_stack(acq_id)
            except Exception as exc:
                print(f"Prefetch error for {acq_id}: {exc}")

    def _store_current_scaling_state(self):
        if not self.custom_scaling_chk.isChecked():
            return

        channel = self.previous_scaling_channel or self.scaling_channel_combo.currentText()
        if not channel:
            return

        current_min = self.min_spinbox.value()
        current_max = self.max_spinbox.value()

        if self.link_chk.isChecked():
            self.channel_linked_scaling[channel] = {"min": current_min, "max": current_max}
            return

        target_acq_id = self._loaded_scaling_image_id or self.image_combo.currentData()
        if not target_acq_id:
            return

        per_image = self.channel_per_image_scaling.setdefault(channel, {})
        per_image[target_acq_id] = {"min": current_min, "max": current_max}

    def _get_current_scaling_channel(self) -> Optional[str]:
        channel = self.scaling_channel_combo.currentText()
        if channel:
            return channel
        channel = self.channel_combo.currentText()
        return channel or None

    def _get_current_arcsinh_state(self, acq_id: str) -> Tuple[bool, float]:
        state = self.image_arcsinh_state.get(acq_id, {})
        enabled = bool(state.get("enabled", False))
        cofactor = float(state.get("cofactor", self.arcsinh_cofactor.value()))
        return enabled, cofactor

    def _should_apply_arcsinh(self, acq_id: str, channel: str) -> Tuple[bool, float]:
        if self.link_chk.isChecked():
            return False, self.arcsinh_cofactor.value()

        current_scaling_channel = self._get_current_scaling_channel()
        if channel != current_scaling_channel:
            return False, self.arcsinh_cofactor.value()

        enabled, cofactor = self._get_current_arcsinh_state(acq_id)
        return enabled, cofactor

    def _get_transformed_channel_image(self, acq_id: str, channel: str) -> np.ndarray:
        image = self._get_channel_image(acq_id, channel).astype(np.float32, copy=False)
        should_apply, cofactor = self._should_apply_arcsinh(acq_id, channel)
        if should_apply:
            return arcsinh_normalize(image, cofactor)
        return image

    def _compute_default_range(self, channel: str, acq_id: Optional[str] = None) -> Tuple[float, float]:
        if not channel:
            return 0.0, 1.0

        if acq_id is None:
            should_apply = False
            cofactor = 0.0
            selection_key = tuple(self.selected_acquisitions)
            cache_key = (channel, None, should_apply, cofactor, selection_key)
            if cache_key in self._range_cache:
                return self._range_cache[cache_key]

            mins: List[float] = []
            maxs: List[float] = []
            for selected_acq in self.selected_acquisitions:
                image = self._get_transformed_channel_image(selected_acq, channel)
                mins.append(float(np.min(image)))
                maxs.append(float(np.max(image)))
            if not mins or not maxs:
                return 0.0, 1.0
            value = (min(mins), max(maxs))
            self._range_cache[cache_key] = value
            return value

        should_apply, cofactor = self._should_apply_arcsinh(acq_id, channel)
        cache_key = (channel, acq_id, should_apply, float(cofactor if should_apply else 0.0), ())
        if cache_key in self._range_cache:
            return self._range_cache[cache_key]

        image = self._get_transformed_channel_image(acq_id, channel)
        value = (float(np.min(image)), float(np.max(image)))
        self._range_cache[cache_key] = value
        return value

    def _resolve_display_range(self, channel: str, acq_id: str) -> Tuple[float, float]:
        if not self.custom_scaling_chk.isChecked():
            if self.link_chk.isChecked():
                return self._compute_default_range(channel, None)
            return self._compute_default_range(channel, acq_id)

        current_scaling_channel = self._get_current_scaling_channel()
        if self.link_chk.isChecked():
            if current_scaling_channel == channel:
                return self.min_spinbox.value(), self.max_spinbox.value()

            values = self.channel_linked_scaling.get(channel)
            if values is None:
                vmin, vmax = self._compute_default_range(channel, None)
                self.channel_linked_scaling[channel] = {"min": vmin, "max": vmax}
                return vmin, vmax
            return float(values["min"]), float(values["max"])

        if current_scaling_channel == channel and self.image_combo.currentData() == acq_id:
            return self.min_spinbox.value(), self.max_spinbox.value()

        per_image = self.channel_per_image_scaling.setdefault(channel, {})
        if acq_id not in per_image:
            vmin, vmax = self._compute_default_range(channel, acq_id)
            per_image[acq_id] = {"min": vmin, "max": vmax}
        values = per_image[acq_id]
        return float(values["min"]), float(values["max"])

    def _set_spinbox_values(self, minimum: float, maximum: float):
        self.min_spinbox.blockSignals(True)
        self.max_spinbox.blockSignals(True)
        self.min_spinbox.setValue(float(minimum))
        self.max_spinbox.setValue(float(maximum))
        self.min_spinbox.blockSignals(False)
        self.max_spinbox.blockSignals(False)

    def _load_image_arcsinh_state(self):
        current_acq_id = self.image_combo.currentData()
        if not current_acq_id:
            return

        enabled, cofactor = self._get_current_arcsinh_state(current_acq_id)
        self.arcsinh_chk.blockSignals(True)
        self.arcsinh_cofactor.blockSignals(True)
        self.arcsinh_chk.setChecked(enabled)
        self.arcsinh_cofactor.setValue(cofactor)
        self.arcsinh_chk.blockSignals(False)
        self.arcsinh_cofactor.blockSignals(False)

    def _load_scaling_controls(self):
        self._update_scaling_widget_states()
        if not self.custom_scaling_chk.isChecked():
            return

        channel = self._get_current_scaling_channel()
        if not channel:
            return

        if self.link_chk.isChecked():
            values = self.channel_linked_scaling.get(channel)
            if values is None:
                vmin, vmax = self._compute_default_range(channel, None)
                self.channel_linked_scaling[channel] = {"min": vmin, "max": vmax}
            else:
                vmin = float(values["min"])
                vmax = float(values["max"])
            self._loaded_scaling_image_id = None
            self._set_spinbox_values(vmin, vmax)
            self.previous_scaling_channel = channel
            self._update_scaling_widget_states()
            return

        current_acq_id = self.image_combo.currentData()
        if not current_acq_id:
            return

        self._load_image_arcsinh_state()
        should_apply, _cofactor = self._should_apply_arcsinh(current_acq_id, channel)
        if should_apply:
            vmin, vmax = self._compute_default_range(channel, current_acq_id)
        else:
            per_image = self.channel_per_image_scaling.setdefault(channel, {})
            if current_acq_id not in per_image:
                vmin, vmax = self._compute_default_range(channel, current_acq_id)
                per_image[current_acq_id] = {"min": vmin, "max": vmax}
            values = per_image[current_acq_id]
            vmin = float(values["min"])
            vmax = float(values["max"])

        self._loaded_scaling_image_id = current_acq_id
        self.previous_scaling_channel = channel
        self._set_spinbox_values(vmin, vmax)
        self._update_scaling_widget_states()

    def _comparison_auto_range(self):
        self._comparison_default_range()

    def _comparison_auto_contrast(self):
        if not self.custom_scaling_chk.isChecked():
            return

        channel = self._get_current_scaling_channel()
        if not channel:
            return

        if self.link_chk.isChecked():
            pixels = [self._get_transformed_channel_image(acq_id, channel).ravel() for acq_id in self.selected_acquisitions]
            if not pixels:
                return
            all_pixels = np.concatenate(pixels)
            vmin = float(np.percentile(all_pixels, 1))
            vmax = float(np.percentile(all_pixels, 99))
            self.channel_linked_scaling[channel] = {"min": vmin, "max": vmax}
            self._set_spinbox_values(vmin, vmax)
        else:
            current_acq_id = self.image_combo.currentData()
            if not current_acq_id:
                return
            image = self._get_transformed_channel_image(current_acq_id, channel)
            vmin = float(np.percentile(image, 1))
            vmax = float(np.percentile(image, 99))
            self.channel_per_image_scaling.setdefault(channel, {})[current_acq_id] = {"min": vmin, "max": vmax}
            self._set_spinbox_values(vmin, vmax)

        self._request_display_update()

    def _comparison_default_range(self):
        if not self.custom_scaling_chk.isChecked():
            return

        channel = self._get_current_scaling_channel()
        if not channel:
            return

        if self.link_chk.isChecked():
            vmin, vmax = self._compute_default_range(channel, None)
            self.channel_linked_scaling[channel] = {"min": vmin, "max": vmax}
        else:
            current_acq_id = self.image_combo.currentData()
            if not current_acq_id:
                return
            vmin, vmax = self._compute_default_range(channel, current_acq_id)
            self.channel_per_image_scaling.setdefault(channel, {})[current_acq_id] = {"min": vmin, "max": vmax}

        self._set_spinbox_values(vmin, vmax)
        self._request_display_update()

    def _restore_default_range(self):
        self._comparison_default_range()

    def _on_comparison_scaling_toggled(self):
        with self._batch_display_updates():
            self._refresh_scaling_channel_options(self.scaling_channel_combo.currentText())
            self._update_image_combo(self.image_combo.currentData())
            self._load_scaling_controls()
        self._request_display_update()

    def _on_scaling_channel_changed(self):
        if self.custom_scaling_chk.isChecked():
            self._store_current_scaling_state()
        self.previous_scaling_channel = self.scaling_channel_combo.currentText() or None
        self._clear_range_cache()

        with self._batch_display_updates():
            self._load_scaling_controls()

        self._request_display_update()

    def _on_arcsinh_toggled(self):
        current_acq_id = self.image_combo.currentData()
        if current_acq_id:
            self.image_arcsinh_state[current_acq_id] = {
                "enabled": self.arcsinh_chk.isChecked(),
                "cofactor": self.arcsinh_cofactor.value(),
            }
        self.channel_per_image_scaling.clear()
        self._clear_range_cache()

        with self._batch_display_updates():
            self._load_scaling_controls()

        self._request_display_update()

    def _on_arcsinh_cofactor_changed(self):
        current_acq_id = self.image_combo.currentData()
        if current_acq_id:
            state = self.image_arcsinh_state.setdefault(
                current_acq_id,
                {"enabled": False, "cofactor": self.arcsinh_cofactor.value()},
            )
            state["cofactor"] = self.arcsinh_cofactor.value()
        self.channel_per_image_scaling.clear()
        self._clear_range_cache()

        with self._batch_display_updates():
            self._load_scaling_controls()

        self._request_display_update()

    def _apply_comparison_scaling(self):
        if not self.custom_scaling_chk.isChecked():
            return

        channel = self._get_current_scaling_channel()
        if not channel:
            return

        if self.link_chk.isChecked():
            self.channel_linked_scaling[channel] = {
                "min": self.min_spinbox.value(),
                "max": self.max_spinbox.value(),
            }
        else:
            current_acq_id = self.image_combo.currentData()
            if current_acq_id:
                self.channel_per_image_scaling.setdefault(channel, {})[current_acq_id] = {
                    "min": self.min_spinbox.value(),
                    "max": self.max_spinbox.value(),
                }
        self._request_display_update()

    def _on_link_contrast_toggled(self):
        if self.custom_scaling_chk.isChecked():
            self._store_current_scaling_state()
        self._clear_range_cache()

        with self._batch_display_updates():
            self._refresh_scaling_channel_options(self.scaling_channel_combo.currentText())
            self._update_image_combo(self.image_combo.currentData())
            self._load_scaling_controls()

        self._request_display_update()

    def _on_image_selection_changed(self):
        if not self.custom_scaling_chk.isChecked() or self.link_chk.isChecked():
            return

        self._store_current_scaling_state()
        self._clear_range_cache()

        with self._batch_display_updates():
            self._load_scaling_controls()

        self._request_display_update()

    def _add_acquisition(self):
        current_item = self.available_acq_list.currentItem()
        if current_item is None:
            return

        acq_id = current_item.data(Qt.UserRole)
        if acq_id in self.selected_acquisitions:
            return

        self.selected_acquisitions.append(acq_id)
        new_item = QtWidgets.QListWidgetItem(current_item.text())
        new_item.setData(Qt.UserRole, acq_id)
        self.acq_list.addItem(new_item)

        self._start_prefetch_selected()
        self._sync_channel_controls()

    def _remove_acquisition(self):
        current_item = self.acq_list.currentItem()
        if current_item is None:
            return

        acq_id = current_item.data(Qt.UserRole)
        if acq_id not in self.selected_acquisitions:
            return

        self.selected_acquisitions.remove(acq_id)
        self.acq_list.takeItem(self.acq_list.row(current_item))

        self._prune_stack_cache()
        self._sync_channel_controls()

    def _normalize_to_unit(self, image: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        if vmax <= vmin:
            return np.zeros_like(image, dtype=np.float32)
        return np.clip((image.astype(np.float32, copy=False) - vmin) / (vmax - vmin), 0.0, 1.0)

    def _build_rgb_image(self, acq_id: str, channels_by_color: Dict[str, List[str]]) -> Optional[np.ndarray]:
        ordered_channels = []
        for color in self.available_colors_rgb:
            ordered_channels.extend(channels_by_color[color])
        if not ordered_channels:
            return None

        base_image = self._get_transformed_channel_image(acq_id, ordered_channels[0])
        rgb_image = np.zeros(base_image.shape + (3,), dtype=np.float32)

        for color_index, color in enumerate(self.available_colors_rgb):
            plane = np.zeros(base_image.shape, dtype=np.float32)
            for channel in channels_by_color[color]:
                image = self._get_transformed_channel_image(acq_id, channel)
                vmin, vmax = self._resolve_display_range(channel, acq_id)
                plane += self._normalize_to_unit(image, vmin, vmax)
            rgb_image[..., color_index] = np.clip(plane, 0.0, 1.0)

        return rgb_image

    def _get_overlay_mask(self, acq_id: str, expected_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        if not self.overlay_chk.isChecked():
            return None

        parent = self.parent()
        masks = getattr(parent, "segmentation_masks", {}) if parent else {}
        if acq_id not in masks:
            return None

        try:
            mask = masks[acq_id]
            mask_bool = mask.astype(bool)
            if mask_bool.ndim == 2 and mask_bool.shape == expected_shape:
                return mask_bool
        except Exception:
            return None
        return None

    def _build_display_specs(self) -> List[Dict[str, object]]:
        specs: List[Dict[str, object]] = []
        grayscale = self.grayscale_chk.isChecked()
        is_rgb = self.rgb_mode_chk.isChecked()

        if is_rgb:
            channels_by_color = self._checked_rgb_channels_by_color()
            if not any(channels_by_color[color] for color in self.available_colors_rgb):
                return specs

            for acq_id in self.selected_acquisitions:
                rgb_image = self._build_rgb_image(acq_id, channels_by_color)
                if rgb_image is None:
                    continue

                if grayscale:
                    image = np.mean(rgb_image, axis=2)
                    specs.append(
                        {
                            "mode": "rgb_gray",
                            "image": image,
                            "cmap": "gray",
                            "vmin": 0.0,
                            "vmax": 1.0,
                            "show_colorbar": True,
                            "title": self._get_acquisition_subtitle(acq_id),
                            "overlay_mask": self._get_overlay_mask(acq_id, image.shape),
                        }
                    )
                else:
                    specs.append(
                        {
                            "mode": "rgb",
                            "image": rgb_image,
                            "cmap": None,
                            "vmin": None,
                            "vmax": None,
                            "show_colorbar": False,
                            "title": self._get_acquisition_subtitle(acq_id),
                            "overlay_mask": self._get_overlay_mask(acq_id, rgb_image.shape[:2]),
                        }
                    )
            return specs

        channel = self.channel_combo.currentText()
        if not channel:
            return specs

        for acq_id in self.selected_acquisitions:
            image = self._get_transformed_channel_image(acq_id, channel)
            vmin, vmax = self._resolve_display_range(channel, acq_id)
            specs.append(
                {
                    "mode": "single",
                    "image": image,
                    "cmap": "gray" if grayscale else "viridis",
                    "vmin": vmin,
                    "vmax": vmax,
                    "show_colorbar": True,
                    "title": self._get_acquisition_subtitle(acq_id),
                    "overlay_mask": self._get_overlay_mask(acq_id, image.shape),
                }
            )

        return specs

    def _create_canvas_slot(self) -> Dict[str, object]:
        canvas = MplCanvas(width=4, height=4, dpi=100)
        canvas.setParent(self.image_widget)
        return {
            "canvas": canvas,
            "image_artist": None,
            "colorbar": None,
            "overlay_artists": [],
            "mode": None,
        }

    def _remove_overlay_artists(self, slot: Dict[str, object]):
        for artist in slot.get("overlay_artists", []):
            try:
                artist.remove()
            except Exception:
                pass
        slot["overlay_artists"] = []

    def _remove_colorbar(self, slot: Dict[str, object]):
        colorbar = slot.get("colorbar")
        if colorbar is not None:
            try:
                colorbar.remove()
            except Exception:
                pass
        slot["colorbar"] = None

    def _release_canvas_slot(self, slot: Dict[str, object]):
        self._remove_overlay_artists(slot)
        self._remove_colorbar(slot)
        canvas = slot.get("canvas")
        if canvas is None:
            return
        self.image_layout.removeWidget(canvas)
        if hasattr(canvas, "fig"):
            try:
                canvas.fig.clear()
                plt.close(canvas.fig)
            except Exception:
                pass
        slot["image_artist"] = None
        slot["mode"] = None
        canvas.deleteLater()

    def _ensure_canvas_slots(self, count: int):
        while len(self._canvas_slots) < count:
            self._canvas_slots.append(self._create_canvas_slot())

        while len(self._canvas_slots) > count:
            slot = self._canvas_slots.pop()
            self._release_canvas_slot(slot)

        cols = max(1, min(3, count))
        for index, slot in enumerate(self._canvas_slots):
            row = index // cols
            col = index % cols
            self.image_layout.addWidget(slot["canvas"], row, col)
        self._update_canvas_geometry()

    def _update_canvas_geometry(self):
        count = len(self._canvas_slots)
        if count == 0:
            return

        cols = max(1, min(3, count))
        rows = (count + cols - 1) // cols

        margins = self.image_layout.contentsMargins()
        h_spacing = self.image_layout.horizontalSpacing()
        v_spacing = self.image_layout.verticalSpacing()
        if h_spacing < 0:
            h_spacing = self.image_layout.spacing()
        if v_spacing < 0:
            v_spacing = self.image_layout.spacing()

        viewport_width = max(1, self.image_scroll.viewport().width())
        viewport_height = max(1, self.image_scroll.viewport().height())
        available_width = max(120, viewport_width - margins.left() - margins.right() - (h_spacing * (cols - 1)))
        available_height = max(120, viewport_height - margins.top() - margins.bottom() - (v_spacing * (rows - 1)))

        tile_width = max(120, available_width // cols)
        tile_height = max(tile_width, min(available_height // rows, int(tile_width * 1.4)))

        for column in range(3):
            self.image_layout.setColumnStretch(column, 1 if column < cols else 0)
        for row in range(max(1, rows)):
            self.image_layout.setRowStretch(row, 1)

        for slot in self._canvas_slots:
            canvas = slot["canvas"]
            canvas.setFixedSize(tile_width, tile_height)

    def _update_canvas_slot(self, slot: Dict[str, object], spec: Dict[str, object]):
        canvas: MplCanvas = slot["canvas"]  # type: ignore[assignment]
        ax = canvas.ax
        mode = spec["mode"]
        image_artist = slot.get("image_artist")
        mode_changed = slot.get("mode") != mode or image_artist is None

        self._remove_overlay_artists(slot)

        if mode_changed:
            self._remove_colorbar(slot)
            ax.clear()
            if mode == "rgb":
                image_artist = ax.imshow(spec["image"], interpolation="nearest")
            else:
                image_artist = ax.imshow(
                    spec["image"],
                    interpolation="nearest",
                    cmap=spec["cmap"],
                    vmin=spec["vmin"],
                    vmax=spec["vmax"],
                )
            slot["image_artist"] = image_artist
            slot["mode"] = mode
        else:
            image_artist = slot["image_artist"]
            image_artist.set_data(spec["image"])
            if mode != "rgb":
                image_artist.set_cmap(spec["cmap"])
                image_artist.set_clim(spec["vmin"], spec["vmax"])

        overlay_mask = spec.get("overlay_mask")
        if overlay_mask is not None:
            try:
                contour = ax.contour(overlay_mask, levels=[0.5], colors="r", linewidths=0.6, alpha=0.7)
                slot["overlay_artists"] = list(contour.collections)
            except Exception:
                slot["overlay_artists"] = []

        ax.set_title(spec["title"], fontsize=10)
        ax.axis("off")

        if spec["show_colorbar"]:
            if slot.get("colorbar") is None or mode_changed:
                self._remove_colorbar(slot)
                slot["colorbar"] = canvas.fig.colorbar(image_artist, ax=ax, shrink=0.8, aspect=20)
            else:
                slot["colorbar"].update_normal(image_artist)
        else:
            self._remove_colorbar(slot)

        canvas.draw()

    def _update_display(self):
        if not self.selected_acquisitions:
            self._ensure_canvas_slots(0)
            return

        try:
            specs = self._build_display_specs()
        except Exception as exc:
            print(f"Comparison display error: {exc}")
            specs = []

        if not specs:
            self._ensure_canvas_slots(0)
            return

        self._ensure_canvas_slots(len(specs))
        for slot, spec in zip(self._canvas_slots, specs):
            self._update_canvas_slot(slot, spec)
        self.image_widget.update()

    def _clear_display(self):
        while self._canvas_slots:
            slot = self._canvas_slots.pop()
            self._release_canvas_slot(slot)
        while self.image_layout.count():
            child = self.image_layout.takeAt(0)
            if child.widget():
                child.widget().setParent(None)
        self.image_widget.update()

    def _refresh_markers(self):
        self._sync_channel_controls()
