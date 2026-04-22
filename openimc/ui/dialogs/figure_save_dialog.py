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
Figure Save Options Dialog

This dialog allows users to configure save options for matplotlib figures:
- DPI (dots per inch)
- Format (PNG, JPG, PDF)
- Font size
- Image size (width and height in inches)
"""

from typing import List, Optional, Tuple
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt


def _draw_figure_if_possible(figure) -> None:
    """Realize pending artists so text sizing can be captured and restored reliably."""
    canvas = getattr(figure, "canvas", None)
    if canvas is None or not hasattr(canvas, "draw"):
        return
    try:
        canvas.draw()
    except Exception:
        pass


def _collect_figure_text_states(figure) -> List[Tuple[object, float]]:
    """Return all current matplotlib Text objects with their original font sizes."""
    import matplotlib.text as mtext

    states = []
    seen = set()
    for text in figure.findobj(mtext.Text):
        if not hasattr(text, "get_fontsize") or not hasattr(text, "set_fontsize"):
            continue
        text_id = id(text)
        if text_id in seen:
            continue
        seen.add(text_id)
        states.append((text, text.get_fontsize()))
    return states


def _apply_fontsize_override(figure, fontsize: float) -> List[Tuple[object, float]]:
    """Apply a temporary font-size override to all visible figure text."""
    _draw_figure_if_possible(figure)
    text_states = _collect_figure_text_states(figure)
    for text, _original_size in text_states:
        text.set_fontsize(fontsize)
    return text_states


def _restore_figure_text_states(text_states: List[Tuple[object, float]]) -> None:
    """Restore text objects to their original font sizes."""
    for text, original_size in text_states:
        if hasattr(text, "set_fontsize"):
            text.set_fontsize(original_size)


class FigureSaveDialog(QtWidgets.QDialog):
    """Dialog for configuring figure save options."""
    
    def __init__(self, default_filename: str = "figure.png", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save Figure Options")
        self.setModal(True)
        self.resize(400, 300)
        
        # Default values
        self.default_dpi = 300
        self.default_format = "PNG"
        self.default_font_size = None  # Will be determined from figure
        self.default_width = None  # Will be determined from figure
        self.default_height = None  # Will be determined from figure
        
        self._setup_ui(default_filename)
        
    def _setup_ui(self, default_filename: str):
        """Set up the user interface."""
        layout = QtWidgets.QVBoxLayout()
        
        # Filename input
        filename_group = QtWidgets.QGroupBox("Filename")
        filename_layout = QtWidgets.QVBoxLayout()
        filename_row = QtWidgets.QHBoxLayout()
        self.filename_edit = QtWidgets.QLineEdit(default_filename)
        filename_row.addWidget(self.filename_edit)
        browse_button = QtWidgets.QPushButton("Browse...")
        browse_button.clicked.connect(self._browse_filename)
        filename_row.addWidget(browse_button)
        filename_layout.addLayout(filename_row)
        filename_group.setLayout(filename_layout)
        layout.addWidget(filename_group)
        
        # Format selection
        format_group = QtWidgets.QGroupBox("Format")
        format_layout = QtWidgets.QVBoxLayout()
        self.format_combo = QtWidgets.QComboBox()
        self.format_combo.addItems(["PNG", "JPG", "PDF"])
        self.format_combo.setCurrentText(self.default_format)
        self.format_combo.currentTextChanged.connect(self._on_format_changed)
        format_layout.addWidget(self.format_combo)
        format_group.setLayout(format_layout)
        layout.addWidget(format_group)
        
        # DPI input
        dpi_group = QtWidgets.QGroupBox("DPI (Dots Per Inch)")
        dpi_layout = QtWidgets.QVBoxLayout()
        self.dpi_spinbox = QtWidgets.QSpinBox()
        self.dpi_spinbox.setMinimum(72)
        self.dpi_spinbox.setMaximum(1200)
        self.dpi_spinbox.setValue(self.default_dpi)
        self.dpi_spinbox.setSuffix(" DPI")
        dpi_layout.addWidget(self.dpi_spinbox)
        dpi_group.setLayout(dpi_layout)
        layout.addWidget(dpi_group)
        
        # Font size input
        font_group = QtWidgets.QGroupBox("Font Size (points)")
        font_layout = QtWidgets.QVBoxLayout()
        self.font_size_spinbox = QtWidgets.QDoubleSpinBox()
        self.font_size_spinbox.setMinimum(6.0)
        self.font_size_spinbox.setMaximum(72.0)
        self.font_size_spinbox.setValue(10.0)
        self.font_size_spinbox.setSuffix(" pt")
        self.font_size_spinbox.setDecimals(1)
        self.font_size_checkbox = QtWidgets.QCheckBox("Override figure font size")
        self.font_size_checkbox.setChecked(False)
        font_layout.addWidget(self.font_size_checkbox)
        font_layout.addWidget(self.font_size_spinbox)
        self.font_size_spinbox.setEnabled(False)
        self.font_size_checkbox.toggled.connect(self.font_size_spinbox.setEnabled)
        font_group.setLayout(font_layout)
        layout.addWidget(font_group)
        
        # Image size inputs
        size_group = QtWidgets.QGroupBox("Image Size (inches)")
        size_layout = QtWidgets.QGridLayout()
        
        size_layout.addWidget(QtWidgets.QLabel("Width:"), 0, 0)
        self.width_spinbox = QtWidgets.QDoubleSpinBox()
        self.width_spinbox.setMinimum(1.0)
        self.width_spinbox.setMaximum(100.0)
        self.width_spinbox.setValue(8.0)
        self.width_spinbox.setSuffix(" in")
        self.width_spinbox.setDecimals(2)
        size_layout.addWidget(self.width_spinbox, 0, 1)
        
        size_layout.addWidget(QtWidgets.QLabel("Height:"), 1, 0)
        self.height_spinbox = QtWidgets.QDoubleSpinBox()
        self.height_spinbox.setMinimum(1.0)
        self.height_spinbox.setMaximum(100.0)
        self.height_spinbox.setValue(6.0)
        self.height_spinbox.setSuffix(" in")
        self.height_spinbox.setDecimals(2)
        size_layout.addWidget(self.height_spinbox, 1, 1)
        
        self.size_checkbox = QtWidgets.QCheckBox("Override figure size")
        self.size_checkbox.setChecked(False)
        size_layout.addWidget(self.size_checkbox, 2, 0, 1, 2)
        self.width_spinbox.setEnabled(False)
        self.height_spinbox.setEnabled(False)
        self.size_checkbox.toggled.connect(self._on_size_checkbox_toggled)
        
        size_group.setLayout(size_layout)
        layout.addWidget(size_group)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.save_button = QtWidgets.QPushButton("Save")
        self.save_button.clicked.connect(self.accept)
        self.save_button.setDefault(True)
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.save_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
    def _browse_filename(self):
        """Open file dialog to choose save location."""
        from PyQt5.QtWidgets import QFileDialog
        current_format = self.format_combo.currentText().upper()
        filter_map = {
            "PNG": "PNG files (*.png)",
            "JPG": "JPEG files (*.jpg *.jpeg)",
            "PDF": "PDF files (*.pdf)"
        }
        file_filter = filter_map.get(current_format, "All files (*.*)")
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Figure",
            self.filename_edit.text(),
            file_filter
        )
        if filename:
            self.filename_edit.setText(filename)
    
    def _on_format_changed(self, format_text: str):
        """Update filename extension when format changes."""
        filename = self.filename_edit.text()
        if filename:
            # Remove existing extension
            base = filename.rsplit('.', 1)[0] if '.' in filename else filename
            # Add new extension
            ext_map = {"PNG": ".png", "JPG": ".jpg", "PDF": ".pdf"}
            new_filename = base + ext_map.get(format_text, ".png")
            self.filename_edit.setText(new_filename)
    
    def _on_size_checkbox_toggled(self, checked: bool):
        """Enable/disable size spinboxes."""
        self.width_spinbox.setEnabled(checked)
        self.height_spinbox.setEnabled(checked)
    
    def set_figure_properties(self, figure):
        """Set default values from the figure."""
        if figure is None:
            return
            
        # Get current figure size
        fig_size = figure.get_size_inches()
        self.default_width = fig_size[0]
        self.default_height = fig_size[1]
        self.width_spinbox.setValue(self.default_width)
        self.height_spinbox.setValue(self.default_height)
        
        # Try to get font size from figure (average of all text elements)
        try:
            import matplotlib.text as mtext
            texts = figure.findobj(mtext.Text)
            if texts:
                font_sizes = [t.get_fontsize() for t in texts if hasattr(t, 'get_fontsize')]
                if font_sizes:
                    avg_font_size = sum(font_sizes) / len(font_sizes)
                    self.default_font_size = avg_font_size
                    self.font_size_spinbox.setValue(avg_font_size)
        except:
            pass
    
    def get_save_options(self) -> Tuple[Optional[str], dict]:
        """
        Get the save options.
        
        Returns:
            Tuple of (filename, save_kwargs) where save_kwargs contains:
            - dpi: int
            - format: str (lowercase)
            - fontsize: Optional[float] (if override is checked)
            - figsize: Optional[Tuple[float, float]] (if override is checked)
        """
        filename = self.filename_edit.text().strip()
        if not filename:
            return None, {}
        
        format_text = self.format_combo.currentText().upper()
        dpi = self.dpi_spinbox.value()
        
        save_kwargs = {
            'dpi': dpi,
            'format': format_text.lower()
        }
        
        # Add font size if override is checked
        if self.font_size_checkbox.isChecked():
            save_kwargs['fontsize'] = self.font_size_spinbox.value()
        
        # Add figure size if override is checked
        if self.size_checkbox.isChecked():
            save_kwargs['figsize'] = (self.width_spinbox.value(), self.height_spinbox.value())
        
        return filename, save_kwargs


def save_figure_with_options(figure, default_filename: str = "figure.png", parent=None) -> bool:
    """
    Show save dialog and save figure with user-selected options.
    
    Args:
        figure: matplotlib Figure object
        default_filename: Default filename to suggest
        parent: Parent widget for the dialog
        
    Returns:
        True if figure was saved, False if cancelled
    """
    dialog = FigureSaveDialog(default_filename, parent)
    dialog.set_figure_properties(figure)
    
    if dialog.exec_() != QtWidgets.QDialog.Accepted:
        return False
    
    filename, save_kwargs = dialog.get_save_options()
    if not filename:
        return False
    
    # Ensure filename extension matches the selected format
    format_text = save_kwargs.get('format', 'png').lower()
    ext_map = {'png': '.png', 'jpg': '.jpg', 'jpeg': '.jpg', 'pdf': '.pdf'}
    expected_ext = ext_map.get(format_text, '.png')
    
    # Update filename extension if it doesn't match
    if not filename.lower().endswith(expected_ext):
        base = filename.rsplit('.', 1)[0] if '.' in filename else filename
        filename = base + expected_ext
    
    original_size = None
    original_text_states: List[Tuple[object, float]] = []
    try:
        if 'figsize' in save_kwargs:
            original_size = tuple(figure.get_size_inches())
            figure.set_size_inches(save_kwargs.pop('figsize'))

        if 'fontsize' in save_kwargs:
            fontsize = save_kwargs.pop('fontsize')
            original_text_states = _apply_fontsize_override(figure, fontsize)

        figure.savefig(filename, bbox_inches='tight', **save_kwargs)
        return True

    except Exception as e:
        QtWidgets.QMessageBox.critical(
            parent, 
            "Save Error", 
            f"Failed to save figure: {str(e)}"
        )
        return False
    finally:
        if original_text_states:
            _restore_figure_text_states(original_text_states)
        if original_size is not None:
            figure.set_size_inches(original_size)
        canvas = getattr(figure, "canvas", None)
        if canvas is not None and hasattr(canvas, "draw_idle"):
            try:
                canvas.draw_idle()
            except Exception:
                pass
