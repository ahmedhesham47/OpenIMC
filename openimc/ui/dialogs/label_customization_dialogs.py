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

"""Reusable label-editing dialogs for cluster and feature display names."""

from typing import Dict, Iterable, Optional, Sequence

from PyQt5 import QtWidgets

from openimc.ui.cluster_utils import (
    format_default_cluster_label,
    normalize_cluster_annotation_map,
    sort_cluster_values,
)


def edit_cluster_annotation_map(
    parent,
    cluster_ids: Sequence,
    current_map: Optional[Dict] = None,
    *,
    title: str = "Customize Cluster Names",
) -> Optional[Dict]:
    """Open a dialog for editing cluster display names."""
    current_map = normalize_cluster_annotation_map(current_map)
    cluster_ids = sort_cluster_values(cluster_ids, annotation_map=current_map, canonical=True)

    dlg = QtWidgets.QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.resize(480, 560)

    layout = QtWidgets.QVBoxLayout(dlg)
    instruction = QtWidgets.QLabel(
        "Set custom display names for clusters. Leave a field blank to use the default cluster label."
    )
    instruction.setWordWrap(True)
    layout.addWidget(instruction)

    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    content = QtWidgets.QWidget()
    form = QtWidgets.QFormLayout(content)

    editors = {}
    for cluster_id in cluster_ids:
        editor = QtWidgets.QLineEdit()
        if cluster_id in current_map:
            editor.setText(str(current_map[cluster_id]))
        form.addRow(format_default_cluster_label(cluster_id), editor)
        editors[cluster_id] = editor

    scroll.setWidget(content)
    layout.addWidget(scroll)

    buttons = QtWidgets.QHBoxLayout()
    apply_button = QtWidgets.QPushButton("Apply")
    cancel_button = QtWidgets.QPushButton("Cancel")
    apply_button.clicked.connect(dlg.accept)
    cancel_button.clicked.connect(dlg.reject)
    buttons.addStretch()
    buttons.addWidget(apply_button)
    buttons.addWidget(cancel_button)
    layout.addLayout(buttons)

    if dlg.exec_() != QtWidgets.QDialog.Accepted:
        return None

    return {
        cluster_id: editors[cluster_id].text().strip()
        for cluster_id in cluster_ids
        if editors[cluster_id].text().strip()
    }


def edit_feature_label_map(
    parent,
    feature_names: Iterable[str],
    current_map: Optional[Dict[str, str]] = None,
    *,
    title: str = "Customize Feature Labels",
) -> Optional[Dict[str, str]]:
    """Open a dialog for editing feature display names."""
    current_map = dict(current_map or {})
    feature_names = sorted(str(feature_name) for feature_name in feature_names)

    dlg = QtWidgets.QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.resize(720, 620)

    layout = QtWidgets.QVBoxLayout(dlg)
    instruction = QtWidgets.QLabel(
        "Set custom display names for features. Leave a field unchanged or blank to use the original feature name."
    )
    instruction.setWordWrap(True)
    layout.addWidget(instruction)

    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    content = QtWidgets.QWidget()
    form = QtWidgets.QFormLayout(content)

    editors = {}
    for feature_name in feature_names:
        editor = QtWidgets.QLineEdit()
        editor.setText(current_map.get(feature_name, feature_name))
        form.addRow(feature_name, editor)
        editors[feature_name] = editor

    scroll.setWidget(content)
    layout.addWidget(scroll)

    buttons = QtWidgets.QHBoxLayout()
    apply_button = QtWidgets.QPushButton("Apply")
    cancel_button = QtWidgets.QPushButton("Cancel")
    apply_button.clicked.connect(dlg.accept)
    cancel_button.clicked.connect(dlg.reject)
    buttons.addStretch()
    buttons.addWidget(apply_button)
    buttons.addWidget(cancel_button)
    layout.addLayout(buttons)

    if dlg.exec_() != QtWidgets.QDialog.Accepted:
        return None

    updated_map = dict(current_map)
    for feature_name in feature_names:
        label_text = editors[feature_name].text().strip()
        if label_text and label_text != feature_name:
            updated_map[feature_name] = label_text
        else:
            updated_map.pop(feature_name, None)
    return updated_map
