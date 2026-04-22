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

import pytest

from tests.unit.test_cluster_explorer_dialog import _build_cluster_explorer


@pytest.mark.unit
def test_cluster_explorer_tiles_expose_source_and_cell_metadata_in_tooltips(qtbot):
    dialog, _parent, _loaders = _build_cluster_explorer(qtbot)

    dialog._settings.sample_count = 2
    dialog._load_cell_images()
    qtbot.wait(80)

    assert dialog.current_preview_records
    first_record = dialog.current_preview_records[0]
    first_canvas = dialog._tile_canvases[0]
    tooltip = first_canvas.toolTip()

    assert f"source_file: {first_record['source_file']}" in tooltip
    assert f"acquisition_id: {first_record['original_acq_id']}" in tooltip
    assert f"cell_id: {first_record['cell_id']}" in tooltip
