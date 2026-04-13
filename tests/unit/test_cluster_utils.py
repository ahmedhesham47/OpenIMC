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

import numpy as np
import pandas as pd

from openimc.ui.cluster_utils import (
    build_cluster_annotation_map,
    extract_cluster_annotation_map_from_dataframe,
    get_cluster_display_name,
    normalize_cluster_annotation_map,
    sort_cluster_values,
)


def test_sort_cluster_values_orders_numeric_ids_before_special_labels():
    annotation_map = normalize_cluster_annotation_map(
        {
            "1": "Alpha",
            np.int64(2): "Beta",
            "10": "Gamma",
        }
    )

    values = ["10", "2", "Unassigned", "1", "Cluster 11", np.int64(3)]

    assert sort_cluster_values(values, annotation_map=annotation_map, canonical=True) == [
        1,
        2,
        3,
        10,
        11,
        "Unassigned",
    ]
    assert get_cluster_display_name("1", annotation_map=annotation_map) == "Alpha"
    assert get_cluster_display_name("Cluster 10", annotation_map=annotation_map) == "Gamma"
def test_sort_cluster_values_remains_stable_for_overlay_style_mixed_types():
    annotation_map = normalize_cluster_annotation_map({1: "Alpha", 2: "Beta", 10: "Gamma"})

    ordered = sort_cluster_values(
        [np.int64(10), "2", 1, "Unassigned"],
        annotation_map=annotation_map,
        canonical=True,
    )

    assert ordered == [1, 2, 10, "Unassigned"]


def test_extract_cluster_annotation_map_from_dataframe_uses_persisted_cluster_phenotypes():
    df = pd.DataFrame(
        {
            "cluster": ["1", "1", 2, 2],
            "cluster_phenotype": ["Edited T cells", "", "Edited B cells", "Edited B cells"],
        }
    )

    assert extract_cluster_annotation_map_from_dataframe(df) == {
        1: "Edited T cells",
        2: "Edited B cells",
    }


def test_build_cluster_annotation_map_merges_loaded_dataframe_annotations_without_overriding_existing_names():
    df = pd.DataFrame(
        {
            "cluster": [1, 1, 2, 2],
            "cluster_phenotype": ["Loaded T cells", "Loaded T cells", "Loaded B cells", "Loaded B cells"],
        }
    )

    annotation_map = build_cluster_annotation_map(
        {1: "Saved T cells"},
        df,
    )

    assert annotation_map == {
        1: "Saved T cells",
        2: "Loaded B cells",
    }
    assert get_cluster_display_name(1, annotation_map=annotation_map) == "Saved T cells"
    assert get_cluster_display_name(2, annotation_map=annotation_map) == "Loaded B cells"
