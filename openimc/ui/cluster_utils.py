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

"""Shared helpers for cluster normalization, ordering, and display names."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional
import re

import numpy as np
import pandas as pd


_CLUSTER_ID_RE = re.compile(r"^[+-]?\d+$")
_CLUSTER_FLOAT_RE = re.compile(r"^[+-]?(?:\d+\.\d*|\d*\.\d+)$")
_DEFAULT_CLUSTER_LABEL_RE = re.compile(
    r"^cluster\s+([+-]?(?:\d+(?:\.\d*)?|\d*\.\d+))$",
    re.IGNORECASE,
)
_SPECIAL_CLUSTER_ORDER = {
    "unassigned": 0,
    "unknown cluster": 1,
    "unknown": 2,
    "__missing__": 3,
    "missing": 4,
}
_CLUSTER_COLUMNS = {"cluster", "cluster_id", "cluster_phenotype"}


def coerce_python_scalar(value: Any) -> Any:
    """Convert numpy/pandas scalars to plain Python scalars when possible."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _normalize_text(text: Any) -> str:
    return " ".join(str(text).replace("_", " ").split()).strip()


def _normalized_text_key(text: Any) -> str:
    return _normalize_text(text).casefold()


def _parse_numeric_string(text: str) -> Optional[Any]:
    stripped = text.strip()
    if not stripped:
        return None
    if _CLUSTER_ID_RE.fullmatch(stripped):
        try:
            return int(stripped)
        except ValueError:
            return None
    if _CLUSTER_FLOAT_RE.fullmatch(stripped):
        try:
            value = float(stripped)
        except ValueError:
            return None
        if value.is_integer():
            return int(value)
        return value
    match = _DEFAULT_CLUSTER_LABEL_RE.fullmatch(stripped)
    if match:
        return _parse_numeric_string(match.group(1))
    return None


def _canonicalize_cluster_id_base(value: Any) -> Any:
    value = coerce_python_scalar(value)
    if _is_missing(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        float_value = float(value)
        if np.isfinite(float_value) and float_value.is_integer():
            return int(float_value)
        return float_value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        parsed_numeric = _parse_numeric_string(stripped)
        if parsed_numeric is not None:
            return parsed_numeric
        return _normalize_text(stripped)
    return value


def normalize_cluster_annotation_map(annotation_map: Optional[Dict[Any, Any]]) -> Dict[Any, str]:
    """Normalize annotation-map keys so equivalent cluster ids share one entry."""
    normalized: Dict[Any, str] = {}
    if not annotation_map:
        return normalized
    for raw_key, raw_value in annotation_map.items():
        if raw_value is None:
            continue
        label = _normalize_text(raw_value)
        if not label:
            continue
        canonical_key = _canonicalize_cluster_id_base(raw_key)
        if canonical_key is None:
            continue
        normalized[canonical_key] = label
    return normalized


def canonicalize_cluster_id(value: Any, annotation_map: Optional[Dict[Any, Any]] = None) -> Any:
    """Normalize a cluster value while preserving non-numeric labels when needed."""
    canonical_value = _canonicalize_cluster_id_base(value)
    if isinstance(canonical_value, str) and annotation_map:
        reverse_map = {
            _normalized_text_key(display_name): canonical_key
            for canonical_key, display_name in normalize_cluster_annotation_map(annotation_map).items()
        }
        mapped_value = reverse_map.get(_normalized_text_key(canonical_value))
        if mapped_value is not None:
            return mapped_value
    return canonical_value


def extract_cluster_annotation_map_from_dataframe(dataframe: Optional[pd.DataFrame]) -> Dict[Any, str]:
    """Recover persisted cluster display names from a dataframe when available."""
    if dataframe is None or dataframe.empty or "cluster_phenotype" not in dataframe.columns:
        return {}

    cluster_col = None
    for candidate in ("cluster", "cluster_id"):
        if candidate in dataframe.columns:
            cluster_col = candidate
            break
    if cluster_col is None:
        return {}

    phenotype_map: Dict[Any, str] = {}
    subset = dataframe.loc[:, [cluster_col, "cluster_phenotype"]]
    for cluster_value, phenotype_value in subset.itertuples(index=False, name=None):
        phenotype_name = _normalize_text(phenotype_value)
        if not phenotype_name:
            continue
        canonical_cluster = canonicalize_cluster_id(cluster_value)
        if canonical_cluster is None or canonical_cluster in phenotype_map:
            continue
        phenotype_map[canonical_cluster] = phenotype_name

    return normalize_cluster_annotation_map(phenotype_map)


def build_cluster_annotation_map(
    annotation_map: Optional[Dict[Any, Any]] = None,
    *dataframes: Optional[pd.DataFrame],
) -> Dict[Any, str]:
    """Merge a base annotation map with persisted cluster phenotypes from dataframes."""
    merged_map = normalize_cluster_annotation_map(annotation_map)
    loaded_annotations: Dict[Any, str] = {}

    for dataframe in dataframes:
        loaded_annotations.update(extract_cluster_annotation_map_from_dataframe(dataframe))

    for cluster_id, display_name in normalize_cluster_annotation_map(loaded_annotations).items():
        merged_map.setdefault(cluster_id, display_name)

    return merged_map


def format_default_cluster_label(cluster_id: Any) -> str:
    """Return the default label for a cluster id."""
    canonical_id = canonicalize_cluster_id(cluster_id)
    if canonical_id is None:
        return "Unknown Cluster"
    if isinstance(canonical_id, str):
        normalized = _normalized_text_key(canonical_id)
        if normalized == "__missing__":
            return "Unassigned"
        if normalized in _SPECIAL_CLUSTER_ORDER:
            return _normalize_text(canonical_id).title()
        return _normalize_text(canonical_id)
    if isinstance(canonical_id, float) and not float(canonical_id).is_integer():
        return f"Cluster {canonical_id:g}"
    return f"Cluster {int(canonical_id)}"


def get_cluster_display_name(cluster_id: Any, annotation_map: Optional[Dict[Any, Any]] = None) -> str:
    """Return the user-facing display name for a cluster value."""
    normalized_map = normalize_cluster_annotation_map(annotation_map)
    canonical_id = canonicalize_cluster_id(cluster_id, annotation_map=normalized_map)
    if canonical_id in normalized_map:
        return _normalize_text(normalized_map[canonical_id])
    return format_default_cluster_label(canonical_id)


def cluster_sort_key(value: Any, annotation_map: Optional[Dict[Any, Any]] = None):
    """Sort numeric cluster ids numerically and keep special labels at the end."""
    canonical_value = canonicalize_cluster_id(value, annotation_map=annotation_map)
    if canonical_value is None:
        return (4, "")
    if isinstance(canonical_value, (bool, np.bool_)):
        return (0, int(canonical_value))
    if isinstance(canonical_value, (int, np.integer)):
        return (0, int(canonical_value))
    if isinstance(canonical_value, (float, np.floating)):
        return (1, float(canonical_value))
    normalized = _normalized_text_key(canonical_value)
    if normalized in _SPECIAL_CLUSTER_ORDER:
        return (3, _SPECIAL_CLUSTER_ORDER[normalized], normalized)
    return (2, normalized)


def sort_cluster_values(
    values: Iterable[Any],
    annotation_map: Optional[Dict[Any, Any]] = None,
    *,
    canonical: bool = False,
) -> list:
    """Sort cluster values consistently while de-duplicating equivalent ids."""
    normalized_map = normalize_cluster_annotation_map(annotation_map)
    unique_entries = []
    seen_keys = set()
    for raw_value in values:
        canonical_value = canonicalize_cluster_id(raw_value, annotation_map=normalized_map)
        if canonical_value is None:
            continue
        key = (
            type(canonical_value).__name__,
            canonical_value,
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_entries.append((raw_value, canonical_value))

    unique_entries.sort(key=lambda item: cluster_sort_key(item[1], annotation_map=normalized_map))
    if canonical:
        return [item[1] for item in unique_entries]
    return [item[0] for item in unique_entries]


def is_cluster_column(column_name: Optional[str]) -> bool:
    """Return True when a dataframe column is one of the cluster label columns."""
    if column_name is None:
        return False
    return str(column_name).strip().lower() in _CLUSTER_COLUMNS
