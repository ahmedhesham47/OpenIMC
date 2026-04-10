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

import json
import shutil
import builtins

import numpy as np
import pandas as pd

import openimc.ui.state_manager as state_manager_module
from openimc.ui.state_manager import StateManager


def test_state_manager_paths_are_relative_to_state_folder(tmp_path):
    sm = StateManager()

    state_dir = tmp_path / "my_state_folder"
    main_window_state = {
        "main_state": {"openimc_version": "test"},
        "images": {},
        "masks": {},
        "features": {},
        "analysis": {
            "clustering": {"clustering_method": "kmeans"},
        },
        "source_files": [],
    }

    assert sm.save_state(state_dir, main_window_state, overwrite=True)

    state_json = json.loads((state_dir / "state.json").read_text())
    # analysis path should be relative to state folder, not include parent folder name
    clustering_path = state_json["analysis_modules"]["clustering"]
    assert not clustering_path.startswith(str(state_dir.name))
    assert clustering_path.startswith("analysis/")
    assert "\\" not in clustering_path

    # Renaming the state folder should not break loads
    renamed = tmp_path / "renamed_state_folder"
    shutil.move(str(state_dir), str(renamed))
    loaded = sm.load_state(renamed)
    assert loaded is not None
    assert loaded["analysis"]["clustering"]["clustering_method"] == "kmeans"


def test_state_manager_persists_large_arrays_and_dataframes(tmp_path):
    sm = StateManager()

    # Large embedding triggers ndarray_file
    rng = np.random.default_rng(123)
    umap = rng.standard_normal((12000, 2)).astype(np.float32)

    # Large DF triggers DataFrame_file (rows * cols > 50_000)
    df = pd.DataFrame(rng.standard_normal((5000, 20)))

    state_dir = tmp_path / "state"
    main_window_state = {
        "main_state": {"openimc_version": "test"},
        "images": {},
        "masks": {},
        "features": {},
        "analysis": {
            "clustering": {"umap_embedding": umap},
            "pixel_correlation": {"aggregated_results": df},
        },
        "source_files": [],
    }

    assert sm.save_state(state_dir, main_window_state, overwrite=True)

    # Ensure blob files exist in expected location
    blobs_dir = state_dir / "analysis" / "_blobs"
    assert blobs_dir.exists()
    # We don't depend on the exact filenames too much, but they should be present.
    assert any(p.suffix == ".npy" for p in blobs_dir.iterdir())
    assert any(p.suffix == ".csv" for p in blobs_dir.iterdir())

    # Renaming the state folder should not break blob loads
    renamed = tmp_path / "state_renamed"
    shutil.move(str(state_dir), str(renamed))
    loaded = sm.load_state(renamed)
    assert loaded is not None

    restored_umap = loaded["analysis"]["clustering"]["umap_embedding"]
    assert isinstance(restored_umap, np.ndarray)
    assert restored_umap.shape == umap.shape
    assert restored_umap.dtype == umap.dtype
    np.testing.assert_allclose(restored_umap, umap, rtol=0, atol=0)

    restored_df = loaded["analysis"]["pixel_correlation"]["aggregated_results"]
    assert isinstance(restored_df, pd.DataFrame)
    assert restored_df.shape == df.shape
    # CSV round-trip loses exact float formatting sometimes; compare numerically.
    np.testing.assert_allclose(restored_df.to_numpy(), df.to_numpy(), rtol=1e-6, atol=1e-6)


def test_state_manager_save_state_uses_utf8_for_text_outputs(tmp_path, monkeypatch):
    sm = StateManager()

    state_dir = tmp_path / "utf8_state"
    main_window_state = {
        "main_state": {"openimc_version": "test"},
        "images": {},
        "masks": {},
        "features": {},
        "analysis": {},
        "source_files": [],
    }

    real_open = builtins.open

    def cp1252_default_open(file, mode="r", *args, **kwargs):
        if "b" not in mode and "encoding" not in kwargs:
            kwargs["encoding"] = "cp1252"
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(state_manager_module, "open", cp1252_default_open, raising=False)

    assert sm.save_state(state_dir, main_window_state, overwrite=True)
    assert (state_dir / "README.md").exists()
    assert (state_dir / "manifest.txt").exists()
