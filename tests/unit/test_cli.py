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
Unit tests for CLI functions.
"""
import pytest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import tifffile

from openimc.cli import load_data, parse_denoise_settings, qc_analysis_command
from openimc.core import qc_analysis
from openimc.data.mcd_loader import AcquisitionInfo


class _DummyQCLoader:
    def __init__(self, stack, acquisition):
        self._stack = stack
        self._acquisition = acquisition

    def list_acquisitions(self):
        return [self._acquisition]

    def get_channels(self, _acq_id):
        return list(self._acquisition.channels)

    def get_all_channels(self, _acq_id):
        return self._stack

    def get_image(self, _acq_id, channel):
        index = self._acquisition.channels.index(channel)
        return self._stack[:, :, index]

    def close(self):
        return None


@pytest.mark.unit
class TestLoadData:
    """Tests for load_data CLI function."""
    
    def test_load_data_invalid_path(self):
        """Test load_data with invalid path raises error."""
        with pytest.raises(ValueError, match="Input path must be"):
            load_data("/nonexistent/path")
    
    def test_load_data_directory(self, mock_ometiff_directory):
        """Test load_data with OME-TIFF directory."""
        loader, loader_type = load_data(str(mock_ometiff_directory))
        
        assert loader_type == 'ometiff'
        assert loader is not None
    
    def test_load_data_channel_format(self, mock_ometiff_directory):
        """Test load_data with custom channel format."""
        loader, loader_type = load_data(str(mock_ometiff_directory), channel_format='HWC')
        
        assert loader_type == 'ometiff'
        assert loader.channel_format == 'HWC'


@pytest.mark.unit
class TestParseDenoiseSettings:
    """Tests for parse_denoise_settings CLI function."""
    
    def test_parse_denoise_settings_none(self):
        """Test parse_denoise_settings with None."""
        result = parse_denoise_settings(None)
        assert result == {}
    
    def test_parse_denoise_settings_empty_string(self):
        """Test parse_denoise_settings with empty string."""
        result = parse_denoise_settings("")
        assert result == {}
    
    def test_parse_denoise_settings_json_string(self):
        """Test parse_denoise_settings with JSON string."""
        json_str = '{"DAPI": {"hot": {"method": "median3"}}}'
        result = parse_denoise_settings(json_str)
        
        assert isinstance(result, dict)
        assert "DAPI" in result
    
    def test_parse_denoise_settings_json_file(self, temp_dir):
        """Test parse_denoise_settings with JSON file."""
        settings = {
            "DAPI": {
                "hot": {"method": "median3"},
                "speckle": {"method": "gaussian", "sigma": 0.8}
            }
        }
        
        json_file = temp_dir / "denoise_settings.json"
        with open(json_file, 'w') as f:
            json.dump(settings, f)
        
        result = parse_denoise_settings(str(json_file))
        
        assert isinstance(result, dict)
        assert "DAPI" in result
        assert "hot" in result["DAPI"]
    
    def test_parse_denoise_settings_invalid_json(self):
        """Test parse_denoise_settings with invalid JSON raises error."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_denoise_settings("{invalid json}")


@pytest.mark.unit
class TestQCAnalysisCLI:
    """Tests for QC CLI signal-definition plumbing."""

    def test_qc_analysis_command_round_trips_cell_signal_options(self, temp_dir, monkeypatch):
        channels = ["MarkerA"]
        img = np.array(
            [
                [12.0, 12.0, 0.0, 2.0, 2.0],
                [12.0, 12.0, 0.0, 2.0, 2.0],
                [1.0, 2.0, 1.0, 2.0, 1.0],
                [2.0, 2.0, 0.0, 2.0, 2.0],
                [2.0, 2.0, 0.0, 2.0, 2.0],
            ],
            dtype=np.float32,
        )
        mask = np.array(
            [
                [1, 1, 0, 2, 2],
                [1, 1, 0, 2, 2],
                [0, 0, 0, 0, 0],
                [3, 3, 0, 4, 4],
                [3, 3, 0, 4, 4],
            ],
            dtype=np.uint32,
        )
        stack = np.stack([img], axis=-1)
        acquisition = AcquisitionInfo(
            id="ROI_1",
            name="ROI_1",
            well="A1",
            size=img.shape,
            channels=channels,
            channel_metals=[""],
            channel_labels=[""],
            metadata={},
            source_file="dummy",
        )
        loader = _DummyQCLoader(stack, acquisition)
        mask_path = temp_dir / "mask.tif"
        output_path = temp_dir / "qc.csv"
        tifffile.imwrite(mask_path, mask)

        monkeypatch.setattr("openimc.cli.load_mcd", lambda *_args, **_kwargs: (loader, "ometiff"))

        args = SimpleNamespace(
            input="dummy_input",
            output=str(output_path),
            channel_format="CHW",
            acquisition=None,
            channels=None,
            mask=str(mask_path),
            mode="cell",
            cell_signal_method="upper_quantile",
            positive_threshold_sd=2.5,
            upper_quantile=0.75,
        )

        qc_analysis_command(args)

        expected = qc_analysis(
            loader=loader,
            acquisition=acquisition,
            channels=channels,
            mode="cell",
            mask=mask,
            cell_signal_method="upper_quantile",
            positive_threshold_sd=2.5,
            upper_quantile=0.75,
        ).reset_index(drop=True)
        actual = pd.read_csv(output_path).reset_index(drop=True)

        assert actual.loc[0, "cell_signal_method"] == "upper_quantile"
        assert actual.loc[0, "signal_quantile"] == pytest.approx(0.75)
        assert actual.loc[0, "snr"] == pytest.approx(expected.loc[0, "snr"])
        assert actual.loc[0, "signal_mean"] == pytest.approx(expected.loc[0, "signal_mean"])
