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

import os
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.unit
@pytest.mark.parametrize(
    "module_name",
    [
        "openimc.ui.dialogs.comparison_dialog",
        "openimc.ui.dialogs.deconvolution_dialog",
    ],
)
def test_dialog_modules_import_under_headless_agg_backend(module_name):
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env.pop("DISPLAY", None)
    env.pop("WAYLAND_DISPLAY", None)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env.setdefault("QT_QUICK_BACKEND", "software")

    result = subprocess.run(
        [sys.executable, "-c", f"import {module_name}"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        f"Failed to import {module_name} with MPLBACKEND=Agg.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
