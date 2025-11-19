# SPDX-License-Identifier: GPL-3.0-or-later
#
# Thin entry script to start the OpenIMC GUI for PyInstaller builds.
from openimc.__main__ import run_gui

if __name__ == "__main__":
    run_gui()
