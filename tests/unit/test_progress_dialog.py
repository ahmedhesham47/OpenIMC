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

from PyQt5 import QtWidgets

from openimc.ui.dialogs.progress_dialog import ProgressDialog, close_progress_dialog


def test_close_progress_dialog_hides_closes_and_processes_events(monkeypatch):
    events = []

    class FakeDialog:
        def hide(self):
            events.append("hide")

        def close(self):
            events.append("close")

    class FakeApplication:
        def processEvents(self):
            events.append("processEvents")

    fake_dialog = FakeDialog()
    fake_app = FakeApplication()

    monkeypatch.setattr(
        QtWidgets.QApplication,
        "instance",
        staticmethod(lambda: fake_app),
    )

    close_progress_dialog(fake_dialog)

    assert events == ["hide", "close", "processEvents"]


def test_progress_dialog_can_disable_cancel_button(qtbot):
    dlg = ProgressDialog("Test Progress")
    qtbot.addWidget(dlg)

    assert dlg.cancel_btn.isEnabled()

    dlg.set_cancel_enabled(False)
    qtbot.wait(1)

    assert not dlg.cancel_btn.isEnabled()
