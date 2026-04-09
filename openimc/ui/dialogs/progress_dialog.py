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

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
import time
from typing import Callable, Optional, TypeVar

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QEventLoop, QThread

T = TypeVar("T")

class ProgressDialog(QtWidgets.QDialog):
    def __init__(self, title: str = "Export Progress", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)
        self.setFixedSize(450, 180)
        self.setWindowFlags(Qt.Dialog | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self.cancelled = False
        self._create_ui()

    def _create_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        self.status_label = QtWidgets.QLabel("Preparing export...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.details_label = QtWidgets.QLabel("")
        self.details_label.setAlignment(Qt.AlignCenter)
        self.details_label.setWordWrap(True)
        self.details_label.setStyleSheet("QLabel { color: #666; }")
        layout.addWidget(self.details_label)

        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._cancel)
        layout.addWidget(self.cancel_btn)

    def _cancel(self):
        self.cancelled = True
        self.status_label.setText("Cancelling...")
        self.cancel_btn.setEnabled(False)

    def update_progress(self, value: int, status: str = "", details: str = ""):
        self.progress_bar.setValue(value)
        if status:
            self.status_label.setText(status)
        if details:
            self.details_label.setText(details)
        QtWidgets.QApplication.processEvents()

    def set_maximum(self, maximum: int):
        self.progress_bar.setMaximum(maximum)

    def set_cancel_enabled(self, enabled: bool):
        self.cancel_btn.setEnabled(enabled)

    def is_cancelled(self) -> bool:
        return self.cancelled


def close_progress_dialog(progress_dialog: Optional[QtWidgets.QDialog]) -> None:
    """Close a progress dialog and flush pending UI updates."""
    if progress_dialog is None:
        return

    progress_dialog.hide()
    progress_dialog.close()

    app = QtWidgets.QApplication.instance()
    if app is not None:
        app.processEvents()


def run_blocking_task_with_progress(
    *,
    parent: Optional[QtWidgets.QWidget],
    window_title: str,
    initial_message: str,
    task: Callable[[], T],
    detail_text: str = "",
    poll_interval_ms: int = 100,
) -> T:
    """Run a blocking task in a worker thread while showing a modal busy dialog."""
    progress = QtWidgets.QProgressDialog(initial_message, None, 0, 0, parent)
    progress.setWindowTitle(window_title)
    progress.setWindowModality(Qt.WindowModal)
    progress.setCancelButton(None)
    progress.setMinimumDuration(0)
    progress.setAutoClose(False)
    progress.setAutoReset(False)
    progress.show()

    app = QtWidgets.QApplication.instance()
    if app is not None:
        app.processEvents(QEventLoop.AllEvents, 20)

    started_at = time.monotonic()
    poll_interval_ms = max(20, min(250, int(poll_interval_ms)))

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(task)
            while not future.done():
                elapsed_s = int(max(0.0, time.monotonic() - started_at))
                dots = "." * ((elapsed_s % 3) + 1)

                lines = [f"{initial_message}{dots}"]
                if detail_text:
                    lines.append(detail_text)
                lines.append(f"Elapsed: {elapsed_s}s")
                progress.setLabelText("\n".join(lines))

                if app is not None:
                    app.processEvents(QEventLoop.AllEvents, poll_interval_ms)
                QThread.msleep(poll_interval_ms)

            return future.result()
    finally:
        close_progress_dialog(progress)


def run_blocking_task_with_progress_then_finalize(
    *,
    parent: Optional[QtWidgets.QWidget],
    window_title: str,
    initial_message: str,
    task: Callable[[], T],
    finalize: Callable[[T, QtWidgets.QProgressDialog], None],
    detail_text: str = "",
    finishing_message: str = "Rendering plot",
    finishing_detail_text: str = "",
    poll_interval_ms: int = 100,
) -> T:
    """Run a worker task with progress, then keep the dialog open through a UI finalize step."""
    progress = QtWidgets.QProgressDialog(initial_message, None, 0, 0, parent)
    progress.setWindowTitle(window_title)
    progress.setWindowModality(Qt.WindowModal)
    progress.setCancelButton(None)
    progress.setMinimumDuration(0)
    progress.setAutoClose(False)
    progress.setAutoReset(False)
    progress.show()

    app = QtWidgets.QApplication.instance()
    if app is not None:
        app.processEvents(QEventLoop.AllEvents, 20)

    started_at = time.monotonic()
    poll_interval_ms = max(20, min(250, int(poll_interval_ms)))

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(task)
            while not future.done():
                elapsed_s = int(max(0.0, time.monotonic() - started_at))
                dots = "." * ((elapsed_s % 3) + 1)

                lines = [f"{initial_message}{dots}"]
                if detail_text:
                    lines.append(detail_text)
                lines.append(f"Elapsed: {elapsed_s}s")
                progress.setLabelText("\n".join(lines))

                if app is not None:
                    app.processEvents(QEventLoop.AllEvents, poll_interval_ms)
                QThread.msleep(poll_interval_ms)

            result = future.result()
            elapsed_s = int(max(0.0, time.monotonic() - started_at))
            final_lines = [f"{finishing_message}..."]
            if finishing_detail_text:
                final_lines.append(finishing_detail_text)
            final_lines.append(f"Elapsed: {elapsed_s}s")
            progress.setLabelText("\n".join(final_lines))
            if app is not None:
                app.processEvents(QEventLoop.AllEvents, 20)
            finalize(result, progress)
            return result
    finally:
        close_progress_dialog(progress)


def run_task_with_event_pump(
    task: Callable[[], T],
    *,
    poll_interval_ms: int = 100,
    use_process: bool = False,
) -> T:
    """Run a blocking task in a worker thread while keeping the GUI event loop responsive.

    This helper is useful when the caller already has its own progress dialog and only
    needs to prevent "application not responding" during long CPU/GPU/network work.
    """
    app = QtWidgets.QApplication.instance()
    poll_interval_ms = max(20, min(250, int(poll_interval_ms)))

    def _wait_future(executor):
        with executor:
            future = executor.submit(task)
            while not future.done():
                if app is not None:
                    app.processEvents(QEventLoop.AllEvents, poll_interval_ms)
                QThread.msleep(poll_interval_ms)
            return future.result()

    if use_process:
        try:
            return _wait_future(
                ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=mp.get_context("spawn"),
                )
            )
        except Exception:
            # Fallback to thread execution if process-based execution fails
            # (e.g. environment restrictions or non-picklable task payloads).
            return _wait_future(ThreadPoolExecutor(max_workers=1))

    return _wait_future(ThreadPoolExecutor(max_workers=1))
