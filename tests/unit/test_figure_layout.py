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
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import openimc.ui.figure_layout as figure_layout
from openimc.ui.figure_layout import (
    fit_canvas_and_draw,
    measure_figure_text_overflow,
    refresh_canvas,
    sync_figure_to_canvas,
)


class _FakeCanvas:
    def __init__(self, width: int, height: int, *, device_pixel_ratio: float = 1.0):
        self._width = width
        self._height = height
        self._device_pixel_ratio = device_pixel_ratio

    def width(self):
        return self._width

    def height(self):
        return self._height

    def devicePixelRatioF(self):
        return self._device_pixel_ratio


class _FakeRefreshCanvas:
    def __init__(self):
        self.calls = []

    def draw(self):
        self.calls.append("draw")

    def draw_idle(self):
        self.calls.append("draw_idle")

    def update(self):
        self.calls.append("update")

    def repaint(self):
        self.calls.append("repaint")


def _build_canvas(qtbot, width: int, height: int):
    canvas = FigureCanvas(Figure(figsize=(3.0, 2.4)))
    qtbot.addWidget(canvas)
    canvas.resize(width, height)
    canvas.show()
    qtbot.wait(50)
    return canvas


def test_fit_canvas_and_draw_syncs_figure_size_and_keeps_heatmap_labels_visible(qtbot):
    canvas = _build_canvas(qtbot, 720, 520)
    fig = canvas.figure
    ax = fig.add_subplot(111)

    data = np.arange(49).reshape(7, 7)
    im = ax.imshow(data, aspect='auto', cmap='viridis')
    labels = [f'Feature {idx} with a deliberately long GUI label' for idx in range(data.shape[0])]
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_title('Heatmap Layout Stress Test')
    fig.colorbar(im, ax=ax, label='Intensity')

    fit_canvas_and_draw(canvas, pad=0.95, allow_text_compaction=True)
    overflow = measure_figure_text_overflow(fig)

    dpr = canvas.devicePixelRatioF() if hasattr(canvas, 'devicePixelRatioF') else 1.0
    expected_width_in = (canvas.width() * dpr) / max(fig.get_dpi(), 1.0)
    expected_height_in = (canvas.height() * dpr) / max(fig.get_dpi(), 1.0)
    assert abs(fig.get_size_inches()[0] - expected_width_in) < 0.2
    assert abs(fig.get_size_inches()[1] - expected_height_in) < 0.2
    assert max(overflow.values()) <= 0.02


def test_sync_figure_to_canvas_accounts_for_hidpi_device_ratio():
    fig = Figure(figsize=(3.0, 2.4), dpi=100.0)
    canvas = _FakeCanvas(720, 520, device_pixel_ratio=2.0)

    sync_figure_to_canvas(fig, canvas)

    assert abs(fig.get_dpi() - 200.0) < 1e-6
    assert abs(fig.get_size_inches()[0] - 7.2) < 1e-6
    assert abs(fig.get_size_inches()[1] - 5.2) < 1e-6
    assert abs(float(fig.bbox.width) - 1440.0) < 1e-6
    assert abs(float(fig.bbox.height) - 1040.0) < 1e-6


def test_refresh_canvas_avoids_blocking_repaint_on_macos_github_actions(monkeypatch):
    canvas = _FakeRefreshCanvas()

    monkeypatch.setattr(figure_layout.sys, "platform", "darwin", raising=False)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")

    refresh_canvas(canvas)

    assert canvas.calls == ["draw", "update", "draw_idle"]


def test_refresh_canvas_uses_repaint_outside_macos_github_actions(monkeypatch):
    canvas = _FakeRefreshCanvas()

    monkeypatch.setattr(figure_layout.sys, "platform", "linux", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)

    refresh_canvas(canvas)

    assert canvas.calls == ["draw", "update", "repaint"]


def test_fit_canvas_and_draw_keeps_external_legend_inside_reserved_space(qtbot):
    canvas = _build_canvas(qtbot, 760, 420)
    fig = canvas.figure
    ax = fig.add_subplot(111)

    x = np.linspace(0.0, 1.0, 50)
    for idx in range(6):
        ax.plot(
            x,
            (idx + 1) * x,
            linewidth=2,
            label=f'Cluster {idx + 1} with a long legend label',
        )

    ax.set_xlabel('Distance')
    ax.set_ylabel('Signal')
    ax.set_title('Legend Layout Stress Test')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True)

    fit_canvas_and_draw(
        canvas,
        rect=[0.0, 0.0, 0.82, 1.0],
        pad=0.95,
        allow_text_compaction=True,
    )
    overflow = measure_figure_text_overflow(fig)

    assert overflow['right'] <= 0.02
    assert max(overflow.values()) <= 0.025
