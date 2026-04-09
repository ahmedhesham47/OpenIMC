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

from openimc.ui.figure_layout import fit_canvas_and_draw, measure_figure_text_overflow


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

    expected_width_in = canvas.width() / max(fig.get_dpi(), 1.0)
    expected_height_in = canvas.height() / max(fig.get_dpi(), 1.0)
    assert abs(fig.get_size_inches()[0] - expected_width_in) < 0.2
    assert abs(fig.get_size_inches()[1] - expected_height_in) < 0.2
    assert max(overflow.values()) <= 0.02


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
