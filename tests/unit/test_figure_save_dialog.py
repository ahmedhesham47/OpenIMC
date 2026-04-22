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
import pytest
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PyQt5 import QtWidgets

from openimc.ui.dialogs import figure_save_dialog as figure_save_dialog_module


def _build_figure_with_legend_and_colorbar() -> Figure:
    fig = Figure(figsize=(4.0, 3.0))
    ax = fig.subplots()
    image = ax.imshow(np.arange(25, dtype=float).reshape(5, 5), cmap="viridis")
    ax.set_title("Cluster Explorer Preview", fontsize=11)
    ax.set_xlabel("X", fontsize=10)
    ax.set_ylabel("Y", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.legend(
        handles=[Line2D([0], [0], color="white", linewidth=2, label="Marker intensity")],
        loc="upper right",
        fontsize=8,
    )
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.ax.tick_params(labelsize=7)
    fig.canvas.draw()
    return fig


def test_save_figure_with_options_restores_font_sizes_between_repeated_exports(qtbot, monkeypatch):
    fig = _build_figure_with_legend_and_colorbar()
    qtbot.addWidget(QtWidgets.QWidget())

    original_size = tuple(fig.get_size_inches())
    original_title_size = fig.axes[0].title.get_fontsize()
    original_legend_sizes = [text.get_fontsize() for text in fig.axes[0].get_legend().get_texts()]
    original_colorbar_sizes = [tick.get_fontsize() for tick in fig.axes[1].get_yticklabels()]

    save_requests = iter(
        [
            ("cluster_grid.png", {"format": "png", "dpi": 300, "fontsize": 6.0, "figsize": (5.5, 4.25)}),
            ("cluster_grid.png", {"format": "png", "dpi": 300, "fontsize": 9.0}),
        ]
    )
    captured = []

    monkeypatch.setattr(
        figure_save_dialog_module.FigureSaveDialog,
        "exec_",
        lambda self: QtWidgets.QDialog.Accepted,
    )
    monkeypatch.setattr(
        figure_save_dialog_module.FigureSaveDialog,
        "get_save_options",
        lambda self: next(save_requests),
    )

    def _fake_savefig(self, filename, *args, **kwargs):
        captured.append(
            {
                "filename": filename,
                "kwargs": dict(kwargs),
                "size": tuple(self.get_size_inches()),
                "title_size": self.axes[0].title.get_fontsize(),
                "legend_sizes": [text.get_fontsize() for text in self.axes[0].get_legend().get_texts()],
                "colorbar_sizes": [tick.get_fontsize() for tick in self.axes[1].get_yticklabels()],
            }
        )

    monkeypatch.setattr(Figure, "savefig", _fake_savefig, raising=True)

    assert figure_save_dialog_module.save_figure_with_options(fig, "cluster_grid.png")
    assert captured[0]["filename"].endswith(".png")
    assert captured[0]["kwargs"]["bbox_inches"] == "tight"
    assert captured[0]["title_size"] == pytest.approx(6.0)
    assert captured[0]["legend_sizes"] == pytest.approx([6.0])
    assert all(size == pytest.approx(6.0) for size in captured[0]["colorbar_sizes"])
    assert captured[0]["size"] == pytest.approx((5.5, 4.25))
    assert tuple(fig.get_size_inches()) == pytest.approx(original_size)
    assert fig.axes[0].title.get_fontsize() == pytest.approx(original_title_size)
    assert [text.get_fontsize() for text in fig.axes[0].get_legend().get_texts()] == pytest.approx(
        original_legend_sizes
    )
    assert [tick.get_fontsize() for tick in fig.axes[1].get_yticklabels()] == pytest.approx(
        original_colorbar_sizes
    )

    assert figure_save_dialog_module.save_figure_with_options(fig, "cluster_grid.png")
    assert captured[1]["title_size"] == pytest.approx(9.0)
    assert captured[1]["legend_sizes"] == pytest.approx([9.0])
    assert all(size == pytest.approx(9.0) for size in captured[1]["colorbar_sizes"])
    assert captured[1]["size"] == pytest.approx(original_size)
    assert tuple(fig.get_size_inches()) == pytest.approx(original_size)
    assert fig.axes[0].title.get_fontsize() == pytest.approx(original_title_size)
    assert [text.get_fontsize() for text in fig.axes[0].get_legend().get_texts()] == pytest.approx(
        original_legend_sizes
    )
    assert [tick.get_fontsize() for tick in fig.axes[1].get_yticklabels()] == pytest.approx(
        original_colorbar_sizes
    )
