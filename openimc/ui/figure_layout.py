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
import sys
from typing import Dict, Iterable, Optional, Sequence
import warnings

from matplotlib import rcParams
from matplotlib.figure import Figure
from matplotlib.text import Text


def _canvas_device_pixel_ratio(canvas) -> float:
    """Return a sane Qt device-pixel ratio for live canvas sizing."""
    if canvas is None:
        return 1.0
    dpr = 1.0
    try:
        if hasattr(canvas, 'devicePixelRatioF'):
            dpr = float(canvas.devicePixelRatioF())
        elif hasattr(canvas, 'devicePixelRatio'):
            dpr = float(canvas.devicePixelRatio())
    except Exception:
        dpr = 1.0
    if dpr < 1.0:
        return 1.0
    return dpr


def _get_base_figure_dpi(figure: Figure, device_pixel_ratio: float) -> float:
    """Recover the figure's logical DPI before any Qt HiDPI scaling."""
    base_dpi = getattr(figure, '_openimc_base_dpi', None)
    if base_dpi is not None:
        try:
            base_dpi = float(base_dpi)
        except Exception:
            base_dpi = None
    if base_dpi is None:
        current_dpi = float(figure.get_dpi() or 100.0)
        default_dpi = float(rcParams.get('figure.dpi', current_dpi) or current_dpi)
        if (
            device_pixel_ratio > 1.0
            and abs(current_dpi - (default_dpi * device_pixel_ratio)) < abs(current_dpi - default_dpi)
        ):
            base_dpi = current_dpi / device_pixel_ratio
        else:
            base_dpi = current_dpi
        setattr(figure, '_openimc_base_dpi', base_dpi)
    return max(36.0, float(base_dpi))


def sync_figure_to_canvas(figure: Figure, canvas) -> None:
    """Resize a matplotlib figure to the current Qt canvas size."""
    if figure is None or canvas is None:
        return
    try:
        dpr = _canvas_device_pixel_ratio(canvas)
        base_dpi = _get_base_figure_dpi(figure, dpr)
        target_dpi = base_dpi * dpr
        width_px = max(1.0, float(canvas.width()) * dpr)
        height_px = max(1.0, float(canvas.height()) * dpr)
        figure.set_dpi(target_dpi)
        figure.set_size_inches(width_px / target_dpi, height_px / target_dpi, forward=False)
    except Exception:
        # Best-effort only.
        pass


def should_use_nonblocking_canvas_refresh() -> bool:
    """Avoid synchronous Qt repaints in environments where they can deadlock."""
    return sys.platform == "darwin" and os.environ.get("GITHUB_ACTIONS") == "true"


def refresh_canvas(canvas, *, draw: bool = True) -> None:
    """Redraw a Qt-backed canvas without forcing unsafe native repaints."""
    if canvas is None:
        return

    try:
        if draw:
            canvas.draw()
    except Exception:
        pass

    try:
        canvas.update()
    except Exception:
        pass

    try:
        if should_use_nonblocking_canvas_refresh():
            if hasattr(canvas, 'draw_idle'):
                canvas.draw_idle()
        else:
            canvas.repaint()
    except Exception:
        pass


def measure_figure_text_overflow(figure: Figure) -> Dict[str, float]:
    """Measure visible text that extends outside the figure canvas."""
    overflow = {'left': 0.0, 'right': 0.0, 'top': 0.0, 'bottom': 0.0}
    if figure is None or getattr(figure, 'canvas', None) is None:
        return overflow

    try:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        fig_bbox = figure.bbox
        fig_width = max(1.0, float(fig_bbox.width))
        fig_height = max(1.0, float(fig_bbox.height))

        for text_artist in figure.findobj(match=Text):
            if not text_artist.get_visible():
                continue
            text_value = text_artist.get_text()
            if not isinstance(text_value, str) or not text_value.strip():
                continue
            try:
                bbox = text_artist.get_window_extent(renderer=renderer)
            except Exception:
                continue
            if bbox.width <= 0 or bbox.height <= 0:
                continue
            if bbox.x0 < fig_bbox.x0:
                overflow['left'] = max(overflow['left'], (fig_bbox.x0 - bbox.x0) / fig_width)
            if bbox.x1 > fig_bbox.x1:
                overflow['right'] = max(overflow['right'], (bbox.x1 - fig_bbox.x1) / fig_width)
            if bbox.y0 < fig_bbox.y0:
                overflow['bottom'] = max(overflow['bottom'], (fig_bbox.y0 - bbox.y0) / fig_height)
            if bbox.y1 > fig_bbox.y1:
                overflow['top'] = max(overflow['top'], (bbox.y1 - fig_bbox.y1) / fig_height)
    except Exception:
        pass

    return overflow


def _clamp_rect(rect: Sequence[float], margin_adjust: Dict[str, float]) -> Sequence[float]:
    left = min(0.45, max(0.0, float(rect[0]) + margin_adjust['left']))
    bottom = min(0.60, max(0.0, float(rect[1]) + margin_adjust['bottom']))
    right = max(left + 0.20, min(1.0, float(rect[2]) - margin_adjust['right']))
    top = max(bottom + 0.20, min(1.0, float(rect[3]) - margin_adjust['top']))
    return [left, bottom, right, top]


def _iter_axis_text(figure: Figure, horizontal: bool, vertical: bool) -> Iterable[Text]:
    for ax in figure.axes:
        if horizontal:
            for text in ax.get_xticklabels():
                yield text
            yield ax.xaxis.label
            yield ax.title
        if vertical:
            for text in ax.get_yticklabels():
                yield text
            yield ax.yaxis.label
            if not horizontal:
                yield ax.title
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                yield text
            yield legend.get_title()
    if getattr(figure, '_suptitle', None) is not None:
        yield figure._suptitle


def _is_axis_heading(text: Text) -> bool:
    ax = getattr(text, 'axes', None)
    if ax is None:
        return False
    return text is ax.xaxis.label or text is ax.yaxis.label or text is ax.title


def _shrink_text(figure: Figure, overflow: Dict[str, float]) -> None:
    max_overflow = max(overflow.values())
    if max_overflow <= 0:
        return

    horizontal = bool(overflow['bottom'] > 0.004 or overflow['top'] > 0.004)
    vertical = bool(overflow['left'] > 0.004 or overflow['right'] > 0.004)
    if not horizontal and not vertical:
        return

    factor = 1.0 - min(0.18, max_overflow * 2.0)
    # Keep paired axes and colorbars on one font scale when compaction is needed.
    shrink_all_axis_text = horizontal or vertical
    for text in _iter_axis_text(
        figure,
        horizontal=shrink_all_axis_text,
        vertical=shrink_all_axis_text,
    ):
        if text is None or not text.get_visible():
            continue
        try:
            current = float(text.get_fontsize())
        except Exception:
            continue
        if current <= 6.0:
            continue
        if text is getattr(figure, '_suptitle', None):
            min_size = 8.0
        elif _is_axis_heading(text):
            min_size = 7.0
        else:
            min_size = 6.0
        try:
            text.set_fontsize(max(min_size, current * factor))
        except Exception:
            continue


def fit_figure_to_canvas(
    figure: Figure,
    canvas=None,
    *,
    rect: Optional[Sequence[float]] = None,
    pad: float = 0.8,
    max_passes: int = 3,
    overflow_threshold: float = 0.0035,
    allow_text_compaction: bool = True,
) -> Dict[str, float]:
    """Fit a figure layout to the live canvas and correct residual text overflow."""
    if figure is None:
        return {'left': 0.0, 'right': 0.0, 'top': 0.0, 'bottom': 0.0}

    if canvas is not None:
        sync_figure_to_canvas(figure, canvas)

    base_rect = rect if rect is not None else [0.0, 0.0, 1.0, 1.0]
    margin_adjust = {'left': 0.0, 'right': 0.0, 'top': 0.0, 'bottom': 0.0}
    overflow = {'left': 0.0, 'right': 0.0, 'top': 0.0, 'bottom': 0.0}

    for pass_idx in range(max(1, int(max_passes))):
        current_rect = _clamp_rect(base_rect, margin_adjust)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                figure.tight_layout(pad=pad, rect=current_rect)
        except Exception:
            try:
                figure.subplots_adjust(
                    left=current_rect[0],
                    bottom=current_rect[1],
                    right=current_rect[2],
                    top=current_rect[3],
                )
            except Exception:
                pass

        overflow = measure_figure_text_overflow(figure)
        if max(overflow.values()) <= overflow_threshold:
            break

        if allow_text_compaction:
            _shrink_text(figure, overflow)

        if pass_idx + 1 < max_passes:
            margin_adjust['left'] = min(0.28, margin_adjust['left'] + overflow['left'] + 0.010)
            margin_adjust['right'] = min(0.28, margin_adjust['right'] + overflow['right'] + 0.010)
            margin_adjust['bottom'] = min(0.34, margin_adjust['bottom'] + overflow['bottom'] + 0.012)
            margin_adjust['top'] = min(0.12, margin_adjust['top'] + overflow['top'] + 0.008)

    return overflow


def fit_canvas_and_draw(
    canvas,
    *,
    rect: Optional[Sequence[float]] = None,
    pad: float = 0.8,
    max_passes: int = 3,
    allow_text_compaction: bool = True,
) -> Dict[str, float]:
    """Resize a figure to its canvas, fit the layout, and redraw."""
    figure = getattr(canvas, 'figure', None)
    overflow = fit_figure_to_canvas(
        figure,
        canvas,
        rect=rect,
        pad=pad,
        max_passes=max_passes,
        allow_text_compaction=allow_text_compaction,
    )
    refresh_canvas(canvas, draw=True)
    return overflow


def dense_heatmap_style(
    *,
    n_rows: int,
    n_cols: int,
    row_labels: Optional[Sequence[str]] = None,
    col_labels: Optional[Sequence[str]] = None,
    base_tick_fontsize: float = 10.0,
    base_annotation_fontsize: float = 9.0,
    allow_annotations: bool = True,
) -> Dict[str, float]:
    """Choose compact, readability-first styling for label-dense heatmaps."""
    row_labels = [str(label) for label in (row_labels or [])]
    col_labels = [str(label) for label in (col_labels or [])]

    max_row_len = max((len(label) for label in row_labels), default=0)
    max_col_len = max((len(label) for label in col_labels), default=0)
    max_label_len = max(max_row_len, max_col_len)
    matrix_span = max(1, int(n_rows), int(n_cols))
    cell_count = max(1, int(n_rows) * int(n_cols))

    dense_penalty = max(0.0, matrix_span - 8) * 0.28
    label_penalty = max(0.0, max_label_len - 12) * 0.11
    tick_fontsize = max(6.0, min(base_tick_fontsize, base_tick_fontsize - dense_penalty - label_penalty))

    axis_fontsize = max(7.0, tick_fontsize + 0.8)
    title_fontsize = max(8.0, axis_fontsize + 1.0)
    colorbar_fontsize = max(7.0, tick_fontsize)
    annotation_fontsize = max(5.0, min(base_annotation_fontsize, tick_fontsize - 0.9))

    show_annotations = bool(
        allow_annotations
        and matrix_span <= 12
        and cell_count <= 144
        and max_label_len <= 18
    )

    if matrix_span <= 10:
        linewidths = 0.50
    elif matrix_span <= 16:
        linewidths = 0.30
    else:
        linewidths = 0.15

    if max_col_len >= 20:
        x_rotation = 50
    elif max_col_len >= 9:
        x_rotation = 40
    else:
        x_rotation = 25

    return {
        'tick_fontsize': tick_fontsize,
        'axis_fontsize': axis_fontsize,
        'title_fontsize': title_fontsize,
        'annotation_fontsize': annotation_fontsize,
        'colorbar_fontsize': colorbar_fontsize,
        'show_annotations': show_annotations,
        'linewidths': linewidths,
        'square_cells': bool(matrix_span <= 9 and max_label_len <= 16),
        'x_rotation': x_rotation,
        'colorbar_fraction': 0.045 if matrix_span <= 12 else 0.038,
        'colorbar_pad': 0.020 if matrix_span <= 12 else 0.028,
        'colorbar_shrink': 0.90 if matrix_span <= 10 else 0.84,
    }
