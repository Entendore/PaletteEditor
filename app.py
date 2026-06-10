#!/usr/bin/env python3
"""
Palette Manager — Color Palette Management Application
Single-file PySide6 + DuckDB + Numba application with CLI support.

Usage:
    python app.py              # Launch GUI (default)
    python app.py gui          # Launch GUI
    python app.py cli <cmd>    # CLI mode (list, generate, export, import, info)
"""

import os
import sys
import time
import random
import math
import json
import argparse
import csv
import struct
import logging
from typing import List, Tuple, Optional, Dict, Any
from collections import Counter

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("DuckPalette")

try:
    import duckdb
except ImportError:
    sys.exit("ERROR: duckdb required. pip install duckdb")

try:
    import numpy as np
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logger.warning("numpy/numba not found. Falling back to pure Python (slower math).")

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton,
    QColorDialog, QListWidget, QFileDialog, QLabel, QListWidgetItem,
    QHBoxLayout, QFrame, QScrollArea, QTabWidget, QFormLayout,
    QDoubleSpinBox, QComboBox, QMessageBox, QGroupBox, QInputDialog,
    QSplitter, QToolBar, QStatusBar, QMenu, QSpinBox, QLineEdit,
    QSizePolicy, QAbstractItemView, QToolTip, QToolButton, QSlider,
    QCheckBox, QAbstractSpinBox,
)
from PySide6.QtGui import (
    QColor, QBrush, QPainter, QPen, QFont, QAction,
    QKeySequence, QPalette, QPixmap, QClipboard, QCursor, QImage,
)
from PySide6.QtCore import Qt, QSize, Signal, QTimer, QPoint


# ══════════════════════════════════════════════════════════════
#  STYLESHEET
# ══════════════════════════════════════════════════════════════

DARK_STYLE = """
QMainWindow, QDialog { background-color: #1e1e2e; }
QWidget { color: #cdd6f4; font-family: 'Segoe UI','Helvetica Neue',sans-serif; font-size: 13px; }
QGroupBox { border:1px solid #45475a; border-radius:6px; margin-top:14px; padding-top:14px; font-weight:bold; }
QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 5px; }
QPushButton { background:#313244; border:1px solid #45475a; border-radius:5px; padding:6px 14px; color:#cdd6f4; }
QPushButton:hover { background:#45475a; border-color:#585b70; }
QPushButton:pressed { background:#585b70; }
QPushButton:disabled { color:#585b70; }
QLineEdit,QSpinBox,QDoubleSpinBox,QComboBox { background:#313244; border:1px solid #45475a; border-radius:4px; padding:4px 8px; color:#cdd6f4; }
QLineEdit:focus,QSpinBox:focus,QDoubleSpinBox:focus { border-color:#89b4fa; }
QComboBox::drop-down { border:none; }
QComboBox QAbstractItemView { background:#313244; selection-background-color:#45475a; }
QListWidget { background:#181825; border:1px solid #313244; border-radius:4px; padding:2px; }
QListWidget::item { padding:5px; border-radius:3px; }
QListWidget::item:selected { background:#45475a; }
QListWidget::item:hover { background:#313244; }
QTabWidget::pane { border:1px solid #313244; border-radius:4px; background:#1e1e2e; top:-1px; }
QTabBar::tab { background:#313244; border:1px solid #45475a; border-bottom:none; border-top-left-radius:6px;
               border-top-right-radius:6px; padding:7px 18px; margin-right:2px; color:#a6adc8; }
QTabBar::tab:selected { background:#1e1e2e; color:#89b4fa; border-bottom:2px solid #89b4fa; }
QTabBar::tab:hover:!selected { background:#45475a; }
QScrollArea { border:none; }
QSplitter::handle { background:#313244; width:3px; }
QToolBar { background:#181825; border-bottom:1px solid #313244; spacing:6px; padding:4px; }
QStatusBar { background:#181825; border-top:1px solid #313244; color:#a6adc8; font-size:12px; }
QMenu { background:#313244; border:1px solid #45475a; border-radius:6px; padding:4px; }
QMenu::item { padding:6px 24px; border-radius:3px; }
QMenu::item:selected { background:#45475a; }
QMenu::separator { height:1px; background:#45475a; margin:4px 8px; }
QToolTip { background:#313244; color:#cdd6f4; border:1px solid #45475a; border-radius:4px; padding:4px 8px; }
QFrame[frameShape="6"] { border:1px solid #313244; border-radius:4px; }
QSlider::groove:horizontal { border: 1px solid #45475a; height: 6px; background: #313244; border-radius: 3px; }
QSlider::handle:horizontal { background: #89b4fa; border: 1px solid #1e1e2e; width: 14px; margin: -5px 0; border-radius: 7px; }
QSlider::sub-page:horizontal { background: #585b70; border-radius: 3px; }
QScrollBar:vertical { background:#181825; width:10px; border-radius:5px; }
QScrollBar::handle:vertical { background:#45475a; border-radius:5px; min-height:20px; }
QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical { height:0; }
QScrollBar:horizontal { background:#181825; height:10px; border-radius:5px; }
QScrollBar::handle:horizontal { background:#45475a; border-radius:5px; min-width:20px; }
QScrollBar::add-line:horizontal,QScrollBar::sub-line:horizontal { width:0; }
"""


# ══════════════════════════════════════════════════════════════
#  ACCELERATED MATH (NUMBA / FALLBACK)
# ══════════════════════════════════════════════════════════════

if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _relative_luminance_numba(r: float, g: float, b: float) -> float:
        c_r, c_g, c_b = r / 255.0, g / 255.0, b / 255.0
        l_r = c_r / 12.92 if c_r <= 0.03928 else ((c_r + 0.055) / 1.055) ** 2.4
        l_g = c_g / 12.92 if c_g <= 0.03928 else ((c_g + 0.055) / 1.055) ** 2.4
        l_b = c_b / 12.92 if c_b <= 0.03928 else ((c_b + 0.055) / 1.055) ** 2.4
        return 0.2126 * l_r + 0.7152 * l_g + 0.0722 * l_b

    @njit(cache=True)
    def calculate_metadata_numba(pal):
        n = pal.shape[0]
        if n == 0:
            return 0.0, 0.0, 0
        brightness = 0.0; max_lum = -1.0; min_lum = 256.0
        sum_r = 0.0; sum_g = 0.0; sum_b = 0.0
        for i in range(n):
            r, g, b = pal[i, 0], pal[i, 1], pal[i, 2]
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            brightness += lum
            if lum > max_lum:
                max_lum = lum
            if lum < min_lum:
                min_lum = lum
            sum_r += r; sum_g += g; sum_b += b
        brightness /= n
        contrast = max_lum - min_lum
        dominant = 0
        if sum_g >= sum_r and sum_g >= sum_b:
            dominant = 1
        elif sum_b >= sum_r and sum_b >= sum_g:
            dominant = 2
        return brightness, contrast, dominant

    @njit(cache=True)
    def palette_wcag_contrast_numba(pal):
        n = pal.shape[0]
        if n < 2:
            return 1.0
        max_lum = -1.0; min_lum = 10.0
        for i in range(n):
            lum = _relative_luminance_numba(pal[i, 0], pal[i, 1], pal[i, 2])
            if lum > max_lum:
                max_lum = lum
            if lum < min_lum:
                min_lum = lum
        return (max_lum + 0.05) / (min_lum + 0.05)

    M_PROTO = np.array([
        [0.152286, 1.052583, -0.204868],
        [0.114503, 0.786281, 0.099216],
        [-0.003882, -0.048116, 1.051998],
    ])
    M_DEUTO = np.array([
        [0.367322, 0.860646, -0.227968],
        [0.280085, 0.672501, 0.047413],
        [-0.011820, 0.042940, 0.968881],
    ])
    M_TRITA = np.array([
        [1.255528, -0.076749, -0.178779],
        [-0.078411, 0.930809, 0.147602],
        [-0.004733, 0.691367, 0.313366],
    ])

    @njit(cache=True)
    def _apply_matrix_numba(pal, matrix):
        res = np.empty_like(pal)
        for i in range(pal.shape[0]):
            r, g, b = pal[i, 0], pal[i, 1], pal[i, 2]
            res[i, 0] = min(255.0, max(0.0, matrix[0, 0] * r + matrix[0, 1] * g + matrix[0, 2] * b))
            res[i, 1] = min(255.0, max(0.0, matrix[1, 0] * r + matrix[1, 1] * g + matrix[1, 2] * b))
            res[i, 2] = min(255.0, max(0.0, matrix[2, 0] * r + matrix[2, 1] * g + matrix[2, 2] * b))
        return res

    def simulate_colorblind(palette_list: List[Tuple[int, int, int]],
                            cb_type: str = "proto") -> List[Tuple[int, int, int]]:
        if not palette_list:
            return []
        pal = np.array(palette_list, dtype=np.float64)
        if cb_type == "proto":
            res = _apply_matrix_numba(pal, M_PROTO)
        elif cb_type == "deuto":
            res = _apply_matrix_numba(pal, M_DEUTO)
        elif cb_type == "trita":
            res = _apply_matrix_numba(pal, M_TRITA)
        else:
            return palette_list
        return [tuple(int(c) for c in row) for row in res]

else:
    # ── Pure-Python fallbacks ────────────────────────────────
    def _relative_luminance_py(r: int, g: int, b: int) -> float:
        srgb = [r / 255, g / 255, b / 255]
        lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb]
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

    def calculate_metadata_numba(pal):
        if not pal:
            return 0.0, 0.0, 0
        brightness = 0.0; max_lum = -1.0; min_lum = 256.0
        sum_r = 0.0; sum_g = 0.0; sum_b = 0.0
        for r, g, b in pal:
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            brightness += lum
            max_lum = max(max_lum, lum)
            min_lum = min(min_lum, lum)
            sum_r += r; sum_g += g; sum_b += b
        brightness /= len(pal)
        contrast = max_lum - min_lum
        dominant = 0
        if sum_g >= sum_r and sum_g >= sum_b:
            dominant = 1
        elif sum_b >= sum_r and sum_b >= sum_g:
            dominant = 2
        return brightness, contrast, dominant

    def palette_wcag_contrast_numba(pal):
        if len(pal) < 2:
            return 1.0
        lums = [_relative_luminance_py(*c) for c in pal]
        return (max(lums) + 0.05) / (min(lums) + 0.05)

    def simulate_colorblind(palette_list: List[Tuple[int, int, int]],
                            cb_type: str = "proto") -> List[Tuple[int, int, int]]:
        return palette_list


# ══════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════

def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02X}{g:02X}{b:02X}"


def hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    if len(h) == 3:
        h = h[0] * 2 + h[1] * 2 + h[2] * 2
    if len(h) != 6 or not all(c in "0123456789abcdefABCDEF" for c in h):
        raise ValueError(f"Invalid hex color: #{h}")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def calculate_metadata(palette: List[Tuple[int, int, int]]) -> Tuple[float, float, str]:
    """Return (brightness, contrast, dominant_channel)."""
    if not palette:
        return 0.0, 0.0, "R"
    if NUMBA_AVAILABLE:
        pal = np.array(palette, dtype=np.float64)
        b, c, d = calculate_metadata_numba(pal)
        return b, c, "RGB"[d]
    b, c, d = calculate_metadata_numba(palette)
    return b, c, "RGB"[d]


def palette_wcag_contrast(palette: List[Tuple[int, int, int]]) -> float:
    if len(palette) < 2:
        return 1.0
    if NUMBA_AVAILABLE:
        pal = np.array(palette, dtype=np.float64)
        return float(palette_wcag_contrast_numba(pal))
    return palette_wcag_contrast_numba(palette)


def rgb_to_hsv(r: int, g: int, b: int) -> Tuple[float, float, float]:
    r1, g1, b1 = r / 255.0, g / 255.0, b / 255.0
    mx, mn = max(r1, g1, b1), min(r1, g1, b1)
    df = mx - mn
    if mx == mn:
        h = 0
    elif mx == r1:
        h = (60 * ((g1 - b1) / df) + 360) % 360
    elif mx == g1:
        h = (60 * ((b1 - r1) / df) + 120) % 360
    else:
        h = (60 * ((r1 - g1) / df) + 240) % 360
    s = 0 if mx == 0 else df / mx
    return h, s, mx


def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
    h %= 360
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c
    if h < 60:
        r, g, b = c, x, 0
    elif h < 120:
        r, g, b = x, c, 0
    elif h < 180:
        r, g, b = 0, c, x
    elif h < 240:
        r, g, b = 0, x, c
    elif h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    return int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)


def rgb_to_hsl(r: int, g: int, b: int) -> Tuple[float, float, float]:
    r1, g1, b1 = r / 255.0, g / 255.0, b / 255.0
    mx, mn = max(r1, g1, b1), min(r1, g1, b1)
    l = (mx + mn) / 2
    if mx == mn:
        h = s = 0.0
    else:
        d = mx - mn
        s = d / (2.0 - mx - mn) if l > 0.5 else d / (mx + mn)
        if mx == r1:
            h = (g1 - b1) / d + (6 if g1 < b1 else 0)
        elif mx == g1:
            h = (b1 - r1) / d + 2
        else:
            h = (r1 - g1) / d + 4
        h /= 6
    return h * 360, s * 100, l * 100


def lerp_color(c1: Tuple[int, int, int],
               c2: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def extract_palette_from_image(qimg: QImage, num_colors: int = 5) -> List[Tuple[int, int, int]]:
    small = qimg.scaled(64, 64, Qt.AspectRatioMode.IgnoreAspectRatio,
                        Qt.TransformationMode.SmoothTransformation)
    pixels: List[Tuple[int, int, int]] = []
    for y in range(small.height()):
        for x in range(small.width()):
            c = small.pixelColor(x, y)
            if c.alpha() > 128:
                pixels.append((c.red(), c.green(), c.blue()))
    if not pixels:
        return []
    rounded_pixels = [(r // 32 * 32, g // 32 * 32, b // 32 * 32) for r, g, b in pixels]
    counts = Counter(rounded_pixels)
    top = counts.most_common(num_colors)
    return [(min(255, r + 16), min(255, g + 16), min(255, b + 16)) for (r, g, b), _ in top]


def extract_palette_kmeans(qimg: QImage, num_colors: int = 5,
                           max_iter: int = 20, seed: Optional[int] = None) -> List[Tuple[int, int, int]]:
    small = qimg.scaled(100, 100, Qt.AspectRatioMode.IgnoreAspectRatio,
                        Qt.TransformationMode.SmoothTransformation)
    pixels: List[List[int]] = []
    for y in range(small.height()):
        for x in range(small.width()):
            c = small.pixelColor(x, y)
            if c.alpha() > 128:
                pixels.append([c.red(), c.green(), c.blue()])
    if not pixels:
        return []
    if NUMBA_AVAILABLE:
        pixels_np = np.array(pixels, dtype=np.float64)
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(pixels_np), min(num_colors, len(pixels_np)), replace=False)
        centers = pixels_np[indices].copy()
        for _ in range(max_iter):
            dists = np.zeros((len(pixels_np), num_colors))
            for k in range(num_colors):
                diff = pixels_np - centers[k]
                dists[:, k] = np.sum(diff ** 2, axis=1)
            labels = np.argmin(dists, axis=1)
            new_centers = np.zeros_like(centers)
            for k in range(num_colors):
                mask = labels == k
                if np.any(mask):
                    new_centers[k] = pixels_np[mask].mean(axis=0)
                else:
                    new_centers[k] = centers[k]
            if np.allclose(centers, new_centers, atol=1.0):
                break
            centers = new_centers
        lums = 0.299 * centers[:, 0] + 0.587 * centers[:, 1] + 0.114 * centers[:, 2]
        centers = centers[np.argsort(lums)]
        return [(int(min(255, max(0, c[0]))), int(min(255, max(0, c[1]))),
                 int(min(255, max(0, c[2])))) for c in centers]
    return extract_palette_from_image(qimg, num_colors)


# ── Pack / Unpack ────────────────────────────────────────────

# BUG-FIX #2: HUGEINT is 128-bit → max 5 colours at 8 bpc.
MAX_PACK_COLORS = 128 // (3 * 8)  # = 5


def pack_palette(palette: List[Tuple[int, int, int]], bits_per_channel: int = 8) -> int:
    max_colors = 128 // (3 * bits_per_channel)
    if len(palette) > max_colors:
        raise ValueError(
            f"Cannot pack {len(palette)} colours into HUGEINT (max {max_colors} at {bits_per_channel} bpc). "
            "Reduce palette size or use fewer bits."
        )
    packed, shift = 0, 0
    for r, g, b in palette:
        r, g, b = max(0, min(r, 255)), max(0, min(g, 255)), max(0, min(b, 255))
        packed |= ((r << (2 * bits_per_channel)) | (g << bits_per_channel) | b) << shift
        shift += 3 * bits_per_channel
    return packed


def unpack_palette(packed: int, num_colors: int = 5, bits_per_channel: int = 8) -> List[Tuple[int, int, int]]:
    palette: List[Tuple[int, int, int]] = []
    mask = (1 << (3 * bits_per_channel)) - 1
    ch_mask = (1 << bits_per_channel) - 1
    for _ in range(num_colors):
        cb = packed & mask
        b_val = cb & ch_mask
        g_val = (cb >> bits_per_channel) & ch_mask
        r_val = (cb >> (2 * bits_per_channel)) & ch_mask
        palette.append((r_val, g_val, b_val))
        packed >>= 3 * bits_per_channel
    return palette


# ── Generators ───────────────────────────────────────────────

# BUG-FIX #7: use local Random instead of poisoning global state
def generate_random_palette(seed: int, num_colors: int = 5) -> List[Tuple[int, int, int]]:
    rng = random.Random(seed)
    return [(rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)) for _ in range(num_colors)]


def generate_harmony_palette(base_hue: float, harmony_type: str,
                             num_colors: int = 5, sat: float = 0.75,
                             val: float = 0.75) -> List[Tuple[int, int, int]]:
    colors: List[Tuple[int, int, int]] = []
    if harmony_type == "complementary":
        for i in range(num_colors):
            t = i / max(num_colors - 1, 1)
            h = (base_hue + t * 180) % 360
            s = min(sat * (0.7 + 0.3 * math.sin(t * math.pi)), 1.0)
            v = min(val * (0.5 + 0.5 * (1 - abs(2 * t - 1))), 1.0)
            colors.append(hsv_to_rgb(h, s, v))
    elif harmony_type == "analogous":
        for i in range(num_colors):
            t = i / max(num_colors - 1, 1)
            h = (base_hue - 30 + t * 60) % 360
            s = min(sat * (0.8 + 0.2 * math.sin(t * math.pi)), 1.0)
            v = min(val * (0.7 + 0.3 * math.sin(t * math.pi + 0.5)), 1.0)
            colors.append(hsv_to_rgb(h, s, v))
    elif harmony_type == "triadic":
        for i in range(num_colors):
            seg = i * 3 // num_colors
            h = (base_hue + seg * 120 + (i % 2) * 15) % 360
            s = min(sat * (0.7 + 0.3 * (i % 2)), 1.0)
            v = min(val * (0.6 + 0.4 * ((i + 1) % 2)), 1.0)
            colors.append(hsv_to_rgb(h, s, v))
    elif harmony_type == "split_complementary":
        angles = [0, 150, 210]
        for i in range(num_colors):
            h = (base_hue + angles[i % 3] + (i // 3) * 12) % 360
            s = min(sat * (0.7 + 0.3 * (i % 2)), 1.0)
            v = min(val * (0.6 + 0.4 * ((i + 1) % 2)), 1.0)
            colors.append(hsv_to_rgb(h, s, v))
    elif harmony_type == "monochromatic":
        for i in range(num_colors):
            t = i / max(num_colors - 1, 1)
            s = min(sat * (0.3 + 0.7 * t), 1.0)
            v = min(0.3 + 0.7 * (1 - t), 1.0)
            colors.append(hsv_to_rgb(base_hue, s, v))
    elif harmony_type == "tetradic":
        for i in range(num_colors):
            h = (base_hue + i * 90) % 360
            s = min(sat * (0.7 + 0.3 * math.sin(i * math.pi / 2)), 1.0)
            v = min(val * (0.6 + 0.4 * math.cos(i * math.pi / 3)), 1.0)
            colors.append(hsv_to_rgb(h, s, v))
    return colors[:num_colors]


def generate_variations(palette: List[Tuple[int, int, int]],
                        variation: str = "lighter",
                        strength: float = 0.3) -> List[Tuple[int, int, int]]:
    result: List[Tuple[int, int, int]] = []
    for r, g, b in palette:
        h, s, v = rgb_to_hsv(r, g, b)
        if variation == "lighter":
            v = min(1.0, v + strength * (1.0 - v))
        elif variation == "darker":
            v = max(0.0, v * (1.0 - strength))
        elif variation == "muted":
            s = max(0.0, s * (1.0 - strength))
        elif variation == "vivid":
            s = min(1.0, s + strength * (1.0 - s))
        elif variation == "pastel":
            s = max(0.0, s * 0.4)
            v = min(1.0, v + 0.3 * (1.0 - v))
        elif variation == "warm":
            h = (h + 15 * strength) % 360
        elif variation == "cool":
            h = (h - 15 * strength) % 360
        result.append(hsv_to_rgb(h, s, v))
    return result


def find_duplicate_colors(palette: List[Tuple[int, int, int]],
                          threshold: float = 15) -> List[Tuple[int, int, float]]:
    dupes: List[Tuple[int, int, float]] = []
    for i in range(len(palette)):
        for j in range(i + 1, len(palette)):
            r1, g1, b1 = palette[i]
            r2, g2, b2 = palette[j]
            dist = math.sqrt((r2 - r1) ** 2 + (g2 - g1) ** 2 + (b2 - b1) ** 2)
            if dist < threshold:
                dupes.append((i, j, dist))
    return dupes


def merge_palettes(*palettes: List[Tuple[int, int, int]],
                   mode: str = "concat",
                   max_colors: int = 12) -> List[Tuple[int, int, int]]:
    if mode == "concat":
        combined: List[Tuple[int, int, int]] = []
        for p in palettes:
            combined.extend(p)
        return combined[:max_colors]
    elif mode == "alternate":
        combined: List[Tuple[int, int, int]] = []
        iters = [iter(p) for p in palettes]
        while len(combined) < max_colors:
            for it in iters:
                try:
                    combined.append(next(it))
                except StopIteration:
                    pass
            if all(it is None for it in iters):
                break
        return combined[:max_colors]
    elif mode == "average":
        # BUG-FIX #15: use round() instead of truncating //
        n = min(len(p) for p in palettes) if palettes else 0
        result: List[Tuple[int, int, int]] = []
        for i in range(n):
            avg_r = round(sum(p[i][0] for p in palettes) / len(palettes))
            avg_g = round(sum(p[i][1] for p in palettes) / len(palettes))
            avg_b = round(sum(p[i][2] for p in palettes) / len(palettes))
            result.append((avg_r, avg_g, avg_b))
        return result
    return palettes[0] if palettes else []


def sort_palette(palette: List[Tuple[int, int, int]],
                 mode: str = "hue") -> List[Tuple[int, int, int]]:
    def key_hue(c: Tuple[int, int, int]) -> float:
        return rgb_to_hsv(*c)[0]

    def key_bright(c: Tuple[int, int, int]) -> float:
        return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]

    def key_sat(c: Tuple[int, int, int]) -> float:
        return rgb_to_hsv(*c)[1]

    return sorted(palette, key={"hue": key_hue, "brightness": key_bright,
                                "saturation": key_sat}.get(mode, key_hue))


# ══════════════════════════════════════════════════════════════
#  FILE I/O
# ══════════════════════════════════════════════════════════════

def parse_map_file(filepath: str) -> List[Tuple[int, int, int]]:
    palette: List[Tuple[int, int, int]] = []
    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line[0] in ("#", ";"):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
                        if 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255:
                            palette.append((r, g, b))
                    except ValueError:
                        continue
    except Exception as e:
        logger.error("Error reading %s: %s", filepath, e)
    return palette


def save_map_file(palette: List[Tuple[int, int, int]], filepath: str) -> bool:
    try:
        with open(filepath, "w") as f:
            f.write("# Generated by DuckPalette\n")
            for r, g, b in palette:
                f.write(f"{r:3d} {g:3d} {b:3d}\n")
        return True
    except Exception as e:
        logger.error("Error writing %s: %s", filepath, e)
        return False


# BUG-FIX #12: ASE parser — use block-data buffer + bounds checking + UTF-16 name
def parse_ase_file(filepath: str) -> List[Tuple[int, int, int]]:
    palette: List[Tuple[int, int, int]] = []
    try:
        with open(filepath, "rb") as f:
            sig = f.read(4)
            if sig != b"ASEF":
                return palette
            f.read(4)  # version
            n_blocks = struct.unpack(">I", f.read(4))[0]
            for _ in range(n_blocks):
                btype = struct.unpack(">H", f.read(2))[0]
                blen = struct.unpack(">I", f.read(4))[0]
                block_data = f.read(blen)
                if btype == 0x0001 and len(block_data) >= 6:
                    offset = 0
                    name_len = struct.unpack(">H", block_data[offset:offset + 2])[0]
                    offset += 2
                    # ASE names are UTF-16-BE → name_len chars × 2 bytes each
                    offset += name_len * 2
                    if offset + 4 <= len(block_data):
                        color_model = block_data[offset:offset + 4].decode(
                            "ascii", errors="replace").strip("\x00")
                        offset += 4
                        if color_model == "RGB " and offset + 12 <= len(block_data):
                            r, g, b = struct.unpack(">fff", block_data[offset:offset + 12])
                            palette.append((
                                min(255, max(0, int(r * 255))),
                                min(255, max(0, int(g * 255))),
                                min(255, max(0, int(b * 255))),
                            ))
                        elif color_model == "CMYK" and offset + 16 <= len(block_data):
                            c, m, y, k = struct.unpack(">ffff", block_data[offset:offset + 16])
                            rv = int(255 * (1 - c) * (1 - k))
                            gv = int(255 * (1 - m) * (1 - k))
                            bv = int(255 * (1 - y) * (1 - k))
                            palette.append((
                                max(0, min(255, rv)),
                                max(0, min(255, gv)),
                                max(0, min(255, bv)),
                            ))
                        elif color_model == "Gray" and offset + 4 <= len(block_data):
                            gv = struct.unpack(">f", block_data[offset:offset + 4])[0]
                            v = min(255, max(0, int(gv * 255)))
                            palette.append((v, v, v))
                # Group start / end blocks are simply skipped (data already read)
    except Exception as e:
        logger.error("ASE parse error: %s", e)
    return palette


# BUG-FIX #5: don't treat comment lines as hex colours
def parse_hex_list(filepath: str) -> List[Tuple[int, int, int]]:
    palette: List[Tuple[int, int, int]] = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip().rstrip(",").rstrip(";")
            if line.startswith("#") and len(line.lstrip("#")) in (3, 6):
                try:
                    palette.append(hex_to_rgb(line))
                except ValueError:
                    continue
            elif len(line) == 6 and all(c in "0123456789abcdefABCDEF" for c in line):
                try:
                    palette.append(hex_to_rgb("#" + line))
                except ValueError:
                    continue
    return palette


def parse_csv_file(filepath: str) -> List[Tuple[int, int, int]]:
    palette: List[Tuple[int, int, int]] = []
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 3:
                try:
                    r, g, b = int(row[0]), int(row[1]), int(row[2])
                    if 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255:
                        palette.append((r, g, b))
                except ValueError:
                    pass
            elif len(row) >= 1:
                try:
                    palette.append(hex_to_rgb(row[0].strip()))
                except ValueError:
                    pass
    return palette


# ── Exporters ────────────────────────────────────────────────

def export_as_css(palette: List[Tuple[int, int, int]], filepath: str) -> bool:
    try:
        with open(filepath, "w") as f:
            f.write(":root {\n")
            for i, (r, g, b) in enumerate(palette):
                f.write(f"  --color-{i + 1}: {rgb_to_hex(r, g, b)};\n")
                f.write(f"  --color-{i + 1}-rgb: {r}, {g}, {b};\n")
            f.write("}\n")
        return True
    except Exception as e:
        logger.error("CSS export error: %s", e)
        return False


def export_as_json(palette: List[Tuple[int, int, int]], filepath: str) -> bool:
    try:
        with open(filepath, "w") as f:
            json.dump([{"hex": rgb_to_hex(*c), "rgb": list(c)} for c in palette], f, indent=2)
        return True
    except Exception as e:
        logger.error("JSON export error: %s", e)
        return False


# BUG-FIX #1: return True must be OUTSIDE the for-loop
def export_as_gpl(palette: List[Tuple[int, int, int]],
                  name: str, filepath: str) -> bool:
    try:
        with open(filepath, "w") as f:
            f.write("GIMP Palette\n")
            f.write(f"Name: {name}\n")
            f.write(f"Columns: {len(palette)}\n#\n")
            for r, g, b in palette:
                f.write(f"{r:3d} {g:3d} {b:3d}\t{rgb_to_hex(r, g, b)}\n")
        return True
    except Exception as e:
        logger.error("GPL export error: %s", e)
        return False


def export_as_scss(palette: List[Tuple[int, int, int]], filepath: str) -> bool:
    try:
        with open(filepath, "w") as f:
            f.write("// Generated by DuckPalette\n$palette: (\n")
            for i, (r, g, b) in enumerate(palette):
                f.write(f'  "{i + 1}": ({r}, {g}, {b}),\n')
            f.write(");\n\n")
            for i, (r, g, b) in enumerate(palette):
                f.write(f"$color-{i + 1}: rgb({r}, {g}, {b});\n")
        return True
    except Exception as e:
        logger.error("SCSS export error: %s", e)
        return False


def export_as_tailwind(palette: List[Tuple[int, int, int]], filepath: str) -> bool:
    try:
        with open(filepath, "w") as f:
            f.write("// tailwind.config.js\nmodule.exports = {\n  theme: {\n    extend: {\n      colors: {\n")
            for i, (r, g, b) in enumerate(palette):
                f.write(f'        "palette-{i + 1}": "rgb({r} {g} {b})",\n')
            f.write("      },\n    },\n  },\n};\n")
        return True
    except Exception as e:
        logger.error("Tailwind export error: %s", e)
        return False


# BUG-FIX #14: last SVG rectangle now fills to the right edge
def export_as_svg(palette: List[Tuple[int, int, int]], filepath: str) -> bool:
    try:
        n = len(palette)
        w, h = max(400, n * 80), 120
        with open(filepath, "w") as f:
            f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">\n')
            for i, (r, g, b) in enumerate(palette):
                x_start = i * (w // n)
                x_end = ((i + 1) * w) // n if i < n - 1 else w
                sw = x_end - x_start
                f.write(f'  <rect x="{x_start}" y="0" width="{sw}" height="{h}" '
                        f'fill="{rgb_to_hex(r, g, b)}"/>\n')
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                tc = "#000" if lum > 128 else "#fff"
                f.write(f'  <text x="{x_start + sw // 2}" y="{h // 2}" fill="{tc}" '
                        f'font-size="11" text-anchor="middle" '
                        f'dominant-baseline="middle">{rgb_to_hex(r, g, b)}</text>\n')
            f.write("</svg>\n")
        return True
    except Exception as e:
        logger.error("SVG export error: %s", e)
        return False


def export_as_android_xml(palette: List[Tuple[int, int, int]], filepath: str) -> bool:
    try:
        with open(filepath, "w") as f:
            f.write('<?xml version="1.0" encoding="utf-8"?>\n<resources>\n')
            for i, (r, g, b) in enumerate(palette):
                f.write(f'    <color name="palette_color_{i + 1}">{rgb_to_hex(r, g, b)}</color>\n')
            f.write("</resources>\n")
        return True
    except Exception as e:
        logger.error("Android XML export error: %s", e)
        return False


def export_as_python(palette: List[Tuple[int, int, int]], filepath: str) -> bool:
    try:
        with open(filepath, "w") as f:
            f.write("# Generated by DuckPalette\n\nPALETTE = [\n")
            for r, g, b in palette:
                f.write(f"    ({r:3d}, {g:3d}, {b:3d}),  # {rgb_to_hex(r, g, b)}\n")
            f.write("]\n\nPALETTE_HEX = [\n")
            for r, g, b in palette:
                f.write(f'    "{rgb_to_hex(r, g, b)}",\n')
            f.write("]\n")
        return True
    except Exception as e:
        logger.error("Python export error: %s", e)
        return False


# ══════════════════════════════════════════════════════════════
#  DATABASE
# ══════════════════════════════════════════════════════════════

class PaletteDB:
    def __init__(self, db_file: str = "palettes.duckdb"):
        self.db_file = db_file
        self.conn = duckdb.connect(database=db_file, read_only=False)
        self._init_db()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

    def __del__(self):
        self.close()

    def _safe_rollback(self):
        """Clear an aborted transaction so subsequent queries can proceed."""
        try:
            self.conn.rollback()
        except Exception:
            pass

    def _column_exists(self, table_name: str, column_name: str) -> bool:
        """Check whether a column already exists in a table."""
        try:
            result = self.conn.execute(
                "SELECT COUNT(*) FROM information_schema.columns "
                "WHERE table_name=? AND column_name=?",
                [table_name.lower(), column_name.lower()],
            ).fetchone()
            return result[0] > 0
        except Exception:
            self._safe_rollback()
            return False

    def _init_db(self):
        # Create sequence — safe to re-run
        try:
            self.conn.execute("CREATE SEQUENCE IF NOT EXISTS palette_id_seq START 1")
        except Exception:
            self._safe_rollback()

        # Create table with ALL known columns up front
        try:
            self.conn.execute('''CREATE TABLE IF NOT EXISTS palettes (
                id INTEGER PRIMARY KEY DEFAULT nextval('palette_id_seq'),
                name TEXT, seed BIGINT, num_colors INTEGER,
                bits_per_channel INTEGER, packed_palette HUGEINT,
                brightness DOUBLE, contrast DOUBLE, dominant TEXT,
                favorite BOOLEAN DEFAULT FALSE, tags TEXT DEFAULT '')''')
        except Exception:
            self._safe_rollback()

        # Migrate older tables that were created without `favorite` / `tags`
        for col_name, col_def in [
            ("favorite", "BOOLEAN DEFAULT FALSE"),
            ("tags", "TEXT DEFAULT ''"),
        ]:
            if not self._column_exists("palettes", col_name):
                try:
                    self.conn.execute(
                        f"ALTER TABLE palettes ADD COLUMN {col_name} {col_def}")
                except duckdb.CatalogException:
                    self._safe_rollback()  # another process added it first
                except duckdb.Error:
                    self._safe_rollback()

        # Sync sequence with existing max id so next INSERT won't collide
        try:
            row = self.conn.execute(
                "SELECT MAX(id) FROM palettes").fetchone()
            if row and row[0] is not None:
                self.conn.execute(
                    f"ALTER SEQUENCE palette_id_seq RESTART WITH {row[0] + 1}")
        except Exception:
            self._safe_rollback()

    # ── CRUD Methods ────────────────────────────────────────

    def insert_palette(self, palette: List[Tuple[int, int, int]],
                       name: str = "User Palette",
                       pid: Optional[int] = None,
                       seed: Optional[int] = None) -> Optional[int]:
        if not palette:
            return None
        b, c, d = calculate_metadata(palette)
        packed = pack_palette(palette)
        if pid is None:
            result = self.conn.execute(
                '''INSERT INTO palettes
                   (name,seed,num_colors,bits_per_channel,packed_palette,brightness,contrast,dominant)
                   VALUES (?,?,?,?,?,?,?,?) RETURNING id''',
                [name, seed, len(palette), 8, packed, b, c, d],
            )
            pid = result.fetchone()[0]
        else:
            self.conn.execute(
                '''INSERT INTO palettes
                   (id,name,seed,num_colors,bits_per_channel,packed_palette,brightness,contrast,dominant)
                   VALUES (?,?,?,?,?,?,?,?,?)''',
                [pid, name, seed, len(palette), 8, packed, b, c, d],
            )
        return pid

    def update_palette(self, pid: int, palette: List[Tuple[int, int, int]],
                       name: str) -> bool:
        if not palette:
            return False
        b, c, d = calculate_metadata(palette)
        packed = pack_palette(palette)
        self.conn.execute(
            '''UPDATE palettes
               SET name=?,packed_palette=?,num_colors=?,brightness=?,contrast=?,dominant=?
               WHERE id=?''',
            [name, packed, len(palette), b, c, d, pid],
        )
        return True

    def rename_palette(self, pid: int, name: str):
        self.conn.execute("UPDATE palettes SET name=? WHERE id=?", [name, pid])

    def delete_palette(self, pid: int):
        self.conn.execute("DELETE FROM palettes WHERE id=?", [pid])

    def toggle_favorite(self, pid: int):
        self.conn.execute(
            "UPDATE palettes SET favorite = NOT favorite WHERE id=?", [pid])

    def set_tags(self, pid: int, tags_str: str):
        self.conn.execute(
            "UPDATE palettes SET tags=? WHERE id=?", [tags_str, pid])

    # BUG-FIX #3: don't use seed as pid
    def generate_and_insert(self, seed: int, num_colors: int = 5) -> int:
        palette = generate_random_palette(seed, num_colors)
        return self.insert_palette(palette, name=f"Gen_{seed}", seed=seed)

    def search(self, *, min_bright: Optional[float] = None,
               max_bright: Optional[float] = None,
               dominant: Optional[str] = None,
               favorite_only: bool = False,
               tag: Optional[str] = None,
               name_query: Optional[str] = None,
               limit: int = 100) -> list:
        q = ('SELECT id,name,brightness,contrast,dominant,num_colors,'
             'packed_palette,tags,favorite FROM palettes WHERE 1=1')
        p: list = []
        if min_bright is not None:
            q += ' AND brightness>=?'
            p.append(min_bright)
        if max_bright is not None:
            q += ' AND brightness<=?'
            p.append(max_bright)
        if dominant:
            q += ' AND dominant=?'
            p.append(dominant)
        if favorite_only:
            q += ' AND favorite=TRUE'
        if tag:
            q += ' AND tags LIKE ?'
            p.append(f'%{tag}%')
        if name_query:
            q += ' AND name LIKE ?'
            p.append(f'%{name_query}%')
        q += ' ORDER BY id DESC LIMIT ?'
        p.append(limit)
        return self.conn.execute(q, p).fetchall()

    def get_palette_by_id(self, pid: int) -> Optional[List[Tuple[int, int, int]]]:
        row = self.conn.execute(
            "SELECT packed_palette,num_colors,bits_per_channel FROM palettes WHERE id=?",
            [pid],
        ).fetchone()
        return unpack_palette(row[0], row[1], row[2]) if row else None

    def get_name_by_id(self, pid: int) -> Optional[str]:
        row = self.conn.execute(
            "SELECT name FROM palettes WHERE id=?", [pid]).fetchone()
        return row[0] if row else None

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM palettes").fetchone()[0]


# ══════════════════════════════════════════════════════════════
#  CONTROLLER
# ══════════════════════════════════════════════════════════════

class PaletteController:
    def __init__(self):
        self.db = PaletteDB()

    def import_file_to_db(self, filepath: str):
        palette: List[Tuple[int, int, int]] = []
        low = filepath.lower()
        if low.endswith('.ase'):
            palette = parse_ase_file(filepath)
        elif low.endswith('.csv'):
            palette = parse_csv_file(filepath)
        elif low.endswith('.hex') or low.endswith('.txt'):
            palette = parse_hex_list(filepath)
        else:
            palette = parse_map_file(filepath)
        if palette:
            name = os.path.basename(filepath)
            pid = self.db.insert_palette(palette, name=name)
            return pid, palette, name
        return None, [], None

    def export_palette_data(self, palette: List[Tuple[int, int, int]],
                            name: str, filepath: str, fmt: str = "map") -> bool:
        if not palette:
            return False
        exporters = {
            "css": export_as_css,
            "json": export_as_json,
            "gpl": lambda p, f: export_as_gpl(p, name, f),
            "scss": export_as_scss,
            "tailwind": export_as_tailwind,
            "svg": export_as_svg,
            "xml": export_as_android_xml,
            "py": export_as_python,
        }
        if fmt in exporters:
            return exporters[fmt](palette, filepath)
        return save_map_file(palette, filepath)

    def create_palette(self, palette_data: List[Tuple[int, int, int]],
                       name: str = "New Palette") -> Optional[int]:
        return self.db.insert_palette(palette_data, name=name)

    def update_palette(self, pid: int,
                       palette_data: List[Tuple[int, int, int]],
                       name: str) -> bool:
        return self.db.update_palette(pid, palette_data, name)

    def rename_palette(self, pid: int, name: str):
        self.db.rename_palette(pid, name)

    def delete_palette(self, pid: int):
        self.db.delete_palette(pid)

    def generate_new_palettes(self, count: int = 10):
        seed = int(time.time())
        for i in range(count):
            self.db.generate_and_insert(seed=seed + i)

    def search_palettes(self, **kwargs):
        return self.db.search(**kwargs)

    def get_palette(self, pid: int):
        return self.db.get_palette_by_id(pid)

    def calculate_metadata(self, palette: List[Tuple[int, int, int]]):
        return calculate_metadata(palette)


controller = PaletteController()


# ══════════════════════════════════════════════════════════════
#  SETTINGS
# ══════════════════════════════════════════════════════════════

class AppSettings:
    _path = os.path.join(os.path.expanduser("~"), ".duckpalette_settings.json")
    _defaults: Dict[str, Any] = {
        "window_geometry": None,
        "splitter_sizes": [320, 780],
        "recent_files": [],
    }

    @classmethod
    def load(cls) -> Dict[str, Any]:
        try:
            with open(cls._path, "r") as f:
                data = json.load(f)
            merged = dict(cls._defaults)
            merged.update(data)
            return merged
        except Exception:
            return dict(cls._defaults)

    @classmethod
    def save(cls, settings: Dict[str, Any]):
        try:
            with open(cls._path, "w") as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save settings: %s", e)

    @classmethod
    def add_recent_file(cls, path: str):
        settings = cls.load()
        recent = settings.get("recent_files", [])
        if path in recent:
            recent.remove(path)
        recent.insert(0, path)
        settings["recent_files"] = recent[:20]
        cls.save(settings)


# ══════════════════════════════════════════════════════════════
#  GUI CUSTOM WIDGETS
# ══════════════════════════════════════════════════════════════

def create_palette_pixmap(palette: List[Tuple[int, int, int]],
                          w: int = 80, h: int = 20) -> QPixmap:
    px = QPixmap(w, h)
    p = QPainter(px)
    p.fillRect(0, 0, w, h, QColor("#181825"))
    if palette:
        n = len(palette)
        x = 0
        for i, col in enumerate(palette):
            nx = ((i + 1) * w) // n
            p.fillRect(x, 0, nx - x, h, QColor(*col))
            x = nx
    p.end()
    return px


class ToastWidget(QFrame):
    def __init__(self, message: str, duration: int = 2500, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.ToolTip)
        lay = QVBoxLayout(self)
        label = QLabel(message)
        label.setStyleSheet("color:#cdd6f4; padding:8px 16px;")
        lay.addWidget(label)
        self.setStyleSheet(
            "background:#313244; border:1px solid #89b4fa; border-radius:8px;")
        self.adjustSize()
        # BUG-FIX #11: use global coordinates for positioning
        if parent:
            center = parent.rect().center()
            global_pt = parent.mapToGlobal(center)
            self.move(
                global_pt.x() - self.width() // 2,
                global_pt.y() + parent.height() // 2 - self.height() - 40,
            )
        self.show()
        QTimer.singleShot(duration, self.close)


class ScreenColorPicker(QWidget):
    color_picked = Signal(tuple)
    cancelled = Signal()

    def __init__(self):
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool,
        )
        self.setWindowState(Qt.WindowState.WindowFullScreen)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._update_color)
        self._current_color = (0, 0, 0)

    def start(self):
        self.show()
        self._timer.start()

    # BUG-FIX #4: use screen-relative coordinates for grabWindow
    def _update_color(self):
        pos = QCursor.pos()
        screen = QApplication.screenAt(pos)
        if screen:
            geo = screen.geometry()
            img = screen.grabWindow(
                0, pos.x() - geo.x(), pos.y() - geo.y(), 1, 1
            ).toImage()
            c = img.pixelColor(0, 0)
            self._current_color = (c.red(), c.green(), c.blue())

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(0, 0, 0, 1))
        pos = QCursor.pos()
        r, g, b = self._current_color
        hx = rgb_to_hex(r, g, b)
        p.setPen(Qt.GlobalColor.white)
        p.drawRect(pos.x() - 20, pos.y() - 30, 120, 24)
        p.fillRect(pos.x() - 19, pos.y() - 29, 118, 22,
                   QColor(*self._current_color))
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        p.setPen(QColor(0, 0, 0) if lum > 128 else QColor(255, 255, 255))
        p.drawText(pos.x() - 15, pos.y() - 13, f"{hx} ({r},{g},{b})")

    def mousePressEvent(self, e):
        self._timer.stop()
        self.close()
        if e.button() == Qt.MouseButton.LeftButton:
            self.color_picked.emit(self._current_color)
        else:
            self.cancelled.emit()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape:
            self._timer.stop()
            self.close()
            self.cancelled.emit()


class ContrastCheckerWidget(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Contrast Checker", parent)
        lay = QVBoxLayout(self)

        fg_row = QHBoxLayout()
        self.fg_swatch = QLabel()
        self.fg_swatch.setFixedSize(40, 28)
        self.fg_swatch.setStyleSheet("background:#ffffff; border-radius:4px;")
        self.fg_hex = QLineEdit("#FFFFFF")
        self.fg_hex.setMaximumWidth(90)
        self.fg_hex.editingFinished.connect(self._recalc)
        fg_btn = QPushButton("Pick")
        fg_btn.clicked.connect(lambda: self._pick("fg"))
        fg_row.addWidget(QLabel("FG:"))
        fg_row.addWidget(self.fg_swatch)
        fg_row.addWidget(self.fg_hex)
        fg_row.addWidget(fg_btn)
        lay.addLayout(fg_row)

        bg_row = QHBoxLayout()
        self.bg_swatch = QLabel()
        self.bg_swatch.setFixedSize(40, 28)
        self.bg_swatch.setStyleSheet("background:#000000; border-radius:4px;")
        self.bg_hex = QLineEdit("#000000")
        self.bg_hex.setMaximumWidth(90)
        self.bg_hex.editingFinished.connect(self._recalc)
        bg_btn = QPushButton("Pick")
        bg_btn.clicked.connect(lambda: self._pick("bg"))
        bg_row.addWidget(QLabel("BG:"))
        bg_row.addWidget(self.bg_swatch)
        bg_row.addWidget(self.bg_hex)
        bg_row.addWidget(bg_btn)
        lay.addLayout(bg_row)

        self.sample = QLabel("Sample Text  Aa Bb Cc  123")
        self.sample.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sample.setMinimumHeight(48)
        self.sample.setFont(QFont("Segoe UI", 16))
        lay.addWidget(self.sample)

        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("font-size:18px; font-weight:bold;")
        lay.addWidget(self.result_label)

        self.detail_label = QLabel()
        self.detail_label.setStyleSheet("color:#a6adc8; font-size:11px;")
        lay.addWidget(self.detail_label)

        swap_btn = QPushButton("⇄ Swap FG/BG")
        swap_btn.clicked.connect(self._swap)
        lay.addWidget(swap_btn)

        self._fg = (255, 255, 255)
        self._bg = (0, 0, 0)
        self._recalc()

    def _pick(self, which: str):
        cur = QColor(*(self._fg if which == "fg" else self._bg))
        c = QColorDialog.getColor(cur, self)
        if c.isValid():
            rgb = (c.red(), c.green(), c.blue())
            if which == "fg":
                self._fg = rgb
            else:
                self._bg = rgb
            self._update_swatches()
            self._recalc()

    def _swap(self):
        self._fg, self._bg = self._bg, self._fg
        self._update_swatches()
        self._recalc()

    def _update_swatches(self):
        self.fg_hex.setText(rgb_to_hex(*self._fg))
        self.bg_hex.setText(rgb_to_hex(*self._bg))
        self.fg_swatch.setStyleSheet(
            f"background:{rgb_to_hex(*self._fg)}; border-radius:4px;")
        self.bg_swatch.setStyleSheet(
            f"background:{rgb_to_hex(*self._bg)}; border-radius:4px;")

    # BUG-FIX #8: visual feedback on invalid hex instead of silent return
    def _recalc(self):
        ok_fg = ok_bg = True
        try:
            self._fg = hex_to_rgb(self.fg_hex.text())
            self.fg_hex.setStyleSheet("")
        except ValueError:
            self.fg_hex.setStyleSheet("border:1px solid #f38ba8;")
            ok_fg = False
        try:
            self._bg = hex_to_rgb(self.bg_hex.text())
            self.bg_hex.setStyleSheet("")
        except ValueError:
            self.bg_hex.setStyleSheet("border:1px solid #f38ba8;")
            ok_bg = False
        if not ok_fg or not ok_bg:
            return

        fg_hex = rgb_to_hex(*self._fg)
        bg_hex = rgb_to_hex(*self._bg)
        self.sample.setStyleSheet(
            f"color:{fg_hex}; background:{bg_hex}; padding:8px; border-radius:4px;")

        if NUMBA_AVAILABLE:
            l1 = float(_relative_luminance_numba(*self._fg))
            l2 = float(_relative_luminance_numba(*self._bg))
        else:
            l1 = _relative_luminance_py(*self._fg)
            l2 = _relative_luminance_py(*self._bg)

        ratio = (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)
        if ratio >= 7:
            rating, color = "AAA ✓", "#a6e3a1"
        elif ratio >= 4.5:
            rating, color = "AA ✓", "#f9e2af"
        elif ratio >= 3:
            rating, color = "AA Large ⚠", "#fab387"
        else:
            rating, color = "Fail ✗", "#f38ba8"
        self.result_label.setText(f"{ratio:.2f}:1  {rating}")
        self.result_label.setStyleSheet(
            f"font-size:18px; font-weight:bold; color:{color};")
        self.detail_label.setText(
            f"Normal text: {'Pass' if ratio >= 4.5 else 'Fail'} AA / "
            f"{'Pass' if ratio >= 7 else 'Fail'} AAA\n"
            f"Large text:  {'Pass' if ratio >= 3 else 'Fail'} AA / "
            f"{'Pass' if ratio >= 4.5 else 'Fail'} AAA"
        )

    def set_colors(self, fg: Tuple[int, int, int], bg: Tuple[int, int, int]):
        self._fg, self._bg = fg, bg
        self._update_swatches()
        self._recalc()


class ColorWheelWidget(QWidget):
    color_selected = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.palette: List[Tuple[int, int, int]] = []
        self.setMinimumSize(180, 180)
        self.setMaximumSize(220, 220)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_palette(self, pal: List[Tuple[int, int, int]]):
        self.palette = pal
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx, cy = self.width() // 2, self.height() // 2
        radius = min(cx, cy) - 12
        for angle in range(360):
            r, g, b = hsv_to_rgb(angle, 0.8, 0.8)
            pen = QPen(QColor(r, g, b), 8)
            p.setPen(pen)
            rad = math.radians(angle - 90)
            x1 = cx + int((radius - 4) * math.cos(rad))
            y1 = cy + int((radius - 4) * math.sin(rad))
            x2 = cx + int((radius + 4) * math.cos(rad))
            y2 = cy + int((radius + 4) * math.sin(rad))
            p.drawLine(x1, y1, x2, y2)
        for i, (r, g, b) in enumerate(self.palette):
            h, s, v = rgb_to_hsv(r, g, b)
            rad = math.radians(h - 90)
            dist = radius * (0.3 + 0.6 * (1.0 - s))
            mx = cx + int(dist * math.cos(rad))
            my = cy + int(dist * math.sin(rad))
            p.setPen(QPen(QColor(255, 255, 255), 2))
            p.setBrush(QBrush(QColor(r, g, b)))
            p.drawEllipse(mx - 7, my - 7, 14, 14)
            p.setPen(QColor("#cdd6f4"))
            font = p.font()
            font.setPixelSize(9)
            p.setFont(font)
            p.drawText(mx + 10, my + 4, f"{i + 1}")
        p.end()

    # BUG-FIX #20: use float-precision position
    def mousePressEvent(self, e):
        cx, cy = self.width() / 2.0, self.height() / 2.0
        dx = e.position().x() - cx
        dy = e.position().y() - cy
        angle = (math.degrees(math.atan2(dy, dx)) + 90) % 360
        self.color_selected.emit(angle)


class PalettePreviewWidget(QFrame):
    color_clicked = Signal(int)

    def __init__(self):
        super().__init__()
        self.palette: List[Tuple[int, int, int]] = []
        self.hover_idx = -1
        self.selected_idx = -1
        self.setMinimumHeight(56)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFrameShape(QFrame.Shape.StyledPanel)

    def set_palette(self, pal: List[Tuple[int, int, int]]):
        self.palette = pal
        self.update()

    def set_selected(self, idx: int):
        self.selected_idx = idx
        self.update()

    def _idx_at(self, x: int) -> int:
        if not self.palette:
            return -1
        n = len(self.palette)
        w = self.width()
        for i in range(n):
            s = (i * w) // n
            e = ((i + 1) * w) // n
            if s <= x < e:
                return i
        return -1

    def mouseMoveEvent(self, e):
        idx = self._idx_at(int(e.position().x()))
        if idx != self.hover_idx:
            self.hover_idx = idx
            self.update()
        if 0 <= idx < len(self.palette):
            r, g, b = self.palette[idx]
            QToolTip.showText(
                e.globalPosition().toPoint(),
                f"{rgb_to_hex(r, g, b)}\nRGB({r}, {g}, {b})",
            )
        else:
            QToolTip.hideText()

    def mousePressEvent(self, e):
        idx = self._idx_at(int(e.position().x()))
        if idx >= 0:
            self.color_clicked.emit(idx)

    def leaveEvent(self, e):
        self.hover_idx = -1
        self.update()
        QToolTip.hideText()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect()
        p.fillRect(r, QColor("#181825"))
        if not self.palette:
            p.setPen(QColor("#6c7086"))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, "No colors")
            return
        n = len(self.palette)
        w = r.width()
        h = r.height()
        x = 0
        for i, col in enumerate(self.palette):
            cr, cg, cb = col if isinstance(col, tuple) else (0, 0, 0)
            qc = QColor(cr, cg, cb)
            nx = ((i + 1) * w) // n
            p.fillRect(x, 0, nx - x, h, qc)
            if i == self.hover_idx:
                p.fillRect(x, 0, nx - x, h, QColor(255, 255, 255, 35))
            if i == self.selected_idx:
                p.setPen(QPen(QColor("#89b4fa"), 2))
                p.drawRect(x + 1, 1, nx - x - 2, h - 2)
            if i < n - 1:
                p.setPen(QPen(QColor(0, 0, 0, 50), 1))
                p.drawLine(nx, 0, nx, h)
            lum = 0.299 * cr + 0.587 * cg + 0.114 * cb
            p.setPen(
                QColor(0, 0, 0, 180) if lum > 128 else QColor(255, 255, 255, 200))
            font = p.font()
            font.setPixelSize(10)
            p.setFont(font)
            p.drawText(x, 0, nx - x, h, Qt.AlignmentFlag.AlignCenter,
                       rgb_to_hex(cr, cg, cb))
            x = nx


class WelcomeWidget(QWidget):
    open_requested = Signal()
    new_requested = Signal()
    generate_requested = Signal()

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon = QLabel("🦆 DuckPalette")
        icon.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet("color:#89b4fa;")
        lay.addWidget(icon)
        sub = QLabel("Color palette management made easy")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setStyleSheet("color:#6c7086; font-size:14px; margin-bottom:20px;")
        lay.addWidget(sub)
        lay.addSpacing(20)
        for text, sig in [
            ("📂 Open File", self.open_requested),
            ("✨ New Empty Palette", self.new_requested),
            ("🎲 Generate Random Palettes", self.generate_requested),
        ]:
            btn = QPushButton(text)
            btn.setFixedWidth(280)
            btn.setMinimumHeight(38)
            btn.clicked.connect(sig.emit)
            lay.addWidget(btn, alignment=Qt.AlignmentFlag.AlignCenter)
        lay.addStretch()


class HarmonyDialog(QWidget):
    palette_ready = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Color Harmony Generator")
        self.setFixedSize(420, 400)
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.base_btn = QPushButton()
        self.base_btn.setFixedSize(60, 28)
        self.base_color = (200, 100, 50)
        self._update_base_btn()
        self.base_btn.clicked.connect(self._pick_base)
        form.addRow("Base Color:", self.base_btn)
        self.base_hex = QLineEdit(rgb_to_hex(*self.base_color))
        self.base_hex.textChanged.connect(self._hex_changed)
        form.addRow("Hex:", self.base_hex)
        self.harmony_combo = QComboBox()
        self.harmony_combo.addItems([
            "complementary", "analogous", "triadic",
            "split_complementary", "monochromatic", "tetradic",
        ])
        form.addRow("Harmony:", self.harmony_combo)
        self.count_spin = QSpinBox()
        self.count_spin.setRange(2, 10)
        self.count_spin.setValue(5)
        form.addRow("Colors:", self.count_spin)
        self.sat_spin = QDoubleSpinBox()
        self.sat_spin.setRange(0.1, 1.0)
        self.sat_spin.setValue(0.75)
        self.sat_spin.setSingleStep(0.05)
        form.addRow("Saturation:", self.sat_spin)
        self.val_spin = QDoubleSpinBox()
        self.val_spin.setRange(0.1, 1.0)
        self.val_spin.setValue(0.75)
        self.val_spin.setSingleStep(0.05)
        form.addRow("Value:", self.val_spin)
        lay.addLayout(form)
        self.preview = PalettePreviewWidget()
        lay.addWidget(self.preview)
        btn_row = QHBoxLayout()
        gen_btn = QPushButton("Generate Preview")
        gen_btn.clicked.connect(self._generate)
        use_btn = QPushButton("Use This Palette")
        use_btn.clicked.connect(self._use)
        btn_row.addWidget(gen_btn)
        btn_row.addWidget(use_btn)
        lay.addLayout(btn_row)
        self._generated: List[Tuple[int, int, int]] = []
        self._generate()

    def _update_base_btn(self):
        self.base_btn.setStyleSheet(
            f"background-color:{rgb_to_hex(*self.base_color)}; "
            "border:1px solid #585b70; border-radius:4px;")

    def _pick_base(self):
        c = QColorDialog.getColor(QColor(*self.base_color), self)
        if c.isValid():
            self.base_color = (c.red(), c.green(), c.blue())
            self.base_hex.blockSignals(True)
            self.base_hex.setText(rgb_to_hex(*self.base_color))
            self.base_hex.blockSignals(False)
            self._update_base_btn()
            self._generate()

    def _hex_changed(self, txt: str):
        try:
            rgb = hex_to_rgb(txt)
            self.base_color = rgb
            self._update_base_btn()
            self._generate()
        except ValueError:
            pass

    def _generate(self):
        h, s, v = rgb_to_hsv(*self.base_color)
        self._generated = generate_harmony_palette(
            h, self.harmony_combo.currentText(),
            self.count_spin.value(), self.sat_spin.value(), self.val_spin.value())
        self.preview.set_palette(self._generated)

    def _use(self):
        if self._generated:
            self.palette_ready.emit(list(self._generated))


# ══════════════════════════════════════════════════════════════
#  PALETTE EDITOR  (completed from truncated original)
# ══════════════════════════════════════════════════════════════

class PaletteEditor(QWidget):
    changed = Signal()
    closed = Signal()

    def __init__(self, name: str = "Untitled", db_id: Optional[int] = None):
        super().__init__()
        self.name = name
        self.db_id = db_id
        self._tuples: List[Tuple[int, int, int]] = []
        self._modified = False
        self._undo_stack: List[List[Tuple[int, int, int]]] = []
        self._redo_stack: List[List[Tuple[int, int, int]]] = []
        self._adj_base: List[Tuple[int, int, int]] = []

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        # ── Header ──────────────────────────────────────────
        hdr = QHBoxLayout()
        self.name_edit = QLineEdit(name)
        self.name_edit.setStyleSheet(
            "font-weight:bold; font-size:15px; background:transparent; border:none;")
        self.name_edit.textChanged.connect(self._name_edited)
        hdr.addWidget(QLabel("Name:"))
        hdr.addWidget(self.name_edit)
        if db_id is not None:
            hdr.addWidget(QLabel(f" [ID:{db_id}]"))
        hdr.addStretch()
        self.close_btn = QPushButton("Close")
        self.close_btn.setFixedSize(60, 26)
        self.close_btn.clicked.connect(self.closed.emit)
        hdr.addWidget(self.close_btn)
        lay.addLayout(hdr)

        # ── Main splitter ──────────────────────────────────
        h_split = QSplitter(Qt.Orientation.Horizontal)

        # Left: color list
        left_w = QWidget()
        left_lay = QVBoxLayout(left_w)
        left_lay.setContentsMargins(0, 0, 0, 0)

        self.color_list = QListWidget()
        self.color_list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.color_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.color_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.color_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.color_list.customContextMenuRequested.connect(self._item_context_menu)
        self.color_list.itemDoubleClicked.connect(self._edit_item_dialog)
        self.color_list.currentRowChanged.connect(self._selection_changed)
        self.color_list.model().rowsMoved.connect(lambda *_: self._sync_from_list())
        left_lay.addWidget(self.color_list)

        # BUG-FIX #6: complete the truncated button row
        r1 = QHBoxLayout()
        self.add_btn = QPushButton("+ Add")
        self.add_btn.clicked.connect(self.add_color)
        self.rm_btn = QPushButton("− Remove")
        self.rm_btn.clicked.connect(self.remove_selected)
        self.rm_btn.setEnabled(False)
        r1.addWidget(self.add_btn)
        r1.addWidget(self.rm_btn)
        r1.addStretch()
        self.undo_btn = QPushButton("↶")
        self.undo_btn.setFixedSize(30, 26)
        self.undo_btn.clicked.connect(self.undo)
        self.undo_btn.setEnabled(False)
        self.redo_btn = QPushButton("↷")
        self.redo_btn.setFixedSize(30, 26)
        self.redo_btn.clicked.connect(self.redo)
        self.redo_btn.setEnabled(False)
        r1.addWidget(self.undo_btn)
        r1.addWidget(self.redo_btn)
        left_lay.addLayout(r1)

        h_split.addWidget(left_w)

        # Right: preview + tools
        right_w = QWidget()
        right_lay = QVBoxLayout(right_w)
        right_lay.setContentsMargins(0, 0, 0, 0)

        self.preview = PalettePreviewWidget()
        self.preview.color_clicked.connect(self._select_color_idx)
        right_lay.addWidget(self.preview)

        self.info_label = QLabel("Select a color")
        self.info_label.setStyleSheet("color:#a6adc8; font-size:12px;")
        self.info_label.setWordWrap(True)
        right_lay.addWidget(self.info_label)

        # Adjustments
        adj_grp = QGroupBox("Adjustments")
        adj_lay = QVBoxLayout(adj_grp)

        adj_row1 = QHBoxLayout()
        for text, var in [("Lighter", "lighter"), ("Darker", "darker"),
                          ("Muted", "muted"), ("Vivid", "vivid")]:
            btn = QPushButton(text)
            btn.clicked.connect(lambda checked, v=var: self._apply_variation(v))
            adj_row1.addWidget(btn)
        adj_lay.addLayout(adj_row1)

        adj_row2 = QHBoxLayout()
        for text, var in [("Pastel", "pastel"), ("Warm", "warm"), ("Cool", "cool")]:
            btn = QPushButton(text)
            btn.clicked.connect(lambda checked, v=var: self._apply_variation(v))
            adj_row2.addWidget(btn)
        adj_lay.addLayout(adj_row2)

        sort_row = QHBoxLayout()
        sort_row.addWidget(QLabel("Sort:"))
        for text, mode in [("Hue", "hue"), ("Bright", "brightness"), ("Sat", "saturation")]:
            btn = QPushButton(text)
            btn.clicked.connect(lambda checked, m=mode: self._sort_colors(m))
            sort_row.addWidget(btn)
        adj_lay.addLayout(sort_row)

        dupe_btn = QPushButton("Find Duplicates")
        dupe_btn.clicked.connect(self._find_duplicates)
        adj_lay.addWidget(dupe_btn)

        right_lay.addWidget(adj_grp)
        right_lay.addStretch()

        h_split.addWidget(right_w)
        h_split.setSizes([220, 420])
        lay.addWidget(h_split)

        # Improvement #27: keyboard shortcuts
        self.undo_shortcut = QKeySequence(QKeySequence.StandardKey.Undo)
        self.redo_shortcut = QKeySequence(QKeySequence.StandardKey.Redo)

    # ── Public API ─────────────────────────────────────────

    def set_palette(self, pal: List[Tuple[int, int, int]]):
        self._push_undo()
        self._tuples = list(pal)
        self._adj_base = list(pal)
        self._sync_list_from_tuples()
        self.preview.set_palette(self._tuples)
        self._modified = False

    def get_palette(self) -> List[Tuple[int, int, int]]:
        return list(self._tuples)

    def add_color(self, color: Optional[Tuple[int, int, int]] = None):
        if color is None:
            c = QColorDialog.getColor(QColor(128, 128, 128), self, "Pick Color")
            if not c.isValid():
                return
            color = (c.red(), c.green(), c.blue())
        self._push_undo()
        self._tuples.append(color)
        self._adj_base = list(self._tuples)
        self._sync_list_from_tuples()
        self._mark_modified()

    def remove_selected(self):
        idx = self.color_list.currentRow()
        if idx < 0:
            return
        self._push_undo()
        self._tuples.pop(idx)
        self._adj_base = list(self._tuples)
        self._sync_list_from_tuples()
        self._mark_modified()

    def undo(self):
        if not self._undo_stack:
            return
        self._redo_stack.append(list(self._tuples))
        self._tuples = self._undo_stack.pop()
        self._adj_base = list(self._tuples)
        self._sync_list_from_tuples()
        self._update_undo_buttons()
        self._mark_modified()

    def redo(self):
        if not self._redo_stack:
            return
        self._undo_stack.append(list(self._tuples))
        self._tuples = self._redo_stack.pop()
        self._adj_base = list(self._tuples)
        self._sync_list_from_tuples()
        self._update_undo_buttons()
        self._mark_modified()

    # ── Private helpers ────────────────────────────────────

    def _push_undo(self):
        self._undo_stack.append(list(self._tuples))
        self._redo_stack.clear()
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)
        self._update_undo_buttons()

    def _update_undo_buttons(self):
        self.undo_btn.setEnabled(bool(self._undo_stack))
        self.redo_btn.setEnabled(bool(self._redo_stack))

    def _name_edited(self, text: str):
        self.name = text
        self._mark_modified()

    def _mark_modified(self):
        self._modified = True
        self.changed.emit()

    def _sync_list_from_tuples(self):
        self.color_list.blockSignals(True)
        self.color_list.clear()
        for i, (r, g, b) in enumerate(self._tuples):
            item = QListWidgetItem(f"  {rgb_to_hex(r, g, b)}   ({r}, {g}, {b})")
            px = QPixmap(24, 24)
            px.fill(QColor(r, g, b))
            item.setIcon(px)
            item.setForeground(QBrush(QColor("#cdd6f4")))
            self.color_list.addItem(item)
        self.color_list.blockSignals(False)
        self.preview.set_palette(self._tuples)
        self._update_undo_buttons()

    def _sync_from_list(self):
        """Rebuild _tuples after drag-drop reorder."""
        new_tuples: List[Tuple[int, int, int]] = []
        for i in range(self.color_list.count()):
            item = self.color_list.item(i)
            text = item.text().strip()
            try:
                hex_part = text.split()[0]
                new_tuples.append(hex_to_rgb(hex_part))
            except (ValueError, IndexError):
                pass
        if new_tuples:
            self._tuples = new_tuples
            self._adj_base = list(self._tuples)
            self.preview.set_palette(self._tuples)
            self._mark_modified()

    def _selection_changed(self, row: int):
        self.rm_btn.setEnabled(row >= 0)
        if 0 <= row < len(self._tuples):
            r, g, b = self._tuples[row]
            h, s, v = rgb_to_hsv(r, g, b)
            hl, sl, ll = rgb_to_hsl(r, g, b)
            self.info_label.setText(
                f"Index {row}  •  {rgb_to_hex(r, g, b)}\n"
                f"RGB({r}, {g}, {b})\n"
                f"HSV({h:.1f}°, {s:.2f}, {v:.2f})\n"
                f"HSL({hl:.1f}°, {sl:.1f}%, {ll:.1f}%)")
            self.preview.set_selected(row)
        else:
            self.info_label.setText("Select a color")
            self.preview.set_selected(-1)

    def _select_color_idx(self, idx: int):
        self.color_list.setCurrentRow(idx)

    def _edit_item_dialog(self, item: QListWidgetItem):
        row = self.color_list.row(item)
        if row < 0 or row >= len(self._tuples):
            return
        old = self._tuples[row]
        c = QColorDialog.getColor(QColor(*old), self, "Edit Color")
        if c.isValid():
            self._push_undo()
            self._tuples[row] = (c.red(), c.green(), c.blue())
            self._adj_base = list(self._tuples)
            self._sync_list_from_tuples()
            self.color_list.setCurrentRow(row)
            self._mark_modified()

    def _item_context_menu(self, pos: QPoint):
        menu = QMenu(self)
        row = self.color_list.currentRow()

        copy_hex = menu.addAction("Copy Hex")
        copy_rgb = menu.addAction("Copy RGB")
        menu.addSeparator()
        replace_action = menu.addAction("Replace Color…")
        insert_before = menu.addAction("Insert Before…")
        insert_after = menu.addAction("Insert After…")
        menu.addSeparator()
        remove_action = menu.addAction("Remove")

        action = menu.exec(self.color_list.mapToGlobal(pos))
        if action is None:
            return

        clipboard = QApplication.clipboard()

        if action == copy_hex and 0 <= row < len(self._tuples):
            clipboard.setText(rgb_to_hex(*self._tuples[row]))
        elif action == copy_rgb and 0 <= row < len(self._tuples):
            r, g, b = self._tuples[row]
            clipboard.setText(f"rgb({r}, {g}, {b})")
        elif action == replace_action and 0 <= row < len(self._tuples):
            c = QColorDialog.getColor(QColor(*self._tuples[row]), self, "Replace")
            if c.isValid():
                self._push_undo()
                self._tuples[row] = (c.red(), c.green(), c.blue())
                self._adj_base = list(self._tuples)
                self._sync_list_from_tuples()
                self._mark_modified()
        elif action == insert_before:
            c = QColorDialog.getColor(QColor(128, 128, 128), self, "Insert Color")
            if c.isValid():
                self._push_undo()
                self._tuples.insert(max(0, row), (c.red(), c.green(), c.blue()))
                self._adj_base = list(self._tuples)
                self._sync_list_from_tuples()
                self._mark_modified()
        elif action == insert_after:
            c = QColorDialog.getColor(QColor(128, 128, 128), self, "Insert Color")
            if c.isValid():
                self._push_undo()
                idx = row + 1 if row >= 0 else len(self._tuples)
                self._tuples.insert(idx, (c.red(), c.green(), c.blue()))
                self._adj_base = list(self._tuples)
                self._sync_list_from_tuples()
                self._mark_modified()
        elif action == remove_action:
            self.remove_selected()

    def _apply_variation(self, variation: str):
        if not self._tuples:
            return
        self._push_undo()
        self._tuples = generate_variations(self._adj_base, variation, 0.3)
        self._sync_list_from_tuples()
        self._mark_modified()

    def _sort_colors(self, mode: str):
        if not self._tuples:
            return
        self._push_undo()
        self._tuples = sort_palette(self._tuples, mode)
        self._adj_base = list(self._tuples)
        self._sync_list_from_tuples()
        self._mark_modified()

    def _find_duplicates(self):
        dupes = find_duplicate_colors(self._tuples)
        if not dupes:
            QMessageBox.information(self, "Duplicates", "No duplicate colors found.")
            return
        lines: List[str] = []
        for i, j, dist in dupes:
            lines.append(
                f"  Colors {i} & {j}: {rgb_to_hex(*self._tuples[i])} ↔ "
                f"{rgb_to_hex(*self._tuples[j])}  (dist={dist:.1f})")
        QMessageBox.information(
            self, "Duplicates", "Near-duplicate colors found:\n" + "\n".join(lines))

    def keyPressEvent(self, e):
        # Improvement #27: Ctrl+Z / Ctrl+Y
        if e.matches(QKeySequence.StandardKey.Undo):
            self.undo()
        elif e.matches(QKeySequence.StandardKey.Redo):
            self.redo()
        elif e.key() == Qt.Key.Key_Delete:
            self.remove_selected()
        else:
            super().keyPressEvent(e)


# ══════════════════════════════════════════════════════════════
#  MAIN WINDOW  (completed from truncated original)
# ══════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🦆 DuckPalette")
        self.setMinimumSize(900, 600)
        self.resize(1100, 700)

        self._editors: Dict[int, PaletteEditor] = {}
        self._screen_picker: Optional[ScreenColorPicker] = None

        self._build_ui()
        self._build_menus()
        self._build_toolbar()
        self._refresh_palette_list()

        self.statusBar().showMessage("Ready")

    # ── UI Construction ────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_lay = QVBoxLayout(central)
        main_lay.setContentsMargins(0, 0, 0, 0)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel: palette browser
        left_w = QWidget()
        left_lay = QVBoxLayout(left_w)
        left_lay.setContentsMargins(6, 6, 6, 6)

        # Search row
        search_row = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search palettes…")
        self.search_edit.textChanged.connect(self._refresh_palette_list)
        search_row.addWidget(self.search_edit)
        left_lay.addLayout(search_row)

        # Filter row
        filter_row = QHBoxLayout()
        self.fav_check = QCheckBox("★ Favs")
        self.fav_check.toggled.connect(self._refresh_palette_list)
        filter_row.addWidget(self.fav_check)
        self.dom_combo = QComboBox()
        self.dom_combo.addItems(["Any", "R", "G", "B"])
        self.dom_combo.currentIndexChanged.connect(self._refresh_palette_list)
        filter_row.addWidget(QLabel("Dom:"))
        filter_row.addWidget(self.dom_combo)
        left_lay.addLayout(filter_row)

        # Palette list
        self.palette_list = QListWidget()
        self.palette_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.palette_list.customContextMenuRequested.connect(self._list_context_menu)
        self.palette_list.itemDoubleClicked.connect(self._open_palette_item)
        left_lay.addWidget(self.palette_list)

        # Palette list buttons
        pl_btn_row = QHBoxLayout()
        gen_btn = QPushButton("🎲 Generate")
        gen_btn.clicked.connect(self._generate_palettes)
        pl_btn_row.addWidget(gen_btn)
        imp_btn = QPushButton("📂 Import")
        imp_btn.clicked.connect(self._import_file)
        pl_btn_row.addWidget(imp_btn)
        left_lay.addLayout(pl_btn_row)

        self.splitter.addWidget(left_w)

        # Right panel: tab widget with editors + welcome
        self.right_stack = QTabWidget()
        self.right_stack.setTabsClosable(True)
        self.right_stack.tabCloseRequested.connect(self._close_tab)

        self.welcome = WelcomeWidget()
        self.welcome.open_requested.connect(self._import_file)
        self.welcome.new_requested.connect(self._new_palette)
        self.welcome.generate_requested.connect(self._generate_palettes)

        # Use a stacked approach: welcome when no tabs, tab widget otherwise
        self.right_container = QWidget()
        self.right_container_lay = QVBoxLayout(self.right_container)
        self.right_container_lay.setContentsMargins(0, 0, 0, 0)
        self.right_container_lay.addWidget(self.welcome)
        self.right_container_lay.addWidget(self.right_stack)
        self.right_stack.hide()

        self.splitter.addWidget(self.right_container)
        self.splitter.setSizes([300, 800])
        main_lay.addWidget(self.splitter)

    def _build_menus(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")
        self._add_action(file_menu, "New Palette", self._new_palette, "Ctrl+N")
        self._add_action(file_menu, "Open File…", self._import_file, "Ctrl+O")
        self._add_action(file_menu, "Save", self._save_current, "Ctrl+S")
        file_menu.addSeparator()
        export_menu = file_menu.addMenu("Export As…")
        for fmt, label in [
            ("map", "PAL / MAP"), ("css", "CSS Variables"), ("json", "JSON"),
            ("gpl", "GIMP Palette"), ("scss", "SCSS"), ("svg", "SVG"),
            ("tailwind", "Tailwind Config"), ("xml", "Android XML"),
            ("py", "Python"),
        ]:
            self._add_action(export_menu, label,
                             lambda checked, f=fmt: self._export_current(f))
        file_menu.addSeparator()
        self._add_action(file_menu, "Exit", self.close, "Ctrl+Q")

        edit_menu = menubar.addMenu("&Edit")
        self._add_action(edit_menu, "Undo", self._undo_current, "Ctrl+Z")
        self._add_action(edit_menu, "Redo", self._redo_current, "Ctrl+Y")
        edit_menu.addSeparator()
        self._add_action(edit_menu, "Add Color…", self._add_color_current, "Ctrl++")
        self._add_action(edit_menu, "Remove Color", self._remove_color_current, "Ctrl+-")

        tools_menu = menubar.addMenu("&Tools")
        self._add_action(tools_menu, "Generate Palettes", self._generate_palettes)
        self._add_action(tools_menu, "Harmony Generator…", self._show_harmony)
        self._add_action(tools_menu, "Screen Color Picker", self._start_screen_picker)
        tools_menu.addSeparator()
        self._add_action(tools_menu, "Contrast Checker…", self._show_contrast_checker)
        tools_menu.addSeparator()
        cb_menu = tools_menu.addMenu("Color Blindness Sim…")
        for cb_type, label in [("proto", "Protanopia"), ("deuto", "Deuteranopia"),
                                ("trita", "Tritanopia")]:
            self._add_action(cb_menu, label,
                             lambda checked, t=cb_type: self._simulate_cb(t))

        help_menu = menubar.addMenu("&Help")
        self._add_action(help_menu, "About", self._show_about)

    def _build_toolbar(self):
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        tb.addAction("📄 New", self._new_palette)
        tb.addAction("📂 Open", self._import_file)
        tb.addAction("💾 Save", self._save_current)
        tb.addSeparator()
        tb.addAction("🎲 Generate", self._generate_palettes)
        tb.addAction("🎨 Pick Screen", self._start_screen_picker)
        tb.addSeparator()
        tb.addAction("📐 Contrast", self._show_contrast_checker)
        tb.addAction("🌈 Harmony", self._show_harmony)

    @staticmethod
    def _add_action(menu: QMenu, text: str, slot, shortcut: str = ""):
        action = QAction(text, menu)
        action.triggered.connect(slot)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        menu.addAction(action)
        return action

    # ── Palette list ───────────────────────────────────────

    def _refresh_palette_list(self):
        self.palette_list.clear()
        name_q = self.search_edit.text().strip() or None
        fav = self.fav_check.isChecked()
        dom_idx = self.dom_combo.currentIndex()
        dom = ["R", "G", "B"][dom_idx - 1] if dom_idx > 0 else None
        rows = controller.search_palettes(
            name_query=name_q, favorite_only=fav, dominant=dom, limit=200)
        for row in rows:
            pid, name, brightness, contrast, dominant, num_colors, packed, tags, favorite = row
            pal = unpack_palette(packed, num_colors, 8)
            item = QListWidgetItem()
            label = f"{'★ ' if favorite else ''}{name}  [{num_colors}c]  B:{brightness:.0f} C:{contrast:.0f} {dominant}"
            item.setText(label)
            item.setData(Qt.ItemDataRole.UserRole, pid)
            item.setIcon(create_palette_pixmap(pal, 60, 16))
            self.palette_list.addItem(item)
        self.statusBar().showMessage(f"{len(rows)} palettes loaded")

    def _list_context_menu(self, pos: QPoint):
        item = self.palette_list.currentItem()
        if item is None:
            return
        pid = item.data(Qt.ItemDataRole.UserRole)
        menu = QMenu(self)
        open_action = menu.addAction("Open")
        rename_action = menu.addAction("Rename…")
        fav_action = menu.addAction("Toggle Favorite ★")
        tags_action = menu.addAction("Edit Tags…")
        menu.addSeparator()
        dup_action = menu.addAction("Duplicate")
        del_action = menu.addAction("Delete")
        action = menu.exec(self.palette_list.mapToGlobal(pos))
        if action is None:
            return
        if action == open_action:
            self._open_palette(pid)
        elif action == rename_action:
            name, ok = QInputDialog.getText(self, "Rename", "New name:",
                                            text=controller.db.get_name_by_id(pid) or "")
            if ok and name.strip():
                controller.rename_palette(pid, name.strip())
                self._refresh_palette_list()
        elif action == fav_action:
            controller.db.toggle_favorite(pid)
            self._refresh_palette_list()
        elif action == tags_action:
            row_data = controller.db.conn.execute(
                "SELECT tags FROM palettes WHERE id=?", [pid]).fetchone()
            current_tags = row_data[0] if row_data else ""
            tags, ok = QInputDialog.getText(self, "Tags", "Tags (comma-separated):",
                                            text=current_tags)
            if ok:
                controller.db.set_tags(pid, tags.strip())
                self._refresh_palette_list()
        elif action == dup_action:
            pal = controller.get_palette(pid)
            name = controller.db.get_name_by_id(pid) or "Palette"
            if pal:
                controller.create_palette(pal, name=f"{name} (copy)")
                self._refresh_palette_list()
        elif action == del_action:
            r = QMessageBox.question(self, "Delete",
                                     "Delete this palette?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if r == QMessageBox.StandardButton.Yes:
                controller.delete_palette(pid)
                self._close_editor_tab(pid)
                self._refresh_palette_list()

    # ── Tab / Editor management ────────────────────────────

    def _open_palette_item(self, item: QListWidgetItem):
        pid = item.data(Qt.ItemDataRole.UserRole)
        if pid is not None:
            self._open_palette(pid)

    def _open_palette(self, pid: int):
        if pid in self._editors:
            # Already open — activate tab
            idx = self.right_stack.indexOf(self._editors[pid])
            if idx >= 0:
                self.right_stack.setCurrentIndex(idx)
            return
        pal = controller.get_palette(pid)
        name = controller.db.get_name_by_id(pid) or "Untitled"
        if pal is None:
            self.statusBar().showMessage(f"Palette {pid} not found")
            return
        editor = PaletteEditor(name=name, db_id=pid)
        editor.set_palette(pal)
        editor.changed.connect(lambda p=pid: self._on_editor_changed(p))
        editor.closed.connect(lambda p=pid: self._close_editor_tab(p))
        self._editors[pid] = editor
        idx = self.right_stack.addTab(editor, f" {name} ")
        self._show_tabs()
        self.right_stack.setCurrentIndex(idx)

    def _close_tab(self, idx: int):
        w = self.right_stack.widget(idx)
        if w is None:
            return
        # Find pid for this widget
        for pid, editor in list(self._editors.items()):
            if editor is w:
                self._close_editor_tab(pid)
                return
        # Could be a non-editor tab
        self.right_stack.removeTab(idx)
        w.deleteLater()
        if self.right_stack.count() == 0:
            self._hide_tabs()

    def _close_editor_tab(self, pid: int):
        if pid not in self._editors:
            return
        editor = self._editors.pop(pid)
        idx = self.right_stack.indexOf(editor)
        if idx >= 0:
            self.right_stack.removeTab(idx)
        editor.deleteLater()
        if self.right_stack.count() == 0:
            self._hide_tabs()

    def _show_tabs(self):
        self.welcome.hide()
        self.right_stack.show()

    def _hide_tabs(self):
        self.right_stack.hide()
        self.welcome.show()

    def _current_editor(self) -> Optional[PaletteEditor]:
        w = self.right_stack.currentWidget()
        if isinstance(w, PaletteEditor):
            return w
        return None

    def _on_editor_changed(self, pid: int):
        if pid not in self._editors:
            return
        editor = self._editors[pid]
        # Update tab title with modification indicator
        idx = self.right_stack.indexOf(editor)
        if idx >= 0:
            name = editor.name
            self.right_stack.setTabText(idx, f" {name}●" if editor._modified else f" {name}")

    # ── Actions ────────────────────────────────────────────

    def _new_palette(self):
        pal: List[Tuple[int, int, int]] = []
        pid = controller.create_palette(pal, name="New Palette")
        if pid is not None:
            self._refresh_palette_list()
            self._open_palette(pid)

    def _generate_palettes(self):
        count, ok = QInputDialog.getInt(self, "Generate", "Number of palettes:", 10, 1, 100)
        if ok:
            controller.generate_new_palettes(count)
            self._refresh_palette_list()
            self.statusBar().showMessage(f"Generated {count} palettes")

    def _import_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Import Palette", "",
            "Palette Files (*.pal *.map *.gpl *.ase *.csv *.hex *.txt);;All Files (*)")
        if not filepath:
            return
        pid, pal, name = controller.import_file_to_db(filepath)
        if pid is not None:
            AppSettings.add_recent_file(filepath)
            self._refresh_palette_list()
            self._open_palette(pid)
            self.statusBar().showMessage(f"Imported {name} ({len(pal)} colors)")
        else:
            QMessageBox.warning(self, "Import", "No valid colors found in file.")

    def _save_current(self):
        editor = self._current_editor()
        if editor is None or editor.db_id is None:
            self.statusBar().showMessage("No palette open to save")
            return
        palette = editor.get_palette()
        name = editor.name
        if len(palette) > MAX_PACK_COLORS:
            QMessageBox.warning(
                self, "Save Error",
                f"Palettes with more than {MAX_PACK_COLORS} colors cannot be saved to the database.\n"
                "Please reduce colors or export to a file instead.")
            return
        controller.update_palette(editor.db_id, palette, name)
        editor._modified = False
        self._on_editor_changed(editor.db_id)
        self._refresh_palette_list()
        self.statusBar().showMessage(f"Saved '{name}'")

    def _export_current(self, fmt: str):
        editor = self._current_editor()
        if editor is None:
            self.statusBar().showMessage("No palette open to export")
            return
        palette = editor.get_palette()
        name = editor.name

        filters = {
            "map": "PAL Files (*.pal *.map)",
            "css": "CSS Files (*.css)",
            "json": "JSON Files (*.json)",
            "gpl": "GIMP Palette (*.gpl)",
            "scss": "SCSS Files (*.scss)",
            "svg": "SVG Files (*.svg)",
            "tailwind": "JS Files (*.js)",
            "xml": "XML Files (*.xml)",
            "py": "Python Files (*.py)",
        }
        filepath, _ = QFileDialog.getSaveFileName(
            self, f"Export as {fmt.upper()}", f"{name}.{fmt}",
            filters.get(fmt, "All Files (*)"))
        if not filepath:
            return
        ok = controller.export_palette_data(palette, name, filepath, fmt)
        if ok:
            self.statusBar().showMessage(f"Exported to {filepath}")
        else:
            QMessageBox.warning(self, "Export", "Export failed.")

    def _add_color_current(self):
        editor = self._current_editor()
        if editor:
            editor.add_color()

    def _remove_color_current(self):
        editor = self._current_editor()
        if editor:
            editor.remove_selected()

    def _undo_current(self):
        editor = self._current_editor()
        if editor:
            editor.undo()

    def _redo_current(self):
        editor = self._current_editor()
        if editor:
            editor.redo()

    def _show_harmony(self):
        dlg = HarmonyDialog(self)
        dlg.palette_ready.connect(self._use_harmony_palette)
        dlg.show()

    def _use_harmony_palette(self, pal: list):
        pid = controller.create_palette(pal, name="Harmony")
        if pid is not None:
            self._refresh_palette_list()
            self._open_palette(pid)

    def _start_screen_picker(self):
        self._screen_picker = ScreenColorPicker()
        self._screen_picker.color_picked.connect(self._on_screen_color_picked)
        self._screen_picker.cancelled.connect(self._on_screen_picker_cancelled)
        self._screen_picker.start()

    def _on_screen_color_picked(self, color: tuple):
        r, g, b = color
        QApplication.clipboard().setText(rgb_to_hex(r, g, b))
        self.statusBar().showMessage(
            f"Picked {rgb_to_hex(r, g, b)} — copied to clipboard")
        editor = self._current_editor()
        if editor:
            editor.add_color(color)
        self._screen_picker = None

    def _on_screen_picker_cancelled(self):
        self.statusBar().showMessage("Screen picker cancelled")
        self._screen_picker = None

    def _show_contrast_checker(self):
        dlg = ContrastCheckerWidget()
        dlg.setWindowTitle("Contrast Checker")
        dlg.setWindowFlags(Qt.WindowType.Window)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dlg.show()

    def _simulate_cb(self, cb_type: str):
        editor = self._current_editor()
        if editor is None:
            self.statusBar().showMessage("No palette open")
            return
        pal = editor.get_palette()
        sim = simulate_colorblind(pal, cb_type)
        pid = controller.create_palette(sim, name=f"CB_{cb_type}_{editor.name}")
        if pid is not None:
            self._refresh_palette_list()
            self._open_palette(pid)

    def _show_about(self):
        QMessageBox.about(
            self, "About DuckPalette",
            "🦆 <b>DuckPalette</b> v1.0<br>"
            "Color palette management with DuckDB + Numba + PySide6<br><br>"
            "Features: generate, import, export, harmony, contrast checker, "
            "color blindness simulation, screen picker, and more.")

    # ── Window events ──────────────────────────────────────

    def closeEvent(self, e):
        # Check for unsaved changes
        unsaved = [ed for ed in self._editors.values() if ed._modified]
        if unsaved:
            r = QMessageBox.question(
                self, "Unsaved Changes",
                f"{len(unsaved)} palette(s) have unsaved changes. Close anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if r == QMessageBox.StandardButton.No:
                e.ignore()
                return
        # Save settings
        settings = AppSettings.load()
        settings["window_geometry"] = self.saveGeometry().data().hex()
        settings["splitter_sizes"] = self.splitter.sizes()
        AppSettings.save(settings)
        e.accept()

    def showEvent(self, e):
        super().showEvent(e)
        settings = AppSettings.load()
        geo = settings.get("window_geometry")
        if geo:
            try:
                self.restoreGeometry(bytes.fromhex(geo))
            except Exception:
                pass
        sizes = settings.get("splitter_sizes")
        if sizes and len(sizes) == 2:
            self.splitter.setSizes(sizes)


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════

def cli_main(args):
    """Handle command-line interface commands."""
    cmd = args.command
    controller_cli = PaletteController()

    if cmd == "list":
        rows = controller_cli.search_palettes(limit=args.limit or 50)
        if not rows:
            print("No palettes found.")
            return
        print(f"{'ID':>6}  {'Name':<30}  {'#':>3}  {'Bright':>7}  {'Contr':>7}  {'Dom':>3}  {'Fav':>3}")
        print("─" * 75)
        for pid, name, brightness, contrast, dominant, num_colors, packed, tags, favorite in rows:
            fav_mark = "★" if favorite else ""
            print(f"{pid:>6}  {name:<30}  {num_colors:>3}  {brightness:>7.1f}  {contrast:>7.1f}  {dominant:>3}  {fav_mark:>3}")

    elif cmd == "generate":
        count = args.count or 10
        controller_cli.generate_new_palettes(count)
        print(f"Generated {count} palettes.")

    elif cmd == "export":
        pid = args.id
        fmt = args.format or "map"
        pal = controller_cli.get_palette(pid)
        name = controller_cli.db.get_name_by_id(pid) or "Palette"
        if pal is None:
            print(f"Palette ID {pid} not found.")
            return
        out = args.output or f"{name}.{fmt}"
        ok = controller_cli.export_palette_data(pal, name, out, fmt)
        if ok:
            print(f"Exported palette '{name}' to {out}")
        else:
            print("Export failed.")

    elif cmd == "import":
        filepath = args.filepath
        pid, pal, name = controller_cli.import_file_to_db(filepath)
        if pid is not None:
            print(f"Imported '{name}' ({len(pal)} colors) as ID {pid}")
        else:
            print("No valid colors found in file.")

    elif cmd == "info":
        pid = args.id
        pal = controller_cli.get_palette(pid)
        name = controller_cli.db.get_name_by_id(pid) or "Unknown"
        if pal is None:
            print(f"Palette ID {pid} not found.")
            return
        brightness, contrast, dominant = calculate_metadata(pal)
        wcag = palette_wcag_contrast(pal)
        print(f"Palette: {name} (ID {pid})")
        print(f"Colors:  {len(pal)}")
        print(f"Brightness: {brightness:.1f}")
        print(f"Contrast:   {contrast:.1f}")
        print(f"Dominant:   {dominant}")
        print(f"WCAG ratio: {wcag:.2f}:1")
        print()
        for i, (r, g, b) in enumerate(pal):
            h, s, v = rgb_to_hsv(r, g, b)
            print(f"  {i + 1:>3}. {rgb_to_hex(r, g, b)}  RGB({r:3d},{g:3d},{b:3d})  "
                  f"HSV({h:5.1f}°,{s:.2f},{v:.2f})")

    else:
        print(f"Unknown command: {cmd}")
        print("Use: list, generate, export, import, info")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DuckPalette — Color Palette Manager")
    sub = parser.add_subparsers(dest="mode")

    # GUI (default)
    sub.add_parser("gui", help="Launch GUI (default)")

    # CLI
    cli = sub.add_parser("cli", help="CLI mode")
    cli_sub = cli.add_subparsers(dest="command")

    ls = cli_sub.add_parser("list", help="List palettes")
    ls.add_argument("--limit", type=int, default=50)

    gen = cli_sub.add_parser("generate", help="Generate random palettes")
    gen.add_argument("--count", type=int, default=10)

    exp = cli_sub.add_parser("export", help="Export a palette")
    exp.add_argument("id", type=int, help="Palette ID")
    exp.add_argument("--format", "-f", default="map",
                     choices=["map", "css", "json", "gpl", "scss", "svg",
                              "tailwind", "xml", "py"])
    exp.add_argument("--output", "-o", help="Output filepath")

    imp = cli_sub.add_parser("import", help="Import palette from file")
    imp.add_argument("filepath", help="File to import")

    info = cli_sub.add_parser("info", help="Show palette details")
    info.add_argument("id", type=int, help="Palette ID")

    return parser


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════

def main():
    parser = build_parser()
    args = parser.parse_args()

    # Default to GUI if no subcommand given
    if args.mode is None or args.mode == "gui":
        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        app.setStyleSheet(DARK_STYLE)

        # Set dark fusion palette for native dialogs
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#1e1e2e"))
        palette.setColor(QPalette.ColorRole.WindowText, QColor("#cdd6f4"))
        palette.setColor(QPalette.ColorRole.Base, QColor("#181825"))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#313244"))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#313244"))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#cdd6f4"))
        palette.setColor(QPalette.ColorRole.Text, QColor("#cdd6f4"))
        palette.setColor(QPalette.ColorRole.Button, QColor("#313244"))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor("#cdd6f4"))
        palette.setColor(QPalette.ColorRole.BrightText, QColor("#f38ba8"))
        palette.setColor(QPalette.ColorRole.Link, QColor("#89b4fa"))
        palette.setColor(QPalette.ColorRole.Highlight, QColor("#45475a"))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#cdd6f4"))
        app.setPalette(palette)

        window = MainWindow()
        window.show()
        sys.exit(app.exec())

    elif args.mode == "cli":
        if not args.command:
            parser.print_help()
            return
        cli_main(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()