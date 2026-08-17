#!/usr/bin/env python3
"""
Palette Manager — Color Palette Management Application
Single-file PySide6 + DuckDB + Numba application with CLI support.
"""

import os, sys, time, random, math, json, argparse, csv, struct, logging
from typing import List, Tuple, Optional, Dict, Any
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
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
    logger.warning("numpy/numba not found. Falling back to pure Python.")

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton,
    QColorDialog, QFileDialog, QLabel, QListWidget, QListWidgetItem,
    QHBoxLayout, QFrame, QScrollArea, QTabWidget, QFormLayout,
    QDoubleSpinBox, QComboBox, QMessageBox, QGroupBox, QInputDialog,
    QSplitter, QToolBar, QStatusBar, QMenu, QSpinBox, QLineEdit,
    QSizePolicy, QAbstractItemView, QToolTip, QToolButton, QSlider,
    QCheckBox, QAbstractSpinBox, QGridLayout, QLayout, QLayoutItem,
    QSpacerItem, QSizePolicy as QSP,
)
from PySide6.QtGui import (
    QColor, QBrush, QPainter, QPen, QFont, QAction, QKeySequence,
    QPalette, QPixmap, QClipboard, QCursor, QImage, QLinearGradient,
    QDrag, QPainterPath, QConicalGradient, QRadialGradient, QIcon,
)
from PySide6.QtCore import Qt, QSize, Signal, QTimer, QPoint, QMimeData, QByteArray, QRect

# ═══════════════════════════════════════════════════════════
#  STYLESHEET (improved)
# ═══════════════════════════════════════════════════════════

DARK_STYLE = """
QMainWindow, QDialog { background-color: #1a1b26; }
QWidget { color: #c0caf5; font-family: 'Segoe UI','Inter','Helvetica Neue',sans-serif; font-size: 13px; }
QGroupBox { border:1px solid #3b4261; border-radius:8px; margin-top:16px; padding-top:16px; font-weight:600; color:#7aa2f7; }
QGroupBox::title { subcontrol-origin:margin; left:12px; padding:0 6px; }
QPushButton { background:#24283b; border:1px solid #3b4261; border-radius:6px; padding:7px 16px; color:#c0caf5; font-weight:500; }
QPushButton:hover { background:#3b4261; border-color:#545c7e; }
QPushButton:pressed { background:#545c7e; }
QPushButton:disabled { color:#3b4261; }
QPushButton[class="accent"] { background:#7aa2f7; color:#1a1b26; border:none; font-weight:600; }
QPushButton[class="accent"]:hover { background:#89b4fa; }
QPushButton[class="accent"]:pressed { background:#5b84d5; }
QPushButton[class="danger"] { background:#f7768e; color:#1a1b26; border:none; font-weight:600; }
QPushButton[class="danger"]:hover { background:#ff9e9e; }
QLineEdit,QSpinBox,QDoubleSpinBox,QComboBox { background:#24283b; border:1px solid #3b4261; border-radius:6px; padding:5px 10px; color:#c0caf5; }
QLineEdit:focus,QSpinBox:focus,QDoubleSpinBox:focus { border-color:#7aa2f7; }
QComboBox::drop-down { border:none; }
QComboBox QAbstractItemView { background:#24283b; selection-background-color:#3b4261; border-radius:4px; }
QListWidget { background:#16161e; border:1px solid #24283b; border-radius:6px; padding:4px; outline:none; }
QListWidget::item { padding:6px; border-radius:4px; margin:1px 0; }
QListWidget::item:selected { background:#3b4261; color:#7aa2f7; }
QListWidget::item:hover:!selected { background:#1f2335; }
QTabWidget::pane { border:1px solid #24283b; border-radius:6px; background:#1a1b26; top:-1px; }
QTabBar::tab { background:#24283b; border:1px solid #3b4261; border-bottom:none; border-top-left-radius:8px;
               border-top-right-radius:8px; padding:8px 20px; margin-right:2px; color:#565f89; font-weight:500; }
QTabBar::tab:selected { background:#1a1b26; color:#7aa2f7; border-bottom:2px solid #7aa2f7; }
QTabBar::tab:hover:!selected { background:#3b4261; color:#c0caf5; }
QScrollArea { border:none; background:transparent; }
QSplitter::handle { background:#3b4261; width:3px; }
QToolBar { background:#16161e; border-bottom:1px solid #24283b; spacing:8px; padding:4px 8px; }
QStatusBar { background:#16161e; border-top:1px solid #24283b; color:#565f89; font-size:12px; }
QMenu { background:#24283b; border:1px solid #3b4261; border-radius:8px; padding:6px; }
QMenu::item { padding:7px 28px; border-radius:4px; }
QMenu::item:selected { background:#3b4261; color:#7aa2f7; }
QMenu::separator { height:1px; background:#3b4261; margin:4px 10px; }
QToolTip { background:#24283b; color:#c0caf5; border:1px solid #3b4261; border-radius:6px; padding:6px 10px; }
QFrame[frameShape="6"] { border:1px solid #24283b; border-radius:4px; }
QSlider::groove:horizontal { border:1px solid #3b4261; height:8px; background:#24283b; border-radius:4px; }
QSlider::handle:horizontal { background:#7aa2f7; border:2px solid #1a1b26; width:16px; margin:-6px 0; border-radius:8px; }
QSlider::handle:horizontal:hover { background:#89b4fa; }
QSlider::sub-page:horizontal { background:#3b4261; border-radius:4px; }
QScrollBar:vertical { background:#16161e; width:10px; border-radius:5px; }
QScrollBar::handle:vertical { background:#3b4261; border-radius:5px; min-height:20px; }
QScrollBar::handle:vertical:hover { background:#545c7e; }
QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical { height:0; }
QScrollBar:horizontal { background:#16161e; height:10px; border-radius:5px; }
QScrollBar::handle:horizontal { background:#3b4261; border-radius:5px; min-width:20px; }
QScrollBar::add-line:horizontal,QScrollBar::sub-line:horizontal { width:0; }
QCheckBox::indicator { width:16px; height:16px; border-radius:4px; border:1px solid #3b4261; background:#24283b; }
QCheckBox::indicator:checked { background:#7aa2f7; border-color:#7aa2f7; }
"""

# ═══════════════════════════════════════════════════════════
#  ACCELERATED MATH (NUMBA / FALLBACK)
# ═══════════════════════════════════════════════════════════

if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _relative_luminance_numba(r, g, b):
        c_r, c_g, c_b = r / 255.0, g / 255.0, b / 255.0
        l_r = c_r / 12.92 if c_r <= 0.03928 else ((c_r + 0.055) / 1.055) ** 2.4
        l_g = c_g / 12.92 if c_g <= 0.03928 else ((c_g + 0.055) / 1.055) ** 2.4
        l_b = c_b / 12.92 if c_b <= 0.03928 else ((c_b + 0.055) / 1.055) ** 2.4
        return 0.2126 * l_r + 0.7152 * l_g + 0.0722 * l_b

    @njit(cache=True)
    def calculate_metadata_numba(pal):
        n = pal.shape[0]
        if n == 0: return 0.0, 0.0, 0
        brightness = 0.0; max_lum = -1.0; min_lum = 256.0
        sum_r = 0.0; sum_g = 0.0; sum_b = 0.0
        for i in range(n):
            r, g, b = pal[i,0], pal[i,1], pal[i,2]
            lum = 0.299*r + 0.587*g + 0.114*b
            brightness += lum
            if lum > max_lum: max_lum = lum
            if lum < min_lum: min_lum = lum
            sum_r += r; sum_g += g; sum_b += b
        brightness /= n
        contrast = max_lum - min_lum
        dominant = 0
        if sum_g >= sum_r and sum_g >= sum_b: dominant = 1
        elif sum_b >= sum_r and sum_b >= sum_g: dominant = 2
        return brightness, contrast, dominant

    @njit(cache=True)
    def palette_wcag_contrast_numba(pal):
        n = pal.shape[0]
        if n < 2: return 1.0
        max_lum = -1.0; min_lum = 10.0
        for i in range(n):
            lum = _relative_luminance_numba(pal[i,0], pal[i,1], pal[i,2])
            if lum > max_lum: max_lum = lum
            if lum < min_lum: min_lum = lum
        return (max_lum + 0.05) / (min_lum + 0.05)

    M_PROTO = np.array([[0.152286,1.052583,-0.204868],[0.114503,0.786281,0.099216],[-0.003882,-0.048116,1.051998]])
    M_DEUTO = np.array([[0.367322,0.860646,-0.227968],[0.280085,0.672501,0.047413],[-0.011820,0.042940,0.968881]])
    M_TRITA = np.array([[1.255528,-0.076749,-0.178779],[-0.078411,0.930809,0.147602],[-0.004733,0.691367,0.313366]])

    @njit(cache=True)
    def _apply_matrix_numba(pal, matrix):
        res = np.empty_like(pal)
        for i in range(pal.shape[0]):
            r,g,b = pal[i,0],pal[i,1],pal[i,2]
            res[i,0] = min(255.0,max(0.0,matrix[0,0]*r+matrix[0,1]*g+matrix[0,2]*b))
            res[i,1] = min(255.0,max(0.0,matrix[1,0]*r+matrix[1,1]*g+matrix[1,2]*b))
            res[i,2] = min(255.0,max(0.0,matrix[2,0]*r+matrix[2,1]*g+matrix[2,2]*b))
        return res

    def simulate_colorblind(palette_list, cb_type="proto"):
        if not palette_list: return []
        pal = np.array(palette_list, dtype=np.float64)
        mats = {"proto": M_PROTO, "deuto": M_DEUTO, "trita": M_TRITA}
        res = _apply_matrix_numba(pal, mats.get(cb_type, M_PROTO)) if cb_type in mats else pal
        return [tuple(int(c) for c in row) for row in res]
else:
    def _relative_luminance_py(r, g, b):
        srgb = [r/255, g/255, b/255]
        lin = [c/12.92 if c<=0.03928 else ((c+0.055)/1.055)**2.4 for c in srgb]
        return 0.2126*lin[0]+0.7152*lin[1]+0.0722*lin[2]

    def calculate_metadata_numba(pal):
        if not pal: return 0.0, 0.0, 0
        brightness=0.0; max_lum=-1.0; min_lum=256.0; sum_r=0.0; sum_g=0.0; sum_b=0.0
        for r,g,b in pal:
            lum=0.299*r+0.587*g+0.114*b; brightness+=lum
            max_lum=max(max_lum,lum); min_lum=min(min_lum,lum)
            sum_r+=r; sum_g+=g; sum_b+=b
        brightness/=len(pal); contrast=max_lum-min_lum
        dominant=0
        if sum_g>=sum_r and sum_g>=sum_b: dominant=1
        elif sum_b>=sum_r and sum_b>=sum_g: dominant=2
        return brightness, contrast, dominant

    def palette_wcag_contrast_numba(pal):
        if len(pal)<2: return 1.0
        lums=[_relative_luminance_py(*c) for c in pal]
        return (max(lums)+0.05)/(min(lums)+0.05)

    def simulate_colorblind(palette_list, cb_type="proto"):
        return palette_list

# ═══════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════

def rgb_to_hex(r, g, b): return f"#{r:02X}{g:02X}{b:02X}"

def hex_to_rgb(h):
    h = h.lstrip("#")
    if len(h)==3: h = h[0]*2+h[1]*2+h[2]*2
    if len(h)!=6 or not all(c in "0123456789abcdefABCDEF" for c in h):
        raise ValueError(f"Invalid hex: #{h}")
    return tuple(int(h[i:i+2],16) for i in (0,2,4))

def calculate_metadata(palette):
    if not palette: return 0.0, 0.0, "R"
    if NUMBA_AVAILABLE:
        pal = np.array(palette, dtype=np.float64)
        b,c,d = calculate_metadata_numba(pal)
        return b,c,"RGB"[d]
    b,c,d = calculate_metadata_numba(palette)
    return b,c,"RGB"[d]

def palette_wcag_contrast(palette):
    if len(palette)<2: return 1.0
    if NUMBA_AVAILABLE:
        return float(palette_wcag_contrast_numba(np.array(palette, dtype=np.float64)))
    return palette_wcag_contrast_numba(palette)

def rgb_to_hsv(r, g, b):
    r1,g1,b1 = r/255.0, g/255.0, b/255.0
    mx,mn = max(r1,g1,b1), min(r1,g1,b1)
    df = mx-mn
    if mx==mn: h=0
    elif mx==r1: h=(60*((g1-b1)/df)+360)%360
    elif mx==g1: h=(60*((b1-r1)/df)+120)%360
    else: h=(60*((r1-g1)/df)+240)%360
    s = 0 if mx==0 else df/mx
    return h, s, mx

def hsv_to_rgb(h, s, v):
    h %= 360; c = v*s; x = c*(1-abs((h/60)%2-1)); m = v-c
    if h<60: r,g,b = c,x,0
    elif h<120: r,g,b = x,c,0
    elif h<180: r,g,b = 0,c,x
    elif h<240: r,g,b = 0,x,c
    elif h<300: r,g,b = x,0,c
    else: r,g,b = c,0,x
    return int((r+m)*255), int((g+m)*255), int((b+m)*255)

def rgb_to_hsl(r, g, b):
    r1,g1,b1 = r/255.0, g/255.0, b/255.0
    mx,mn = max(r1,g1,b1), min(r1,g1,b1)
    l = (mx+mn)/2
    if mx==mn: h=s=0.0
    else:
        d = mx-mn
        s = d/(2.0-mx-mn) if l>0.5 else d/(mx+mn)
        if mx==r1: h=(g1-b1)/d+(6 if g1<b1 else 0)
        elif mx==g1: h=(b1-r1)/d+2
        else: h=(r1-g1)/d+4
        h/=6
    return h*360, s*100, l*100

def lerp_color(c1, c2, t):
    return (int(c1[0]+(c2[0]-c1[0])*t), int(c1[1]+(c2[1]-c1[1])*t), int(c1[2]+(c2[2]-c1[2])*t))

def extract_palette_from_image(qimg, num_colors=5):
    small = qimg.scaled(64,64,Qt.AspectRatioMode.IgnoreAspectRatio,Qt.TransformationMode.SmoothTransformation)
    pixels = []
    for y in range(small.height()):
        for x in range(small.width()):
            c = small.pixelColor(x,y)
            if c.alpha()>128: pixels.append((c.red(),c.green(),c.blue()))
    if not pixels: return []
    rounded = [(r//32*32,g//32*32,b//32*32) for r,g,b in pixels]
    counts = Counter(rounded)
    return [(min(255,r+16),min(255,g+16),min(255,b+16)) for (r,g,b),_ in counts.most_common(num_colors)]

def extract_palette_kmeans(qimg, num_colors=5, max_iter=20, seed=None):
    small = qimg.scaled(100,100,Qt.AspectRatioMode.IgnoreAspectRatio,Qt.TransformationMode.SmoothTransformation)
    pixels = []
    for y in range(small.height()):
        for x in range(small.width()):
            c = small.pixelColor(x,y)
            if c.alpha()>128: pixels.append([c.red(),c.green(),c.blue()])
    if not pixels: return []
    if NUMBA_AVAILABLE:
        pixels_np = np.array(pixels, dtype=np.float64)
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(pixels_np), min(num_colors,len(pixels_np)), replace=False)
        centers = pixels_np[indices].copy()
        for _ in range(max_iter):
            dists = np.zeros((len(pixels_np),num_colors))
            for k in range(num_colors):
                diff = pixels_np - centers[k]
                dists[:,k] = np.sum(diff**2, axis=1)
            labels = np.argmin(dists, axis=1)
            new_centers = np.zeros_like(centers)
            for k in range(num_colors):
                mask = labels==k
                if np.any(mask): new_centers[k] = pixels_np[mask].mean(axis=0)
                else: new_centers[k] = centers[k]
            if np.allclose(centers, new_centers, atol=1.0): break
            centers = new_centers
        lums = 0.299*centers[:,0]+0.587*centers[:,1]+0.114*centers[:,2]
        centers = centers[np.argsort(lums)]
        return [(int(min(255,max(0,c[0]))),int(min(255,max(0,c[1]))),int(min(255,max(0,c[2])))) for c in centers]
    return extract_palette_from_image(qimg, num_colors)

MAX_PACK_COLORS = 128 // (3*8)

def pack_palette(palette, bits_per_channel=8):
    max_colors = 128 // (3*bits_per_channel)
    if len(palette) > max_colors:
        raise ValueError(f"Cannot pack {len(palette)} colours into HUGEINT (max {max_colors})")
    packed, shift = 0, 0
    for r,g,b in palette:
        r,g,b = max(0,min(r,255)),max(0,min(g,255)),max(0,min(b,255))
        packed |= ((r<<(2*bits_per_channel))|(g<<bits_per_channel)|b) << shift
        shift += 3*bits_per_channel
    return packed

def unpack_palette(packed, num_colors=5, bits_per_channel=8):
    palette = []; mask = (1<<(3*bits_per_channel))-1; ch_mask = (1<<bits_per_channel)-1
    for _ in range(num_colors):
        cb = packed & mask
        b_val = cb & ch_mask; g_val = (cb>>bits_per_channel)&ch_mask; r_val = (cb>>(2*bits_per_channel))&ch_mask
        palette.append((r_val,g_val,b_val))
        packed >>= 3*bits_per_channel
    return palette

def generate_random_palette(seed, num_colors=5):
    rng = random.Random(seed)
    return [(rng.randint(0,255),rng.randint(0,255),rng.randint(0,255)) for _ in range(num_colors)]

def generate_harmony_palette(base_hue, harmony_type, num_colors=5, sat=0.75, val=0.75):
    colors = []
    if harmony_type == "complementary":
        for i in range(num_colors):
            t = i/max(num_colors-1,1); h = (base_hue+t*180)%360
            s = min(sat*(0.7+0.3*math.sin(t*math.pi)),1.0)
            v = min(val*(0.5+0.5*(1-abs(2*t-1))),1.0)
            colors.append(hsv_to_rgb(h,s,v))
    elif harmony_type == "analogous":
        for i in range(num_colors):
            t = i/max(num_colors-1,1); h = (base_hue-30+t*60)%360
            s = min(sat*(0.8+0.2*math.sin(t*math.pi)),1.0)
            v = min(val*(0.7+0.3*math.sin(t*math.pi+0.5)),1.0)
            colors.append(hsv_to_rgb(h,s,v))
    elif harmony_type == "triadic":
        for i in range(num_colors):
            seg = i*3//num_colors; h = (base_hue+seg*120+(i%2)*15)%360
            s = min(sat*(0.7+0.3*(i%2)),1.0); v = min(val*(0.6+0.4*((i+1)%2)),1.0)
            colors.append(hsv_to_rgb(h,s,v))
    elif harmony_type == "split_complementary":
        angles = [0,150,210]
        for i in range(num_colors):
            h = (base_hue+angles[i%3]+(i//3)*12)%360
            s = min(sat*(0.7+0.3*(i%2)),1.0); v = min(val*(0.6+0.4*((i+1)%2)),1.0)
            colors.append(hsv_to_rgb(h,s,v))
    elif harmony_type == "monochromatic":
        for i in range(num_colors):
            t = i/max(num_colors-1,1); s = min(sat*(0.3+0.7*t),1.0); v = min(0.3+0.7*(1-t),1.0)
            colors.append(hsv_to_rgb(base_hue,s,v))
    elif harmony_type == "tetradic":
        for i in range(num_colors):
            h = (base_hue+i*90)%360; s = min(sat*(0.7+0.3*math.sin(i*math.pi/2)),1.0)
            v = min(val*(0.6+0.4*math.cos(i*math.pi/3)),1.0)
            colors.append(hsv_to_rgb(h,s,v))
    return colors[:num_colors]

def generate_variations(palette, variation="lighter", strength=0.3):
    result = []
    for r,g,b in palette:
        h,s,v = rgb_to_hsv(r,g,b)
        if variation=="lighter": v = min(1.0, v+strength*(1.0-v))
        elif variation=="darker": v = max(0.0, v*(1.0-strength))
        elif variation=="muted": s = max(0.0, s*(1.0-strength))
        elif variation=="vivid": s = min(1.0, s+strength*(1.0-s))
        elif variation=="pastel": s = max(0.0, s*0.4); v = min(1.0, v+0.3*(1.0-v))
        elif variation=="warm": h = (h+15*strength)%360
        elif variation=="cool": h = (h-15*strength)%360
        result.append(hsv_to_rgb(h,s,v))
    return result

def find_duplicate_colors(palette, threshold=15):
    dupes = []
    for i in range(len(palette)):
        for j in range(i+1,len(palette)):
            r1,g1,b1 = palette[i]; r2,g2,b2 = palette[j]
            dist = math.sqrt((r2-r1)**2+(g2-g1)**2+(b2-b1)**2)
            if dist < threshold: dupes.append((i,j,dist))
    return dupes

def merge_palettes(*palettes, mode="concat", max_colors=12):
    if mode=="concat":
        combined = []
        for p in palettes: combined.extend(p)
        return combined[:max_colors]
    elif mode=="alternate":
        combined = []; iters = [iter(p) for p in palettes]
        while len(combined) < max_colors:
            for it in iters:
                try: combined.append(next(it))
                except StopIteration: pass
        return combined[:max_colors]
    elif mode=="average":
        n = min(len(p) for p in palettes) if palettes else 0; result = []
        for i in range(n):
            avg_r = round(sum(p[i][0] for p in palettes)/len(palettes))
            avg_g = round(sum(p[i][1] for p in palettes)/len(palettes))
            avg_b = round(sum(p[i][2] for p in palettes)/len(palettes))
            result.append((avg_r,avg_g,avg_b))
        return result
    return palettes[0] if palettes else []

def sort_palette(palette, mode="hue"):
    def key_hue(c): return rgb_to_hsv(*c)[0]
    def key_bright(c): return 0.299*c[0]+0.587*c[1]+0.114*c[2]
    def key_sat(c): return rgb_to_hsv(*c)[1]
    return sorted(palette, key={"hue":key_hue,"brightness":key_bright,"saturation":key_sat}.get(mode, key_hue))

# ═══════════════════════════════════════════════════════════
#  FILE I/O
# ═══════════════════════════════════════════════════════════

def parse_map_file(filepath):
    palette = []
    try:
        with open(filepath,"r") as f:
            for line in f:
                line = line.strip()
                if not line or line[0] in ("#",";"): continue
                parts = line.split()
                if len(parts)>=3:
                    try:
                        r,g,b = int(parts[0]),int(parts[1]),int(parts[2])
                        if 0<=r<=255 and 0<=g<=255 and 0<=b<=255: palette.append((r,g,b))
                    except ValueError: continue
    except Exception as e: logger.error("Error reading %s: %s",filepath,e)
    return palette

def save_map_file(palette, filepath):
    try:
        with open(filepath,"w") as f:
            f.write("# Generated by DuckPalette\n")
            for r,g,b in palette: f.write(f"{r:3d} {g:3d} {b:3d}\n")
        return True
    except Exception as e: logger.error("Error writing %s: %s",filepath,e); return False

def parse_ase_file(filepath):
    palette = []
    try:
        with open(filepath,"rb") as f:
            sig = f.read(4)
            if sig != b"ASEF": return palette
            f.read(4); n_blocks = struct.unpack(">I",f.read(4))[0]
            for _ in range(n_blocks):
                btype = struct.unpack(">H",f.read(2))[0]
                blen = struct.unpack(">I",f.read(4))[0]
                block_data = f.read(blen)
                if btype==0x0001 and len(block_data)>=6:
                    offset = 0
                    name_len = struct.unpack(">H",block_data[offset:offset+2])[0]; offset += 2
                    offset += name_len*2
                    if offset+4 <= len(block_data):
                        color_model = block_data[offset:offset+4].decode("ascii",errors="replace").strip("\x00"); offset += 4
                        if color_model=="RGB " and offset+12<=len(block_data):
                            r,g,b = struct.unpack(">fff",block_data[offset:offset+12])
                            palette.append((min(255,max(0,int(r*255))),min(255,max(0,int(g*255))),min(255,max(0,int(b*255)))))
                        elif color_model=="CMYK" and offset+16<=len(block_data):
                            c,m,y,k = struct.unpack(">ffff",block_data[offset:offset+16])
                            palette.append((max(0,min(255,int(255*(1-c)*(1-k)))),max(0,min(255,int(255*(1-m)*(1-k)))),max(0,min(255,int(255*(1-y)*(1-k))))))
                        elif color_model=="Gray" and offset+4<=len(block_data):
                            gv = struct.unpack(">f",block_data[offset:offset+4])[0]; v = min(255,max(0,int(gv*255)))
                            palette.append((v,v,v))
    except Exception as e: logger.error("ASE parse error: %s",e)
    return palette

def parse_hex_list(filepath):
    palette = []
    with open(filepath,"r") as f:
        for line in f:
            line = line.strip().rstrip(",").rstrip(";")
            if line.startswith("#") and len(line.lstrip("#")) in (3,6):
                try: palette.append(hex_to_rgb(line))
                except ValueError: continue
            elif len(line)==6 and all(c in "0123456789abcdefABCDEF" for c in line):
                try: palette.append(hex_to_rgb("#"+line))
                except ValueError: continue
    return palette

def parse_csv_file(filepath):
    palette = []
    with open(filepath,"r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row)>=3:
                try:
                    r,g,b = int(row[0]),int(row[1]),int(row[2])
                    if 0<=r<=255 and 0<=g<=255 and 0<=b<=255: palette.append((r,g,b))
                except ValueError: pass
            elif len(row)>=1:
                try: palette.append(hex_to_rgb(row[0].strip()))
                except ValueError: pass
    return palette

def export_as_css(palette, filepath):
    try:
        with open(filepath,"w") as f:
            f.write(":root {\n")
            for i,(r,g,b) in enumerate(palette):
                f.write(f"  --color-{i+1}: {rgb_to_hex(r,g,b)};\n")
                f.write(f"  --color-{i+1}-rgb: {r}, {g}, {b};\n")
            f.write("}\n")
        return True
    except Exception as e: logger.error("CSS export error: %s",e); return False

def export_as_json(palette, filepath):
    try:
        with open(filepath,"w") as f: json.dump([{"hex":rgb_to_hex(*c),"rgb":list(c)} for c in palette],f,indent=2)
        return True
    except Exception as e: logger.error("JSON export error: %s",e); return False

def export_as_gpl(palette, name, filepath):
    try:
        with open(filepath,"w") as f:
            f.write("GIMP Palette\n"); f.write(f"Name: {name}\n"); f.write(f"Columns: {len(palette)}\n#\n")
            for r,g,b in palette: f.write(f"{r:3d} {g:3d} {b:3d}\t{rgb_to_hex(r,g,b)}\n")
        return True
    except Exception as e: logger.error("GPL export error: %s",e); return False

def export_as_scss(palette, filepath):
    try:
        with open(filepath,"w") as f:
            f.write("// Generated by DuckPalette\n$palette: (\n")
            for i,(r,g,b) in enumerate(palette): f.write(f'  "{i+1}": ({r}, {g}, {b}),\n')
            f.write(");\n\n")
            for i,(r,g,b) in enumerate(palette): f.write(f"$color-{i+1}: rgb({r}, {g}, {b});\n")
        return True
    except Exception as e: logger.error("SCSS export error: %s",e); return False

def export_as_tailwind(palette, filepath):
    try:
        with open(filepath,"w") as f:
            f.write("// tailwind.config.js\nmodule.exports = {\n  theme: {\n    extend: {\n      colors: {\n")
            for i,(r,g,b) in enumerate(palette): f.write(f'        "palette-{i+1}": "rgb({r} {g} {b})",\n')
            f.write("      },\n    },\n  },\n};\n")
        return True
    except Exception as e: logger.error("Tailwind export error: %s",e); return False

def export_as_svg(palette, filepath):
    try:
        n = len(palette); w,h = max(400,n*80),120
        with open(filepath,"w") as f:
            f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">\n')
            for i,(r,g,b) in enumerate(palette):
                x_start = i*(w//n); x_end = ((i+1)*w)//n if i<n-1 else w; sw = x_end-x_start
                f.write(f'  <rect x="{x_start}" y="0" width="{sw}" height="{h}" fill="{rgb_to_hex(r,g,b)}"/>\n')
                lum = 0.299*r+0.587*g+0.114*b; tc = "#000" if lum>128 else "#fff"
                f.write(f'  <text x="{x_start+sw//2}" y="{h//2}" fill="{tc}" font-size="11" text-anchor="middle" dominant-baseline="middle">{rgb_to_hex(r,g,b)}</text>\n')
            f.write("</svg>\n")
        return True
    except Exception as e: logger.error("SVG export error: %s",e); return False

def export_as_android_xml(palette, filepath):
    try:
        with open(filepath,"w") as f:
            f.write('<?xml version="1.0" encoding="utf-8"?>\n<resources>\n')
            for i,(r,g,b) in enumerate(palette): f.write(f'    <color name="palette_color_{i+1}">{rgb_to_hex(r,g,b)}</color>\n')
            f.write("</resources>\n")
        return True
    except Exception as e: logger.error("Android XML export error: %s",e); return False

def export_as_python(palette, filepath):
    try:
        with open(filepath,"w") as f:
            f.write("# Generated by DuckPalette\n\nPALETTE = [\n")
            for r,g,b in palette: f.write(f"    ({r:3d}, {g:3d}, {b:3d}),  # {rgb_to_hex(r,g,b)}\n")
            f.write("]\n\nPALETTE_HEX = [\n")
            for r,g,b in palette: f.write(f'    "{rgb_to_hex(r,g,b)}",\n')
            f.write("]\n")
        return True
    except Exception as e: logger.error("Python export error: %s",e); return False

# ═══════════════════════════════════════════════════════════
#  DATABASE
# ═══════════════════════════════════════════════════════════

class PaletteDB:
    def __init__(self, db_file="palettes.duckdb"):
        self.db_file = db_file
        self.conn = duckdb.connect(database=db_file, read_only=False)
        self._init_db()

    def __enter__(self): return self
    def __exit__(self,*exc): self.close()
    def close(self):
        try: self.conn.close()
        except: pass
    def __del__(self): self.close()

    def _safe_rollback(self):
        try: self.conn.rollback()
        except: pass

    def _column_exists(self, table_name, column_name):
        try:
            result = self.conn.execute("SELECT COUNT(*) FROM information_schema.columns WHERE table_name=? AND column_name=?",
                [table_name.lower(), column_name.lower()]).fetchone()
            return result[0] > 0
        except: self._safe_rollback(); return False

    def _init_db(self):
        try: self.conn.execute("CREATE SEQUENCE IF NOT EXISTS palette_id_seq START 1")
        except: self._safe_rollback()
        try:
            self.conn.execute('''CREATE TABLE IF NOT EXISTS palettes (
                id INTEGER PRIMARY KEY DEFAULT nextval('palette_id_seq'),
                name TEXT, seed BIGINT, num_colors INTEGER,
                bits_per_channel INTEGER, packed_palette HUGEINT,
                brightness DOUBLE, contrast DOUBLE, dominant TEXT,
                favorite BOOLEAN DEFAULT FALSE, tags TEXT DEFAULT '')''')
        except: self._safe_rollback()
        for col_name, col_def in [("favorite","BOOLEAN DEFAULT FALSE"),("tags","TEXT DEFAULT ''")]:
            if not self._column_exists("palettes",col_name):
                try: self.conn.execute(f"ALTER TABLE palettes ADD COLUMN {col_name} {col_def}")
                except: self._safe_rollback()
        try:
            row = self.conn.execute("SELECT MAX(id) FROM palettes").fetchone()
            if row and row[0] is not None: self.conn.execute(f"ALTER SEQUENCE palette_id_seq RESTART WITH {row[0]+1}")
        except: self._safe_rollback()

    def insert_palette(self, palette, name="User Palette", pid=None, seed=None):
        if not palette: return None
        b,c,d = calculate_metadata(palette); packed = pack_palette(palette)
        if pid is None:
            result = self.conn.execute(
                'INSERT INTO palettes (name,seed,num_colors,bits_per_channel,packed_palette,brightness,contrast,dominant) VALUES (?,?,?,?,?,?,?,?) RETURNING id',
                [name,seed,len(palette),8,packed,b,c,d])
            pid = result.fetchone()[0]
        else:
            self.conn.execute(
                'INSERT INTO palettes (id,name,seed,num_colors,bits_per_channel,packed_palette,brightness,contrast,dominant) VALUES (?,?,?,?,?,?,?,?,?)',
                [pid,name,seed,len(palette),8,packed,b,c,d])
        return pid

    def update_palette(self, pid, palette, name):
        if not palette: return False
        b,c,d = calculate_metadata(palette); packed = pack_palette(palette)
        self.conn.execute('UPDATE palettes SET name=?,packed_palette=?,num_colors=?,brightness=?,contrast=?,dominant=? WHERE id=?',
            [name,packed,len(palette),b,c,d,pid])
        return True

    def rename_palette(self, pid, name): self.conn.execute("UPDATE palettes SET name=? WHERE id=?", [name,pid])
    def delete_palette(self, pid): self.conn.execute("DELETE FROM palettes WHERE id=?", [pid])
    def toggle_favorite(self, pid): self.conn.execute("UPDATE palettes SET favorite = NOT favorite WHERE id=?", [pid])
    def set_tags(self, pid, tags_str): self.conn.execute("UPDATE palettes SET tags=? WHERE id=?", [tags_str,pid])

    def generate_and_insert(self, seed, num_colors=5):
        palette = generate_random_palette(seed, num_colors)
        return self.insert_palette(palette, name=f"Gen_{seed}", seed=seed)

    def search(self, *, min_bright=None, max_bright=None, dominant=None, favorite_only=False, tag=None, name_query=None, limit=100):
        q = 'SELECT id,name,brightness,contrast,dominant,num_colors,packed_palette,tags,favorite FROM palettes WHERE 1=1'
        p = []
        if min_bright is not None: q+=' AND brightness>=?'; p.append(min_bright)
        if max_bright is not None: q+=' AND brightness<=?'; p.append(max_bright)
        if dominant: q+=' AND dominant=?'; p.append(dominant)
        if favorite_only: q+=' AND favorite=TRUE'
        if tag: q+=' AND tags LIKE ?'; p.append(f'%{tag}%')
        if name_query: q+=' AND name LIKE ?'; p.append(f'%{name_query}%')
        q += ' ORDER BY id DESC LIMIT ?'; p.append(limit)
        return self.conn.execute(q,p).fetchall()

    def get_palette_by_id(self, pid):
        row = self.conn.execute("SELECT packed_palette,num_colors,bits_per_channel FROM palettes WHERE id=?",[pid]).fetchone()
        return unpack_palette(row[0],row[1],row[2]) if row else None

    def get_name_by_id(self, pid):
        row = self.conn.execute("SELECT name FROM palettes WHERE id=?",[pid]).fetchone()
        return row[0] if row else None

    def count(self): return self.conn.execute("SELECT COUNT(*) FROM palettes").fetchone()[0]

# ═══════════════════════════════════════════════════════════
#  CONTROLLER
# ═══════════════════════════════════════════════════════════

class PaletteController:
    def __init__(self): self.db = PaletteDB()

    def import_file_to_db(self, filepath):
        low = filepath.lower()
        if low.endswith('.ase'): palette = parse_ase_file(filepath)
        elif low.endswith('.csv'): palette = parse_csv_file(filepath)
        elif low.endswith('.hex') or low.endswith('.txt'): palette = parse_hex_list(filepath)
        else: palette = parse_map_file(filepath)
        if palette:
            name = os.path.basename(filepath)
            pid = self.db.insert_palette(palette, name=name)
            return pid, palette, name
        return None, [], None

    def export_palette_data(self, palette, name, filepath, fmt="map"):
        if not palette: return False
        exporters = {
            "css": export_as_css, "json": export_as_json,
            "gpl": lambda p,f: export_as_gpl(p,name,f),
            "scss": export_as_scss, "tailwind": export_as_tailwind,
            "svg": export_as_svg, "xml": export_as_android_xml, "py": export_as_python,
        }
        if fmt in exporters: return exporters[fmt](palette, filepath)
        return save_map_file(palette, filepath)

    def create_palette(self, palette_data, name="New Palette"): return self.db.insert_palette(palette_data, name=name)
    def update_palette(self, pid, palette_data, name): return self.db.update_palette(pid, palette_data, name)
    def rename_palette(self, pid, name): self.db.rename_palette(pid, name)
    def delete_palette(self, pid): self.db.delete_palette(pid)

    def generate_new_palettes(self, count=10):
        seed = int(time.time())
        for i in range(count): self.db.generate_and_insert(seed=seed+i)

    def search_palettes(self, **kwargs): return self.db.search(**kwargs)
    def get_palette(self, pid): return self.db.get_palette_by_id(pid)
    def calculate_metadata(self, palette): return calculate_metadata(palette)

controller = PaletteController()

# ═══════════════════════════════════════════════════════════
#  SETTINGS
# ═══════════════════════════════════════════════════════════

class AppSettings:
    _path = os.path.join(os.path.expanduser("~"), ".duckpalette_settings.json")
    _defaults = {"window_geometry": None, "splitter_sizes": [360, 840], "recent_files": []}

    @classmethod
    def load(cls):
        try:
            with open(cls._path,"r") as f: data = json.load(f)
            merged = dict(cls._defaults); merged.update(data); return merged
        except: return dict(cls._defaults)

    @classmethod
    def save(cls, settings):
        try:
            with open(cls._path,"w") as f: json.dump(settings,f,indent=2)
        except Exception as e: logger.warning("Failed to save settings: %s",e)

    @classmethod
    def add_recent_file(cls, path):
        settings = cls.load(); recent = settings.get("recent_files",[])
        if path in recent: recent.remove(path)
        recent.insert(0,path); settings["recent_files"] = recent[:20]
        cls.save(settings)

# ═══════════════════════════════════════════════════════════
#  GUI CUSTOM WIDGETS
# ═══════════════════════════════════════════════════════════

def create_palette_pixmap(palette, w=80, h=20):
    px = QPixmap(w,h); p = QPainter(px)
    p.fillRect(0,0,w,h,QColor("#16161e"))
    if palette:
        n = len(palette); x = 0
        for i,col in enumerate(palette):
            nx = ((i+1)*w)//n; p.fillRect(x,0,nx-x,h,QColor(*col)); x = nx
    p.end(); return px


class ToastWidget(QFrame):
    def __init__(self, message, duration=2500, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.ToolTip)
        lay = QVBoxLayout(self)
        label = QLabel(message)
        label.setStyleSheet("color:#c0caf5; padding:8px 16px; font-weight:500;")
        lay.addWidget(label)
        self.setStyleSheet("background:#24283b; border:1px solid #7aa2f7; border-radius:8px;")
        self.adjustSize()
        if parent:
            center = parent.rect().center(); global_pt = parent.mapToGlobal(center)
            self.move(global_pt.x()-self.width()//2, global_pt.y()+parent.height()//2-self.height()-40)
        self.show()
        QTimer.singleShot(duration, self.close)


class ScreenColorPicker(QWidget):
    color_picked = Signal(tuple)
    cancelled = Signal()

    def __init__(self):
        super().__init__(None, Qt.WindowType.FramelessWindowHint|Qt.WindowType.WindowStaysOnTopHint|Qt.WindowType.Tool)
        self.setWindowState(Qt.WindowState.WindowFullScreen)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._timer = QTimer(self); self._timer.setInterval(50)
        self._timer.timeout.connect(self._update_color)
        self._current_color = (0,0,0)

    def start(self): self.show(); self._timer.start()

    def _update_color(self):
        pos = QCursor.pos(); screen = QApplication.screenAt(pos)
        if screen:
            geo = screen.geometry()
            img = screen.grabWindow(0,pos.x()-geo.x(),pos.y()-geo.y(),1,1).toImage()
            c = img.pixelColor(0,0); self._current_color = (c.red(),c.green(),c.blue())

    def paintEvent(self, e):
        p = QPainter(self); p.fillRect(self.rect(),QColor(0,0,0,1))
        pos = QCursor.pos(); r,g,b = self._current_color; hx = rgb_to_hex(r,g,b)
        p.setPen(Qt.GlobalColor.white); p.drawRect(pos.x()-20,pos.y()-30,140,24)
        p.fillRect(pos.x()-19,pos.y()-29,138,22,QColor(*self._current_color))
        lum = 0.299*r+0.587*g+0.114*b
        p.setPen(QColor(0,0,0) if lum>128 else QColor(255,255,255))
        p.drawText(pos.x()-15,pos.y()-13,f"{hx} ({r},{g},{b})")

    def mousePressEvent(self, e):
        self._timer.stop(); self.close()
        if e.button()==Qt.MouseButton.LeftButton: self.color_picked.emit(self._current_color)
        else: self.cancelled.emit()

    def keyPressEvent(self, e):
        if e.key()==Qt.Key.Key_Escape: self._timer.stop(); self.close(); self.cancelled.emit()


# ── Gradient Slider ───────────────────────────────────────

class GradientSlider(QWidget):
    """A slider with a gradient-filled track."""
    valueChanged = Signal(int)

    def __init__(self, orientation=Qt.Orientation.Horizontal, parent=None):
        super().__init__(parent)
        self._value = 0
        self._min = 0
        self._max = 255
        self._gradient_stops = []  # list of (pos, QColor)
        self.setFixedHeight(28)
        self.setMinimumWidth(200)
        self._dragging = False

    def set_range(self, mn, mx):
        self._min = mn; self._max = mx; self.update()

    def set_value(self, v):
        self._value = max(self._min, min(self._max, v)); self.update(); self.valueChanged.emit(self._value)

    def value(self): return self._value

    def set_gradient(self, stops):
        """stops: list of (position_0_to_1, QColor)"""
        self._gradient_stops = stops; self.update()

    def _handle_x(self):
        span = self.width() - 16
        ratio = (self._value - self._min) / max(1, self._max - self._min)
        return 8 + int(ratio * span)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        try:
            track_rect = self.rect().adjusted(8, 6, -8, -6)
            grad = QLinearGradient(track_rect.left(), 0, track_rect.right(), 0)
            for pos, color in self._gradient_stops:
                grad.setColorAt(pos, color)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(grad)
            p.drawRoundedRect(track_rect, 4, 4)

            hx = self._handle_x()
            p.setBrush(QColor("#7aa2f7"))
            p.setPen(QPen(QColor("#1a1b26"), 2))
            p.drawEllipse(QPoint(hx, self.height() // 2), 7, 7)
        finally:
            p.end()

    def mousePressEvent(self, e):
        if e.button()==Qt.MouseButton.LeftButton: self._dragging=True; self._set_from_x(e.position().x())

    def mouseMoveEvent(self, e):
        if self._dragging: self._set_from_x(e.position().x())

    def mouseReleaseEvent(self, e): self._dragging = False

    def _set_from_x(self, x):
        span = self.width()-16; ratio = max(0, min(1, (x-8)/max(1,span)))
        self.set_value(int(self._min + ratio*(self._max-self._min)))


# ── Color Swatch Widget ───────────────────────────────────

class ColorSwatchWidget(QFrame):
    """A single interactive color swatch with hex label."""
    clicked = Signal(int)
    remove_requested = Signal(int)
    color_changed = Signal(int, tuple)

    def __init__(self, index, color=(128,128,128), parent=None):
        super().__init__(parent)
        self.index = index
        self.color = color
        self.selected = False
        self._hover = False
        self.setFixedSize(72, 90)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)
        self.setToolTip(rgb_to_hex(*self.color))

    def set_color(self, color):
        self.color = color; self.setToolTip(rgb_to_hex(*self.color)); self.update()

    def set_selected(self, sel):
        self.selected = sel; self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        try:
            r, g, b = self.color
            hx = rgb_to_hex(r, g, b)

            sw_rect = self.rect().adjusted(4, 4, -4, -24)

            # Shadow
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(0, 0, 0, 60))
            p.drawRoundedRect(sw_rect.adjusted(2, 2, 2, 2), 8, 8)

            # Swatch fill
            p.setBrush(QColor(r, g, b))
            if self.selected:
                p.setPen(QPen(QColor("#7aa2f7"), 2))
            elif self._hover:
                p.setPen(QPen(QColor("#545c7e"), 1))
            else:
                p.setPen(QPen(QColor("#3b4261"), 1))
            p.drawRoundedRect(sw_rect, 8, 8)

            # Hex label
            p.setPen(QColor("#565f89"))
            p.setFont(QFont("Segoe UI", 8))
            label_rect = QRect(0, self.height() - 20, self.width(), 20)
            p.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, hx)
        finally:
            p.end()

    def enterEvent(self, e): self._hover=True; self.update()
    def leaveEvent(self, e): self._hover=False; self.update()

    def mousePressEvent(self, e):
        if e.button()==Qt.MouseButton.LeftButton:
            if e.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self.remove_requested.emit(self.index)
            else:
                self.clicked.emit(self.index)

    def contextMenuEvent(self, e):
        menu = QMenu(self)
        copy_act = menu.addAction("Copy HEX")
        copy_rgb_act = menu.addAction("Copy RGB")
        menu.addSeparator()
        remove_act = menu.addAction("Remove Color")
        action = menu.exec(e.globalPos())
        if action == copy_act:
            QApplication.clipboard().setText(rgb_to_hex(*self.color))
        elif action == copy_rgb_act:
            QApplication.clipboard().setText(f"rgb({self.color[0]}, {self.color[1]}, {self.color[2]})")
        elif action == remove_act:
            self.remove_requested.emit(self.index)


# ── Palette Strip Widget ──────────────────────────────────

class PaletteStripWidget(QWidget):
    """Displays the full palette as a horizontal strip with gradient between colors."""
    selection_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.palette = []
        self.selected_index = -1
        self.setMinimumHeight(60)
        self.setMaximumHeight(80)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_palette(self, palette):
        self.palette = list(palette); self.update()

    def set_selected(self, idx):
        self.selected_index = idx; self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        try:
            if not self.palette:
                p.fillRect(self.rect(), QColor("#16161e"))
                p.setPen(QColor("#3b4261"))
                p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No colors yet")
                return

            n = len(self.palette)
            w = self.width()
            h = self.height()

            grad = QLinearGradient(0, 0, w, 0)
            for i, col in enumerate(self.palette):
                grad.setColorAt(i / max(n - 1, 1), QColor(*col))
            if n == 1:
                grad.setColorAt(1.0, QColor(*self.palette[0]))
            p.fillRect(self.rect(), grad)

            if 0 <= self.selected_index < n:
                x_start = self.selected_index * w // n
                x_end = (self.selected_index + 1) * w // n
                p.setPen(QPen(QColor("#7aa2f7"), 3))
                p.drawLine(x_start, 0, x_end, 0)
                p.drawLine(x_start, h - 1, x_end, h - 1)

            p.setPen(QPen(QColor(0, 0, 0, 40), 1))
            for i in range(1, n):
                x = i * w // n
                p.drawLine(x, 0, x, h)
        finally:
            p.end()

    def mousePressEvent(self, e):
        if e.button()==Qt.MouseButton.LeftButton and self.palette:
            n = len(self.palette); idx = e.position().x() * n // self.width()
            idx = max(0, min(n-1, idx))
            self.selected_index = idx; self.selection_changed.emit(idx); self.update()


# ── Color Editor Panel ────────────────────────────────────

class ColorEditorPanel(QWidget):
    """Full-featured color editor with RGB/HSV sliders and hex input."""
    color_changed = Signal(tuple)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._color = (128, 128, 128)
        self._updating = False
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(12)

        # ── Top: Large preview + hex ─────────────────────
        top_row = QHBoxLayout()
        top_row.setSpacing(12)

        # Large color preview
        self.preview = QLabel()
        self.preview.setFixedSize(80, 80)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet("border-radius:12px; font-size:11px; font-weight:600;")
        top_row.addWidget(self.preview)

        # Color info column
        info_col = QVBoxLayout()
        info_col.setSpacing(6)

        # Hex input
        hex_row = QHBoxLayout()
        hex_row.addWidget(QLabel("HEX:"))
        self.hex_input = QLineEdit("#808080")
        self.hex_input.setMaximumWidth(100)
        self.hex_input.editingFinished.connect(self._on_hex_changed)
        hex_row.addWidget(self.hex_input)
        copy_btn = QToolButton()
        copy_btn.setText("📋")
        copy_btn.setToolTip("Copy HEX to clipboard")
        copy_btn.setFixedSize(28, 28)
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(self.hex_input.text()))
        hex_row.addWidget(copy_btn)
        info_col.addLayout(hex_row)

        # RGB / HSL labels
        self.rgb_label = QLabel("RGB: 128, 128, 128")
        self.rgb_label.setStyleSheet("color:#565f89; font-size:11px;")
        info_col.addWidget(self.rgb_label)

        self.hsl_label = QLabel("HSL: 0°, 0%, 50%")
        self.hsl_label.setStyleSheet("color:#565f89; font-size:11px;")
        info_col.addWidget(self.hsl_label)

        self.lum_label = QLabel("Luminance: 0.22")
        self.lum_label.setStyleSheet("color:#565f89; font-size:11px;")
        info_col.addWidget(self.lum_label)

        top_row.addLayout(info_col, 1)
        main_layout.addLayout(top_row)

        # ── Separator ────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background:#3b4261; max-height:1px;")
        main_layout.addWidget(sep)

        # ── HSV Sliders ──────────────────────────────────
        hsv_group = QGroupBox("HSV")
        hsv_lay = QVBoxLayout(hsv_group)
        hsv_lay.setSpacing(8)

        # Hue
        self.hue_slider = GradientSlider()
        self.hue_slider.set_range(0, 360)
        self._setup_hue_gradient()
        self.hue_label = QLabel("0°")
        self.hue_label.setFixedWidth(36)
        self.hue_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        h_row = QHBoxLayout(); h_row.addWidget(QLabel("H")); h_row.addWidget(self.hue_slider, 1); h_row.addWidget(self.hue_label)
        hsv_lay.addLayout(h_row)

        # Saturation
        self.sat_slider = GradientSlider()
        self.sat_slider.set_range(0, 100)
        self.sat_label = QLabel("0%")
        self.sat_label.setFixedWidth(36)
        self.sat_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        s_row = QHBoxLayout(); s_row.addWidget(QLabel("S")); s_row.addWidget(self.sat_slider, 1); s_row.addWidget(self.sat_label)
        hsv_lay.addLayout(s_row)

        # Value
        self.val_slider = GradientSlider()
        self.val_slider.set_range(0, 100)
        self.val_label = QLabel("0%")
        self.val_label.setFixedWidth(36)
        self.val_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        v_row = QHBoxLayout(); v_row.addWidget(QLabel("V")); v_row.addWidget(self.val_slider, 1); v_row.addWidget(self.val_label)
        hsv_lay.addLayout(v_row)

        main_layout.addWidget(hsv_group)

        # ── RGB Sliders ──────────────────────────────────
        rgb_group = QGroupBox("RGB")
        rgb_lay = QVBoxLayout(rgb_group)
        rgb_lay.setSpacing(8)

        self.r_slider = GradientSlider(); self.r_slider.set_range(0, 255)
        self.g_slider = GradientSlider(); self.g_slider.set_range(0, 255)
        self.b_slider = GradientSlider(); self.b_slider.set_range(0, 255)

        self.r_label = QLabel("0"); self.r_label.setFixedWidth(30); self.r_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.g_label = QLabel("0"); self.g_label.setFixedWidth(30); self.g_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.b_label = QLabel("0"); self.b_label.setFixedWidth(30); self.b_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        r_row = QHBoxLayout(); r_row.addWidget(QLabel("R")); r_row.addWidget(self.r_slider,1); r_row.addWidget(self.r_label)
        g_row = QHBoxLayout(); g_row.addWidget(QLabel("G")); g_row.addWidget(self.g_slider,1); g_row.addWidget(self.g_label)
        b_row = QHBoxLayout(); b_row.addWidget(QLabel("B")); b_row.addWidget(self.b_slider,1); b_row.addWidget(self.b_label)
        rgb_lay.addLayout(r_row); rgb_lay.addLayout(g_row); rgb_lay.addLayout(b_row)

        main_layout.addWidget(rgb_group)

        # ── Connections ──────────────────────────────────
        self.hue_slider.valueChanged.connect(self._on_hsv_changed)
        self.sat_slider.valueChanged.connect(self._on_hsv_changed)
        self.val_slider.valueChanged.connect(self._on_hsv_changed)
        self.r_slider.valueChanged.connect(self._on_rgb_changed)
        self.g_slider.valueChanged.connect(self._on_rgb_changed)
        self.b_slider.valueChanged.connect(self._on_rgb_changed)

        main_layout.addStretch()
        self.set_color(self._color)

    def _setup_hue_gradient(self):
        stops = []
        for i in range(7):
            h = i * 60
            r, g, b = hsv_to_rgb(h, 1.0, 1.0)
            stops.append((i / 6, QColor(r, g, b)))
        self.hue_slider.set_gradient(stops)

    def _update_slider_gradients(self):
        h, s, v = rgb_to_hsv(*self._color)
        r, g, b = self._color
        # Saturation gradient: gray → full hue
        gray = hsv_to_rgb(h, 0, v)
        full = hsv_to_rgb(h, 1, v)
        self.sat_slider.set_gradient([(0, QColor(*gray)), (1, QColor(*full))])
        # Value gradient: black → full
        self.val_slider.set_gradient([(0, QColor(0,0,0)), (1, QColor(*hsv_to_rgb(h, s, 1)))])
        # RGB gradients
        self.r_slider.set_gradient([(0, QColor(0,g,b)), (1, QColor(255,g,b))])
        self.g_slider.set_gradient([(0, QColor(r,0,b)), (1, QColor(r,255,b))])
        self.b_slider.set_gradient([(0, QColor(r,g,0)), (1, QColor(r,g,255))])

    def set_color(self, color):
        self._updating = True
        self._color = color
        r, g, b = color
        h, s, v = rgb_to_hsv(r, g, b)
        hx = rgb_to_hex(r, g, b)
        hsl = rgb_to_hsl(r, g, b)
        lum = 0.299*r + 0.587*g + 0.114*b
        tc = "#1a1b26" if lum > 128 else "#c0caf5"

        self.preview.setStyleSheet(
            f"background:{hx}; border-radius:12px; color:{tc}; font-size:11px; font-weight:600;")
        self.preview.setText(hx)
        self.hex_input.setText(hx)
        self.rgb_label.setText(f"RGB: {r}, {g}, {b}")
        self.hsl_label.setText(f"HSL: {hsl[0]:.0f}°, {hsl[1]:.0f}%, {hsl[2]:.0f}%")
        self.lum_label.setText(f"Luminance: {lum/255:.2f}")

        self.hue_slider.set_value(int(h))
        self.sat_slider.set_value(int(s * 100))
        self.val_slider.set_value(int(v * 100))
        self.hue_label.setText(f"{int(h)}°")
        self.sat_label.setText(f"{int(s*100)}%")
        self.val_label.setText(f"{int(v*100)}%")

        self.r_slider.set_value(r)
        self.g_slider.set_value(g)
        self.b_slider.set_value(b)
        self.r_label.setText(str(r))
        self.g_label.setText(str(g))
        self.b_label.setText(str(b))

        self._update_slider_gradients()
        self._updating = False

    def _on_hsv_changed(self, _=None):
        if self._updating: return
        h = self.hue_slider.value()
        s = self.sat_slider.value() / 100
        v = self.val_slider.value() / 100
        self._color = hsv_to_rgb(h, s, v)
        self.color_changed.emit(self._color)
        self.set_color(self._color)

    def _on_rgb_changed(self, _=None):
        if self._updating: return
        r = self.r_slider.value()
        g = self.g_slider.value()
        b = self.b_slider.value()
        self._color = (r, g, b)
        self.color_changed.emit(self._color)
        self.set_color(self._color)

    def _on_hex_changed(self):
        try:
            c = hex_to_rgb(self.hex_input.text())
            self._color = c
            self.color_changed.emit(c)
            self.set_color(c)
        except ValueError:
            self.hex_input.setText(rgb_to_hex(*self._color))


# ── Contrast Checker Widget ───────────────────────────────

class ContrastCheckerWidget(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("♿ Contrast Checker", parent)
        lay = QVBoxLayout(self)
        lay.setSpacing(10)

        # FG row
        fg_row = QHBoxLayout()
        self.fg_swatch = QLabel()
        self.fg_swatch.setFixedSize(36, 24)
        self.fg_swatch.setStyleSheet("background:#ffffff; border-radius:4px;")
        self.fg_hex = QLineEdit("#FFFFFF")
        self.fg_hex.setMaximumWidth(85)
        self.fg_hex.editingFinished.connect(self._recalc)
        fg_btn = QPushButton("Pick")
        fg_btn.setFixedWidth(44)
        fg_btn.clicked.connect(lambda: self._pick("fg"))
        fg_row.addWidget(QLabel("FG:")); fg_row.addWidget(self.fg_swatch)
        fg_row.addWidget(self.fg_hex); fg_row.addWidget(fg_btn)
        lay.addLayout(fg_row)

        # BG row
        bg_row = QHBoxLayout()
        self.bg_swatch = QLabel()
        self.bg_swatch.setFixedSize(36, 24)
        self.bg_swatch.setStyleSheet("background:#000000; border-radius:4px;")
        self.bg_hex = QLineEdit("#000000")
        self.bg_hex.setMaximumWidth(85)
        self.bg_hex.editingFinished.connect(self._recalc)
        bg_btn = QPushButton("Pick")
        bg_btn.setFixedWidth(44)
        bg_btn.clicked.connect(lambda: self._pick("bg"))
        bg_row.addWidget(QLabel("BG:")); bg_row.addWidget(self.bg_swatch)
        bg_row.addWidget(self.bg_hex); bg_row.addWidget(bg_btn)
        lay.addLayout(bg_row)

        # Swap button
        swap_row = QHBoxLayout()
        swap_btn = QPushButton("⇄ Swap")
        swap_btn.setFixedWidth(70)
        swap_btn.clicked.connect(self._swap)
        swap_row.addStretch()
        swap_row.addWidget(swap_btn)
        swap_row.addStretch()
        lay.addLayout(swap_row)

        # Sample text
        self.sample = QLabel("Sample Text  Aa Bb Cc  123")
        self.sample.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sample.setMinimumHeight(44)
        self.sample.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        lay.addWidget(self.sample)

        # Result
        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("font-weight:700; font-size:15px;")
        lay.addWidget(self.result_label)

        self._recalc()

    def _pick(self, target):
        dlg = QColorDialog(self)
        if dlg.exec() == QColorDialog.DialogCode.Accepted:
            c = dlg.selectedColor()
            hx = c.name().upper()
            if target == "fg":
                self.fg_hex.setText(hx); self.fg_swatch.setStyleSheet(f"background:{hx}; border-radius:4px;")
            else:
                self.bg_hex.setText(hx); self.bg_swatch.setStyleSheet(f"background:{hx}; border-radius:4px;")
            self._recalc()

    def _swap(self):
        fg = self.fg_hex.text(); bg = self.bg_hex.text()
        self.fg_hex.setText(bg); self.bg_hex.setText(fg)
        self.fg_swatch.setStyleSheet(f"background:{bg}; border-radius:4px;")
        self.bg_swatch.setStyleSheet(f"background:{fg}; border-radius:4px;")
        self._recalc()

    def _recalc(self):
        try:
            fg = hex_to_rgb(self.fg_hex.text()); bg = hex_to_rgb(self.bg_hex.text())
        except ValueError:
            return
        fg_lum = _relative_luminance_py(*fg) if not NUMBA_AVAILABLE else float(_relative_luminance_numba(*fg))
        bg_lum = _relative_luminance_py(*bg) if not NUMBA_AVAILABLE else float(_relative_luminance_numba(*bg))
        l1, l2 = max(fg_lum,bg_lum), min(fg_lum,bg_lum)
        ratio = (l1+0.05)/(l2+0.05)
        self.sample.setStyleSheet(
            f"background:{self.bg_hex.text()}; color:{self.fg_hex.text()}; "
            f"border-radius:6px; padding:8px;")
        aa = ratio >= 4.5; aaa = ratio >= 7.0
        aa_large = ratio >= 3.0; aaa_large = ratio >= 4.5
        color = "#9ece6a" if aaa else "#e0af68" if aa else "#f7768e"
        self.result_label.setText(f"Contrast: {ratio:.2f}:1")
        badges = []
        badges.append(f"AA {'✓' if aa else '✗'}")
        badges.append(f"AAA {'✓' if aaa else '✗'}")
        badges.append(f"AA-L {'✓' if aa_large else '✗'}")
        badges.append(f"AAA-L {'✓' if aaa_large else '✗'}")
        self.result_label.setStyleSheet(f"color:{color}; font-weight:700; font-size:14px;")
        self.result_label.setText(f"{ratio:.2f}:1  {'  '.join(badges)}")


# ── Color Blindness Preview ───────────────────────────────

class ColorBlindnessPreview(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("👁 Color Blindness Simulation", parent)
        lay = QVBoxLayout(self)
        lay.setSpacing(8)
        self.previews = {}
        for cb_type, label in [("proto","Protanopia"),("deuto","Deuteranopia"),("trita","Tritanopia")]:
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setFixedWidth(110)
            lbl.setStyleSheet("color:#565f89; font-size:11px;")
            strip = QLabel()
            strip.setFixedHeight(24)
            strip.setStyleSheet("border-radius:4px;")
            self.previews[cb_type] = strip
            row.addWidget(lbl); row.addWidget(strip, 1)
            lay.addLayout(row)

    def set_palette(self, palette):
        for cb_type, strip in self.previews.items():
            simulated = simulate_colorblind(palette, cb_type)
            px = create_palette_pixmap(simulated, w=300, h=24)
            strip.setPixmap(px)


# ── Variation Preview ─────────────────────────────────────

class VariationPreview(QGroupBox):
    variation_applied = Signal(str, float)

    def __init__(self, parent=None):
        super().__init__("✨ Variations", parent)
        lay = QGridLayout(self)
        lay.setSpacing(6)
        self.var_strips = {}
        variations = [
            ("lighter", "☀ Lighter"), ("darker", "🌙 Darker"),
            ("muted", "🔇 Muted"), ("vivid", "📢 Vivid"),
            ("pastel", "🍭 Pastel"), ("warm", "🔥 Warm"),
            ("cool", "❄ Cool"),
        ]
        for i, (key, label) in enumerate(variations):
            row, col = i // 2, (i % 2) * 2
            lbl = QLabel(label)
            lbl.setFixedWidth(85)
            lbl.setStyleSheet("color:#565f89; font-size:11px;")
            strip = QLabel()
            strip.setFixedHeight(20)
            strip.setStyleSheet("border-radius:3px;")
            strip.setCursor(Qt.CursorShape.PointingHandCursor)
            strip.mousePressEvent = lambda e, k=key: self.variation_applied.emit(k, 0.3)
            self.var_strips[key] = strip
            lay.addWidget(lbl, row, col); lay.addWidget(strip, row, col+1)

    def set_palette(self, palette):
        for key, strip in self.var_strips.items():
            varied = generate_variations(palette, key, 0.3)
            px = create_palette_pixmap(varied, w=160, h=20)
            strip.setPixmap(px)


# ═══════════════════════════════════════════════════════════
#  MAIN PALETTE EDITOR TAB (IMPROVED)
# ═══════════════════════════════════════════════════════════

class PaletteEditorTab(QWidget):
    palette_saved = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._palette = [(94,129,172),(166,227,161),(249,226,175),(247,118,142),(187,154,247)]
        self._selected_idx = 0
        self._pid = None  # palette id in DB
        self._name = "New Palette"
        self._modified = False
        self._init_ui()

    def _init_ui(self):
        outer = QHBoxLayout(self)
        outer.setSpacing(0)
        outer.setContentsMargins(0, 0, 0, 0)

        # ── Left: Scroll area with swatches + editor ────
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(16, 16, 16, 16)
        left_layout.setSpacing(12)

        # Palette name
        name_row = QHBoxLayout()
        name_label = QLabel("Palette:")
        name_label.setStyleSheet("color:#565f89; font-weight:600;")
        self.name_edit = QLineEdit(self._name)
        self.name_edit.setStyleSheet("font-weight:600; font-size:14px;")
        self.name_edit.textChanged.connect(lambda _: self._mark_modified())
        name_row.addWidget(name_label); name_row.addWidget(self.name_edit, 1)
        left_layout.addLayout(name_row)

        # ── Palette strip (gradient preview) ────────────
        strip_label = QLabel("Preview")
        strip_label.setStyleSheet("color:#7aa2f7; font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:1px;")
        left_layout.addWidget(strip_label)

        self.strip_widget = PaletteStripWidget()
        self.strip_widget.selection_changed.connect(self._on_strip_select)
        left_layout.addWidget(self.strip_widget)

        # ── Swatches row ────────────────────────────────
        swatch_label = QLabel("Colors")
        swatch_label.setStyleSheet("color:#7aa2f7; font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:1px;")
        left_layout.addWidget(swatch_label)

        self.swatch_container = QWidget()
        self.swatch_layout = QHBoxLayout(self.swatch_container)
        self.swatch_layout.setContentsMargins(0, 0, 0, 0)
        self.swatch_layout.setSpacing(4)
        self.swatch_layout.addStretch()
        left_layout.addWidget(self.swatch_container)

        # Swatch action buttons
        swatch_btns = QHBoxLayout()
        self.add_btn = QPushButton("+ Add Color")
        self.add_btn.setProperty("class", "accent")
        self.add_btn.clicked.connect(self._add_color)
        self.pick_screen_btn = QPushButton("📺 Pick Screen")
        self.pick_screen_btn.clicked.connect(self._pick_screen_color)
        self.pick_btn = QPushButton("🎨 Pick Color")
        self.pick_btn.clicked.connect(self._pick_color)
        self.remove_btn = QPushButton("✕ Remove")
        self.remove_btn.setProperty("class", "danger")
        self.remove_btn.clicked.connect(self._remove_color)
        swatch_btns.addWidget(self.add_btn)
        swatch_btns.addWidget(self.pick_screen_btn)
        swatch_btns.addWidget(self.pick_btn)
        swatch_btns.addWidget(self.remove_btn)
        left_layout.addLayout(swatch_btns)

        # ── Separator ────────────────────────────────────
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background:#24283b; max-height:1px;")
        left_layout.addWidget(sep)

        # ── Color Editor Panel ──────────────────────────
        editor_label = QLabel("Edit Selected Color")
        editor_label.setStyleSheet("color:#7aa2f7; font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:1px;")
        left_layout.addWidget(editor_label)

        self.color_editor = ColorEditorPanel()
        self.color_editor.color_changed.connect(self._on_color_edited)
        left_layout.addWidget(self.color_editor)

        left_layout.addStretch()

        left_scroll = QScrollArea()
        left_scroll.setWidget(left_widget)
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # ── Right: Sidebar with tools ────────────────────
        right_widget = QWidget()
        right_widget.setMaximumWidth(320)
        right_widget.setMinimumWidth(280)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(12, 16, 12, 16)
        right_layout.setSpacing(10)

        # Palette actions
        actions_label = QLabel("Actions")
        actions_label.setStyleSheet("color:#7aa2f7; font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:1px;")
        right_layout.addWidget(actions_label)

        act_grid = QGridLayout()
        act_grid.setSpacing(6)
        self.sort_hue_btn = QPushButton("Sort by Hue")
        self.sort_bright_btn = QPushButton("Sort by Brightness")
        self.sort_sat_btn = QPushButton("Sort by Saturation")
        self.reverse_btn = QPushButton("Reverse")
        self.harmony_btn = QPushButton("🎵 Harmony")
        self.import_img_btn = QPushButton("🖼 From Image")

        self.sort_hue_btn.clicked.connect(lambda: self._sort_palette("hue"))
        self.sort_bright_btn.clicked.connect(lambda: self._sort_palette("brightness"))
        self.sort_sat_btn.clicked.connect(lambda: self._sort_palette("saturation"))
        self.reverse_btn.clicked.connect(self._reverse_palette)
        self.harmony_btn.clicked.connect(self._show_harmony_dialog)
        self.import_img_btn.clicked.connect(self._import_from_image)

        act_grid.addWidget(self.sort_hue_btn, 0, 0)
        act_grid.addWidget(self.sort_bright_btn, 0, 1)
        act_grid.addWidget(self.sort_sat_btn, 1, 0)
        act_grid.addWidget(self.reverse_btn, 1, 1)
        act_grid.addWidget(self.harmony_btn, 2, 0)
        act_grid.addWidget(self.import_img_btn, 2, 1)
        right_layout.addLayout(act_grid)

        # Save / Export
        save_export_row = QHBoxLayout()
        self.save_btn = QPushButton("💾 Save")
        self.save_btn.setProperty("class", "accent")
        self.save_btn.clicked.connect(self._save_palette)
        self.export_btn = QPushButton("📤 Export")
        self.export_btn.clicked.connect(self._export_palette)
        save_export_row.addWidget(self.save_btn)
        save_export_row.addWidget(self.export_btn)
        right_layout.addLayout(save_export_row)

        # ── Variation Preview ───────────────────────────
        self.variation_preview = VariationPreview()
        self.variation_preview.variation_applied.connect(self._apply_variation)
        right_layout.addWidget(self.variation_preview)

        # ── Contrast Checker ────────────────────────────
        self.contrast_checker = ContrastCheckerWidget()
        right_layout.addWidget(self.contrast_checker)

        # ── Color Blindness Preview ─────────────────────
        self.cb_preview = ColorBlindnessPreview()
        right_layout.addWidget(self.cb_preview)

        # ── Palette Metadata ────────────────────────────
        self.meta_group = QGroupBox("📊 Metadata")
        meta_lay = QFormLayout(self.meta_group)
        meta_lay.setSpacing(4)
        self.meta_bright = QLabel("—")
        self.meta_contrast = QLabel("—")
        self.meta_dominant = QLabel("—")
        self.meta_wcag = QLabel("—")
        self.meta_dupes = QLabel("—")
        meta_lay.addRow("Brightness:", self.meta_bright)
        meta_lay.addRow("Contrast:", self.meta_contrast)
        meta_lay.addRow("Dominant:", self.meta_dominant)
        meta_lay.addRow("WCAG Ratio:", self.meta_wcag)
        meta_lay.addRow("Duplicates:", self.meta_dupes)
        right_layout.addWidget(self.meta_group)

        right_layout.addStretch()

        right_scroll = QScrollArea()
        right_scroll.setWidget(right_widget)
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # ── Splitter ────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(right_scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([600, 320])

        outer.addWidget(splitter)

        # ── Screen picker ───────────────────────────────
        self._screen_picker = None

        # Initial render
        self._rebuild_swatches()
        self._select_index(0)
        self._update_metadata()

    # ── Swatch Management ────────────────────────────────

    def _rebuild_swatches(self):
        # Clear existing
        while self.swatch_layout.count() > 1:
            item = self.swatch_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        # Add swatches
        for i, col in enumerate(self._palette):
            sw = ColorSwatchWidget(i, col)
            sw.clicked.connect(self._on_swatch_click)
            sw.remove_requested.connect(self._remove_at_index)
            self.swatch_layout.insertWidget(i, sw)
        self.strip_widget.set_palette(self._palette)
        self.variation_preview.set_palette(self._palette)
        self.cb_preview.set_palette(self._palette)

    def _select_index(self, idx):
        if not self._palette: return
        self._selected_idx = max(0, min(idx, len(self._palette)-1))
        # Update swatch selection visuals
        for i in range(self.swatch_layout.count()):
            w = self.swatch_layout.itemAt(i).widget()
            if isinstance(w, ColorSwatchWidget):
                w.set_selected(w.index == self._selected_idx)
        self.strip_widget.set_selected(self._selected_idx)
        self.color_editor.set_color(self._palette[self._selected_idx])

    def _on_swatch_click(self, idx):
        self._select_index(idx)

    def _on_strip_select(self, idx):
        self._select_index(idx)

    def _on_color_edited(self, color):
        if 0 <= self._selected_idx < len(self._palette):
            self._palette[self._selected_idx] = color
            self._mark_modified()
            self._refresh_current_swatch()
            self.strip_widget.set_palette(self._palette)
            self.variation_preview.set_palette(self._palette)
            self.cb_preview.set_palette(self._palette)
            self._update_metadata()

    def _refresh_current_swatch(self):
        if 0 <= self._selected_idx < len(self._palette):
            w = self.swatch_layout.itemAt(self._selected_idx).widget()
            if isinstance(w, ColorSwatchWidget):
                w.set_color(self._palette[self._selected_idx])

    def _add_color(self):
        color = QColorDialog.getColor(QColor(*self._palette[-1] if self._palette else (128,128,128)), self)
        if color.isValid():
            self._palette.append((color.red(), color.green(), color.blue()))
            self._mark_modified()
            self._rebuild_swatches()
            self._select_index(len(self._palette)-1)
            self._update_metadata()

    def _pick_color(self):
        if 0 <= self._selected_idx < len(self._palette):
            color = QColorDialog.getColor(QColor(*self._palette[self._selected_idx]), self)
            if color.isValid():
                self._palette[self._selected_idx] = (color.red(), color.green(), color.blue())
                self._mark_modified()
                self.color_editor.set_color(self._palette[self._selected_idx])
                self._refresh_current_swatch()
                self.strip_widget.set_palette(self._palette)
                self.variation_preview.set_palette(self._palette)
                self.cb_preview.set_palette(self._palette)
                self._update_metadata()

    def _pick_screen_color(self):
        self._screen_picker = ScreenColorPicker()
        self._screen_picker.color_picked.connect(self._on_screen_color_picked)
        self._screen_picker.start()

    def _on_screen_color_picked(self, color):
        if 0 <= self._selected_idx < len(self._palette):
            self._palette[self._selected_idx] = color
            self._mark_modified()
            self.color_editor.set_color(color)
            self._refresh_current_swatch()
            self.strip_widget.set_palette(self._palette)
            self.variation_preview.set_palette(self._palette)
            self.cb_preview.set_palette(self._palette)
            self._update_metadata()

    def _remove_color(self):
        self._remove_at_index(self._selected_idx)

    def _remove_at_index(self, idx):
        if len(self._palette) <= 1:
            QMessageBox.information(self, "Info", "Palette must have at least one color.")
            return
        self._palette.pop(idx)
        self._mark_modified()
        self._rebuild_swatches()
        self._select_index(min(idx, len(self._palette)-1))
        self._update_metadata()

    # ── Actions ───────────────────────────────────────────

    def _sort_palette(self, mode):
        self._palette = sort_palette(self._palette, mode)
        self._mark_modified(); self._rebuild_swatches()
        self._select_index(0); self._update_metadata()

    def _reverse_palette(self):
        self._palette.reverse()
        self._mark_modified(); self._rebuild_swatches()
        self._select_index(0); self._update_metadata()

    def _apply_variation(self, variation, strength):
        self._palette = generate_variations(self._palette, variation, strength)
        self._mark_modified(); self._rebuild_swatches()
        self._select_index(0); self._update_metadata()

    def _show_harmony_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Generate Harmony Palette")
        dlg.setMinimumWidth(360)
        lay = QVBoxLayout(dlg)

        form = QFormLayout()
        hue_spin = QDoubleSpinBox(); hue_spin.setRange(0,360); hue_spin.setValue(180)
        harmony_combo = QComboBox()
        harmony_combo.addItems(["complementary","analogous","triadic","split_complementary","monochromatic","tetradic"])
        num_spin = QSpinBox(); num_spin.setRange(2,12); num_spin.setValue(5)
        sat_spin = QDoubleSpinBox(); sat_spin.setRange(0,1); sat_spin.setSingleStep(0.05); sat_spin.setValue(0.75)
        val_spin = QDoubleSpinBox(); val_spin.setRange(0,1); val_spin.setSingleStep(0.05); val_spin.setValue(0.75)
        form.addRow("Base Hue:", hue_spin)
        form.addRow("Harmony:", harmony_combo)
        form.addRow("Colors:", num_spin)
        form.addRow("Saturation:", sat_spin)
        form.addRow("Value:", val_spin)
        lay.addLayout(form)

        # Preview
        preview_strip = QLabel()
        preview_strip.setFixedHeight(40)
        preview_strip.setStyleSheet("border-radius:6px;")
        lay.addWidget(preview_strip)

        def _update_preview():
            colors = generate_harmony_palette(
                hue_spin.value(), harmony_combo.currentText(),
                num_spin.value(), sat_spin.value(), val_spin.value())
            px = create_palette_pixmap(colors, w=320, h=40)
            preview_strip.setPixmap(px)

        hue_spin.valueChanged.connect(_update_preview)
        harmony_combo.currentTextChanged.connect(_update_preview)
        num_spin.valueChanged.connect(_update_preview)
        sat_spin.valueChanged.connect(_update_preview)
        val_spin.valueChanged.connect(_update_preview)
        _update_preview()

        btn_row = QHBoxLayout()
        ok_btn = QPushButton("Apply"); ok_btn.setProperty("class","accent")
        cancel_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(dlg.accept)
        cancel_btn.clicked.connect(dlg.reject)
        btn_row.addStretch(); btn_row.addWidget(ok_btn); btn_row.addWidget(cancel_btn)
        lay.addLayout(btn_row)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._palette = generate_harmony_palette(
                hue_spin.value(), harmony_combo.currentText(),
                num_spin.value(), sat_spin.value(), val_spin.value())
            self._mark_modified(); self._rebuild_swatches()
            self._select_index(0); self._update_metadata()

    def _import_from_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import from Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.webp);;All Files (*)")
        if path:
            img = QImage(path)
            if img.isNull():
                QMessageBox.warning(self, "Error", "Cannot load image."); return
            colors = extract_palette_kmeans(img, 5)
            if not colors:
                QMessageBox.warning(self, "Error", "No colors extracted."); return
            self._palette = colors
            self._mark_modified(); self._rebuild_swatches()
            self._select_index(0); self._update_metadata()
            AppSettings.add_recent_file(path)

    # ── Save / Export ─────────────────────────────────────

    def _save_palette(self):
        name = self.name_edit.text().strip() or "Untitled"
        if self._pid is not None:
            controller.update_palette(self._pid, self._palette, name)
        else:
            self._pid = controller.create_palette(self._palette, name)
        self._name = name; self._modified = False
        self.palette_saved.emit()
        ToastWidget(f"Saved: {name}", parent=self.window())

    def _export_palette(self):
        name = self.name_edit.text().strip() or "palette"
        formats = ";;".join([
            "CSS (*.css)", "JSON (*.json)", "GIMP Palette (*.gpl)",
            "SCSS (*.scss)", "Tailwind JS (*.js)", "SVG (*.svg)",
            "Android XML (*.xml)", "Python (*.py)", "MAP (*.map)",
        ])
        path, _ = QFileDialog.getSaveFileName(self, "Export Palette", f"{name}", formats)
        if path:
            ext = path.rsplit(".", 1)[-1].lower() if "." in path else "map"
            fmt_map = {"css":"css","json":"json","gpl":"gpl","scss":"scss","js":"tailwind",
                       "svg":"svg","xml":"xml","py":"py","map":"map"}
            fmt = fmt_map.get(ext, "map")
            if controller.export_palette_data(self._palette, name, path, fmt):
                ToastWidget(f"Exported to {os.path.basename(path)}", parent=self.window())
            else:
                QMessageBox.warning(self, "Error", "Export failed.")

    # ── Metadata ──────────────────────────────────────────

    def _update_metadata(self):
        if not self._palette:
            self.meta_bright.setText("—")
            self.meta_contrast.setText("—")
            self.meta_dominant.setText("—")
            self.meta_wcag.setText("—")
            self.meta_dupes.setText("—")
            return
        b, c, d = calculate_metadata(self._palette)
        wcag = palette_wcag_contrast(self._palette)
        dupes = find_duplicate_colors(self._palette)
        self.meta_bright.setText(f"{b:.1f}")
        self.meta_contrast.setText(f"{c:.1f}")
        self.meta_dominant.setText(d)
        wcag_color = "#9ece6a" if wcag >= 7 else "#e0af68" if wcag >= 4.5 else "#f7768e"
        self.meta_wcag.setText(f"{wcag:.2f}:1")
        self.meta_wcag.setStyleSheet(f"color:{wcag_color}; font-weight:600;")
        self.meta_dupes.setText(f"{len(dupes)} pairs" if dupes else "None")
        if dupes:
            self.meta_dupes.setStyleSheet("color:#f7768e;")
        else:
            self.meta_dupes.setStyleSheet("color:#9ece6a;")

    def _mark_modified(self):
        self._modified = True

    # ── Public API ────────────────────────────────────────

    def load_palette(self, pid):
        palette = controller.get_palette(pid)
        if palette:
            self._pid = pid
            self._palette = list(palette)
            name = controller.db.get_name_by_id(pid) or "Untitled"
            self._name = name
            self.name_edit.setText(name)
            self._rebuild_swatches()
            self._select_index(0)
            self._update_metadata()
            self._modified = False

    def new_palette(self):
        self._pid = None
        self._palette = [(94,129,172),(166,227,161),(249,226,175),(247,118,142),(187,154,247)]
        self.name_edit.setText("New Palette")
        self._rebuild_swatches()
        self._select_index(0)
        self._update_metadata()
        self._modified = False


# ═══════════════════════════════════════════════════════════
#  BROWSE TAB
# ═══════════════════════════════════════════════════════════

class BrowseTab(QWidget):
    palette_selected = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12,12,12,12)
        lay.setSpacing(8)

        # Search / filter row
        filter_row = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("🔍 Search palettes...")
        self.search_edit.textChanged.connect(self._refresh)

        self.dominant_combo = QComboBox()
        self.dominant_combo.addItem("All"); self.dominant_combo.addItems(["R","G","B"])
        self.dominant_combo.currentTextChanged.connect(self._refresh)

        self.fav_check = QCheckBox("★ Favorites")
        self.fav_check.toggled.connect(self._refresh)

        filter_row.addWidget(self.search_edit, 1)
        filter_row.addWidget(QLabel("Dominant:"))
        filter_row.addWidget(self.dominant_combo)
        filter_row.addWidget(self.fav_check)
        lay.addLayout(filter_row)

        # Brightness range
        bright_row = QHBoxLayout()
        bright_row.addWidget(QLabel("Brightness:"))
        self.bright_min = QSpinBox(); self.bright_min.setRange(0,255); self.bright_min.setSpecialValueText("Min")
        self.bright_max = QSpinBox(); self.bright_max.setRange(0,255); self.bright_max.setValue(255); self.bright_max.setSpecialValueText("Max")
        self.bright_min.valueChanged.connect(self._refresh)
        self.bright_max.valueChanged.connect(self._refresh)
        bright_row.addWidget(self.bright_min); bright_row.addWidget(QLabel("—")); bright_row.addWidget(self.bright_max)
        bright_row.addStretch()
        lay.addLayout(bright_row)

        # Palette list
        self.palette_list = QListWidget()
        self.palette_list.setIconSize(QSize(120, 24))
        self.palette_list.setAlternatingRowColors(True)
        self.palette_list.itemDoubleClicked.connect(self._on_double_click)
        self.palette_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.palette_list.customContextMenuRequested.connect(self._context_menu)
        lay.addWidget(self.palette_list, 1)

        # Actions
        btn_row = QHBoxLayout()
        gen_btn = QPushButton("🎲 Generate 10")
        gen_btn.clicked.connect(self._generate)
        import_btn = QPushButton("📂 Import File")
        import_btn.clicked.connect(self._import_file)
        btn_row.addWidget(gen_btn); btn_row.addWidget(import_btn); btn_row.addStretch()
        lay.addLayout(btn_row)

        self._refresh()

    def _refresh(self):
        self.palette_list.clear()
        kwargs = {}
        q = self.search_edit.text().strip()
        if q: kwargs["name_query"] = q
        dom = self.dominant_combo.currentText()
        if dom != "All": kwargs["dominant"] = dom
        if self.fav_check.isChecked(): kwargs["favorite_only"] = True
        bmin = self.bright_min.value()
        bmax = self.bright_max.value()
        if bmin > 0: kwargs["min_bright"] = float(bmin)
        if bmax < 255: kwargs["max_bright"] = float(bmax)

        rows = controller.search_palettes(**kwargs)
        for row in rows:
            pid, name, bright, contrast, dominant, num_colors, packed, tags, favorite = row
            palette = unpack_palette(packed, num_colors)
            px = create_palette_pixmap(palette, 120, 24)
            item = QListWidgetItem(QIcon(px), f"{'★ ' if favorite else ''}{name}  ({num_colors} colors)")
            item.setData(Qt.ItemDataRole.UserRole, pid)
            self.palette_list.addItem(item)

    def _on_double_click(self, item):
        pid = item.data(Qt.ItemDataRole.UserRole)
        if pid: self.palette_selected.emit(pid)

    def _context_menu(self, pos):
        item = self.palette_list.itemAt(pos)
        if not item: return
        pid = item.data(Qt.ItemDataRole.UserRole)
        menu = QMenu(self)
        open_act = menu.addAction("Edit")
        fav_act = menu.addAction("★ Toggle Favorite")
        rename_act = menu.addAction("Rename")
        del_act = menu.addAction("Delete")
        action = menu.exec(self.palette_list.mapToGlobal(pos))
        if action == open_act: self.palette_selected.emit(pid)
        elif action == fav_act: controller.db.toggle_favorite(pid); self._refresh()
        elif action == rename_act:
            new_name, ok = QInputDialog.getText(self, "Rename", "Name:", text=controller.db.get_name_by_id(pid))
            if ok and new_name.strip(): controller.rename_palette(pid, new_name.strip()); self._refresh()
        elif action == del_act:
            if QMessageBox.question(self, "Delete", "Delete this palette?") == QMessageBox.StandardButton.Yes:
                controller.delete_palette(pid); self._refresh()

    def _generate(self):
        controller.generate_new_palettes(10); self._refresh()

    def _import_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import Palette", "",
            "Palette Files (*.map *.gpl *.ase *.csv *.hex *.txt);;All Files (*)")
        if path:
            pid, palette, name = controller.import_file_to_db(path)
            if pid:
                AppSettings.add_recent_file(path)
                self._refresh()
                ToastWidget(f"Imported: {name}", parent=self.window())
            else:
                QMessageBox.warning(self, "Error", "No colors found in file.")


# ═══════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🦆 DuckPalette — Palette Manager")
        self.setMinimumSize(1100, 750)
        self.resize(1280, 800)

        # Central tabs
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.setCentralWidget(self.tabs)

        # Editor tab
        self.editor_tab = PaletteEditorTab()
        self.editor_tab.palette_saved.connect(self._on_saved)

        # Browse tab
        self.browse_tab = BrowseTab()
        self.browse_tab.palette_selected.connect(self._open_palette)

        self.tabs.addTab(self.editor_tab, "🎨 Editor")
        self.tabs.addTab(self.browse_tab, "📚 Browse")

        # Toolbar
        self._build_toolbar()

        # Status bar
        self.statusBar().showMessage("Ready")

        # Settings
        settings = AppSettings.load()
        geo = settings.get("window_geometry")
        if geo is not None:
            try:
                if isinstance(geo, list):
                    # Old format: list of byte integers [23, 0, 255, ...]
                    self.restoreGeometry(QByteArray(bytes(geo)))
                elif isinstance(geo, str):
                    # New format: base64-encoded string
                    self.restoreGeometry(QByteArray.fromBase64(geo.encode("utf-8")))
            except Exception as e:
                logger.warning("Failed to restore window geometry: %s", e)    

    def _build_toolbar(self):
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(20, 20))
        self.addToolBar(toolbar)

        new_act = QAction("📄 New", self)
        new_act.setShortcut(QKeySequence("Ctrl+N"))
        new_act.triggered.connect(self._new_palette)
        toolbar.addAction(new_act)

        save_act = QAction("💾 Save", self)
        save_act.setShortcut(QKeySequence("Ctrl+S"))
        save_act.triggered.connect(self.editor_tab._save_palette)
        toolbar.addAction(save_act)

        toolbar.addSeparator()

        gen_act = QAction("🎲 Generate", self)
        gen_act.triggered.connect(lambda: (controller.generate_new_palettes(5), self.browse_tab._refresh()))
        toolbar.addAction(gen_act)

        toolbar.addSeparator()

        import_act = QAction("📂 Import", self)
        import_act.setShortcut(QKeySequence("Ctrl+O"))
        import_act.triggered.connect(self._import_file)
        toolbar.addAction(import_act)

        export_act = QAction("📤 Export", self)
        export_act.triggered.connect(self.editor_tab._export_palette)
        toolbar.addAction(export_act)

        toolbar.addSeparator()

        screen_pick_act = QAction("📺 Screen Pick", self)
        screen_pick_act.triggered.connect(self.editor_tab._pick_screen_color)
        toolbar.addAction(screen_pick_act)

        # Spacer
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        db_count_label = QLabel(f"DB: {controller.db.count()} palettes")
        db_count_label.setStyleSheet("color:#565f89; font-size:11px; padding-right:8px;")
        toolbar.addWidget(db_count_label)

    def _new_palette(self):
        self.editor_tab.new_palette()
        self.tabs.setCurrentIndex(0)

    def _open_palette(self, pid):
        self.editor_tab.load_palette(pid)
        self.tabs.setCurrentIndex(0)

    def _import_file(self):
        self.browse_tab._import_file()

    def _on_saved(self):
        self.browse_tab._refresh()

    def closeEvent(self, e):
        settings = {
            "window_geometry": bytes(self.saveGeometry().toBase64()).decode("utf-8"),
        }
        AppSettings.save(settings)
        super().closeEvent(e)


# ═══════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════

def cli_main(args):
    if args.cmd == "list":
        rows = controller.search_palettes(limit=args.limit or 50)
        for row in rows:
            pid, name, bright, contrast, dominant, num_colors, *_ = row
            print(f"  [{pid:>4}] {name:<30} colors={num_colors} bright={bright:.1f} contrast={contrast:.1f} dom={dominant}")
    elif args.cmd == "generate":
        count = args.count or 10
        controller.generate_new_palettes(count)
        print(f"Generated {count} palettes.")
    elif args.cmd == "export":
        pid = args.id
        fmt = args.format or "map"
        palette = controller.get_palette(pid)
        if not palette:
            print(f"Palette {pid} not found."); return
        name = controller.db.get_name_by_id(pid) or "palette"
        out = args.output or f"palette_{pid}.{fmt}"
        ok = controller.export_palette_data(palette, name, out, fmt)
        print(f"Export {'OK' if ok else 'FAILED'}: {out}")
    elif args.cmd == "import":
        path = args.file
        pid, palette, name = controller.import_file_to_db(path)
        if pid:
            print(f"Imported '{name}' as palette #{pid} ({len(palette)} colors)")
        else:
            print("No colors found in file.")
    elif args.cmd == "info":
        pid = args.id
        palette = controller.get_palette(pid)
        if not palette:
            print(f"Palette {pid} not found."); return
        b, c, d = calculate_metadata(palette)
        wcag = palette_wcag_contrast(palette)
        print(f"Palette #{pid}: {len(palette)} colors")
        print(f"  Brightness: {b:.1f}  Contrast: {c:.1f}  Dominant: {d}  WCAG: {wcag:.2f}")
        for i, col in enumerate(palette):
            print(f"  [{i}] {rgb_to_hex(*col)}  rgb{col}")


# ═══════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="DuckPalette — Palette Manager")
    sub = parser.add_subparsers(dest="mode")
    sub.default = "gui"

    cli_parser = sub.add_parser("cli", help="CLI mode")
    cli_sub = cli_parser.add_subparsers(dest="cmd")
    list_p = cli_sub.add_parser("list")
    list_p.add_argument("--limit", type=int, default=50)
    gen_p = cli_sub.add_parser("generate")
    gen_p.add_argument("--count", type=int, default=10)
    exp_p = cli_sub.add_parser("export")
    exp_p.add_argument("id", type=int)
    exp_p.add_argument("--format", default="map")
    exp_p.add_argument("--output")
    imp_p = cli_sub.add_parser("import")
    imp_p.add_argument("file")
    info_p = cli_sub.add_parser("info")
    info_p.add_argument("id", type=int)

    args = parser.parse_args()

    if args.mode == "cli" and args.cmd:
        cli_main(args)
    else:
        app = QApplication(sys.argv)
        app.setStyleSheet(DARK_STYLE)
        app.setStyle("Fusion")
        window = MainWindow()
        window.show()
        sys.exit(app.exec())


if __name__ == "__main__":
    main()