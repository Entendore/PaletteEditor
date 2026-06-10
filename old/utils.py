import random
import math
from typing import List, Tuple

def pack_palette(palette: List[Tuple[int, int, int]], bits_per_channel: int = 8) -> int:
    """
    Packs a list of (R,G,B) tuples into a single integer.
    Supports up to 5 colors at 8-bits per channel (120 bits) using HUGEINT.
    """
    max_val = (1 << bits_per_channel) - 1
    packed = 0
    shift = 0
    
    for r, g, b in palette:
        # Clamp values
        r = max(0, min(r, 255))
        g = max(0, min(g, 255))
        b = max(0, min(b, 255))
        
        # Scale down if bits_per_channel changes, though 8 is standard
        r_scaled = int(r * max_val / 255)
        g_scaled = int(g * max_val / 255)
        b_scaled = int(b * max_val / 255)
        
        color_bits = (r_scaled << (2*bits_per_channel)) | (g_scaled << bits_per_channel) | b_scaled
        packed |= color_bits << shift
        shift += 3 * bits_per_channel
    return packed

def unpack_palette(packed: int, num_colors: int = 5, bits_per_channel: int = 8) -> List[Tuple[int, int, int]]:
    """Unpacks an integer into a list of (R,G,B) tuples."""
    max_val = (1 << bits_per_channel) - 1
    palette = []
    for _ in range(num_colors):
        color_bits = packed & ((1 << (3*bits_per_channel)) - 1)
        b = color_bits & max_val
        g = (color_bits >> bits_per_channel) & max_val
        r = (color_bits >> (2*bits_per_channel)) & max_val
        palette.append((
            int(r * 255 / max_val),
            int(g * 255 / max_val),
            int(b * 255 / max_val)
        ))
        packed >>= 3*bits_per_channel
    return palette

def calculate_metadata(palette: List[Tuple[int, int, int]]) -> Tuple[float, float, str]:
    """
    Calculates brightness (perceived), contrast, and dominant channel.
    """
    if not palette:
        return 0.0, 0.0, 'N'
    
    # Perceived luminance weights: R=0.299, G=0.587, B=0.114
    luminances = []
    sums = [0, 0, 0]
    
    for r, g, b in palette:
        luminances.append(0.299 * r + 0.587 * g + 0.114 * b)
        sums[0] += r
        sums[1] += g
        sums[2] += b
    
    brightness = sum(luminances) / len(palette)
    contrast = max(luminances) - min(luminances) if luminances else 0.0
    
    dominant_index = sums.index(max(sums))
    dominant = ['R', 'G', 'B'][dominant_index]
    
    return brightness, contrast, dominant

def generate_random_palette(seed: int, num_colors: int = 5) -> List[Tuple[int, int, int]]:
    random.seed(seed)
    return [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(num_colors)]