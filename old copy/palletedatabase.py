import duckdb
import random
import os

# ----------------------------
# Palette Bitmasking Utilities
# ----------------------------
def pack_palette(palette, bits_per_channel=5):
    max_val = (1 << bits_per_channel) - 1
    packed = 0
    shift = 0
    for r, g, b in palette:
        r = r * max_val // 255
        g = g * max_val // 255
        b = b * max_val // 255
        color_bits = (r << (2*bits_per_channel)) | (g << bits_per_channel) | b
        packed |= color_bits << shift
        shift += 3 * bits_per_channel
    return packed

def unpack_palette(packed, num_colors=5, bits_per_channel=5):
    max_val = (1 << bits_per_channel) - 1
    palette = []
    for _ in range(num_colors):
        color_bits = packed & ((1 << (3*bits_per_channel)) - 1)
        b = color_bits & max_val
        g = (color_bits >> bits_per_channel) & max_val
        r = (color_bits >> (2*bits_per_channel)) & max_val
        palette.append((r * 255 // max_val, g * 255 // max_val, b * 255 // max_val))
        packed >>= 3*bits_per_channel
    return palette

# ----------------------------
# Palette Metadata Functions
# ----------------------------
def palette_metadata(palette):
    # Average brightness
    brightness = sum(sum(color)/3 for color in palette)/len(palette)
    # Contrast: max luminance - min luminance
    luminances = [sum(color)/3 for color in palette]
    contrast = max(luminances) - min(luminances)
    # Dominant color (R,G,B channel with max sum)
    sums = [sum(c) for c in zip(*palette)]
    dominant_index = sums.index(max(sums))
    dominant = ['R','G','B'][dominant_index]
    return brightness, contrast, dominant

# ----------------------------
# Procedural Palette Generation
# ----------------------------
def generate_palette(seed, num_colors=5):
    random.seed(seed)
    return [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(num_colors)]

# ----------------------------
# DuckDB Setup
# ----------------------------
conn = duckdb.connect(database='palettes.duckdb', read_only=False)
conn.execute('''
CREATE TABLE IF NOT EXISTS palettes (
    id BIGINT PRIMARY KEY,
    seed BIGINT,
    num_colors INTEGER,
    bits_per_channel INTEGER,
    packed_palette BIGINT,
    brightness DOUBLE,
    contrast DOUBLE,
    dominant TEXT
)
''')

# ----------------------------
# Insert Palette
# ----------------------------
def insert_palette(seed, palette_id, num_colors=5, bits_per_channel=5):
    palette = generate_palette(seed, num_colors)
    packed = pack_palette(palette, bits_per_channel)
    brightness, contrast, dominant = palette_metadata(palette)
    conn.execute('''
        INSERT INTO palettes (id, seed, num_colors, bits_per_channel, packed_palette, brightness, contrast, dominant)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', [palette_id, seed, num_colors, bits_per_channel, packed, brightness, contrast, dominant])

# ----------------------------
# Export Single Palette to .map
# ----------------------------
def export_palette_to_map(palette_id, filename):
    row = conn.execute('''
        SELECT num_colors, bits_per_channel, packed_palette
        FROM palettes WHERE id=?
    ''', [palette_id]).fetchone()
    if not row:
        raise ValueError(f"Palette ID {palette_id} not found")
    num_colors, bits, packed = row
    colors = unpack_palette(packed, num_colors, bits)
    with open(filename, 'w') as f:
        for r, g, b in colors:
            f.write(f'{r} {g} {b}\n')

# ----------------------------
# Batch Export Palettes
# ----------------------------
def export_collection_to_map(folder='maps', limit=None):
    os.makedirs(folder, exist_ok=True)
    query = 'SELECT id FROM palettes'
    if limit:
        query += f' LIMIT {limit}'
    ids = conn.execute(query).fetchall()
    for pid, in ids:
        export_palette_to_map(pid, f'{folder}/palette_{pid}.map')

# ----------------------------
# Query Palettes by Metadata
# ----------------------------
def query_palettes(min_brightness=None, max_brightness=None,
                   min_contrast=None, max_contrast=None,
                   dominant_color=None, limit=10):
    query = 'SELECT id, brightness, contrast, dominant FROM palettes WHERE 1=1'
    params = []
    if min_brightness is not None:
        query += ' AND brightness >= ?'
        params.append(min_brightness)
    if max_brightness is not None:
        query += ' AND brightness <= ?'
        params.append(max_brightness)
    if min_contrast is not None:
        query += ' AND contrast >= ?'
        params.append(min_contrast)
    if max_contrast is not None:
        query += ' AND contrast <= ?'
        params.append(max_contrast)
    if dominant_color is not None:
        query += ' AND dominant = ?'
        params.append(dominant_color)
    query += f' ORDER BY brightness DESC LIMIT {limit}'
    return conn.execute(query, params).fetchall()

# ----------------------------
# Example Usage
# ----------------------------
if __name__ == '__main__':
    # Insert sample palettes
    for i in range(1000):
        insert_palette(seed=i, palette_id=i+1)
    print("Inserted 1000 procedural palettes.")

    # Query bright palettes
    results = query_palettes(min_brightness=180, min_contrast=50, dominant_color='R', limit=5)
    print("Sample query results:")
    for row in results:
        print(row)

    # Export a single palette
    export_palette_to_map(42, 'palette42.map')
    print("Exported palette42.map")

    # Batch export first 10 palettes
    export_collection_to_map(folder='maps', limit=10)
    print("Exported first 10 palettes to 'maps/' folder")
