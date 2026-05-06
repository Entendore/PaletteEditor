import duckdb
from utils import pack_palette, calculate_metadata, generate_random_palette, unpack_palette

class PaletteDB:
    def __init__(self, db_file='palettes.duckdb'):
        self.conn = duckdb.connect(database=db_file, read_only=False)
        self._init_db()

    def _init_db(self):
        # Using HUGEINT for packed_palette to fit 5 colors (120 bits)
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS palettes (
            id INTEGER PRIMARY KEY,
            name TEXT,
            seed BIGINT,
            num_colors INTEGER,
            bits_per_channel INTEGER,
            packed_palette HUGEINT,
            brightness DOUBLE,
            contrast DOUBLE,
            dominant TEXT
        )
        ''')

    def insert_palette(self, palette, name="User Palette", pid=None, seed=None):
        if not palette:
            return None

        # Auto-increment logic
        if pid is None:
            res = self.conn.execute("SELECT MAX(id) FROM palettes").fetchone()
            pid = (res[0] or 0) + 1

        brightness, contrast, dominant = calculate_metadata(palette)
        packed = pack_palette(palette, bits_per_channel=8)

        self.conn.execute('''
            INSERT INTO palettes (id, name, seed, num_colors, bits_per_channel, packed_palette, brightness, contrast, dominant)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', [pid, name, seed, len(palette), 8, packed, brightness, contrast, dominant])
        
        return pid

    def update_palette(self, pid, palette, name):
        if not palette: return False
        brightness, contrast, dominant = calculate_metadata(palette)
        packed = pack_palette(palette)
        self.conn.execute('''
            UPDATE palettes 
            SET name=?, packed_palette=?, num_colors=?, brightness=?, contrast=?, dominant=?
            WHERE id=?
        ''', [name, packed, len(palette), brightness, contrast, dominant, pid])
        return True

    def delete_palette(self, pid):
        self.conn.execute("DELETE FROM palettes WHERE id = ?", [pid])

    def generate_and_insert(self, seed, num_colors=5):
        palette = generate_random_palette(seed, num_colors)
        return self.insert_palette(palette, name=f"Gen_{seed}", pid=seed, seed=seed)

    def search(self, min_bright=None, max_bright=None, dominant=None, limit=50):
        query = 'SELECT id, name, brightness, contrast, dominant, num_colors, packed_palette FROM palettes WHERE 1=1'
        params = []
        
        if min_bright is not None:
            query += ' AND brightness >= ?'
            params.append(min_bright)
        if max_bright is not None:
            query += ' AND brightness <= ?'
            params.append(max_bright)
        if dominant:
            query += ' AND dominant = ?'
            params.append(dominant)
            
        query += ' ORDER BY id DESC LIMIT ?'
        params.append(limit)
        
        return self.conn.execute(query, params).fetchall()

    def get_palette_by_id(self, pid):
        row = self.conn.execute("SELECT packed_palette, num_colors, bits_per_channel FROM palettes WHERE id=?", [pid]).fetchone()
        if row:
            packed, num, bits = row
            return unpack_palette(packed, num, bits)
        return None

# Singleton instance
db = PaletteDB()