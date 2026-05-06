import os
import time
import sys
from database import db
import file_io
import utils

class PaletteController:
    def __init__(self):
        self.db = db

    # --- Actions for GUI and CLI ---
    
    def import_file_to_db(self, filepath):
        """Reads a file and saves it to the database."""
        palette = file_io.parse_map_file(filepath)
        if palette:
            name = os.path.basename(filepath)
            pid = self.db.insert_palette(palette, name=name)
            return pid, palette, name
        return None, [], None

    def export_db_to_file(self, palette_id, filepath):
        """Retrieves a palette from DB and saves to file."""
        palette = self.db.get_palette_by_id(palette_id)
        if palette:
            return file_io.save_map_file(palette, filepath)
        return False

    def create_palette(self, palette_data, name="New Palette"):
        """Saves a user-created palette to DB."""
        return self.db.insert_palette(palette_data, name=name)

    def update_palette(self, pid, palette_data, name):
        return self.db.update_palette(pid, palette_data, name)

    def delete_palette(self, pid):
        self.db.delete_palette(pid)

    def generate_new_palettes(self, count=10):
        """Generates random palettes into DB."""
        start_seed = int(time.time())
        for i in range(count):
            self.db.generate_and_insert(seed=start_seed + i)

    def search_palettes(self, min_bright=None, max_bright=None, dominant=None):
        """Search wrapper."""
        return self.db.search(min_bright, max_bright, dominant)

    def get_palette(self, pid):
        """Get palette by ID wrapper."""
        return self.db.get_palette_by_id(pid)

    def calculate_metadata(self, palette):
        """Utility wrapper."""
        return utils.calculate_metadata(palette)

# Singleton instance to be used by GUI and CLI
controller = PaletteController()

# --- Entry Point Logic ---
if __name__ == "__main__":
    # Check if the first argument is 'cli'
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        # Import CLI module
        import cli
        
        # Remove the 'cli' argument so argparse in cli.py works correctly
        # sys.argv becomes: [script_name, command, ...args]
        sys.argv.pop(1)
        
        # Run CLI main function
        cli.main()
        
    else:
        # Default to GUI
        # Allow 'python app.py gui' by removing 'gui' if present
        if len(sys.argv) > 1 and sys.argv[1] == "gui":
            sys.argv.pop(1)
            
        # Import GUI module (lazy import to speed up CLI startup)
        import gui
        gui.main()