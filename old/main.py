#!/usr/bin/env python3
import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QColorDialog,
    QListWidget, QFileDialog, QLabel, QListWidgetItem, QHBoxLayout,
    QFrame, QScrollArea, QGroupBox, QGridLayout
)
from PyQt6.QtGui import QColor, QBrush, QPainter
from PyQt6.QtCore import Qt

class PaletteEditor(QWidget):
    """Single palette editor with list and preview"""
    def __init__(self, name, palette=None):
        super().__init__()
        self.name = name
        self.palette = palette or []

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.label = QLabel(name)
        self.layout.addWidget(self.label)

        # Color list with drag & drop
        self.color_list = QListWidget()
        self.color_list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.color_list.model().rowsMoved.connect(self.update_palette_order)
        self.layout.addWidget(self.color_list)

        # Buttons to edit colors
        button_layout = QHBoxLayout()
        self.add_button = QPushButton("Add Color")
        self.remove_button = QPushButton("Remove Selected")
        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.remove_button)
        self.layout.addLayout(button_layout)

        # Palette preview
        self.preview = QFrame()
        self.preview.setMinimumHeight(30)
        self.preview.setFrameShape(QFrame.Shape.Box)
        self.layout.addWidget(self.preview)

        # Signals
        self.add_button.clicked.connect(self.add_color)
        self.remove_button.clicked.connect(self.remove_selected)

        # Load initial palette
        self.load_palette(self.palette)

    def load_palette(self, palette):
        self.palette = palette
        self.color_list.clear()
        for color in self.palette:
            item = QListWidgetItem(color.name())
            item.setBackground(QBrush(color))
            self.color_list.addItem(item)
        self.preview.update()

    def add_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.palette.append(color)
            item = QListWidgetItem(color.name())
            item.setBackground(QBrush(color))
            self.color_list.addItem(item)
            self.preview.update()

    def remove_selected(self):
        selected_items = self.color_list.selectedItems()
        for item in selected_items:
            idx = self.color_list.row(item)
            self.color_list.takeItem(idx)
            del self.palette[idx]
        self.preview.update()

    def update_palette_order(self):
        """Update internal palette after drag & drop"""
        new_palette = []
        for i in range(self.color_list.count()):
            item = self.color_list.item(i)
            color = item.background().color()
            new_palette.append(color)
        self.palette = new_palette
        self.preview.update()

    def paintEvent(self, event):
        """Draw horizontal preview"""
        painter = QPainter(self.preview)
        if not self.palette:
            return
        width_per_color = self.preview.width() / len(self.palette)
        for i, color in enumerate(self.palette):
            painter.fillRect(int(i * width_per_color), 0, int(width_per_color), self.preview.height(), color)

class MultiPaletteEditor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Palette Editor")
        self.setGeometry(100, 100, 700, 500)

        self.palettes = []  # list of PaletteEditor

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Buttons
        button_layout = QHBoxLayout()
        self.import_button = QPushButton("Import Palette(s)")
        self.export_button = QPushButton("Export All")
        self.clear_button = QPushButton("Clear All")
        button_layout.addWidget(self.import_button)
        button_layout.addWidget(self.export_button)
        button_layout.addWidget(self.clear_button)
        self.layout.addLayout(button_layout)

        # Scrollable area for palettes
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QVBoxLayout()
        self.scroll_widget = QGroupBox()
        self.scroll_widget.setLayout(self.scroll_content)
        self.scroll_area.setWidget(self.scroll_widget)
        self.layout.addWidget(self.scroll_area)

        # Signals
        self.import_button.clicked.connect(self.import_palettes)
        self.export_button.clicked.connect(self.export_all)
        self.clear_button.clicked.connect(self.clear_palettes)

    def add_palette_editor(self, name, palette):
        editor = PaletteEditor(name, palette)
        self.palettes.append(editor)
        self.scroll_content.addWidget(editor)

    def import_palettes(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Import Palette(s)", "", "MAP Files (*.map);;TXT Files (*.txt)")
        if not paths:
            return

        for path in paths:
            palette = []
            try:
                with open(path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 3:
                            r, g, b = map(int, parts)
                            palette.append(QColor(r, g, b))
                name = path.split("/")[-1]
                self.add_palette_editor(name, palette)
            except Exception as e:
                print(f"Failed to import {path}: {e}")

    def export_all(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if not folder:
            return

        for editor in self.palettes:
            path_map = f"{folder}/{editor.name}.map"
            path_txt = f"{folder}/{editor.name}.txt"
            path_pal = f"{folder}/{editor.name}.pal"

            # Save .map and .txt (same format)
            for path in [path_map, path_txt]:
                with open(path, 'w') as f:
                    for color in editor.palette:
                        r, g, b, _ = color.getRgb()
                        f.write(f"{r} {g} {b}\n")
            # Save .pal (Adobe-style)
            with open(path_pal, 'w') as f:
                f.write(f"{len(editor.palette)}\n")
                for color in editor.palette:
                    r, g, b, _ = color.getRgb()
                    f.write(f"{r} {g} {b}\n")

        print(f"Exported {len(self.palettes)} palettes to {folder}")

    def clear_palettes(self):
        self.palettes.clear()
        while self.scroll_content.count():
            item = self.scroll_content.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MultiPaletteEditor()
    window.show()
    sys.exit(app.exec())
