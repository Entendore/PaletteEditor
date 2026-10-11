#!/usr/bin/env python3
import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QColorDialog,
    QListWidget, QFileDialog, QLabel, QListWidgetItem, QHBoxLayout,
    QFrame, QScrollArea, QTabWidget, QFormLayout, QDoubleSpinBox,
    QComboBox, QMessageBox, QGroupBox, QInputDialog
)
from PySide6.QtGui import QColor, QBrush, QPainter
from PySide6.QtCore import Qt

import app

class PalettePreviewWidget(QFrame):
    def __init__(self):
        super().__init__()
        self.palette = []
        self.setMinimumHeight(40)
        self.setFrameShape(QFrame.Shape.StyledPanel)

    def set_palette(self, palette):
        self.palette = palette
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#2d2d2d"))
        if not self.palette:
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Empty")
            return

        total_w = self.width()
        count = len(self.palette)
        if count == 0: return
        x_pos = 0
        for i, color in enumerate(self.palette):
            qcol = QColor(*color) if isinstance(color, tuple) else color
            next_x = ((i + 1) * total_w) // count
            painter.fillRect(x_pos, 0, next_x - x_pos, self.height(), qcol)
            x_pos = next_x

class PaletteEditor(QWidget):
    def __init__(self, name="New Palette", db_id=None):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.name = name
        self.db_id = db_id
        self._palette_tuples = []

        # Header
        header = QHBoxLayout()
        self.name_label = QLabel(f"<b>{name}</b>")
        header.addWidget(self.name_label)
        header.addStretch()
        self.layout.addLayout(header)

        # List
        self.color_list = QListWidget()
        self.color_list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.color_list.model().rowsMoved.connect(self._update_order)
        self.color_list.itemDoubleClicked.connect(self.edit_color_item)
        self.layout.addWidget(self.color_list)

        # Controls
        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton("Add Color")
        self.rm_btn = QPushButton("Remove")
        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.rm_btn)
        self.layout.addLayout(btn_layout)

        # DB Actions
        db_btn_layout = QHBoxLayout()
        self.save_new_btn = QPushButton("Save as New")
        self.update_btn = QPushButton("Update DB")
        self.export_btn = QPushButton("Export .map")
        db_btn_layout.addWidget(self.save_new_btn)
        db_btn_layout.addWidget(self.update_btn)
        db_btn_layout.addWidget(self.export_btn)
        self.layout.addLayout(db_btn_layout)

        # Preview
        self.preview = PalettePreviewWidget()
        self.layout.addWidget(self.preview)
        self.meta_label = QLabel("Metadata: --")
        self.layout.addWidget(self.meta_label)

        # Connections
        self.add_btn.clicked.connect(self.add_color)
        self.rm_btn.clicked.connect(self.remove_selected)
        self.save_new_btn.clicked.connect(self.save_to_db)
        self.update_btn.clicked.connect(self.update_db)
        self.export_btn.clicked.connect(self.export_file)

        self._refresh_ui_state()

    def load_palette(self, palette, name=None, db_id=None):
        if name:
            self.name = name
            self.name_label.setText(f"<b>{name}</b>")
        self.db_id = db_id
        
        self._palette_tuples = palette
        self.color_list.clear()
        for col in palette:
            c = QColor(*col)
            item = QListWidgetItem(f"R:{col[0]} G:{col[1]} B:{col[2]}")
            item.setBackground(QBrush(c))
            item.setForeground(QBrush(Qt.GlobalColor.white if c.lightness() < 128 else Qt.GlobalColor.black))
            self.color_list.addItem(item)
        self._refresh()

    def get_palette(self):
        updated = []
        for i in range(self.color_list.count()):
            c = self.color_list.item(i).background().color()
            updated.append((c.red(), c.green(), c.blue()))
        self._palette_tuples = updated
        return self._palette_tuples

    def add_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self._palette_tuples.append((color.red(), color.green(), color.blue()))
            self.load_palette(self._palette_tuples, db_id=self.db_id)

    def edit_color_item(self, item):
        current_bg = item.background().color()
        color = QColorDialog.getColor(current_bg)
        if color.isValid():
            idx = self.color_list.row(item)
            if 0 <= idx < len(self._palette_tuples):
                self._palette_tuples[idx] = (color.red(), color.green(), color.blue())
                self.load_palette(self._palette_tuples, db_id=self.db_id)

    def remove_selected(self):
        idx = self.color_list.currentRow()
        if idx >= 0:
            del self._palette_tuples[idx]
            self.load_palette(self._palette_tuples, db_id=self.db_id)

    def _update_order(self):
        self.get_palette()
        self._refresh()

    def _refresh(self):
        self.preview.set_palette(self._palette_tuples)
        b, c, d = app.controller.calculate_metadata(self._palette_tuples)
        self.meta_label.setText(f"Brightness: {b:.1f} | Contrast: {c:.1f} | Dominant: {d}")
        self._refresh_ui_state()

    def _refresh_ui_state(self):
        self.update_btn.setVisible(self.db_id is not None)

    def save_to_db(self):
        data = self.get_palette()
        if data:
            name, ok = QInputDialog.getText(self, "Save Palette", "Palette Name:", text=self.name)
            if ok and name:
                pid = app.controller.create_palette(data, name=name)
                self.db_id = pid
                self.name = name
                self.name_label.setText(f"<b>{name}</b>")
                self._refresh_ui_state()
                QMessageBox.information(self, "Saved", f"Saved as ID {pid}.")
    
    def update_db(self):
        if not self.db_id: return
        data = self.get_palette()
        app.controller.update_palette(self.db_id, data, self.name)
        QMessageBox.information(self, "Updated", f"Palette ID {self.db_id} updated.")

    def export_file(self):
        data = self.get_palette()
        path, _ = QFileDialog.getSaveFileName(self, "Export", f"{self.name}.map", "*.map")
        if path:
            import file_io
            if file_io.save_map_file(data, path):
                QMessageBox.information(self, "Done", f"Saved to {path}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DuckPalette Manager")
        self.resize(900, 600)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # --- Tab 1: Editor ---
        editor_tab = QWidget()
        elayout = QVBoxLayout(editor_tab)
        
        io_layout = QHBoxLayout()
        btn_import = QPushButton("Import .map File")
        btn_new = QPushButton("New Empty Palette")
        io_layout.addWidget(btn_import)
        io_layout.addWidget(btn_new)
        io_layout.addStretch()
        elayout.addLayout(io_layout)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.container = QWidget()
        self.container_layout = QVBoxLayout(self.container)
        self.scroll.setWidget(self.container)
        elayout.addWidget(self.scroll)

        btn_import.clicked.connect(self.import_file)
        btn_new.clicked.connect(lambda: self.add_editor("Untitled"))
        self.tabs.addTab(editor_tab, "Editor")

        # --- Tab 2: Database ---
        db_tab = QWidget()
        dlayout = QVBoxLayout(db_tab)
        
        search_group = QGroupBox("Search Filters")
        slayout = QFormLayout(search_group)
        self.spin_min = QDoubleSpinBox()
        self.spin_min.setRange(0, 255)
        self.spin_max = QDoubleSpinBox()
        self.spin_max.setRange(0, 255)
        self.spin_max.setValue(255)
        self.combo_dom = QComboBox()
        self.combo_dom.addItems(["Any", "R", "G", "B"])
        
        slayout.addRow("Min Bright:", self.spin_min)
        slayout.addRow("Max Bright:", self.spin_max)
        slayout.addRow("Dominant:", self.combo_dom)
        
        btn_search = QPushButton("Search")
        btn_gen = QPushButton("Generate 10 Random")
        dlayout.addWidget(search_group)
        
        act_layout = QHBoxLayout()
        act_layout.addWidget(btn_search)
        act_layout.addWidget(btn_gen)
        dlayout.addLayout(act_layout)

        self.db_list = QListWidget()
        dlayout.addWidget(self.db_list)

        db_actions = QHBoxLayout()
        btn_load = QPushButton("Load Selected to Editor")
        btn_delete = QPushButton("Delete Selected")
        db_actions.addWidget(btn_load)
        db_actions.addWidget(btn_delete)
        dlayout.addLayout(db_actions)

        btn_search.clicked.connect(self.search_db)
        btn_gen.clicked.connect(self.gen_db)
        btn_load.clicked.connect(self.load_selected)
        btn_delete.clicked.connect(self.delete_selected)
        
        self.tabs.addTab(db_tab, "Database")

    def add_editor(self, name, palette=None, db_id=None):
        ed = PaletteEditor(name, db_id=db_id)
        if palette:
            ed.load_palette(palette, name, db_id)
        self.container_layout.addWidget(ed)
        return ed

    def import_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import", "", "*.map *.txt")
        if path:
            pid, palette, name = app.controller.import_file_to_db(path)
            if palette:
                self.add_editor(name, palette, db_id=pid)

    def search_db(self):
        self.db_list.clear()
        dom = self.combo_dom.currentText()
        rows = app.controller.search_palettes(
            min_bright=self.spin_min.value(),
            max_bright=self.spin_max.value(),
            dominant=dom if dom != "Any" else None
        )
        for r in rows:
            pid, name, b, c, d, num, _ = r
            item = QListWidgetItem(f"ID:{pid} | {name} | B:{b:.0f} C:{c:.0f} [{d}] ({num} colors)")
            item.setData(Qt.ItemDataRole.UserRole, pid)
            self.db_list.addItem(item)

    def gen_db(self):
        app.controller.generate_new_palettes(10)
        QMessageBox.information(self, "Done", "Generated 10 palettes.")
        self.search_db()

    def load_selected(self):
        item = self.db_list.currentItem()
        if not item: return
        pid = item.data(Qt.ItemDataRole.UserRole)
        name_parts = item.text().split('|')
        name = name_parts[1].strip() if len(name_parts) > 1 else f"ID {pid}"
        
        palette = app.controller.get_palette(pid)
        if palette:
            self.tabs.setCurrentIndex(0)
            self.add_editor(name, palette, db_id=pid)

    def delete_selected(self):
        item = self.db_list.currentItem()
        if not item: return
        pid = item.data(Qt.ItemDataRole.UserRole)
        reply = QMessageBox.question(self, "Confirm Delete", f"Delete Palette ID {pid}?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            app.controller.delete_palette(pid)
            self.search_db()

def main():
    app_qt = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app_qt.exec())