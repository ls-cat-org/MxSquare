# widgets/data_browser_window.py
from __future__ import annotations
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QPushButton, QFileDialog, QMessageBox, QLabel
)

# ---- Theme (matches your main window palette) ----
THEME = {
    "bg": "#e8f0ff",     # soft blue
    "panel": "#f5f8ff",
    "accent": "#ffcc33", # warm yellow
    "text": "#0a1a40",   # dark navy
    "border": "#c7d3ff",
}
STYLE = f"""
QWidget {{ background: {THEME['bg']}; color: {THEME['text']}; font-size: 14px; }}
QPushButton {{
  background: {THEME['accent']}; color: {THEME['text']};
  border: 1px solid {THEME['border']}; border-radius: 10px; padding: 8px 12px;
}}
QPushButton:pressed {{ background: #e6b82e; }}
QLabel {{
  background: {THEME['panel']}; border: 1px solid {THEME['border']};
  border-radius: 8px; padding: 8px;
}}
"""

class _EigerChooserWidget(QWidget):
    """Inner widget: one button, native file dialog, .h5 enforcement."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_file: Path | None = None

        self.btn_choose = QPushButton("Choose File")
        self.label = QLabel("No file selected")
        self.label.setWordWrap(True)

        lay = QVBoxLayout(self)
        lay.addWidget(self.btn_choose, alignment=Qt.AlignmentFlag.AlignLeft)
        lay.addWidget(self.label)

        self.btn_choose.clicked.connect(self._choose_file)

    def _choose_file(self):
        start = str(Path.home())
        fp, _ = QFileDialog.getOpenFileName(
            self,
            "Select Eiger HDF5 (.h5)",
            start,
            "Eiger HDF5 (*.h5);;All Files (*)",
        )
        if not fp:
            return
        p = Path(fp)
        if p.suffix.lower() != ".h5":
            QMessageBox.warning(self, "Unsupported file", "Sorry, I only accept .h5 files right now.")
            return

        self.selected_file = p
        self.label.setText(str(p))
        QMessageBox.information(self, "Ready", "time to code more")


class DataBrowserWindow(QMainWindow):
    """Public window class you already import in MxSquare.py"""
    def __init__(self, parent=None, apply_style: bool = True):
        super().__init__(parent)
        if apply_style:
            # Set once at the window level so it inherits to children
            self.setStyleSheet(STYLE)

        self.setWindowTitle("MxSquare – Data Browser")
        self._widget = _EigerChooserWidget(self)
        self.setCentralWidget(self._widget)
        self.resize(600, 220)

    # Optional: expose the selected file path to the rest of the app
    def selected_file(self) -> Path | None:
        return self._widget.selected_file

