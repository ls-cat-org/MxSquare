#!/usr/bin/env python3
from __future__ import annotations

import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
)
from PyQt6.QtCore import Qt

# Windows
from widgets.eiger_window import EigerWindow
from widgets.omega_window import OmegaWindow
from widgets.microscope_window import MicroscopeWindow
from widgets.data_browser_window import DataBrowserWindow


APP_STYLES = """
QWidget {
    background-color: #e8f0ff;   /* soft blue */
    color: #0a1a40;              /* dark navy */
    font-size: 14px;
}
QPushButton {
    background-color: #ffcc33;   /* warm yellow */
    border-radius: 8px;
    padding: 6px 12px;
    font-weight: bold;
}
QPushButton:hover { background-color: #ffdb4d; }
"""

class ControlPanel(QWidget):
    """Parent control panel to launch user-essential windows."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MxSquare — Control Panel")
        self.setStyleSheet(APP_STYLES)

        self._eiger = None
        self._omega = None
        self._microscope = None
        self._browser = None

        lay = QVBoxLayout(self)

        title = QLabel("<h2>MxSquare • User Essentials</h2>")
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        lay.addWidget(title)

        # Row 1: Detector + Omega
        row1 = QHBoxLayout()
        btn_eiger = QPushButton("Detector (Eiger2)")
        btn_eiger.clicked.connect(self.open_eiger)
        row1.addWidget(btn_eiger)

        btn_omega = QPushButton("Omega (MD3)")
        btn_omega.clicked.connect(self.open_omega)
        row1.addWidget(btn_omega)

        lay.addLayout(row1)

        # Row 2: Microscope + Recent Images
        row2 = QHBoxLayout()
        btn_micro = QPushButton("Microscope")
        btn_micro.clicked.connect(self.open_microscope)
        row2.addWidget(btn_micro)

        btn_browser = QPushButton("Recent Images")
        btn_browser.clicked.connect(self.open_browser)
        row2.addWidget(btn_browser)

        lay.addLayout(row2)

        # Hint
        hint = QLabel("Each window is self-contained. No crosstalk required.")
        hint.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        lay.addWidget(hint)

    def open_eiger(self):
        if self._eiger is None:
            self._eiger = EigerWindow()
        self._eiger.show()
        self._eiger.raise_()
        self._eiger.activateWindow()

    def open_omega(self):
        if self._omega is None:
            self._omega = OmegaWindow()
        self._omega.show()
        self._omega.raise_()
        self._omega.activateWindow()

    def open_microscope(self):
        if self._microscope is None:
            self._microscope = MicroscopeWindow()
        self._microscope.show()
        self._microscope.raise_()
        self._microscope.activateWindow()

    def open_browser(self):
        if self._browser is None:
            self._browser = DataBrowserWindow()
        self._browser.show()
        self._browser.raise_()
        self._browser.activateWindow()


def main():
    app = QApplication(sys.argv)
    panel = ControlPanel()
    panel.resize(520, 220)
    panel.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

