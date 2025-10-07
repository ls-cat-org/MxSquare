from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt

class OmegaWindow(QWidget):
    """Omega (MD3) — Placeholder.
    Will own: start angle, range, step, velocity readback; simple jog buttons
    (+/- 90°, +/- 180°) later. Omega is through MD3 (no MRES/HLM/LLM here).
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Omega (MD3) — Placeholder")
        lay = QVBoxLayout(self)
        title = QLabel("<h3>Omega (MD3)</h3>")
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        lay.addWidget(title)
        lay.addWidget(QLabel("Planned controls (later):"))
        lay.addWidget(QLabel("• Start angle, Range (°), Step (°/frame)"))
        lay.addWidget(QLabel("• Readbacks: Ω.RBV, χ/φ optional"))
        lay.addWidget(QLabel("• Buttons: +90, -90, +180, -180 (not yet)"))

