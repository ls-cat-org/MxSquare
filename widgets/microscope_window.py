from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt

class MicroscopeWindow(QWidget):
    """Microscope/MD3-UP — Placeholder.
    Will host video feed, zoom/focus/light controls, XYZ readbacks.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Microscope — Placeholder")
        lay = QVBoxLayout(self)
        title = QLabel("<h3>Microscope</h3>")
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        lay.addWidget(title)
        lay.addWidget(QLabel("Future items: live video, zoom/focus, light, XYZ readbacks"))

