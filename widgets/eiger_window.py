from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt

class EigerWindow(QWidget):
    """Detector (Eiger2) — Placeholder.
    Will own: Filename/Path, FileNumber, AcquireTime, NumImages, TriggerMode,
    Arm/Start/Abort, FrameCounter display, Status text.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Detector — Eiger2 (Placeholder)")
        lay = QVBoxLayout(self)
        title = QLabel("<h3>Detector (Eiger2)</h3>")
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        lay.addWidget(title)
        lay.addWidget(QLabel("This window will host user fields like:"))
        lay.addWidget(QLabel("• FilePath / FileName / FileNumber"))
        lay.addWidget(QLabel("• Exposure (AcquireTime), NumImages, TriggerMode"))
        lay.addWidget(QLabel("• Buttons: Arm, Start, Abort"))
        lay.addWidget(QLabel("• StatusMessage, FrameCounter"))

