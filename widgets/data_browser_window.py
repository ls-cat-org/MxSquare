from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt

class DataBrowserWindow(QWidget):
    """Recent Images — Placeholder.
    Will read FilePath/FileName from detector PVs and preview recent HDF5 images.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Recent Images — Placeholder")
        lay = QVBoxLayout(self)
        title = QLabel("<h3>Recent Images</h3>")
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        lay.addWidget(title)
        lay.addWidget(QLabel("Future items: read HDF5 path, list latest files, show thumbnail"))

