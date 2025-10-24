from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGridLayout, QFrame
from PyQt6.QtCore import Qt, QTimer
from utils.epics_tools import get_caget_value


class OmegaWindow(QWidget):
    """Omega (MD3) — Read-only status display for MD3 rotation and peripherals."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Omega (MD3)")
        self.resize(420, 640)

        # ---- Layout ----
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title = QLabel("<h3>Omega (MD3)</h3>")
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(title)

        sub = QLabel("Live readouts from MD3 — read-only for now.")
        sub.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(sub)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line)
        # ---- PV definitions ----
        self.pv_prefix = "21:F:MD3:"
        self.pvs = {
            "Omega (deg)":           "BeamstopPosition",
            "Omega SP (deg)":        "BeamstopPosition",
            "Omega Vel (deg/s)":     "BeamstopPosition",
            "Omega Acc (deg/s²)":    "BeamstopPosition",
            "Omega Moving":          "BeamstopPosition",
            "Omega Status":          "BeamstopPosition",
            "Beamstop Pos (mm)":     "BeamstopPosition",
            "Beamstop In":           "BeamstopPosition",
            "Backlight On":          "BeamstopPosition",
            "Backlight Intensity":   "BeamstopPosition",
            "Atten Material":        "BeamstopPosition",
            "Atten State":           "BeamstopPosition",
            "Transmission (%)":      "BeamstopPosition",
            "Shutter":               "BeamstopPosition",
            "Aperture (µm)":         "BeamstopPosition"
        }

        # ---- Grid for labels ----
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)
        self.value_labels = {}

        for row, (name, suffix) in enumerate(self.pvs.items()):
            lbl = QLabel(name)
            lbl.setStyleSheet("color:#0a1a40;")
            val = QLabel("—")
            val.setStyleSheet(
                "font-family:monospace; color:#0a1a40; "
                "background-color:#f5f8ff; padding:2px 6px; border-radius:4px;"
            )
            grid.addWidget(lbl, row, 0)
            grid.addWidget(val, row, 1)
            self.value_labels[name] = val

        layout.addLayout(grid)
        layout.addStretch(1)
        self.setLayout(layout)

        # ---- Timer for auto-updates ----
        self.timer = QTimer(self)
        self.timer.setInterval(3000)  # 3 seconds
        self.timer.timeout.connect(self.refresh_values)
        self.timer.start()

        self.refresh_values()

    # ------------------------------------------------------------------
    def refresh_values(self):
        """Poll all PVs via epics_tools.get_caget_value and update labels."""
        for label, suffix in self.pvs.items():
            pv = self.pv_prefix + suffix
            ok, val = get_caget_value(pv)
            if not ok:
                text = f"(err) {val}"
            else:
                text = str(val) if val else "—"

            # limit very long strings
            if len(text) > 40:
                text = text[:37] + "..."
            self.value_labels[label].setText(text)

    # ------------------------------------------------------------------
    def closeEvent(self, event):
        """Stop the update timer cleanly on window close."""
        try:
            self.timer.stop()
        except Exception:
            pass
        super().closeEvent(event)

