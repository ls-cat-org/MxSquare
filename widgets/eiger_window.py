# MxSquare/widgets/eiger_window.py
from __future__ import annotations

import time
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGridLayout
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from concurrent.futures import ThreadPoolExecutor, Future

from utils.epics_tools import get_caget_value

DEFAULT_PV_PREFIX = "21EIG2:cam1:"

# Suffixes you want to display (readbacks only)
# Removed FilePath (array PV)
PV_ITEMS = [
    ("Manufacturer",  "Manufacturer_RBV"),
    ("Model",         "Model_RBV"),
    ("StatusMessage", "StatusMessage_RBV"),
    ("FrameCounter",  "NumImagesCounter_RBV"),
    ("AcquireTime",   "AcquireTime_RBV"),
    ("TriggerMode",   "TriggerMode_RBV"),
    ("FileName",      "FileName_RBV"),
    ("FileNumber",    "FileNumber_RBV"),
]

class EigerWindow(QWidget):
    """Detector (Eiger2) — always-on PV polling with main-window theme."""

    valueReady = pyqtSignal(object, str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Detector — Eiger2")
        self.setStyleSheet("""
            QWidget {
                background-color: #e8f0ff;   /* soft blue background */
                color: #0a1a40;             /* dark navy text */
                font-size: 13px;
            }
            QLabel {
                padding: 2px 6px;
            }
            QLabel[role="key"] {
                font-weight: bold;
            }
            QLabel[role="val"] {
                color: #0a1a40;
            }
            QLabel[role="val"]:hover {
                color: #ffcc33;             /* warm yellow hover */
            }
            QToolTip {
                background-color: #ffcc33;
                color: #0a1a40;
                border: 1px solid #0a1a40;
            }
        """)

        outer = QVBoxLayout(self)
        title = QLabel("<h3>Eiger2 Detector</h3>")
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        outer.addWidget(title)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)

        self._rows: list[tuple[str, str, QLabel]] = []

        for i, (label_text, suffix) in enumerate(PV_ITEMS):
            pv = DEFAULT_PV_PREFIX + suffix
            key = QLabel(label_text)
            key.setProperty("role", "key")
            key.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            key.setToolTip(pv)

            val = QLabel("—")
            val.setProperty("role", "val")
            val.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            val.setMinimumWidth(260)

            grid.addWidget(key, i, 0)
            grid.addWidget(val, i, 1)
            self._rows.append((label_text, pv, val))

        # Test row to prove updates (remove later if you want)
        test_key = QLabel("TEST (seconds)")
        test_key.setProperty("role", "key")
        test_val = QLabel("—")
        test_val.setProperty("role", "val")
        grid.addWidget(test_key, len(PV_ITEMS), 0)
        grid.addWidget(test_val, len(PV_ITEMS), 1)
        self._test_label = test_val
        self._t0 = time.time()

        outer.addLayout(grid)

        # Polling infra
        self._pool = ThreadPoolExecutor(max_workers=6)
        self._inflight: list[Future] = []
        self._timer = QTimer(self)
        self._timer.setInterval(1000)  # 1s
        self._timer.timeout.connect(self._poll_once)
        self._timer.start()

        self.valueReady.connect(self._on_value_ready)

    def _poll_once(self):
        # update test row every tick
        elapsed = f"{int(time.time() - self._t0)}"
        self.valueReady.emit(self._test_label, elapsed)

        if any(f for f in self._inflight if not f.done()):
            return
        self._inflight.clear()

        for name, pv, label in self._rows:
            fut = self._pool.submit(self._safe_caget, name, pv)
            fut.add_done_callback(lambda f, lab=label: self._deliver(lab, f))
            self._inflight.append(fut)

    @staticmethod
    def _safe_caget(name: str, pv: str) -> tuple[str, str]:
        try:
            ok, val = get_caget_value(pv)
            return pv, (val if ok else "")
        except Exception as e:
            print(f"[DEBUG] exception calling caget({pv}): {e}", flush=True)
            return pv, ""

    def _deliver(self, label: QLabel, fut: Future):
        pv, val = fut.result()
        self.valueReady.emit(label, "—" if not val else val)

    def _on_value_ready(self, label: QLabel, value: str):
        label.setText(value)

    def closeEvent(self, event):
        try:
            self._timer.stop()
            self._pool.shutdown(wait=False, cancel_futures=True)
        finally:
            super().closeEvent(event)

