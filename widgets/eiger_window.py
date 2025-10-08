# MxSquare/widgets/eiger_window.py
from __future__ import annotations

import time
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGridLayout
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from concurrent.futures import ThreadPoolExecutor, Future

from utils.epics_tools import get_caget_value

DEFAULT_PV_PREFIX = "21EIG2:cam1:"

PV_ITEMS = [
    ("Manufacturer",  "Manufacturer_RBV"),
    ("Model",         "Model_RBV"),
    ("StatusMessage", "StatusMessage_RBV"),
    ("FrameCounter",  "NumImagesCounter_RBV"),
    ("AcquireTime",   "AcquireTime_RBV"),
    ("TriggerMode",   "TriggerMode_RBV"),
    ("FilePath",      "FilePath_RBV"),
    ("FileName",      "FileName_RBV"),
    ("FileNumber",    "FileNumber_RBV"),
]

class EigerWindow(QWidget):
    """Detector (Eiger2) — always-on PV polling via utils.epics_tools.get_caget_value."""

    # Strongly-typed signal so UI updates happen on the GUI thread reliably
    valueReady = pyqtSignal(object, str)   # (QLabel, value)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Detector")

        outer = QVBoxLayout(self)
        title = QLabel("<h3> Eiger2 </h3>")
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        outer.addWidget(title)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)

        # rows: list of (name, pv, label_widget)
        self._rows: list[tuple[str, str, QLabel]] = []

        # Real PV rows
        for i, (label_text, suffix) in enumerate(PV_ITEMS):
            pv = DEFAULT_PV_PREFIX + suffix
            key = QLabel(f"<b>{label_text}</b>")
            key.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            key.setToolTip(pv)

            val = QLabel("—")
            val.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            val.setMinimumWidth(260)

            grid.addWidget(key, i, 0)
            grid.addWidget(val, i, 1)
            self._rows.append((label_text, pv, val))

        # Always-on TEST row (no EPICS) so we can see UI updates:
        test_key = QLabel("<b>TEST (seconds since start)</b>")
        test_val = QLabel("—")
        test_key.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        test_val.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        test_val.setMinimumWidth(260)
        grid.addWidget(test_key, len(PV_ITEMS), 0)
        grid.addWidget(test_val, len(PV_ITEMS), 1)
        self._test_label = test_val
        self._t0 = time.time()

        outer.addLayout(grid)

        # Polling infra (always on)
        self._pool = ThreadPoolExecutor(max_workers=6)
        self._inflight: list[Future] = []

        self._timer = QTimer(self)
        self._timer.setInterval(1000)  # 1s
        self._timer.timeout.connect(self._poll_once)
        self._timer.start()

        # UI update signal
        self.valueReady.connect(self._on_value_ready)

        print("[DEBUG] EigerWindow started; polling every 1s", flush=True)

    # -------- polling --------
    def _poll_once(self):
        # Update TEST row so we can see something change even if PVs fail
        elapsed = f"{int(time.time() - self._t0)}"
        self.valueReady.emit(self._test_label, elapsed)

        # Avoid overlapping batches
        if any(f for f in self._inflight if not f.done()):
            print("[DEBUG] Skipping tick: previous batch still in-flight", flush=True)
            return
        self._inflight.clear()

        for name, pv, label in self._rows:
            fut = self._pool.submit(self._safe_caget, name, pv)
            fut.add_done_callback(lambda f, lab=label: self._deliver(lab, f))
            self._inflight.append(fut)

    @staticmethod
    def _safe_caget(name: str, pv: str) -> tuple[str, str]:
        """Return (pv, value_text). Never raise."""
        try:
            print(f"[DEBUG] caget -> {pv}  [{name}]", flush=True)
            ok, val = get_caget_value(pv)
            print(f"[DEBUG] result <- ok={ok}, val={val!r}  [{name}]", flush=True)
            return pv, (val if ok else "")
        except Exception as e:
            print(f"[DEBUG] exception calling caget({pv}): {e}", flush=True)
            return pv, ""

    def _deliver(self, label: QLabel, fut: Future):
        pv, val = fut.result()
        print(f"[DEBUG] deliver {pv} -> {val!r}", flush=True)
        # Emit signal to update label on GUI thread
        self.valueReady.emit(label, "—" if not val else val)

    # -------- UI slot --------
    def _on_value_ready(self, label: QLabel, value: str):
        label.setText(value)

    # -------- cleanup --------
    def closeEvent(self, event):
        try:
            self._timer.stop()
            self._pool.shutdown(wait=False, cancel_futures=True)
        finally:
            super().closeEvent(event)

