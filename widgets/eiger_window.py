# MxSquare/widgets/eiger_window.py
from __future__ import annotations
import functools
from concurrent.futures import ThreadPoolExecutor, Future
from typing import List, Tuple

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGridLayout, QHBoxLayout,
    QLineEdit, QSpinBox, QPushButton, QCheckBox
)
from utils.epics_tools import get_caget_value
try:
    from utils.epics_tools import set_caput_value
except Exception:
    set_caput_value = None

DEFAULT_PV_PREFIX = "21EIG2:cam1:"

STATUS_ITEMS: List[Tuple[str, str]] = [
    ("State",            "DetectorState_RBV"),
    ("Status",           "StatusMessage_RBV"),
    ("Armed",            "Armed"),
    ("Acquire",          "Acquire_RBV"),
    ("ImageMode",        "ImageMode_RBV"),
    ("TriggerMode",      "TriggerMode_RBV"),
    ("NumImages",        "NumImages_RBV"),
    ("Img Counter",      "NumImagesCounter_RBV"),
    ("Count Time (s)",   "AcquireTime_RBV"),
    ("Frame Period (s)", "AcquirePeriod_RBV"),
    ("FW State",         "FWState_RBV"),
]

FW_NAME_PATTERN_PV = "FWNamePattern"
FW_LOCAL_PATH_PV   = "FWLocalPathPrefix"
FW_IMGS_PER_FILE_PV= "FWImagesPerFile"
FW_ENABLE_PV       = "FWEnable"
STREAM_ENABLE_PV   = "StreamEnable"

class EigerWindow(QWidget):
    valueReady = pyqtSignal(object, str)

    def __init__(self, prefix: str = DEFAULT_PV_PREFIX):
        super().__init__()
        self.setWindowTitle("Eiger — Data Collection")
        self._prefix = prefix
        self._pool = ThreadPoolExecutor(max_workers=8)
        self._timer = QTimer(self)
        self._timer.setInterval(750)

        # --- Apply color theme ---
        self.setStyleSheet("""
            QWidget {
                background-color: #e8f0ff;
                color: #0a1a40;
                font-size: 12pt;
            }
            QLabel {
                color: #0a1a40;
            }
            QLineEdit, QSpinBox {
                background-color: white;
                border: 1px solid #0a1a40;
                border-radius: 4px;
                padding: 3px;
            }
            QPushButton {
                background-color: #ffcc33;
                color: #0a1a40;
                font-weight: bold;
                border: 1px solid #0a1a40;
                border-radius: 6px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #ffe066;
            }
            QPushButton:pressed {
                background-color: #ffd633;
            }
            QCheckBox {
                font-weight: bold;
            }
            QCheckBox::indicator:checked {
                background-color: #ffcc33;
                border: 1px solid #0a1a40;
            }
            QCheckBox::indicator:unchecked {
                background-color: white;
                border: 1px solid #0a1a40;
            }
        """)

        # --- Layout ---
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)

        self._labels = []
        for row, (title, suffix) in enumerate(STATUS_ITEMS):
            name_lbl = QLabel(title + ":")
            name_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            val_lbl = QLabel("—")
            val_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            grid.addWidget(name_lbl, row, 0)
            grid.addWidget(val_lbl,  row, 1)
            self._labels.append((name_lbl, val_lbl, self._prefix + suffix))

        root.addLayout(grid)

        # --- File-Writer editable controls ---
        fw_box = QGridLayout()
        row = 0

        self.in_name = QLineEdit()
        self.in_path = QLineEdit()
        self.in_imgs = QSpinBox()
        self.in_imgs.setRange(1, 10_000_000)
        self.in_imgs.setValue(10)

        fw_box.addWidget(QLabel("FW Name Pattern:"), row, 0)
        fw_box.addWidget(self.in_name, row, 1); row += 1
        fw_box.addWidget(QLabel("FW Local Path:"), row, 0)
        fw_box.addWidget(self.in_path, row, 1); row += 1
        fw_box.addWidget(QLabel("FW Images/File:"), row, 0)
        fw_box.addWidget(self.in_imgs, row, 1); row += 1

        self.chk_force_h5 = QCheckBox("Force HDF5 (FWEnable=1, StreamEnable=0)")
        fw_box.addWidget(self.chk_force_h5, row, 0, 1, 2); row += 1

        btns = QHBoxLayout()
        self.btn_refresh_fw = QPushButton("Read FW → inputs")
        self.btn_apply_fw   = QPushButton("Apply FW settings")
        btns.addWidget(self.btn_refresh_fw)
        btns.addWidget(self.btn_apply_fw)
        fw_box.addLayout(btns, row, 0, 1, 2); row += 1

        root.addLayout(fw_box)
        root.addSpacing(10)

        self.status_line = QLabel("")
        root.addWidget(self.status_line)

        # --- Signals ---
        self.valueReady.connect(self._on_value_ready)
        self.btn_refresh_fw.clicked.connect(self._read_fw_into_inputs)
        self.btn_apply_fw.clicked.connect(self._apply_fw_settings)
        self._timer.timeout.connect(self._poll_once)
        self._timer.start()

        self._read_fw_into_inputs()

    # ---------- polling ----------
    def _poll_once(self):
        for (_, val_lbl, pv) in self._labels:
            fut: Future = self._pool.submit(get_caget_value, pv)
            fut.add_done_callback(functools.partial(self._deliver, val_lbl))

    def _deliver(self, label: QLabel, fut: Future):
        ok, out = fut.result()
        self.valueReady.emit(label, out if ok else f"(err) {out}")

    def _on_value_ready(self, label: QLabel, value: str):
        label.setText(value)

    # ---------- FW helpers ----------
    def _pv(self, suffix: str) -> str:
        return self._prefix + suffix

    def _safe_caput(self, pv: str, val, as_string=True):
        if set_caput_value:
            return set_caput_value(pv, val, as_string=as_string, timeout=2.0)
        import shutil, subprocess
        cap = shutil.which("caput")
        if not cap:
            return False, "caput not found"
        try:
            args = [cap, "-w", "2"]
            if as_string:
                args.append("-S")
            args += [pv, str(val)]
            p = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return (p.returncode == 0, p.stdout.strip() or p.stderr.strip())
        except Exception as e:
            return False, f"caput exception: {e}"

    def _read_fw_into_inputs(self):
        for widget, suff in (
            (self.in_name, FW_NAME_PATTERN_PV),
            (self.in_path, FW_LOCAL_PATH_PV),
        ):
            ok, out = get_caget_value(self._pv(suff))
            widget.setText(out if ok else "")
        ok, out = get_caget_value(self._pv(FW_IMGS_PER_FILE_PV))
        try:
            self.in_imgs.setValue(int(float(out))) if ok else None
        except Exception:
            pass
        ok1, fw_en = get_caget_value(self._pv(FW_ENABLE_PV))
        ok2, st_en = get_caget_value(self._pv(STREAM_ENABLE_PV))
        try:
            self.chk_force_h5.setChecked(
                (fw_en.strip() in ("1", "Yes", "Enable")) and
                (st_en.strip() in ("0", "No", "Disable"))
            )
        except Exception:
            self.chk_force_h5.setChecked(False)
        self._set_status("FW inputs refreshed.")

    def _apply_fw_settings(self):
        errs = []
        ok, msg = self._safe_caput(self._pv(FW_NAME_PATTERN_PV), self.in_name.text(), as_string=True)
        if not ok: errs.append(f"FWNamePattern: {msg}")
        ok, msg = self._safe_caput(self._pv(FW_LOCAL_PATH_PV), self.in_path.text(), as_string=True)
        if not ok: errs.append(f"FWLocalPathPrefix: {msg}")
        ok, msg = self._safe_caput(self._pv(FW_IMGS_PER_FILE_PV), self.in_imgs.value(), as_string=False)
        if not ok: errs.append(f"FWImagesPerFile: {msg}")

        if self.chk_force_h5.isChecked():
            ok, msg = self._safe_caput(self._pv(FW_ENABLE_PV), 1, as_string=False)
            if not ok: errs.append(f"FWEnable: {msg}")
            ok, msg = self._safe_caput(self._pv(STREAM_ENABLE_PV), 0, as_string=False)
            if not ok: errs.append(f"StreamEnable: {msg}")

        if errs:
            self._set_status("Apply finished with errors: " + " | ".join(errs))
        else:
            self._set_status("Apply OK — HDF5 preference applied if checked.")

    def _set_status(self, text: str):
        self.status_line.setText(text)

    def closeEvent(self, e):
        try:
            self._timer.stop()
            self._pool.shutdown(wait=False, cancel_futures=True)
        finally:
            super().closeEvent(e)

