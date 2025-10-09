# widgets/data_browser_window.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QPushButton, QFileDialog, QMessageBox,
    QLabel, QDialog, QScrollArea, QHBoxLayout
)

try:
    import hdf5plugin  # registers bitshuffle/lz4, etc.
except Exception:
    pass

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

# --------- HDF5 utilities ---------
def _import_h5() -> Tuple:
    try:
        import h5py  # type: ignore
        import numpy as np  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency. Please install: pip install h5py numpy"
        ) from e
    return h5py, np

def _find_first_image_dataset(h5) -> Optional[Tuple[str, "h5py.Dataset"]]:
    """
    Search common Eiger/NeXus locations, then fall back to first dataset with ndim>=2.
    Returns (path, dataset) or None.
    """
    # Fast pass: common paths
    common = [
        "/entry/data/data",
        "/entry/instrument/detector/data",
    ]
    for p in common:
        if p in h5 and hasattr(h5[p], "shape") and h5[p].ndim >= 2:
            return p, h5[p]

    # Eiger pattern: /entry/data/data_000001, data_000002, ...
    if "/entry/data" in h5:
        grp = h5["/entry/data"]
        for k in sorted(grp.keys()):
            d = grp.get(k)
            if hasattr(d, "shape") and getattr(d, "ndim", 0) >= 2:
                return f"/entry/data/{k}", d

    # Fallback: first dataset anywhere with ndim >= 2
    def visitor(name, obj):
        if isinstance(obj, h5.__dict__.get("Dataset")) and getattr(obj, "ndim", 0) >= 2:
            raise StopIteration((name, obj))

    try:
        h5.visititems(visitor)
    except StopIteration as stop:
        return stop.value

    return None

def _frame_to_qimage(frame) -> QImage:
    """
    Convert a 2D numpy array (uint16/uint32/float/etc.) to an 8-bit grayscale QImage
    using robust percentile-based contrast.
    """
    _, np = _import_h5()
    if frame.ndim != 2:
        raise ValueError("Expected a 2D array for display.")

    # Robust min/max (ignore NaN/Inf)
    finite = np.isfinite(frame)
    if not finite.any():
        vmin, vmax = 0.0, 1.0
    else:
        vals = frame[finite]
        # Protect against edge cases with constant arrays
        vmin = float(np.percentile(vals, 1))
        vmax = float(np.percentile(vals, 99))
        if vmax <= vmin:
            vmin, vmax = float(vals.min()), float(vals.max())
            if vmax <= vmin:
                vmax = vmin + 1.0

    # Scale to 0..255
    scaled = (np.clip(frame, vmin, vmax) - vmin) / (vmax - vmin)
    # NumPy 2.0-safe: allow copy if needed, then ensure C-contiguous
    img8 = np.asarray(scaled * 255.0, dtype=np.uint8)
    img8 = np.ascontiguousarray(img8)
    h, w = img8.shape

    qimg = QImage(img8.data, w, h, w, QImage.Format.Format_Grayscale8)
    # Keep a copy so the memory stays valid after function returns
    return qimg.copy()

def _load_first_frame(filepath: Path) -> Tuple[QImage, str, Tuple[int, ...], str]:
    """
    Opens file, finds first dataset with images, returns (QImage, dset_path, shape, dtype_str).
    Reads only the first frame if dataset is 3D.
    """
    h5py, np = _import_h5()
    with h5py.File(str(filepath), "r") as h5:
        found = _find_first_image_dataset(h5)
        if not found:
            raise RuntimeError("No 2D/3D image dataset found in this HDF5 file.")
        dpath, dset = found
        shape = tuple(dset.shape)
        dtype_str = str(dset.dtype)

        # Slice first frame if 3D (N,H,W) or (H,W,N) corner case
        if dset.ndim == 3:
            # Assume (N, H, W) first; if that fails, try (H, W, N)
            try:
                frame = dset[0, ...]
            except Exception:
                frame = dset[..., 0]
        elif dset.ndim == 2:
            frame = dset[...]
        else:
            raise RuntimeError(f"Unsupported dataset ndim: {dset.ndim}")

        # Ensure float for scaling
        frame = np.asarray(frame, dtype=np.float32)
        qimg = _frame_to_qimage(frame)
        return qimg, dpath, shape, dtype_str

# --------- UI ---------
class _EigerChooserWidget(QWidget):
    """Inner widget: one button, native file dialog, .h5 enforcement; displays first image."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_file: Optional[Path] = None

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

        # Try to load and show the first image
        try:
            qimg, dpath, shape, dtype_str = _load_first_frame(p)
            self._show_image_dialog(qimg, p, dpath, shape, dtype_str)
        except Exception as e:
            QMessageBox.critical(self, "Failed to load image", f"{type(e).__name__}: {e}")

    def _show_image_dialog(self, qimg: QImage, filepath: Path, dpath: str,
                           shape: Tuple[int, ...], dtype_str: str):
        dlg = QDialog(self)
        dlg.setWindowTitle("First Image Preview")
        dlg.setModal(True)

        # Image label inside a scroll area
        img_label = QLabel()
        img_label.setPixmap(QPixmap.fromImage(qimg))
        img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        scroll = QScrollArea()
        scroll.setWidget(img_label)
        scroll.setWidgetResizable(True)

        # Metadata bar
        meta = QLabel(
            f"<b>File:</b> {filepath.name}<br>"
            f"<b>Dataset:</b> {dpath}<br>"
            f"<b>Shape:</b> {shape} &nbsp; <b>dtype:</b> {dtype_str}"
        )
        meta.setWordWrap(True)

        v = QVBoxLayout()
        v.addWidget(meta)
        v.addWidget(scroll, 1)

        # Optional close button row (click anywhere X to close anyway)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dlg.accept)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(btn_close)
        v.addLayout(row)

        dlg.setLayout(v)
        dlg.resize(900, 700)
        dlg.exec()

class DataBrowserWindow(QMainWindow):
    """Public window class you already import in MxSquare.py"""
    def __init__(self, parent=None, apply_style: bool = True):
        super().__init__(parent)
        if apply_style:
            self.setStyleSheet(STYLE)

        self.setWindowTitle("MxSquare – Data Browser")
        self._widget = _EigerChooserWidget(self)
        self.setCentralWidget(self._widget)
        self.resize(640, 260)

    # Optional: expose the selected file path to the rest of the app
    def selected_file(self) -> Optional[Path]:
        return self._widget.selected_file

