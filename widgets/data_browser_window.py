# widgets/data_browser_window.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Callable

from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QImage, QPixmap, QPainter

try:
    # Optional; used for magma colormap
    import matplotlib.cm as _mpl_cm  # type: ignore
except Exception:
    _mpl_cm = None



from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QPushButton, QFileDialog, QMessageBox,
    QLabel, QDialog, QHBoxLayout, QSlider, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QFrame
)

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
def _import_h5():
    try:
        import h5py  # type: ignore
        import numpy as np  # type: ignore
        try:
            import hdf5plugin  # type: ignore  # noqa: F401
        except Exception:
            pass
    except Exception as e:
        raise RuntimeError("Missing dependency. Please install: conda install -c conda-forge h5py numpy (and optionally hdf5plugin)") from e
    return h5py, np

def _find_first_image_dataset(h5) -> Optional[Tuple[str, "h5py.Dataset"]]:
    common = [
        "/entry/data/data",
        "/entry/instrument/detector/data",
    ]
    for p in common:
        if p in h5 and hasattr(h5[p], "shape") and getattr(h5[p], "ndim", 0) >= 2:
            return p, h5[p]
    if "/entry/data" in h5:
        grp = h5["/entry/data"]
        for k in sorted(grp.keys()):
            d = grp.get(k)
            if hasattr(d, "shape") and getattr(d, "ndim", 0) >= 2:
                return f"/entry/data/{k}", d
    def visitor(name, obj):
        if isinstance(obj, h5.__dict__.get("Dataset")) and getattr(obj, "ndim", 0) >= 2:
            raise StopIteration((name, obj))
    try:
        h5.visititems(visitor)
    except StopIteration as stop:
        return stop.value
    return None


def _frame_to_qimage(
    frame,
    lo_pct: float = 2.0,
    hi_pct: float = 99.5,
    gamma: float = 1.6,
    colormap: str = "magma",
) -> QImage:
    """
    Convert a 2D numpy array to an 8-bit RGB QImage using a perceptually-uniform colormap.
    - Percentile scaling (lo_pct..hi_pct)
    - Gamma > 1 compresses low-end (noise suppression)
    - Uses matplotlib colormap if available; otherwise falls back to grayscale
    """
    _, np = _import_h5()
    if frame.ndim != 2:
        raise ValueError("Expected a 2D array for display.")

    finite = np.isfinite(frame)
    if not finite.any():
        vmin, vmax = 0.0, 1.0
    else:
        vals = frame[finite]
        vmin = float(np.percentile(vals, lo_pct))
        vmax = float(np.percentile(vals, hi_pct))
        if vmax <= vmin:
            vmin, vmax = float(vals.min()), float(vals.max())
            if vmax <= vmin:
                vmax = vmin + 1.0

    # Normalize 0..1
    norm = (np.clip(frame, vmin, vmax) - vmin) / (vmax - vmin)
    # Noise suppression: compress lows with gamma > 1
    if gamma and gamma > 0:
        norm = np.power(norm, gamma)

    if _mpl_cm is not None and colormap:
        cmap = _mpl_cm.get_cmap(colormap)
        rgba = cmap(norm)  # (H,W,4), floats 0..1
        rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
        rgb = np.ascontiguousarray(rgb)
        h, w, _ = rgb.shape
        bytes_per_line = 3 * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return qimg.copy()
    else:
        # Fallback: grayscale
        img8 = np.asarray(norm * 255.0, dtype=np.uint8)
        img8 = np.ascontiguousarray(img8)
        h, w = img8.shape
        qimg = QImage(img8.data, w, h, w, QImage.Format.Format_Grayscale8)
        return qimg.copy()


def _open_dataset(filepath: Path):
    h5py, np = _import_h5()
    h5 = h5py.File(str(filepath), "r")
    found = _find_first_image_dataset(h5)
    if not found:
        h5.close()
        raise RuntimeError("No 2D/3D image dataset found in this HDF5 file.")
    dpath, dset = found
    shape = tuple(dset.shape)
    dtype_str = str(dset.dtype)
    if dset.ndim == 3:
        try:
            _ = dset[0, ...]
            first = dset[0, ...]
            frame_get: Callable[[int], "np.ndarray"] = lambda idx: dset[int(idx), ...]
            nframes = shape[0]
        except Exception:
            _ = dset[..., 0]
            first = dset[..., 0]
            frame_get = lambda idx: dset[..., int(idx)]
            nframes = shape[-1]
    elif dset.ndim == 2:
        first = dset[...]
        frame_get = lambda idx: dset[...]
        nframes = 1
    else:
        h5.close()
        raise RuntimeError(f"Unsupported dataset ndim: {dset.ndim}")
    return h5, dpath, shape, dtype_str, nframes, frame_get, first

def _qimage_from_frame(frame: "np.ndarray") -> QImage:
    _, np = _import_h5()
    frame = np.asarray(frame, dtype=np.float32)
    return _frame_to_qimage(frame)

# --------- Robust zoomable view ---------
class _ZoomView(QGraphicsView):
    """
    QGraphicsView-based image viewer with stable wheel zoom.
    - Transforms view; does NOT recreate pixmaps each wheel step.
    - Clamped zoom to avoid runaway allocations.
    """
    def __init__(self):
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pix_item: Optional[QGraphicsPixmapItem] = None
        self._scale = 1.0
        self._min_scale = 0.05
        self._max_scale = 10.0

        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)  # Pan with mouse

        # Aesthetic: no frame
        self.setFrameShape(QFrame.Shape.NoFrame)

    def set_image(self, pm: QPixmap):
        self._scene.clear()
        self._pix_item = self._scene.addPixmap(pm)
        self._scene.setSceneRect(QRectF(pm.rect()))
        self._scale = 1.0
        self.resetTransform()

    def wheelEvent(self, e):
        if self._pix_item is None:
            return
        # Gentle scaling: factor based on wheel delta (120 per notch)
        delta = e.angleDelta().y()
        if delta == 0:
            return
        factor = pow(1.0015, delta)  # ~1.18x per notch; smoother for large deltas
        new_scale = self._scale * factor
        # Clamp to safe range
        if new_scale < self._min_scale:
            factor = self._min_scale / self._scale
            self._scale = self._min_scale
        elif new_scale > self._max_scale:
            factor = self._max_scale / self._scale
            self._scale = self._max_scale
        else:
            self._scale = new_scale
        self.scale(factor, factor)

# --------- Main widget/window ---------
class _EigerChooserWidget(QWidget):
    """One button, native .h5 picker; opens preview dialog with slider + zoom."""
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
            self, "Select Eiger HDF5 (.h5)", start, "Eiger HDF5 (*.h5);;All Files (*)",
        )
        if not fp:
            return
        p = Path(fp)
        if p.suffix.lower() != ".h5":
            QMessageBox.warning(self, "Unsupported file", "Sorry, I only accept .h5 files right now.")
            return
        self.selected_file = p
        self.label.setText(str(p))
        self._show_image_dialog(p)

    def _show_image_dialog(self, filepath: Path):
        try:
            h5, dpath, shape, dtype_str, nframes, frame_get, first = _open_dataset(filepath)
        except Exception as e:
            QMessageBox.critical(self, "Failed to load image", f"{type(e).__name__}: {e}")
            return
    
        dlg = QDialog(self)
        dlg.setWindowTitle("Image Preview")
        dlg.setModal(True)
    
        # Ensure HDF5 closes when dialog finishes
        def _cleanup():
            try:
                h5.close()
            except Exception:
                pass
        dlg.finished.connect(lambda _: _cleanup())
    
        meta = QLabel(
            f"<b>File:</b> {filepath.name}<br>"
            f"<b>Dataset:</b> {dpath}<br>"
            f"<b>Shape:</b> {shape} &nbsp; <b>dtype:</b> {dtype_str}<br>"
            f"<b>Frames:</b> {nframes}"
        )
        meta.setWordWrap(True)
    
        view = _ZoomView()
    
        # --- Navigation buttons instead of slider ---
        btn_prev = QPushButton("◀ Prev")
        btn_next = QPushButton("Next ▶")
        # Disable if single frame
        btn_prev.setEnabled(nframes > 1)
        btn_next.setEnabled(nframes > 1)
    
        # Current frame index + label
        idx_label = QLabel("")
        idx_label.setMinimumWidth(120)  # keep layout tidy
    
        # state
        current = {"i": 0}
    
        def set_frame(i: int):
            # clamp to bounds
            i = max(0, min(nframes - 1, i))
            current["i"] = i
            idx_label.setText(f"Frame {i+1} / {nframes}")
            try:
                qimg = _qimage_from_frame(frame_get(i))
                view.set_image(QPixmap.fromImage(qimg))
            except Exception as e:
                QMessageBox.critical(dlg, "Read error", f"{type(e).__name__}: {e}")
    
        def go_prev():
            if nframes <= 1:
                return
            set_frame(current["i"] - 1)
    
        def go_next():
            if nframes <= 1:
                return
            set_frame(current["i"] + 1)
    
        btn_prev.clicked.connect(go_prev)
        btn_next.clicked.connect(go_next)
    
        # Optional: keyboard shortcuts (← / →)
        dlg.keyPressEvent = (  # type: ignore
            lambda ev:
                go_prev() if ev.key() == Qt.Key.Key_Left else
                go_next() if ev.key() == Qt.Key.Key_Right else
                QDialog.keyPressEvent(dlg, ev)
        )
    
        # First frame
        qimg0 = _qimage_from_frame(first)
        view.set_image(QPixmap.fromImage(qimg0))
        idx_label.setText(f"Frame 1 / {nframes}")
    
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dlg.accept)
    
        # Layout
        v = QVBoxLayout()
        v.addWidget(meta)
        v.addWidget(view, 1)
    
        nav = QHBoxLayout()
        nav.addWidget(btn_prev)
        nav.addWidget(btn_next)
        nav.addSpacing(12)
        nav.addWidget(idx_label)
        nav.addStretch(1)
        nav.addWidget(btn_close)
        v.addLayout(nav)
    
        dlg.setLayout(v)
        dlg.resize(1000, 750)
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

    def selected_file(self) -> Optional[Path]:
        return self._widget.selected_file

