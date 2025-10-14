# widgets/data_browser_window.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QImage, QPixmap, QPainter
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QPushButton, QFileDialog, QMessageBox,
    QLabel, QDialog, QHBoxLayout, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QFrame, QSpinBox, QCheckBox
)

# Optional matplotlib colormap; falls back to grayscale if missing
try:
    import matplotlib.cm as _mpl_cm  # type: ignore
except Exception:
    _mpl_cm = None


# ---- Theme ----
THEME = {
    "bg": "#e8f0ff",
    "panel": "#f5f8ff",
    "accent": "#ffcc33",
    "text": "#0a1a40",
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

# ---------------- HDF5 + NumPy import ----------------
def _import_h5():
    try:
        import h5py  # type: ignore
        import numpy as np  # type: ignore
        try:
            import hdf5plugin  # type: ignore  # noqa: F401 (registers bitshuffle/lz4 if present)
        except Exception:
            pass
        return h5py, np
    except Exception as e:
        raise RuntimeError(
            "Missing dependency. Install: conda install -c conda-forge h5py numpy (and optionally hdf5plugin)"
        ) from e

# ---------------- Dataset discovery ----------------
def _find_first_image_dataset(h5) -> Optional[Tuple[str, "h5py.Dataset"]]:
    # common Eiger/NeXus
    common = ["/entry/data/data", "/entry/instrument/detector/data"]
    for p in common:
        if p in h5 and hasattr(h5[p], "shape") and getattr(h5[p], "ndim", 0) >= 2:
            return p, h5[p]
    # eiger-style numbered under /entry/data
    if "/entry/data" in h5:
        grp = h5["/entry/data"]
        for k in sorted(grp.keys()):
            d = grp.get(k)
            if hasattr(d, "shape") and getattr(d, "ndim", 0) >= 2:
                return f"/entry/data/{k}", d
    # fallback: first dataset anywhere with ndim>=2
    def visitor(name, obj):
        if hasattr(obj, "shape") and getattr(obj, "ndim", 0) >= 2:
            raise StopIteration((name, obj))
    try:
        h5.visititems(visitor)
    except StopIteration as stop:
        return stop.value
    return None

def _open_dataset(filepath: Path):
    """
    Open HDF5 file and return:
      h5, dset_path, shape, dtype_str, nframes, frame_get(idx), first_frame
    """
    h5py, np = _import_h5()
    h5 = h5py.File(str(filepath), "r")

    found = _find_first_image_dataset(h5)
    if not found:
        h5.close()
        raise RuntimeError("No 2D/3D image dataset found in this HDF5 file.")
    dpath, dset = found
    shape = tuple(int(s) for s in dset.shape)
    dtype_str = str(dset.dtype)

    if dset.ndim == 2:
        nframes = 1
        def frame_get(_idx: int):
            return dset[...]
        first_frame = dset[...]

    elif dset.ndim == 3:
        # Fix recursion: use a raw getter and wrap once
        try:
            _ = dset[0, ...]  # (N, H, W)
            nframes = shape[0]
            def _raw_get(idx: int):
                i = max(0, min(nframes - 1, int(idx)))
                return dset[i, ...]
        except Exception:
            _ = dset[..., 0]  # (H, W, N)
            nframes = shape[-1]
            def _raw_get(idx: int):
                i = max(0, min(nframes - 1, int(idx)))
                return dset[..., i]

        def frame_get(idx: int):
            arr = _raw_get(idx)
            arr = np.squeeze(np.asarray(arr))
            if arr.ndim != 2:
                raise RuntimeError(f"Expected 2D frame, got {arr.shape}")
            return arr

        first_frame = frame_get(0)

    else:
        h5.close()
        raise RuntimeError(f"Unsupported dataset ndim: {dset.ndim}")

    return h5, dpath, shape, dtype_str, nframes, frame_get, first_frame

# ---------------- Rendering helpers ----------------
def _qimage_from_norm01(norm01, colormap: Optional[str]):
    """Convert a normalized 0..1 image to QImage (optionally using a matplotlib colormap)."""
    _, np = _import_h5()
    norm01 = np.clip(np.asarray(norm01, dtype=np.float32), 0.0, 1.0)
    if _mpl_cm is not None and colormap:
        cmap = _mpl_cm.get_cmap(colormap)
        rgba = cmap(norm01)
        rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
        rgb = np.ascontiguousarray(rgb)
        h, w, _ = rgb.shape
        return QImage(rgb.data, w, h, 3*w, QImage.Format.Format_RGB888).copy()
    else:
        img8 = np.ascontiguousarray((norm01 * 255.0).astype(np.uint8))
        h, w = img8.shape
        return QImage(img8.data, w, h, w, QImage.Format.Format_Grayscale8).copy()

# ---------------- Zoomable viewer ----------------
class _ZoomView(QGraphicsView):
    """Stable wheel zoom using view transforms; clamps scale to 0.05x–10x."""
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
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
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
        delta = e.angleDelta().y()
        if delta == 0:
            return
        factor = pow(1.0015, delta)  # smooth; ~1.18x per notch
        new_scale = self._scale * factor
        if new_scale < self._min_scale:
            factor = self._min_scale / self._scale
            self._scale = self._min_scale
        elif new_scale > self._max_scale:
            factor = self._max_scale / self._scale
            self._scale = self._max_scale
        else:
            self._scale = new_scale
        self.scale(factor, factor)

# ---------------- Main widget/window ----------------
class _EigerChooserWidget(QWidget):
    """One button, native .h5 picker; preview with sane defaults + hot mask."""
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
            self, "Select Eiger HDF5 (.h5)", start, "Eiger HDF5 (*.h5);;All Files (*)"
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
        h5py, np = _import_h5()
        try:
            h5, dpath, shape, dtype_str, nframes, frame_get, first = _open_dataset(filepath)
        except Exception as e:
            QMessageBox.critical(self, "Failed to load image", f"{type(e).__name__}: {e}")
            return

        dlg = QDialog(self); dlg.setWindowTitle("Image Preview"); dlg.setModal(True)
        dlg.finished.connect(lambda _: h5.close())

        view = _ZoomView()
        meta = QLabel(); meta.setWordWrap(True)
        idx_label = QLabel(""); idx_label.setMinimumWidth(160)

        # state + defaults
        # manual_range set to (5, 50) as requested
        state = {
            "cmap": "magma",
            "gamma": 0.8,
            "transpose": False,
            "i": 0,
            "manual_range": (5, 50),
            "mask_hot": True,
            "hot_value": 65535,  # will be overwritten below if integer dtype
            "spot_boost": False,   # NEW
        }

        # Controls
        btn_prev = QPushButton("◀ Prev"); btn_next = QPushButton("Next ▶")
        btn_prev.setEnabled(nframes > 1); btn_next.setEnabled(nframes > 1)
        btn_cmap = QPushButton("Colormap: magma")
        btn_gm_down = QPushButton("γ−"); btn_gm_up = QPushButton("γ+")
        btn_trans = QPushButton("Transpose X↔Y")
        btn_hist = QPushButton("Histogram")
        btn_spot = QPushButton("Spot ×10: Off")   # NEW
        btn_close = QPushButton("Close")

        spin_jump = QSpinBox(); spin_jump.setRange(1, max(1, nframes)); spin_jump.setValue(1)
        btn_go = QPushButton("Go")

        lbl_min = QLabel("Min:"); lbl_max = QLabel("Max:")
        spin_min = QSpinBox(); spin_max = QSpinBox()
        spin_min.setRange(-100000, 100000); spin_max.setRange(-100000, 100000)
        spin_min.setValue(5); spin_max.setValue(50)
        btn_apply_range = QPushButton("Apply")


        # Hot pixel mask controls
        chk_mask_hot = QCheckBox("Mask hot pixels")
        spin_hot = QSpinBox(); spin_hot.setRange(0, 10**9)
        # Detect a sensible default hot value
        try:
            ff = np.asarray(first)
            if np.issubdtype(ff.dtype, np.integer):
                hot_default = int(np.iinfo(ff.dtype).max)
            else:
                hot_default = int(np.nanmax(ff))
        except Exception:
            hot_default = 65535
        spin_hot.setValue(hot_default)
        chk_mask_hot.setChecked(True)
        state["hot_value"] = hot_default

        def toggle_mask_hot(checked: bool):
            state["mask_hot"] = bool(checked); _render(state["i"])
        def set_hot_value():
            state["hot_value"] = int(spin_hot.value()); _render(state["i"])
        chk_mask_hot.toggled.connect(toggle_mask_hot)
        spin_hot.editingFinished.connect(set_hot_value)
        def toggle_spot():                        # NEW
            state["spot_boost"] = not state["spot_boost"]
            btn_spot.setText(f"Spot ×10: {'On' if state['spot_boost'] else 'Off'}")
            _render(state["i"])

        def _enhance_small_spots(x01):
            # Visual-only: gently dilate mid-bright pixels to make small spots pop.
            # x01 is expected in [0,1]
            p70 = float(np.percentile(x01, 70.0))
            p98 = float(np.percentile(x01, 98.0))
            if not np.isfinite(p70) or not np.isfinite(p98) or p98 <= p70:
                return x01
        
            mask = (x01 > p70) & (x01 < p98)
            spot = np.where(mask, x01, 0.0)
        
            def dilate8(img):
                up    = np.pad(img[1: , : ], ((0,1),(0,0)))
                down  = np.pad(img[:-1, : ], ((1,0),(0,0)))
                left  = np.pad(img[: , 1: ], ((0,0),(0,1)))
                right = np.pad(img[: , :-1], ((0,0),(1,0)))
                ul    = np.pad(img[1: , 1: ], ((0,1),(0,1)))
                ur    = np.pad(img[1: , :-1], ((0,1),(1,0)))
                dl    = np.pad(img[:-1, 1: ], ((1,0),(0,1)))
                dr    = np.pad(img[:-1, :-1], ((1,0),(1,0)))
                return np.maximum.reduce([img, up, down, left, right, ul, ur, dl, dr])
        
            dil = dilate8(dilate8(dilate8(spot)))  # ~3 iters = strong but still gentle
            out = np.clip(x01 + dil, 0.0, 1.0)
            return out

        # --- rendering with hot mask + fixed range ---
        def _render(frame_index: int):
            i = max(0, min(nframes - 1, frame_index))
            state["i"] = i
            fr = frame_get(i)

            vmin_user, vmax_user = state.get("manual_range", (5, 50))
            inside_low = 0.0   # map min to 0 (use 0.15 if you want a floor)
            inside_high = 1.0  # map max to full brightness
            very_dark = 0.06   # masked areas & below-min

            a = np.asarray(fr, dtype=np.float32)
            out = np.full_like(a, very_dark, dtype=np.float32)

            finite = np.isfinite(a)
            a = np.where(finite, a, np.float32(vmin_user - 1))

            # Hot pixel mask (exact match)
            if state.get("mask_hot", False):
                hv = float(state.get("hot_value", 65535))
                mask_hot = (a == hv)
            else:
                mask_hot = np.zeros_like(a, dtype=bool)

            # In-range linear mapping [min..max] -> [inside_low..inside_high]
            if vmax_user > vmin_user:
                lin = (a - vmin_user) / (vmax_user - vmin_user)
                lin = np.clip(lin, 0.0, 1.0)
                in_or_above_min = (a >= vmin_user) & (~mask_hot)
                out = np.where(in_or_above_min, inside_low + lin * (inside_high - inside_low), out)

            # Values > max are clipped by lin=1 → remain bright; hot stay dark
            out = np.where(mask_hot, very_dark, out)

            # gamma
            g = max(0.1, float(state.get("gamma", 1.0)))
            if abs(g - 1.0) > 1e-3:
                out = np.power(out, g)

            # NEW: spot boost toggle
            if state.get("spot_boost", False):
                out = _enhance_small_spots(out)

            if state.get("transpose", False):
                out = out.T

            qimg = _qimage_from_norm01(out, state.get("cmap"))
            view.set_image(QPixmap.fromImage(qimg))
            idx_label.setText(f"Frame {i+1}/{nframes}")
            meta_info = f"<b>{filepath.name}</b> — {dpath} | shape {shape}, dtype {dtype_str}"
            hot_txt = f", hot={int(state['hot_value'])}" if state.get("mask_hot", False) else ""
            meta.setText(
                meta_info + "<br/>"
                + f"<b>Display</b>: min={vmin_user}, max={vmax_user}{hot_txt}; "
                  f"below-min/NaN/hot→very dark; >max→clipped bright; "
                  f"cmap={state.get('cmap') or 'gray'}; γ={state.get('gamma'):.2f}"
            )

        def go_prev():
            if state["i"] > 0:
                _render(state["i"] - 1); spin_jump.setValue(state["i"] + 1)
        def go_next():
            if state["i"] < nframes - 1:
                _render(state["i"] + 1); spin_jump.setValue(state["i"] + 1)
        def go_to_frame():
            _render(int(spin_jump.value()) - 1)

        def toggle_cmap():
            if state["cmap"] == "magma":
                state["cmap"] = None; btn_cmap.setText("Colormap: grayscale")
            else:
                state["cmap"] = "magma"; btn_cmap.setText("Colormap: magma")
            _render(state["i"])
        def decrease_gamma():
            state["gamma"] = max(0.1, state["gamma"] - 0.2); _render(state["i"])
        def increase_gamma():
            state["gamma"] = state["gamma"] + 0.2; _render(state["i"])
        def toggle_transpose():
            state["transpose"] = not state["transpose"]; _render(state["i"])

        def show_histogram():
            """Histogram (log-y), ignoring zeros and exact hot pixels; lines at min/max."""
            import matplotlib.pyplot as plt
            fr = frame_get(state["i"])
            vals = np.asarray(fr, dtype=np.float32).ravel()
            vals = vals[np.isfinite(vals)]
            vals = vals[vals > 0]
            if state.get("mask_hot", False):
                hv = float(state.get("hot_value", 65535))
                vals = vals[vals != hv]
            if vals.size == 0:
                QMessageBox.information(dlg, "Histogram", "No in-range pixel values to plot.")
                return
            # Trim high tail for visibility
            hi = float(np.percentile(vals, 99.5))
            vals = vals[vals <= hi]

            vmin_user, vmax_user = state.get("manual_range", (5, 50))
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(vals, bins="auto", log=True)
            ax.axvline(vmin_user, color="k", linestyle="--", linewidth=1, label=f"min={vmin_user}")
            ax.axvline(vmax_user, color="k", linestyle="-",  linewidth=1, label=f"max={vmax_user}")
            ax.legend(frameon=False, fontsize=9)
            ax.set_xlabel("Pixel Value (ADU)")
            ax.set_ylabel("Count (log)")
            ax.set_title(f"Histogram – Frame {state['i']+1}")
            fig.tight_layout()
            plt.show()

        def apply_manual_range():
            vmin_val = spin_min.value(); vmax_val = spin_max.value()
            if vmin_val >= vmax_val:
                QMessageBox.warning(dlg, "Invalid range", "Max must be greater than Min.")
                return
            state["manual_range"] = (vmin_val, vmax_val)
            _render(state["i"])

        # Connect
        btn_prev.clicked.connect(go_prev)
        btn_next.clicked.connect(go_next)
        btn_go.clicked.connect(go_to_frame)
        btn_cmap.clicked.connect(toggle_cmap)
        btn_gm_down.clicked.connect(decrease_gamma)
        btn_gm_up.clicked.connect(increase_gamma)
        btn_trans.clicked.connect(toggle_transpose)
        btn_hist.clicked.connect(show_histogram)
        btn_apply_range.clicked.connect(apply_manual_range)
        btn_close.clicked.connect(dlg.accept)
        chk_mask_hot.toggled.connect(toggle_mask_hot)
        spin_hot.editingFinished.connect(set_hot_value)
        btn_spot.clicked.connect(toggle_spot)     # NEW

        # keyboard ← / →
        dlg.keyPressEvent = (  # IMPORTANT: PyQt6 enums use Qt.Key.Key_*
            lambda ev:
                go_prev() if ev.key() == Qt.Key.Key_Left else
                go_next() if ev.key() == Qt.Key.Key_Right else
                QDialog.keyPressEvent(dlg, ev)
        )

        # first frame
        _render(0)

        # layout
        v = QVBoxLayout()
        v.addWidget(meta)
        v.addWidget(view, 1)

        row1 = QHBoxLayout()
        row1.addWidget(btn_prev); row1.addWidget(btn_next); row1.addSpacing(12); row1.addWidget(idx_label)
        row1.addStretch(1)
        row1.addWidget(btn_cmap); row1.addWidget(btn_gm_down); row1.addWidget(btn_gm_up); row1.addWidget(btn_trans); row1.addWidget(btn_spot) 

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Go to:")); row2.addWidget(spin_jump); row2.addWidget(btn_go)
        row2.addSpacing(15)
        row2.addWidget(lbl_min); row2.addWidget(spin_min)
        row2.addWidget(lbl_max); row2.addWidget(spin_max)
        row2.addWidget(btn_apply_range)
        row2.addSpacing(15)
        row2.addWidget(chk_mask_hot); row2.addWidget(QLabel("Hot value:")); row2.addWidget(spin_hot)
        row2.addSpacing(15)
        row2.addWidget(btn_hist)
        row2.addStretch(1)
        row2.addWidget(btn_close)

        v.addLayout(row1)
        v.addLayout(row2)
        dlg.setLayout(v)
        dlg.resize(1600, 1600)
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

