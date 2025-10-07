# Last Author
Darren Sherrell October 7th 2025

# MxSquare — User Essentials GUI

Four-window PyQt6 app (blue/yellow theme), each window self-contained:
- Detector (Eiger2) — owns filename/path, exposure, frames, trigger, arm/start/abort, status.
- Omega (MD3) — rotation setup (later: ±90/180 jog).
- Microscope — MD3-UP video/controls (future).
- Recent Images — quick HDF5 previews (future).

## Setup
```bash
conda env create -f environment.yml
conda activate mxsquare
python MxSquare.py

