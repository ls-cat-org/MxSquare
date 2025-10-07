import shutil
import subprocess
from typing import Tuple

def get_caget_value(pvname: str) -> Tuple[bool, str]:
    """Read PV via subprocess caget, ignoring benign CA warnings on stderr."""
    caget_path = shutil.which("caget")
    if not caget_path:
        return False, "caget not found in PATH"

    try:
        proc = subprocess.run(
            [caget_path, "-t", "-S", pvname],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=2.0,
            check=False,
        )
        out = (proc.stdout or "").strip()
        # Strip any accidental banners on stdout (rare but safe).
        out = "\n".join(
            ln for ln in out.splitlines()
            if not ln.startswith("CA.Client.Exception")
        ).strip()
        if out:
            return True, out

        err = (proc.stderr or "").strip()
        if "Identical process variable names on multiple servers" in err:
            # Treat as benign; just report a neutral message for now.
            return True, "(duplicate PV warning ignored)"

        return False, f"caget error (rc={proc.returncode}): {err.splitlines()[-1] if err else 'no stderr'}"

    except subprocess.TimeoutExpired:
        return False, "caget timeout"
    except Exception as e:
        return False, f"caget exception: {e}"

