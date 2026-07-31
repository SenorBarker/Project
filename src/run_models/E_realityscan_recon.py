import subprocess
import time
from pathlib import Path


def _run(cmd, cwd, desc=""):
    print(f"$ {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, cwd=str(cwd), check=True)
    print(f"  {desc} done in {time.time() - t0:.0f}s")


def run_realityscan(frames_dir, output_dir, rs_exe, project_name="scene"):
    """
    Launch RealityScan headless to register a folder of frames and export a
    camera-pose registration CSV, consumable by
    F_post_recon_processing.load_reality_scan_trace().

    Args:
        frames_dir   : path-like -- directory of input frames (*.jpg/*.png)
        output_dir   : path-like -- where the registration CSV is written
        rs_exe       : path-like -- path to RealityScan.exe (Linux/Wine build)
        project_name : str -- used for the output CSV filename

    Returns:
        Path to the exported registration CSV in output_dir.
    """
    frames_dir = Path(frames_dir).resolve()
    output_dir = Path(output_dir).resolve()
    rs_exe = Path(rs_exe).resolve()

    assert frames_dir.exists(), f"frames_dir not found: {frames_dir}"
    assert rs_exe.exists(), f"RealityScan executable not found: {rs_exe}"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{project_name}_registration.csv"

    cmd = [
        str(rs_exe),
        "-headless",
        "-newScene",
        "-addFolder", str(frames_dir),
        # TODO: confirm exact alignment command -- RealityScan's CLI docs
        # page for "Alignment Commands" wasn't available when this was
        # written; only the basic project/image commands were confirmed.
        # Likely candidates: "-align", or "-calculateFeatures" followed by
        # "-registerImages". Check against the CLI docs before first run.
        "-align",
        # TODO: confirm exact export command/params for a registration CSV.
        # The exported column layout must match what load_reality_scan_trace
        # expects: x, y, alt, yaw, pitch, roll, #name
        # (src/F_post_recon_processing.py:17-41).
        "-exportRegistration", str(csv_path),
        "-quit",
    ]
    _run(cmd, cwd=frames_dir, desc="RealityScan registration")

    assert csv_path.exists(), f"Expected registration CSV not found: {csv_path}"
    return csv_path
