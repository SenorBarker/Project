import os
import shutil
import subprocess
import time
from pathlib import Path


# Vars from the calling Jupyter kernel's own env (a different conda env, e.g.
# Msc2) that have no business leaking into the mega_sam subprocess: MPLBACKEND
# points at matplotlib_inline, which isn't installed there and crashes any
# script that imports matplotlib; PYTHONPATH/PYTHONHOME would point at the
# wrong env's site-packages entirely.
_ENV_BLOCKLIST = ("MPLBACKEND", "PYTHONPATH", "PYTHONHOME")


def _clean_env(extra=None):
    env = {k: v for k, v in os.environ.items() if k not in _ENV_BLOCKLIST}
    if extra:
        env.update(extra)
    return env


def _run(cmd, cwd, env=None, desc=""):
    print(f"$ {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, cwd=str(cwd), env=env or _clean_env(), check=True)
    print(f"  {desc} done in {time.time() - t0:.0f}s")


def run_megasam(frames_dir, output_dir, mega_sam_dir, scene_name="scene",
                 conda_env="mega_sam"):
    """
    Run the full MegaSaM pipeline (mono-depth -> camera tracking -> RAFT flow
    + consistent video depth optimisation) on a directory of frames.

    MegaSaM needs its own conda env (torch 2.0.1 / CUDA 11.8 — incompatible
    with this notebook's kernel env), so each stage runs out-of-process via
    `conda run -n <conda_env>`, with cwd set to mega_sam_dir since the
    underlying scripts use paths relative to the repo root.

    Args:
        frames_dir   : path-like — directory of input frames (*.jpg/*.png)
        output_dir   : path-like — where the final {scene_name}_sgd_cvd_hr.npz
                        is copied to
        mega_sam_dir : path-like — root of the mega-sam repo
        scene_name   : str — key threading through mega-sam's intermediate
                        reconstructions/, cache_flow/, outputs/ dirs
        conda_env    : str — name of the conda env built by setup_linux_env.sh

    Returns:
        Path to the copied {scene_name}_sgd_cvd_hr.npz in output_dir
        (keys: images, depths, intrinsic, cam_c2w)
    """
    frames_dir   = Path(frames_dir).resolve()
    output_dir   = Path(output_dir).resolve()
    mega_sam_dir = Path(mega_sam_dir).resolve()

    assert frames_dir.exists(), f"frames_dir not found: {frames_dir}"
    ckpt = mega_sam_dir / "checkpoints" / "megasam_final.pth"
    assert ckpt.exists(), f"MegaSaM checkpoint not found: {ckpt}"

    output_dir.mkdir(parents=True, exist_ok=True)

    def conda_run(*args):
        return ["conda", "run", "--no-capture-output", "-n", conda_env, *args]

    # 1. mono-depth precompute (Depth-Anything + UniDepth)
    _run(conda_run(
        "python", "Depth-Anything/run_videos.py",
        "--encoder", "vitl",
        "--load-from", "Depth-Anything/checkpoints/depth_anything_vitl14.pth",
        "--img-path", str(frames_dir),
        "--outdir", f"Depth-Anything/video_visualization/{scene_name}",
    ), cwd=mega_sam_dir, desc="Depth-Anything")

    unidepth_env = _clean_env({"PYTHONPATH": str(mega_sam_dir / "UniDepth")})
    _run(conda_run(
        "python", "UniDepth/scripts/demo_mega-sam.py",
        "--scene-name", scene_name,
        "--img-path", str(frames_dir),
        "--outdir", "UniDepth/outputs",
    ), cwd=mega_sam_dir, env=unidepth_env, desc="UniDepth")

    # 2. camera tracking (DROID-style bundle adjustment)
    _run(conda_run(
        "python", "camera_tracking_scripts/test_demo.py",
        f"--datapath={frames_dir}",
        f"--weights={ckpt}",
        "--scene_name", scene_name,
        "--mono_depth_path", str(mega_sam_dir / "Depth-Anything" / "video_visualization"),
        "--metric_depth_path", str(mega_sam_dir / "UniDepth" / "outputs"),
        "--disable_vis",
    ), cwd=mega_sam_dir, desc="camera tracking")

    # 3. RAFT flow + consistent video depth optimisation
    _run(conda_run(
        "python", "cvd_opt/preprocess_flow.py",
        f"--datapath={frames_dir}",
        "--model=cvd_opt/raft-things.pth",
        "--scene_name", scene_name,
        "--mixed_precision",
    ), cwd=mega_sam_dir, desc="RAFT flow")

    _run(conda_run(
        "python", "cvd_opt/cvd_opt.py",
        "--scene_name", scene_name,
        "--w_grad", "2.0",
        "--w_normal", "5.0",
    ), cwd=mega_sam_dir, desc="CVD optimisation")

    src = mega_sam_dir / "outputs_cvd" / f"{scene_name}_sgd_cvd_hr.npz"
    assert src.exists(), f"Expected output not found: {src}"
    dst = output_dir / src.name
    shutil.copy(src, dst)
    print(f"Output: {dst}")
    return dst
