"""
G_confidence_miner.py -- walks Data/<case>/<recon-type-folder>/... to find every
reconstruction run in the project (each case's reference `batch` run plus every
Cam_pose_vNN/poses_XXXX parameter-sweep run), builds a ReconInstance for each one,
and writes one row per instance to a CSV of confidence percentiles + frame
range/density, so the result can be grouped by recon_type.

5 of the 6 recon techniques carry confidence data on disk: VGGT-O's depth_conf,
CUT3R's per-frame conf/*.npy, CUT3R_Revisit -- CUT3R's own second pass, written to a
nested revisit/ subfolder with its own conf/*.npy, NOT the same data as the first
pass -- lingbot-map's depth_conf/world_points_conf, and MegaSaM's `uncertainty` key
in {scene}_sgd_cvd_hr.npz (misleadingly named -- it's confidence, same polarity as
everyone else's, not inverted; only present in ~55% of MegaSaM runs, added partway
through the project, so older runs still fall back to confidence_available=False).
Plain VGGT has no confidence data at all -- always confidence_available=False.
RealityScan isn't included -- no consistent numbered output folder for it was found
anywhere in Data/.
"""
import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from A_Config import REPO_ROOT

DATA_DIR = Path(REPO_ROOT) / "Data"
OUT_CSV = DATA_DIR / "confidence_report.csv"
NON_CASE_DIRS = {"FIGS"}

PERCENTILES = [1, 5, 25, 50, 75, 95, 99]

RECON_FOLDERS = {
    "VGGT-O": "035_VGGT_O_output",
    "CUT3R": "035_CUT3R_output",
    "CUT3R_Revisit": "035_CUT3R_output",  # same tree as CUT3R -- distinguished by the nested revisit/ subfolder
    "lingbot-map": "036_lingbot_map_output",
    "MegaSaM": "036_MEGASAM_output",
    "VGGT": "034_VGGT_output",
}


@dataclass
class ReconInstance:
    case: str
    recon_type: str
    experiment: str
    experiment_tag: str | None
    path: Path
    confidence: np.ndarray | None
    frame_start: int | None
    frame_end: int | None
    frame_interval: float | None
    n_frames_used: int | None


# ---------------------------------------------------------------- walker ----

def _is_run_dir(dirpath: Path, filenames: list[str], dirnames: list[str], recon_type: str) -> bool:
    """Whether this folder is itself a complete reconstruction run for recon_type,
    identified by the artifact each technique's own producer script writes directly
    into the run folder (see F_post_recon_processing.py's load_* functions)."""
    if recon_type == "VGGT-O":
        return "predictions.npz" in filenames
    if recon_type == "CUT3R":
        return dirpath.name != "revisit" and "conf" in dirnames
    if recon_type == "CUT3R_Revisit":
        return dirpath.name == "revisit" and "conf" in dirnames
    if recon_type == "lingbot-map":
        return any(f.endswith(".npz") for f in filenames)
    if recon_type == "MegaSaM":
        return any(f.endswith("_sgd_cvd_hr.npz") for f in filenames)
    if recon_type == "VGGT":
        return "extrinsic.npy" in filenames
    return False


def _experiment_and_tag(run_dir: Path, type_root: Path):
    """case/<recon-folder>/<experiment>[/<experiment_tag>][/revisit]/... -> (experiment, tag).
    Depth isn't fixed: `batch` runs hold output directly (no tag level), sweep runs
    nest an extra poses_XXXX tag underneath their Cam_pose_vNN experiment. A trailing
    `revisit` component (CUT3R_Revisit's own subfolder) is stripped first so its
    experiment/tag line up with the matching plain-CUT3R run at the same density --
    recon_type is what tells the two apart, not the tag."""
    parts = run_dir.relative_to(type_root).parts
    if parts and parts[-1] == "revisit":
        parts = parts[:-1]
    experiment = parts[0]
    experiment_tag = parts[1] if len(parts) > 1 else None
    return experiment, experiment_tag


def _find_runs(type_root: Path, recon_type: str):
    for dirpath, dirnames, filenames in os.walk(type_root):
        d = Path(dirpath)
        if d.name == "CAM_poses":
            dirnames[:] = []  # sibling metadata folder, not a run itself
            continue
        if _is_run_dir(d, filenames, dirnames, recon_type):
            yield d


def iter_recon_runs(data_dir: Path = DATA_DIR):
    for case_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        if case_dir.name in NON_CASE_DIRS:
            continue
        case = case_dir.name
        for recon_type, folder_name in RECON_FOLDERS.items():
            type_root = case_dir / folder_name
            if not type_root.is_dir():
                continue
            for run_dir in _find_runs(type_root, recon_type):
                experiment, experiment_tag = _experiment_and_tag(run_dir, type_root)
                yield case, recon_type, experiment, experiment_tag, run_dir


# ---------------------------------------------------------- frame stats -----

def _sorted_frames_from_csv(csv_path: Path):
    with open(csv_path, newline="") as f:
        frames = [int(row["frame"]) for row in csv.DictReader(f)]
    return sorted(frames)


def _sorted_frames_from_filenames(dir_path: Path, glob_pattern: str):
    return sorted(int(f.stem) for f in dir_path.glob(glob_pattern))


def _frame_stats(frames: list[int]):
    if not frames:
        return None, None, None, None
    diffs = np.diff(frames)
    interval = float(np.median(diffs)) if len(diffs) else None
    return frames[0], frames[-1], interval, len(frames)


def frame_stats(run_dir: Path, recon_type: str):
    """(frame_start, frame_end, frame_interval, n_frames_used) for a run.
    Primary source is the sibling CAM_poses/poses.csv that every sweep run has;
    `batch` runs don't have one, so fall back to the run's own per-frame filenames
    where the technique names files by frame number (CUT3R, lingbot-map). VGGT-O's
    `batch` folder has neither (single predictions.npz, no per-frame files) --
    falls through to (None, None, None, None)."""
    poses_csv = run_dir / "CAM_poses" / "poses.csv"
    if poses_csv.exists():
        return _frame_stats(_sorted_frames_from_csv(poses_csv))
    if recon_type in ("CUT3R", "CUT3R_Revisit") and (run_dir / "conf").is_dir():
        return _frame_stats(_sorted_frames_from_filenames(run_dir / "conf", "*.npy"))
    if recon_type == "lingbot-map":
        return _frame_stats(_sorted_frames_from_filenames(run_dir, "*.npz"))
    return None, None, None, None


# --------------------------------------------------------- confidence -------
# Deliberately independent of F_post_recon_processing.load_VGGT_O_predictions:
# that helper pulls depth/images/extrinsic/intrinsic onto a torch device too --
# overkill (and a GPU dependency) when only depth_conf is needed here.

def _vggt_o_confidence(run_dir: Path) -> np.ndarray:
    d = np.load(run_dir / "predictions.npz")
    return np.asarray(d["depth_conf"]).reshape(-1)


def _cut3r_confidence(run_dir: Path) -> np.ndarray:
    files = sorted((run_dir / "conf").glob("*.npy"))
    if not files:
        return np.array([])
    return np.concatenate([np.load(f).reshape(-1) for f in files])


def _lingbot_map_confidence(run_dir: Path) -> np.ndarray:
    parts = []
    for f in sorted(run_dir.glob("*.npz")):
        d = np.load(f)
        if "depth_conf" in d:
            parts.append(np.asarray(d["depth_conf"]).reshape(-1))
        if "world_points_conf" in d:
            parts.append(np.asarray(d["world_points_conf"]).reshape(-1))
    return np.concatenate(parts) if parts else np.array([])


def _megasam_confidence(run_dir: Path) -> np.ndarray:
    # Two copies sometimes sit side by side in the same run_dir (an older scene_name-less
    # naming convention alongside the current one) -- same run, so use the newest file only
    # to avoid double-counting identical data.
    files = sorted(run_dir.glob("*_sgd_cvd_hr.npz"), key=lambda f: f.stat().st_mtime)
    if not files:
        return np.array([])
    d = np.load(files[-1])
    if "uncertainty" not in d:
        return np.array([])
    return np.asarray(d["uncertainty"]).reshape(-1)


CONFIDENCE_BUILDERS = {
    "VGGT-O": _vggt_o_confidence,
    "CUT3R": _cut3r_confidence,
    "CUT3R_Revisit": _cut3r_confidence,  # same shape of data, just under revisit/conf/*.npy
    "lingbot-map": _lingbot_map_confidence,
    "MegaSaM": _megasam_confidence,  # only present in ~55% of runs -- see module docstring
    # VGGT (plain): no confidence data exists -- rows stay confidence_available=False.
}


def build_instance(case, recon_type, experiment, experiment_tag, run_dir: Path) -> ReconInstance:
    frame_start, frame_end, frame_interval, n_frames_used = frame_stats(run_dir, recon_type)

    confidence = None
    builder = CONFIDENCE_BUILDERS.get(recon_type)
    if builder is not None:
        try:
            arr = builder(run_dir)
            confidence = arr if arr.size > 0 else None
        except (OSError, KeyError, ValueError) as e:
            print(f"WARN: couldn't read confidence for {run_dir}: {e}")

    return ReconInstance(
        case=case, recon_type=recon_type, experiment=experiment, experiment_tag=experiment_tag,
        path=run_dir, confidence=confidence,
        frame_start=frame_start, frame_end=frame_end,
        frame_interval=frame_interval, n_frames_used=n_frames_used,
    )


# -------------------------------------------------------------- mining ------

CSV_COLUMNS = [
    "case", "recon_type", "experiment", "experiment_tag",
    "frame_start", "frame_end", "frame_interval", "n_frames_used",
    "confidence_available", "n_samples", "mean", "min", "max",
    *[f"p{p}" for p in PERCENTILES],
]


def _row_for(instance: ReconInstance) -> dict:
    row = {
        "case": instance.case,
        "recon_type": instance.recon_type,
        "experiment": instance.experiment,
        "experiment_tag": instance.experiment_tag or "",
        "frame_start": instance.frame_start,
        "frame_end": instance.frame_end,
        "frame_interval": instance.frame_interval,
        "n_frames_used": instance.n_frames_used,
    }
    if instance.confidence is None:
        row["confidence_available"] = False
        for col in ("n_samples", "mean", "min", "max", *[f"p{p}" for p in PERCENTILES]):
            row[col] = ""
        return row

    flat = instance.confidence.astype(np.float64)
    row["confidence_available"] = True
    row["n_samples"] = flat.size
    row["mean"] = float(flat.mean())
    row["min"] = float(flat.min())
    row["max"] = float(flat.max())
    for p, v in zip(PERCENTILES, np.percentile(flat, PERCENTILES)):
        row[f"p{p}"] = float(v)
    return row


def _completed_keys(out_csv: Path) -> set[tuple[str, str, str, str | None]]:
    if not out_csv.exists():
        return set()
    with open(out_csv, newline="") as f:
        return {
            (row["case"], row["recon_type"], row["experiment"], row["experiment_tag"] or None)
            for row in csv.DictReader(f)
        }


def mine(data_dir: Path = DATA_DIR, out_csv: Path = OUT_CSV, resume: bool = True) -> Path:
    """Processes one run at a time (build -> row -> discard) so peak memory stays
    bounded despite individual VGGT-O predictions.npz files running ~300MB, and
    flushes each row as it's written so an interrupted run doesn't lose progress.

    resume=True (default) skips any (case, recon_type, experiment, experiment_tag)
    already present in out_csv from a prior run and appends rather than overwriting --
    safe to interrupt/kill and re-run from the top. resume=False always starts fresh."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_keys(out_csv) if resume else set()
    append = resume and out_csv.exists()
    with open(out_csv, "a" if append else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not completed:
            writer.writeheader()
        for case, recon_type, experiment, experiment_tag, run_dir in iter_recon_runs(data_dir):
            if (case, recon_type, experiment, experiment_tag) in completed:
                continue
            instance = build_instance(case, recon_type, experiment, experiment_tag, run_dir)
            row = _row_for(instance)
            writer.writerow(row)
            f.flush()
            print(f"{case}/{recon_type}/{experiment}/{experiment_tag or '-'}: "
                  f"{'ok' if row['confidence_available'] else 'N/A'}")
    return out_csv


if __name__ == "__main__":
    mine()
