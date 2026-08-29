# Process code for the AA_Cam_pose_sweeps.ipynb family of notebooks -- scoring,
# resumability/status tracking, disk-retention cleanup, and the Part 2 sweep loop
# itself. Per-case parameterization (case_name, frame range, CAM_POSES, FLAGS,
# DATA_STORAGE) and Part 3 (load + plot) stay in the notebook -- see
# AA_Cam_pose_sweeps.ipynb.

import argparse
import gc
import shutil
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from A_Config import (
    set_case, case_dir, cut3r_output_dir, vggt_o_output_dir, predictions_path,
    lingbot_map_dir, frames_for_cam_poses_dir, FRAME_NAME_FMT,
)
from Two2D.B_video_processing import image_sequencer, video_fps, available_frames
from run_models.E_cut3r_recon import run_cut3r
from run_models.E_VGGT_omega import run_vggt_omega
from run_models.E_megasam_recon import run_megasam
from run_models.B_lingbot_map import run_lingbot_map
from Thr3D.F_post_recon_processing import (
    Reconstruction, load_cut3r_trace_v2, load_megasam_trace, load_VGGT_trace,
    load_lingbot_map_trace,
)
from Thr3D.F_transpose_to_recon_objects import (
    cut3r_to_reconstruction, megasam_to_reconstruction, lingbot_map_to_reconstruction,
    vggt_plain_to_reconstruction,
)
from P_projection_mapping import BEV_tile_render_PM
from Q_Metric_georeferencing import GPS_camerapose_matcher
from Thr3D.G_transforms_alignments import umeyama_align


#------------------------------------------------------------------
# Scoring -- align_and_score (reused verbatim from AA_Cam_pose_estimate_batches.ipynb)
#------------------------------------------------------------------

def trajectory_length(positions):
    return float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))


def spatial_extent(positions):
    centroid = positions.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((positions - centroid) ** 2, axis=1))))


def align_and_score(technique, cam_pose_dict, runtime_minutes, gps_dict):
    GPS_locs, lat_0, lon_0, object_locs = GPS_camerapose_matcher(gps_dict, cam_pose_dict)
    R, s, t, B_aligned, rmse = umeyama_align(
        GPS_locs, object_locs, label_A="GT", label_B=technique,
        out_path=None,  # no per-iteration PNGs during the sweep -- see plt.close('all') in run_attempt
        
    )
    mean_dist = float(np.mean(np.linalg.norm(GPS_locs - B_aligned, axis=1)))
    traj_len = trajectory_length(GPS_locs)
    return {
        "rmse": rmse,
        "mean dist": mean_dist,
        "correspondences": len(GPS_locs),
        "cam poses (n)": len(cam_pose_dict),  # actual reconstructed pose count -- distinct from
                                               # the sweep loop's "cam_poses" (target density), which
                                               # run_attempt sets separately on this same result dict
        "trajectory length (m)": traj_len,
        "spatial_extent": spatial_extent(GPS_locs),
        "rmse cm/m": 100 * rmse / traj_len if traj_len > 0 else float("nan"),
        "runtime (min)": runtime_minutes,
    }


#------------------------------------------------------------------
# Resumability infrastructure
#
# A GPU OOM kills the whole kernel -- it does not raise a catchable Python
# exception. So a 'pending' status row is written to disk *before* each risky
# call starts, not just a result written after it succeeds. On restart, any row
# still 'pending' means the kernel died mid-attempt (presumed OOM); every larger
# cam_poses for that same technique is then preemptively marked failed/skipped
# rather than re-attempted, since more frames means more GPU memory and the
# same wall will just get hit again.
#------------------------------------------------------------------

STATUS_COLUMNS = ["technique", "cam_poses", "status", "error_type", "error_message", "timestamp"]

# Only these error_types are treated as a hard wall that skips future re-runs --
# an OOM at this density will predictably OOM again, so there's no point retrying.
# Any other failure (a code bug, a data-availability issue like too few GT
# correspondences, etc.) is NOT terminal -- it gets retried every time the
# notebook re-runs, until it either succeeds or the underlying bug is fixed.
_TERMINAL_ERROR_TYPES = {"oom", "presumed_oom_kernel_death", "skipped_precautionary"}


def _load_status(status_path):
    if status_path.exists():
        # keep_default_na=False: without it, an all-blank column (e.g. every
        # "pending"/"success" row's error_type="") round-trips through CSV as
        # NaN and gets inferred as float64 -- then the next _set_status() call
        # that tries to write a real string into that column crashes.
        return pd.read_csv(status_path, keep_default_na=False, dtype=str)
    return pd.DataFrame(columns=STATUS_COLUMNS)


def _set_status(status_path, technique, cam_poses, status, error_type="", error_message=""):
    df = _load_status(status_path)
    mask = (df["technique"] == technique) & (df["cam_poses"] == str(cam_poses))
    row = {
        "technique": technique, "cam_poses": str(cam_poses), "status": status,
        "error_type": error_type, "error_message": error_message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if mask.any():
        for k, v in row.items():
            df.loc[mask, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(status_path, index=False)  # flushed immediately -- this IS the crash-recovery record


def _get_status_row(status_path, technique, cam_poses):
    df = _load_status(status_path)
    mask = (df["technique"] == technique) & (df["cam_poses"] == str(cam_poses))
    if not mask.any():
        return None
    return df.loc[mask].iloc[0]


def _get_status(status_path, technique, cam_poses):
    row = _get_status_row(status_path, technique, cam_poses)
    return None if row is None else row["status"]


def _should_skip(status_path, technique, cam_poses):
    """Success is always skipped. A failure is only skipped if it was terminal
    (OOM-related) -- an ordinary/code-bug failure keeps being retried on every
    re-run, since nothing about re-running would predictably fail the same way
    once the actual bug is fixed."""
    row = _get_status_row(status_path, technique, cam_poses)
    if row is None:
        return False, None
    if row["status"] == "success":
        return True, "success"
    if row["status"] == "failed" and row["error_type"] in _TERMINAL_ERROR_TYPES:
        return True, f"failed ({row['error_type']})"
    return False, None


def _append_result(results_path, row):
    if results_path.exists():
        df = pd.concat([pd.read_csv(results_path), pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(results_path, index=False)  # written immediately after every success, not batched


_OOM_MARKERS = ("out of memory", "cuda out of memory", "cudnn_status_alloc_failed", "cublas_status_alloc_failed")


def classify_error(e):
    if hasattr(torch.cuda, "OutOfMemoryError") and isinstance(e, torch.cuda.OutOfMemoryError):
        return "oom"
    if any(m in str(e).lower() for m in _OOM_MARKERS):
        return "oom"
    return "other"


def resolve_stale_pending(status_path, cam_poses_list):
    """Run once at sweep start: a leftover 'pending' row means the kernel died
    mid-attempt last time (almost certainly a GPU OOM) -- nothing ever ran the
    except block that would have recorded a normal failure. Mark it as a presumed
    OOM and preemptively skip every larger cam_poses for that technique."""
    df = _load_status(status_path)
    pending = df[df["status"] == "pending"]
    for _, r in pending.iterrows():
        technique, cam_poses = r["technique"], int(r["cam_poses"])
        print(f"[{technique} @ {cam_poses}] kernel died mid-attempt last run -- presumed GPU OOM "
              f"(restart the kernel now if you haven't already)")
        _set_status(status_path, technique, cam_poses, "failed", "presumed_oom_kernel_death",
                    "kernel died before status could be resolved")
        for cp in cam_poses_list:
            if cp > cam_poses and _get_status(status_path, technique, cp) is None:
                _set_status(status_path, technique, cp, "failed", "skipped_precautionary",
                            f"skipped: {technique} OOM'd at cam_poses={cam_poses}")
                print(f"[{technique} @ {cp}] precautionary skip (OOM upstream at cam_poses={cam_poses})")


def run_attempt(status_path, results_path, cam_poses_list, technique, cam_poses, run_fn):
    """run_fn() does the actual reconstruction + scoring and returns the
    align_and_score(...) result dict (technique/cam_poses added by this wrapper),
    or raises on a catchable failure."""
    skip, reason = _should_skip(status_path, technique, cam_poses)
    if skip:
        print(f"[{technique} @ {cam_poses}] already {reason}, skipping")
        return

    _set_status(status_path, technique, cam_poses, "pending")
    try:
        result = run_fn()
        result["technique"] = technique
        result["cam_poses"] = cam_poses
        _append_result(results_path, result)
        _set_status(status_path, technique, cam_poses, "success")
        print(f"[{technique} @ {cam_poses}] success, rmse={result['rmse']:.3f}")
    except Exception as e:
        error_type = classify_error(e)
        _set_status(status_path, technique, cam_poses, "failed", error_type, str(e)[:500])
        print(f"[{technique} @ {cam_poses}] FAILED ({error_type}): {e}")
        if error_type == "oom":
            for cp in cam_poses_list:
                if cp > cam_poses and _get_status(status_path, technique, cp) is None:
                    _set_status(status_path, technique, cp, "failed", "skipped_precautionary",
                                f"skipped: {technique} OOM'd at cam_poses={cam_poses}")
                    print(f"[{technique} @ {cp}] precautionary skip (OOM upstream at cam_poses={cam_poses})")
    finally:
        plt.close("all")  # umeyama_align/ortho_charts and the trace loaders never close their figures
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


#------------------------------------------------------------------
# Data retention -- what the sweep keeps on disk vs deletes per density/technique
#------------------------------------------------------------------

def _positions_only(cam_pose_dict):
    """cam_pose_dict values are either (3,) positions or (4,4) c2w pose matrices
    (full_pose=True) -- scoring always needs position-only, regardless of which
    the loader was asked for."""
    sample = next(iter(cam_pose_dict.values()))
    if sample.ndim == 1:
        return cam_pose_dict
    return {k: v[:3, 3] for k, v in cam_pose_dict.items()}


def _render_bev_tiles(technique, recon, cam_pose_dict, out_dir, n_frames, thresholds=(0, 5, 10, 20, 40, 60,80,90)):
    """BEV tiles are a diagnostic render, not a correctness check -- a threshold with
    no points above it (e.g. a genuinely low-confidence recon) means a blank tile,
    not a failed technique, so this must not raise and fail the whole run_attempt."""
    for c in thresholds:
        try:
            ct = np.percentile(recon.preds["depth_conf"].cpu().numpy(), c)
            BEV_tile_render_PM(
                recon, extra_indices=list(cam_pose_dict.keys()), masks_dir=None,
                confidence_threshold=ct, out_path=out_dir / f"{technique}_{n_frames}_BEV_c{c}.png",
            )
        except Exception as e:
            print(f"[{technique} @ {n_frames}] BEV render skipped for conf_thresh={c}: {e}")


def _save_cam_pose_csv(path, cam_pose_dict, full_pose):
    """frame,x,y,z -- or frame,x,y,z,r00..r22 (flattened rotation) when full_pose."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for frame in sorted(cam_pose_dict):
        pose = cam_pose_dict[frame]
        if full_pose:
            rotation, translation = pose[:3, :3], pose[:3, 3]
            rows.append([frame, *translation, *rotation.flatten()])
        else:
            rows.append([frame, *pose])
    columns = ["frame", "x", "y", "z"]
    if full_pose:
        columns += [f"r{i}{j}" for i in range(3) for j in range(3)]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


# Subfolders written inside technique_output_dir that survive a keep_recon_data=False
# cleanup, each gated by its own data_storage toggle. BEV pngs are NOT here -- they're
# written to run_density_sweep's shared out_dir (080_Experiments/{experiment}/), not
# inside technique_output_dir, so they're never touched by this cleanup at all.
_RETAINED_SUBDIRS = {"CAM_poses": "keep_cam_poses"}


def _cleanup_technique_output(technique_output_dir, data_storage):
    """Deletes technique_output_dir's contents per data_storage's keep_recon_data/
    keep_cam_poses toggles. CAM_poses/ (written by _save_cam_pose_csv below) lives
    inside technique_output_dir, so deleting recon data can't be a blind rmtree of
    the whole dir -- everything except that subfolder is removed, then it's removed
    separately too if keep_cam_poses says not to keep it."""
    if technique_output_dir is None or not technique_output_dir.exists():
        return
    if not data_storage["keep_recon_data"]:
        for child in technique_output_dir.iterdir():
            if child.name in _RETAINED_SUBDIRS:
                continue
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    for name, flag in _RETAINED_SUBDIRS.items():
        if not data_storage[flag]:
            subdir = technique_output_dir / name
            if subdir.exists():
                shutil.rmtree(subdir)
    if not any(technique_output_dir.iterdir()):
        technique_output_dir.rmdir()


#------------------------------------------------------------------
# Reset sweep state
#------------------------------------------------------------------

def reset_sweep_state(case_name, experiment, confirm):
    """Deletes only this sweep's own output -- everything from running this sweep,
    nothing more, nothing less: every poses_* frame/reconstruction folder under
    experiment, plus the status/results CSVs and per-technique BEV pngs under
    080_Experiments/{experiment}/ (must match run_density_sweep's own out_dir
    exactly, or this silently looks in the wrong place and finds nothing to delete).

    Safe by default: with confirm=False this only prints what it *would* delete."""
    set_case(case_name, experiment)
    out_dir = case_dir() / "080_Experiments" / f"{experiment}"

    # Every stage folder the sweep writes into, scoped to this sweep's own
    # experiment tag -- deleting the whole subtree here is equivalent to deleting
    # every poses_XXXX folder under it, without having to glob for them individually.
    # out_dir itself covers status/results CSVs + BEV pngs, all written there now.
    sweep_dirs = [
        case_dir() / "020_frames_for_cam_poses" / experiment,
        case_dir() / "034_VGGT_output" / experiment,
        case_dir() / "035_CUT3R_output" / experiment,
        case_dir() / "035_VGGT_O_output" / experiment,
        case_dir() / "036_lingbot_map_output" / experiment,
        case_dir() / "036_MEGASAM_output" / experiment,
        out_dir,
    ]

    targets_dirs = [d for d in sweep_dirs if d.exists()]

    if not confirm:
        print("confirm=False -- dry run only, nothing deleted. Would remove:")
        for d in targets_dirs:
            print(f"  rmtree {d}")
        if not targets_dirs:
            print("  (nothing to delete)")
    else:
        for d in targets_dirs:
            shutil.rmtree(d)
            print(f"deleted {d}")
        print("Sweep state reset.")


#------------------------------------------------------------------
# Part 2 -- Sweep (expensive, resumable, run once/rarely)
#
# Each cam_poses value gets its own frames/output folders via a nested
# experiment tag (f"{experiment}/poses_{cam_poses:04d}" passed to set_case) --
# this makes every A_Config getter (including run_lingbot_map's, which has no
# path parameters at all and only reads the zero-arg getters) automatically
# density-scoped, with no per-function path-wrangling needed.
#
# Safe to interrupt/kill and re-run from the top at any point.
#------------------------------------------------------------------

def run_density_sweep(case_name, experiment, video_path, start_3D_recon, end_3D_recon,
                       cam_poses_list, technique_order, flags, data_storage, project_root, gps_dict=None, VGGT_O_res = 512):
    set_case(case_name, experiment)
    out_dir = case_dir() / "080_Experiments" /f"{experiment}"
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / f"{experiment}_status.csv"
    results_path = out_dir / f"{experiment}_results.csv"

    fps = video_fps(video_path)

    resolve_stale_pending(status_path, cam_poses_list)

    for cam_poses in cam_poses_list:
        experiment_tag = f"{experiment}/poses_{cam_poses:04d}"
        set_case(case_name, experiment_tag)

        recon_dur = end_3D_recon - start_3D_recon
        stride_frames = (recon_dur + cam_poses - 1) // cam_poses  # ceil division
        print("stride" , stride_frames)
        interval_sec = stride_frames / fps

        frames_dir = frames_for_cam_poses_dir()
        if not list(frames_dir.glob("*.jpg")):
            image_sequencer(
                video_path=str(video_path), output_dir=str(frames_dir),
                interval_sec=interval_sec, search_window=stride_frames/2 , start_frame=start_3D_recon, end_frame=end_3D_recon,
            )
        else:
            print(f"cam_poses={cam_poses}: frames already present in {frames_dir}, skipping extraction")
        n_frames = len(available_frames(frames_dir, FRAME_NAME_FMT))

        for technique in technique_order:
            if not flags[technique]:
                continue

            # Recon-output dir for this technique/density -- only ever read inside this
            # technique's own _run() call below (by the matching load_*_trace right after
            # run_*() finishes), so it's safe to clean up the moment run_attempt returns,
            # regardless of success/failure, per data_storage's retention toggles.
            technique_output_dir = None

            if technique == "CUT3R":
                technique_output_dir = cut3r_output_dir()
                def _run(frames_dir=frames_dir, n_frames=n_frames, stride_frames=stride_frames):
                    t0 = time.perf_counter()
                    run_cut3r(
                        frames_dir=frames_dir, output_dir=cut3r_output_dir(),
                        ckpt_path=project_root / "Models" / "CUT3R" / "src" / "cut3r_512_dpt_4_64.pth",
                        render_3rdperson=False,
                    )
                    runtime = (time.perf_counter() - t0) / 60
                    _, cam_pose_dict = load_cut3r_trace_v2(
                        cut3r_output_dir() / "camera", full_pose=data_storage["full_pose"])
                    if data_storage["keep_cam_poses"]:
                        _save_cam_pose_csv(cut3r_output_dir() / "CAM_poses" / "poses.csv",
                                           cam_pose_dict, data_storage["full_pose"])
                    if data_storage["render_bev"]:
                        recon = cut3r_to_reconstruction(frames_dir, cut3r_output_dir())
                        _render_bev_tiles(technique, recon, cam_pose_dict, out_dir, n_frames)
                    if gps_dict:
                        r = align_and_score("CUT3R", _positions_only(cam_pose_dict), runtime, gps_dict)
                        r["n_frames"] = n_frames
                        r["recon_interframe_interval"] = stride_frames
                        return r
                    else:
                        return

            elif technique == "CUT3R_Revisit":
                technique_output_dir = cut3r_output_dir() / "revisit"
                def _run(frames_dir=frames_dir, n_frames=n_frames, stride_frames=stride_frames):
                    t0 = time.perf_counter()
                    run_cut3r(
                        frames_dir=frames_dir, output_dir=cut3r_output_dir() / "revisit",
                        ckpt_path=project_root / "Models" / "CUT3R" / "src" / "cut3r_512_dpt_4_64.pth",
                        render_3rdperson=False, revisit=True,
                    )
                    runtime = (time.perf_counter() - t0) / 60
                    _, cam_pose_dict = load_cut3r_trace_v2(
                        cut3r_output_dir() / "revisit" / "camera", full_pose=data_storage["full_pose"])
                    if data_storage["keep_cam_poses"]:
                        _save_cam_pose_csv(cut3r_output_dir() / "revisit" / "CAM_poses" / "poses.csv",
                                           cam_pose_dict, data_storage["full_pose"])
                    if data_storage["render_bev"]:
                        recon = cut3r_to_reconstruction(frames_dir, cut3r_output_dir() / "revisit")
                        _render_bev_tiles(technique, recon, cam_pose_dict, out_dir, n_frames)
                    if gps_dict:
                        r = align_and_score("CUT3R_Revisit", _positions_only(cam_pose_dict), runtime, gps_dict)
                        r["n_frames"] = n_frames
                        r["recon_interframe_interval"] = stride_frames
                        return r
                    else:
                        return

            elif technique == "lingbot-map":
                technique_output_dir = lingbot_map_dir()
                def _run(frames_dir=frames_dir, n_frames=n_frames, stride_frames=stride_frames):
                    t0 = time.perf_counter()
                    run_lingbot_map()
                    runtime = (time.perf_counter() - t0) / 60
                    _, cam_pose_dict = load_lingbot_map_trace(
                        lingbot_map_dir(), full_pose=data_storage["full_pose"])
                    if data_storage["keep_cam_poses"]:
                        _save_cam_pose_csv(lingbot_map_dir() / "CAM_poses" / "poses.csv",
                                           cam_pose_dict, data_storage["full_pose"])
                    if data_storage["render_bev"]:
                        recon = lingbot_map_to_reconstruction(frames_dir, lingbot_map_dir())
                        _render_bev_tiles(technique, recon, cam_pose_dict, out_dir, n_frames)
                    if gps_dict:
                        r = align_and_score("lingbot-map", _positions_only(cam_pose_dict), runtime, gps_dict)
                        r["n_frames"] = n_frames
                        r["recon_interframe_interval"] = stride_frames
                        return r
                    else:
                        return

            elif technique == "VGGT-Omega":
                technique_output_dir = vggt_o_output_dir()
                def _run(frames_dir=frames_dir, n_frames=n_frames, stride_frames=stride_frames):
                    gc.collect(); torch.cuda.empty_cache()
                    t0 = time.perf_counter()
                    run_vggt_omega(
                        image_dir=str(frames_dir), output_dir=str(vggt_o_output_dir()),
                        checkpoint_path=str(project_root / "Models" / "vggt-omega" / "checkpoints"
                                            / "VGGT-Omega-1B-512" / "vggt_omega_1b_512.pt"),
                        vggt_omega_dir=str(project_root / "Models" / "vggt-omega"),
                        image_resolution=VGGT_O_res, conf_thres=20.0, max_points=0, show_cam=True,
                    )
                    runtime = (time.perf_counter() - t0) / 60
                    recon = Reconstruction.load(frames_dir, predictions_path(), FRAME_NAME_FMT)
                    cam_pose_dict = recon.cam_pos_dict(full_pose=data_storage["full_pose"])
                    if data_storage["keep_cam_poses"]:
                        _save_cam_pose_csv(vggt_o_output_dir() / "CAM_poses" / "poses.csv",
                                           cam_pose_dict, data_storage["full_pose"])
                    if data_storage["render_bev"]:
                        _render_bev_tiles(technique, recon, cam_pose_dict, out_dir, n_frames)
                    if gps_dict:
                        r = align_and_score("VGGT-Omega", _positions_only(cam_pose_dict), runtime, gps_dict)
                        r["n_frames"] = n_frames
                        r["recon_interframe_interval"] = stride_frames
                        return r
                    else:
                        return
            elif technique == "MegaSaM":
                technique_output_dir = case_dir() / "036_MEGASAM_output" / experiment_tag
                def _run(frames_dir=frames_dir, n_frames=n_frames, stride_frames=stride_frames,
                         experiment_tag=experiment_tag):
                    megasam_output_dir = case_dir() / "036_MEGASAM_output" / experiment_tag
                    megasam_output_dir.mkdir(parents=True, exist_ok=True)
                    t0 = time.perf_counter()
                    megasam_npz_path = run_megasam(
                        mega_sam_dir=project_root / "Models" / "mega-sam", scene_name=case_name,
                    )
                    runtime = (time.perf_counter() - t0) / 60
                    # frames_dir passed explicitly -- load_megasam_trace's default
                    # (frames_for_recon_dir()) points at a different, density-unaware folder.
                    _, cam_pose_dict = load_megasam_trace(
                        megasam_npz_path, frames_dir=frames_dir, full_pose=data_storage["full_pose"])
                    if data_storage["keep_cam_poses"]:
                        _save_cam_pose_csv(megasam_output_dir / "CAM_poses" / "poses.csv",
                                           cam_pose_dict, data_storage["full_pose"])
                    if data_storage["render_bev"]:
                        recon = megasam_to_reconstruction(frames_dir, megasam_npz_path)
                        _render_bev_tiles(technique, recon, cam_pose_dict, out_dir, n_frames)
                    if gps_dict:
                        r = align_and_score("MegaSaM", _positions_only(cam_pose_dict), runtime, gps_dict)
                        r["n_frames"] = n_frames
                        r["recon_interframe_interval"] = stride_frames
                        return r
                    return
            elif technique == "VGGT":
                technique_output_dir = case_dir() / "034_VGGT_output" / experiment_tag
                def _run(frames_dir=frames_dir, n_frames=n_frames, stride_frames=stride_frames,
                         experiment_tag=experiment_tag):
                    sys.path.insert(0, str(project_root / "Models" / "VGGT" / "vggt"))
                    from demo_colmap import demo_fn

                    vggt_output_dir = case_dir() / "034_VGGT_output" / experiment_tag
                    vggt_args = argparse.Namespace(
                        scene_dir=str(case_dir()), image_dir=str(frames_dir), output_dir=str(vggt_output_dir),
                        max_frames=500, load_resolution=512, seed=42, use_ba=False, max_reproj_error=8.0,
                        shared_camera=False, camera_type="SIMPLE_PINHOLE", vis_thresh=0.2, query_frame_num=8,
                        max_query_pts=4096, fine_tracking=True, conf_thres_value=5.0,
                    )
                    t0 = time.perf_counter()
                    demo_fn(vggt_args)
                    runtime = (time.perf_counter() - t0) / 60
                    _, cam_pose_dict = load_VGGT_trace(
                        vggt_output_dir, frames_dir=frames_dir, full_pose=data_storage["full_pose"])
                    if data_storage["keep_cam_poses"]:
                        _save_cam_pose_csv(vggt_output_dir / "CAM_poses" / "poses.csv",
                                           cam_pose_dict, data_storage["full_pose"])
                    if data_storage["render_bev"]:
                        recon = vggt_plain_to_reconstruction(frames_dir, vggt_output_dir)
                        _render_bev_tiles(technique, recon, cam_pose_dict, out_dir, n_frames)
                    if gps_dict:
                        r = align_and_score("VGGT", _positions_only(cam_pose_dict), runtime, gps_dict)
                        r["n_frames"] = n_frames
                        r["recon_interframe_interval"] = stride_frames
                        return r
                    else:
                        return
            run_attempt(status_path, results_path, cam_poses_list, technique, cam_poses, _run)
            _cleanup_technique_output(technique_output_dir, data_storage)

        # frames_dir is shared by every technique in technique_order at this density --
        # only safe to delete once the whole inner technique loop above has finished,
        # not between individual techniques.
        if not data_storage["keep_inputs"] and frames_dir.exists():
            shutil.rmtree(frames_dir)


# Process code for the AA_Cam_pose_sweeps.ipynb family of notebooks -- scoring,
# resumability/status tracking, disk-retention cleanup, and the Part 2 sweep loop
# itself. Per-case parameterization (case_name, frame range, CAM_POSES, FLAGS,
# DATA_STORAGE) and Part 3 (load + plot) stay in the notebook -- see
# AA_Cam_pose_sweeps.ipynb.

import argparse
import gc
import shutil
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from A_Config import (
    set_case, case_dir, cut3r_output_dir, vggt_o_output_dir, predictions_path,
    lingbot_map_dir, frames_for_cam_poses_dir, FRAME_NAME_FMT,
)
from Two2D.B_video_processing import image_sequencer, video_fps, available_frames
from run_models.E_cut3r_recon import run_cut3r
from run_models.E_VGGT_omega import run_vggt_omega
from run_models.E_megasam_recon import run_megasam
from run_models.B_lingbot_map import run_lingbot_map
from Thr3D.F_post_recon_processing import (
    Reconstruction, load_cut3r_trace_v2, load_megasam_trace, load_VGGT_trace,
    load_lingbot_map_trace,
)
from Thr3D.F_transpose_to_recon_objects import (
    cut3r_to_reconstruction, megasam_to_reconstruction, lingbot_map_to_reconstruction,
    vggt_plain_to_reconstruction,
)
from P_projection_mapping import BEV_tile_render_PM
from Q_Metric_georeferencing import GPS_camerapose_matcher
from Thr3D.G_transforms_alignments import umeyama_align


#------------------------------------------------------------------
# Scoring -- align_and_score (reused verbatim from AA_Cam_pose_estimate_batches.ipynb)
#------------------------------------------------------------------

def trajectory_length(positions):
    return float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))


def spatial_extent(positions):
    centroid = positions.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((positions - centroid) ** 2, axis=1))))


def align_and_score(technique, cam_pose_dict, runtime_minutes, gps_dict):
    GPS_locs, lat_0, lon_0, object_locs = GPS_camerapose_matcher(gps_dict, cam_pose_dict)
    R, s, t, B_aligned, rmse = umeyama_align(
        GPS_locs, object_locs, label_A="GT", label_B=technique,
        out_path=None,  # no per-iteration PNGs during the sweep -- see plt.close('all') in run_attempt
        
    )
    mean_dist = float(np.mean(np.linalg.norm(GPS_locs - B_aligned, axis=1)))
    traj_len = trajectory_length(GPS_locs)
    return {
        "rmse": rmse,
        "mean dist": mean_dist,
        "correspondences": len(GPS_locs),
        "cam poses (n)": len(cam_pose_dict),  # actual reconstructed pose count -- distinct from
                                               # the sweep loop's "cam_poses" (target density), which
                                               # run_attempt sets separately on this same result dict
        "trajectory length (m)": traj_len,
        "spatial_extent": spatial_extent(GPS_locs),
        "rmse cm/m": 100 * rmse / traj_len if traj_len > 0 else float("nan"),
        "runtime (min)": runtime_minutes,
    }


#------------------------------------------------------------------
# Resumability infrastructure
#
# A GPU OOM kills the whole kernel -- it does not raise a catchable Python
# exception. So a 'pending' status row is written to disk *before* each risky
# call starts, not just a result written after it succeeds. On restart, any row
# still 'pending' means the kernel died mid-attempt (presumed OOM); every larger
# cam_poses for that same technique is then preemptively marked failed/skipped
# rather than re-attempted, since more frames means more GPU memory and the
# same wall will just get hit again.
#------------------------------------------------------------------

STATUS_COLUMNS = ["technique", "cam_poses", "status", "error_type", "error_message", "timestamp"]

# Only these error_types are treated as a hard wall that skips future re-runs --
# an OOM at this density will predictably OOM again, so there's no point retrying.
# Any other failure (a code bug, a data-availability issue like too few GT
# correspondences, etc.) is NOT terminal -- it gets retried every time the
# notebook re-runs, until it either succeeds or the underlying bug is fixed.
_TERMINAL_ERROR_TYPES = {"oom", "presumed_oom_kernel_death", "skipped_precautionary"}


def _load_status(status_path):
    if status_path.exists():
        # keep_default_na=False: without it, an all-blank column (e.g. every
        # "pending"/"success" row's error_type="") round-trips through CSV as
        # NaN and gets inferred as float64 -- then the next _set_status() call
        # that tries to write a real string into that column crashes.
        return pd.read_csv(status_path, keep_default_na=False, dtype=str)
    return pd.DataFrame(columns=STATUS_COLUMNS)


def _set_status(status_path, technique, cam_poses, status, error_type="", error_message=""):
    df = _load_status(status_path)
    mask = (df["technique"] == technique) & (df["cam_poses"] == str(cam_poses))
    row = {
        "technique": technique, "cam_poses": str(cam_poses), "status": status,
        "error_type": error_type, "error_message": error_message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if mask.any():
        for k, v in row.items():
            df.loc[mask, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(status_path, index=False)  # flushed immediately -- this IS the crash-recovery record


def _get_status_row(status_path, technique, cam_poses):
    df = _load_status(status_path)
    mask = (df["technique"] == technique) & (df["cam_poses"] == str(cam_poses))
    if not mask.any():
        return None
    return df.loc[mask].iloc[0]


def _get_status(status_path, technique, cam_poses):
    row = _get_status_row(status_path, technique, cam_poses)
    return None if row is None else row["status"]


def _should_skip(status_path, technique, cam_poses):
    """Success is always skipped. A failure is only skipped if it was terminal
    (OOM-related) -- an ordinary/code-bug failure keeps being retried on every
    re-run, since nothing about re-running would predictably fail the same way
    once the actual bug is fixed."""
    row = _get_status_row(status_path, technique, cam_poses)
    if row is None:
        return False, None
    if row["status"] == "success":
        return True, "success"
    if row["status"] == "failed" and row["error_type"] in _TERMINAL_ERROR_TYPES:
        return True, f"failed ({row['error_type']})"
    return False, None


def _append_result(results_path, row):
    if results_path.exists():
        df = pd.concat([pd.read_csv(results_path), pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(results_path, index=False)  # written immediately after every success, not batched


_OOM_MARKERS = ("out of memory", "cuda out of memory", "cudnn_status_alloc_failed", "cublas_status_alloc_failed")


def classify_error(e):
    if hasattr(torch.cuda, "OutOfMemoryError") and isinstance(e, torch.cuda.OutOfMemoryError):
        return "oom"
    if any(m in str(e).lower() for m in _OOM_MARKERS):
        return "oom"
    return "other"


def resolve_stale_pending(status_path, cam_poses_list):
    """Run once at sweep start: a leftover 'pending' row means the kernel died
    mid-attempt last time (almost certainly a GPU OOM) -- nothing ever ran the
    except block that would have recorded a normal failure. Mark it as a presumed
    OOM and preemptively skip every larger cam_poses for that technique."""
    df = _load_status(status_path)
    pending = df[df["status"] == "pending"]
    for _, r in pending.iterrows():
        technique, cam_poses = r["technique"], int(r["cam_poses"])
        print(f"[{technique} @ {cam_poses}] kernel died mid-attempt last run -- presumed GPU OOM "
              f"(restart the kernel now if you haven't already)")
        _set_status(status_path, technique, cam_poses, "failed", "presumed_oom_kernel_death",
                    "kernel died before status could be resolved")
        for cp in cam_poses_list:
            if cp > cam_poses and _get_status(status_path, technique, cp) is None:
                _set_status(status_path, technique, cp, "failed", "skipped_precautionary",
                            f"skipped: {technique} OOM'd at cam_poses={cam_poses}")
                print(f"[{technique} @ {cp}] precautionary skip (OOM upstream at cam_poses={cam_poses})")


def run_attempt(status_path, results_path, cam_poses_list, technique, cam_poses, run_fn):
    """run_fn() does the actual reconstruction + scoring and returns the
    align_and_score(...) result dict (technique/cam_poses added by this wrapper),
    or raises on a catchable failure."""
    skip, reason = _should_skip(status_path, technique, cam_poses)
    if skip:
        print(f"[{technique} @ {cam_poses}] already {reason}, skipping")
        return

    _set_status(status_path, technique, cam_poses, "pending")
    try:
        result = run_fn()
        result["technique"] = technique
        result["cam_poses"] = cam_poses
        _append_result(results_path, result)
        _set_status(status_path, technique, cam_poses, "success")
        print(f"[{technique} @ {cam_poses}] success, rmse={result['rmse']:.3f}")
    except Exception as e:
        error_type = classify_error(e)
        _set_status(status_path, technique, cam_poses, "failed", error_type, str(e)[:500])
        print(f"[{technique} @ {cam_poses}] FAILED ({error_type}): {e}")
        if error_type == "oom":
            for cp in cam_poses_list:
                if cp > cam_poses and _get_status(status_path, technique, cp) is None:
                    _set_status(status_path, technique, cp, "failed", "skipped_precautionary",
                                f"skipped: {technique} OOM'd at cam_poses={cam_poses}")
                    print(f"[{technique} @ {cp}] precautionary skip (OOM upstream at cam_poses={cam_poses})")
    finally:
        plt.close("all")  # umeyama_align/ortho_charts and the trace loaders never close their figures
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


#------------------------------------------------------------------
# Data retention -- what the sweep keeps on disk vs deletes per density/technique
#------------------------------------------------------------------

def _positions_only(cam_pose_dict):
    """cam_pose_dict values are either (3,) positions or (4,4) c2w pose matrices
    (full_pose=True) -- scoring always needs position-only, regardless of which
    the loader was asked for."""
    sample = next(iter(cam_pose_dict.values()))
    if sample.ndim == 1:
        return cam_pose_dict
    return {k: v[:3, 3] for k, v in cam_pose_dict.items()}


def _render_bev_tiles(technique, recon, cam_pose_dict, out_dir, n_frames, thresholds=(0, 5, 10, 20, 40, 60,80,90)):
    """BEV tiles are a diagnostic render, not a correctness check -- a threshold with
    no points above it (e.g. a genuinely low-confidence recon) means a blank tile,
    not a failed technique, so this must not raise and fail the whole run_attempt."""
    for c in thresholds:
        try:
            ct = np.percentile(recon.preds["depth_conf"].cpu().numpy(), c)
            BEV_tile_render_PM(
                recon, extra_indices=list(cam_pose_dict.keys()), masks_dir=None,
                confidence_threshold=ct, out_path=out_dir / f"{technique}_{n_frames}_BEV_c{c}.png",
            )
        except Exception as e:
            print(f"[{technique} @ {n_frames}] BEV render skipped for conf_thresh={c}: {e}")


def _save_cam_pose_csv(path, cam_pose_dict, full_pose):
    """frame,x,y,z -- or frame,x,y,z,r00..r22 (flattened rotation) when full_pose."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for frame in sorted(cam_pose_dict):
        pose = cam_pose_dict[frame]
        if full_pose:
            rotation, translation = pose[:3, :3], pose[:3, 3]
            rows.append([frame, *translation, *rotation.flatten()])
        else:
            rows.append([frame, *pose])
    columns = ["frame", "x", "y", "z"]
    if full_pose:
        columns += [f"r{i}{j}" for i in range(3) for j in range(3)]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


# Subfolders written inside technique_output_dir that survive a keep_recon_data=False
# cleanup, each gated by its own data_storage toggle. BEV pngs are NOT here -- they're
# written to run_density_sweep's shared out_dir (080_Experiments/{experiment}/), not
# inside technique_output_dir, so they're never touched by this cleanup at all.
_RETAINED_SUBDIRS = {"CAM_poses": "keep_cam_poses"}


def _cleanup_technique_output(technique_output_dir, data_storage):
    """Deletes technique_output_dir's contents per data_storage's keep_recon_data/
    keep_cam_poses toggles. CAM_poses/ (written by _save_cam_pose_csv below) lives
    inside technique_output_dir, so deleting recon data can't be a blind rmtree of
    the whole dir -- everything except that subfolder is removed, then it's removed
    separately too if keep_cam_poses says not to keep it."""
    if technique_output_dir is None or not technique_output_dir.exists():
        return
    if not data_storage["keep_recon_data"]:
        for child in technique_output_dir.iterdir():
            if child.name in _RETAINED_SUBDIRS:
                continue
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    for name, flag in _RETAINED_SUBDIRS.items():
        if not data_storage[flag]:
            subdir = technique_output_dir / name
            if subdir.exists():
                shutil.rmtree(subdir)
    if not any(technique_output_dir.iterdir()):
        technique_output_dir.rmdir()


#------------------------------------------------------------------
# Reset sweep state
#------------------------------------------------------------------

def reset_sweep_state(case_name, experiment, confirm):
    """Deletes only this sweep's own output -- everything from running this sweep,
    nothing more, nothing less: every poses_* frame/reconstruction folder under
    experiment, plus the status/results CSVs and per-technique BEV pngs under
    080_Experiments/{experiment}/ (must match run_density_sweep's own out_dir
    exactly, or this silently looks in the wrong place and finds nothing to delete).

    Safe by default: with confirm=False this only prints what it *would* delete."""
    set_case(case_name, experiment)
    out_dir = case_dir() / "080_Experiments" / f"{experiment}"

    # Every stage folder the sweep writes into, scoped to this sweep's own
    # experiment tag -- deleting the whole subtree here is equivalent to deleting
    # every poses_XXXX folder under it, without having to glob for them individually.
    # out_dir itself covers status/results CSVs + BEV pngs, all written there now.
    sweep_dirs = [
        case_dir() / "020_frames_for_cam_poses" / experiment,
        case_dir() / "034_VGGT_output" / experiment,
        case_dir() / "035_CUT3R_output" / experiment,
        case_dir() / "035_VGGT_O_output" / experiment,
        case_dir() / "036_lingbot_map_output" / experiment,
        case_dir() / "036_MEGASAM_output" / experiment,
        out_dir,
    ]

    targets_dirs = [d for d in sweep_dirs if d.exists()]

    if not confirm:
        print("confirm=False -- dry run only, nothing deleted. Would remove:")
        for d in targets_dirs:
            print(f"  rmtree {d}")
        if not targets_dirs:
            print("  (nothing to delete)")
    else:
        for d in targets_dirs:
            shutil.rmtree(d)
            print(f"deleted {d}")
        print("Sweep state reset.")


#------------------------------------------------------------------
# Part 2 -- Sweep (expensive, resumable, run once/rarely)
#
# Each cam_poses value gets its own frames/output folders via a nested
# experiment tag (f"{experiment}/poses_{cam_poses:04d}" passed to set_case) --
# this makes every A_Config getter (including run_lingbot_map's, which has no
# path parameters at all and only reads the zero-arg getters) automatically
# density-scoped, with no per-function path-wrangling needed.
#
# Safe to interrupt/kill and re-run from the top at any point.
#------------------------------------------------------------------

def run_distance_sweep(case_name, experiment, video_path, start_3D_recon,
                       end_3D_recon_list,interframe_interval, technique_order, flags, data_storage, project_root, gps_dict=None, VGGT_O_res = 512):
    set_case(case_name, experiment)
    out_dir = case_dir() / "080_Experiments" /f"{experiment}"
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / f"{experiment}_status.csv"
    results_path = out_dir / f"{experiment}_results.csv"

    fps = video_fps(video_path)

    resolve_stale_pending(status_path, end_3D_recon_list)

    for end_3D_recon in end_3D_recon_list:
        experiment_tag = f"{experiment}/poses_{end_3D_recon:04d}"
        set_case(case_name, experiment_tag)

        recon_dur = end_3D_recon - start_3D_recon
        stride_frames = interframe_interval
        interval_sec = stride_frames / fps

        frames_dir = frames_for_cam_poses_dir()
        if not list(frames_dir.glob("*.jpg")):
            image_sequencer(
                video_path=str(video_path), output_dir=str(frames_dir),
                interval_sec=interval_sec, search_window=stride_frames/2 , start_frame=start_3D_recon, end_frame=end_3D_recon,
            )
        else:
            print(f"end frame ={end_3D_recon}: frames already present in {frames_dir}, skipping extraction")
        n_frames = len(available_frames(frames_dir, FRAME_NAME_FMT))

        for technique in technique_order:
            if not flags[technique]:
                continue

            # Recon-output dir for this technique/density -- only ever read inside this
            # technique's own _run() call below (by the matching load_*_trace right after
            # run_*() finishes), so it's safe to clean up the moment run_attempt returns,
            # regardless of success/failure, per data_storage's retention toggles.
            technique_output_dir = None

            if technique == "CUT3R":
                technique_output_dir = cut3r_output_dir()
                def _run(frames_dir=frames_dir, n_frames=n_frames, stride_frames=stride_frames):
                    t0 = time.perf_counter()
                    run_cut3r(
                        frames_dir=frames_dir, output_dir=cut3r_output_dir(),
                        ckpt_path=project_root / "Models" / "CUT3R" / "src" / "cut3r_512_dpt_4_64.pth",
                        render_3rdperson=False,
                    )
                    runtime = (time.perf_counter() - t0) / 60
                    _, cam_pose_dict = load_cut3r_trace_v2(
                        cut3r_output_dir() / "camera", full_pose=data_storage["full_pose"])
                    if data_storage["keep_cam_poses"]:
                        _save_cam_pose_csv(cut3r_output_dir() / "CAM_poses" / "poses.csv",
                                           cam_pose_dict, data_storage["full_pose"])
                    if data_storage["render_bev"]:
                        recon = cut3r_to_reconstruction(frames_dir, cut3r_output_dir())
                        _render_bev_tiles(technique, recon, cam_pose_dict, out_dir, n_frames)
                    if gps_dict:
                        r = align_and_score("CUT3R", _positions_only(cam_pose_dict), runtime, gps_dict)
                        r["n_frames"] = n_frames
                        r["recon_interframe_interval"] = stride_frames
                        return r
                    else:
                        return

            elif technique == "CUT3R_Revisit":
                technique_output_dir = cut3r_output_dir() / "revisit"
                def _run(frames_dir=frames_dir, n_frames=n_frames, stride_frames=stride_frames):
                    t0 = time.perf_counter()
                    run_cut3r(
                        frames_dir=frames_dir, output_dir=cut3r_output_dir() / "revisit",
                        ckpt_path=project_root / "Models" / "CUT3R" / "src" / "cut3r_512_dpt_4_64.pth",
                        render_3rdperson=False, revisit=True,
                    )
                    runtime = (time.perf_counter() - t0) / 60
                    _, cam_pose_dict = load_cut3r_trace_v2(
                        cut3r_output_dir() / "revisit" / "camera", full_pose=data_storage["full_pose"])
                    if data_storage["keep_cam_poses"]:
                        _save_cam_pose_csv(cut3r_output_dir() / "revisit" / "CAM_poses" / "poses.csv",
                                           cam_pose_dict, data_storage["full_pose"])
                    if data_storage["render_bev"]:
                        recon = cut3r_to_reconstruction(frames_dir, cut3r_output_dir() / "revisit")
                        _render_bev_tiles(technique, recon, cam_pose_dict, out_dir, n_frames)
                    if gps_dict:
                        r = align_and_score("CUT3R_Revisit", _positions_only(cam_pose_dict), runtime, gps_dict)
                        r["n_frames"] = n_frames
                        r["recon_interframe_interval"] = stride_frames
                        return r
                    else:
                        return

            elif technique == "lingbot-map":
                technique_output_dir = lingbot_map_dir()
                def _run(frames_dir=frames_dir, n_frames=n_frames, stride_frames=stride_frames):
                    t0 = time.perf_counter()
                    run_lingbot_map()
                    runtime = (time.perf_counter() - t0) / 60
                    _, cam_pose_dict = load_lingbot_map_trace(
                        lingbot_map_dir(), full_pose=data_storage["full_pose"])
                    if data_storage["keep_cam_poses"]:
                        _save_cam_pose_csv(lingbot_map_dir() / "CAM_poses" / "poses.csv",
                                           cam_pose_dict, data_storage["full_pose"])
                    if data_storage["render_bev"]:
                        recon = lingbot_map_to_reconstruction(frames_dir, lingbot_map_dir())
                        _render_bev_tiles(technique, recon, cam_pose_dict, out_dir, n_frames)
                    if gps_dict:
                        r = align_and_score("lingbot-map", _positions_only(cam_pose_dict), runtime, gps_dict)
                        r["n_frames"] = n_frames
                        r["recon_interframe_interval"] = stride_frames
                        return r
                    else:
                        return

            elif technique == "VGGT-Omega":
                technique_output_dir = vggt_o_output_dir()
                def _run(frames_dir=frames_dir, n_frames=n_frames, stride_frames=stride_frames):
                    gc.collect(); torch.cuda.empty_cache()
                    t0 = time.perf_counter()
                    run_vggt_omega(
                        image_dir=str(frames_dir), output_dir=str(vggt_o_output_dir()),
                        checkpoint_path=str(project_root / "Models" / "vggt-omega" / "checkpoints"
                                            / "VGGT-Omega-1B-512" / "vggt_omega_1b_512.pt"),
                        vggt_omega_dir=str(project_root / "Models" / "vggt-omega"),
                        image_resolution=VGGT_O_res, conf_thres=20.0, max_points=0, show_cam=True,
                    )
                    runtime = (time.perf_counter() - t0) / 60
                    recon = Reconstruction.load(frames_dir, predictions_path(), FRAME_NAME_FMT)
                    cam_pose_dict = recon.cam_pos_dict(full_pose=data_storage["full_pose"])
                    if data_storage["keep_cam_poses"]:
                        _save_cam_pose_csv(vggt_o_output_dir() / "CAM_poses" / "poses.csv",
                                           cam_pose_dict, data_storage["full_pose"])
                    if data_storage["render_bev"]:
                        _render_bev_tiles(technique, recon, cam_pose_dict, out_dir, n_frames)
                    if gps_dict:
                        r = align_and_score("VGGT-Omega", _positions_only(cam_pose_dict), runtime, gps_dict)
                        r["n_frames"] = n_frames
                        r["recon_interframe_interval"] = stride_frames
                        return r
                    else:
                        return
            elif technique == "MegaSaM":
                technique_output_dir = case_dir() / "036_MEGASAM_output" / experiment_tag
                def _run(frames_dir=frames_dir, n_frames=n_frames, stride_frames=stride_frames,
                         experiment_tag=experiment_tag):
                    megasam_output_dir = case_dir() / "036_MEGASAM_output" / experiment_tag
                    megasam_output_dir.mkdir(parents=True, exist_ok=True)
                    t0 = time.perf_counter()
                    megasam_npz_path = run_megasam(
                        mega_sam_dir=project_root / "Models" / "mega-sam", scene_name=case_name,
                    )
                    runtime = (time.perf_counter() - t0) / 60
                    # frames_dir passed explicitly -- load_megasam_trace's default
                    # (frames_for_recon_dir()) points at a different, density-unaware folder.
                    _, cam_pose_dict = load_megasam_trace(
                        megasam_npz_path, frames_dir=frames_dir, full_pose=data_storage["full_pose"])
                    if data_storage["keep_cam_poses"]:
                        _save_cam_pose_csv(megasam_output_dir / "CAM_poses" / "poses.csv",
                                           cam_pose_dict, data_storage["full_pose"])
                    if data_storage["render_bev"]:
                        recon = megasam_to_reconstruction(frames_dir, megasam_npz_path)
                        _render_bev_tiles(technique, recon, cam_pose_dict, out_dir, n_frames)
                    if gps_dict:
                        r = align_and_score("MegaSaM", _positions_only(cam_pose_dict), runtime, gps_dict)
                        r["n_frames"] = n_frames
                        r["recon_interframe_interval"] = stride_frames
                        return r
                    return
            elif technique == "VGGT":
                technique_output_dir = case_dir() / "034_VGGT_output" / experiment_tag
                def _run(frames_dir=frames_dir, n_frames=n_frames, stride_frames=stride_frames,
                         experiment_tag=experiment_tag):
                    sys.path.insert(0, str(project_root / "Models" / "VGGT" / "vggt"))
                    from demo_colmap import demo_fn

                    vggt_output_dir = case_dir() / "034_VGGT_output" / experiment_tag
                    vggt_args = argparse.Namespace(
                        scene_dir=str(case_dir()), image_dir=str(frames_dir), output_dir=str(vggt_output_dir),
                        max_frames=500, load_resolution=512, seed=42, use_ba=False, max_reproj_error=8.0,
                        shared_camera=False, camera_type="SIMPLE_PINHOLE", vis_thresh=0.2, query_frame_num=8,
                        max_query_pts=4096, fine_tracking=True, conf_thres_value=5.0,
                    )
                    t0 = time.perf_counter()
                    demo_fn(vggt_args)
                    runtime = (time.perf_counter() - t0) / 60
                    _, cam_pose_dict = load_VGGT_trace(
                        vggt_output_dir, frames_dir=frames_dir, full_pose=data_storage["full_pose"])
                    if data_storage["keep_cam_poses"]:
                        _save_cam_pose_csv(vggt_output_dir / "CAM_poses" / "poses.csv",
                                           cam_pose_dict, data_storage["full_pose"])
                    if data_storage["render_bev"]:
                        recon = vggt_plain_to_reconstruction(frames_dir, vggt_output_dir)
                        _render_bev_tiles(technique, recon, cam_pose_dict, out_dir, n_frames)
                    if gps_dict:
                        r = align_and_score("VGGT", _positions_only(cam_pose_dict), runtime, gps_dict)
                        r["n_frames"] = n_frames
                        r["recon_interframe_interval"] = stride_frames
                        return r
                    else:
                        return
            run_attempt(status_path, results_path, end_3D_recon_list, technique, end_3D_recon, _run)
            _cleanup_technique_output(technique_output_dir, data_storage)

        # frames_dir is shared by every technique in technique_order at this density --
        # only safe to delete once the whole inner technique loop above has finished,
        # not between individual techniques.
        if not data_storage["keep_inputs"] and frames_dir.exists():
            shutil.rmtree(frames_dir)
