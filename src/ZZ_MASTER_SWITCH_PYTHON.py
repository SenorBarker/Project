"""502_EGO_BIKE pipeline -- one def per notebook cell.

Every cell of 502.ipynb is a zero-arg function here. They share one module-level
state dict, S, instead of notebook globals: each step reads what it needs out of S
and writes what it produced back into S. That means a kernel restart is not a
re-run of the notebook top to bottom -- import this module and call restore(),
which re-derives everything that is cheap to re-derive (config, paths, flags,
paper edit, recon jobs, loaded recons, GPS transform) and leaves the expensive
work (Gemini, ASR, SAM3, VGGT solves, the Producer agent) alone on disk.

    from pipe_502 import *
    restore()          # after a crash: back to where you were, no GPU/API work
    bev_hires()        # re-run just the cell that died

Anything not in S yet that CAN be cheaply resolved is resolved on demand
(see _PROVIDERS), so calling a late step directly still works. Anything that
cannot -- an API call, a GPU solve -- raises telling you which step to run.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# path bootstrap -- from this file, not from cwd, so the module imports the same
# whether it's pulled in from the notebook, a script, or a fresh interpreter.
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve()
while not (project_root / "src" / "A_Config.py").exists():
    if project_root == project_root.parent:
        raise RuntimeError("couldn't find repo root (src/A_Config.py) above this file")
    project_root = project_root.parent
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))  # A_Config adds the rest on import

import gc
import json
import os
import time
from pprint import pprint

import numpy as np


FRAME_NAME_FMT = "{:04d}.jpg"

# ---------------------------------------------------------------------------
# shared state
# ---------------------------------------------------------------------------
S = {}          # everything the cells used to leave lying around as globals
_timers = {}


def state():
    """What's currently in S -- the equivalent of `%whos` for this pipeline."""
    for k, v in S.items():
        shown = v if isinstance(v, (str, int, float, bool, Path, type(None))) else type(v).__name__
        if k == "recon_jobs":
            shown = f"{len(v)} job(s): " + ", ".join(f"{j['beat_key']}/{j['kind']}" for j in v)
        print(f"{k:<28} {shown}")
    return S


def _need(key, hint):
    """Fetch key from S, resolving it on demand when it's cheap to."""
    if key in S:
        return S[key]
    provider = _PROVIDERS.get(key)
    if provider is not None:
        provider()
        if key in S:
            return S[key]
    raise RuntimeError(f"'{key}' is not in S -- {hint}")


def timer_start(name="default"):
    _timers[name] = time.perf_counter()


def timer_end(name="default"):
    dt = time.perf_counter() - _timers.pop(name)
    print(f"[{name}] {dt:.2f}s")
    return dt


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
def config(case_name, experiment, report=True):
    """Cell 2 -- case, paths, video, fps. Cheap and idempotent; everything else
    resolves this first, so it never has to be called by hand."""
    from dotenv import load_dotenv
    from A_Config import (
        set_case, case_dir, source_dir, query_dir, sam3_masks_dir,
        frames_for_cam_poses_dir, frames_for_recon_dir, vggt_o_output_dir,
        ply_dir, reg_dir, csv_path_gps, csv_path_pose,
    )
    import cv2
    from Two2D.B_video_processing import video_fps

    load_dotenv(project_root / ".env")

    # S is module state and outlives the case that filled it. Pointing config() at
    # a different case without this leaves the previous one's paper edit, a2d,
    # recon jobs and masks in S, and every step below reads them back as if they
    # belonged to the new case -- silently, since the paths in S all still exist.
    if S.get("case_name") not in (None, case_name) or S.get("experiment") not in (None, experiment):
        print(f"case change {S.get('case_name')}/{S.get('experiment')} -> "
              f"{case_name}/{experiment}: clearing S")
        S.clear()

    set_case(case_name, experiment)

    videos = list(source_dir().glob("*.mp4"))
    assert len(videos) == 1, f"Expected 1 video, found {len(videos)}: {videos}"
    video_path = videos[0]

    S.update(
        case_name=case_name,
        experiment=experiment,
        project_root=project_root,
        case_dir=case_dir(),
        source_dir=source_dir(),
        query_dir=query_dir(),
        masks_dir=sam3_masks_dir(),
        # PARENT stage folders -- each recon's own frame set lives in a beat-keyed
        # subfolder, handed out by build_recon_requests as job["frames_dir"].
        frames_for_cam_poses=frames_for_cam_poses_dir(),
        frames_for_recon=frames_for_recon_dir(),
        VGGT_O_output_dir=vggt_o_output_dir(),
        ply_dir=ply_dir(),
        reg_dir=reg_dir(),
        csv_path_GPS=csv_path_gps(),
        csv_path_pose=csv_path_pose(),
        video_path=video_path,
        fps=video_fps(video_path),
        total_frames=int(cv2.VideoCapture(str(video_path)).get(cv2.CAP_PROP_FRAME_COUNT)),
        api_key=os.getenv("STREETVIEW_API_KEY"),
        FRAME_NAME_FMT=FRAME_NAME_FMT,
    )

    print(f"project_root: {project_root}")
    print(f"Case    : {case_name}")
    print(f"Video   : {video_path.name}")
    print(f"video fps: {S['fps']:.3f}")
    print(f"total_frames {S['total_frames']}")

    if report:
        from C_CSV_report import add_to_report
        add_to_report({"Video_name": video_path.name,
                       "Title": f"{case_name}_{experiment}"})
    return S


def set_flags(**overrides):
    """Cell 3 -- operator flags + the Producer's menu subset."""
    flags = {
        "VID_TO_TEXT": True,
        "TRACKING": False,
        "FRAME_SELECTION": True,
        "RECON_CAM_POSES": False,
        "RECON_3D": False,
        "GOOGLE_MAP": False,
        "BEV_MAP": True,
        "PROJECTION": False,
        "MAKE_MOVIE": False,
        "MAKE_MOVIE_CUTS": False,
        "MASK_OVERLAYS": False,
        "PHYSICS_OVERLAYS": False,
        "MULTICAM": False,
    }
    flags.update(overrides)
    S["flags"] = flags
    return producer_flags()


MENU_KEYS = {"TRACKING", "RECON_CAM_POSES", "RECON_3D",
             "GOOGLE_MAP", "BEV_MAP", "PROJECTION", "MASK_OVERLAYS", "MULTICAM"}


def producer_flags():
    """Cells 3 + 69 -- the flags the Producer is allowed to set."""
    flags = _need("flags", "call set_flags()")
    S["producer_flags"] = {k: v for k, v in flags.items() if k in MENU_KEYS}
    print(flags)
    return S["producer_flags"]


def passer(PASS=1):
    """Cell 10 -- bump each time you hand the Producer feedback. Every paper-edit /
    draft written after this gets "-{PASS}" on its name, so passes accumulate on
    disk instead of overwriting."""
    from A_Config import set_pass
    config_if_needed()
    set_pass(PASS)
    S["PASS"] = PASS
    print(PASS)
    return PASS


def question(Question="Make a video summary of this video"):
    """Cell 11."""
    S["Question"] = Question
    print(Question)
    return Question


def config_if_needed():
    """config() takes the case and experiment, so there is nothing to fall back on
    here -- re-seed S from whatever set_case() is already holding (the case survives
    an S wipe, e.g. an autoreload), and say which call is missing if it holds nothing."""
    if "video_path" in S:
        return S
    from A_Config import case_name as active_case, experiment as active_experiment
    if active_case() is None:
        raise RuntimeError(
            "no case set -- call config(case_name, experiment) first "
            "(or tracking_recovery(case_name, experiment) / restore(...))"
        )
    return config(active_case(), active_experiment(), report=False)


# ---------------------------------------------------------------------------
# AI FIRST PASS
# ---------------------------------------------------------------------------
def gps():
    """Cell 5 -- 2D analysis, just GPS now. Seeds analysis_2d_for_decisions."""
    from D_2d_analysis import analysis_2D
    config_if_needed()
    S["analysis_2d_for_decisions"] = analysis_2D(S["video_path"], S["api_key"])
    print(S["analysis_2d_for_decisions"])
    return S["analysis_2d_for_decisions"]


def vid_to_text():
    """Cell 6 -- turn a video into a text description (Gemini)."""
    from A_Config import query_dir
    from Two2D.A_gemini_v04 import gemini_vid_to_text
    config_if_needed()
    if not _need("flags", "call set_flags()")["VID_TO_TEXT"]:
        print("Cell disabled")
        return None
    S["gem_summaries"] = gemini_vid_to_text(S["video_path"], S["case_name"], query_dir())
    return S["gem_summaries"]


def people():
    """Cell 7 -- Gemini people pass (was gated on `repeat`)."""
    from Two2D.A_gemini_v04 import gemini_people
    config_if_needed()
    S["people"] = gemini_people()
    print(json.dumps(S["people"], indent=2))
    return S["people"]


def transcript():
    """Cell 8 -- whisper words + pyannote speakers + audio events."""
    from run_models.A_ASR_diarize import transcribe_and_diarize
    config_if_needed()
    S["asr_transcript"] = transcribe_and_diarize()
    print(S["asr_transcript"].read_text()[:2000])
    return S["asr_transcript"]


# ---------------------------------------------------------------------------
# PRODUCER 1
# ---------------------------------------------------------------------------
def producer_draft1():
    """Cell 12 -- first paper edit."""
    from claude_agent.producer_agent_execution import (
        run_producer_agent, build_producer_prompt_draft1)
    prompt = build_producer_prompt_draft1(
        _need("Question", "call question()"),
        _need("analysis_2d_for_decisions", "call gps()"),
        _need("producer_flags", "call set_flags()"),
    )
    S["paper_edit_json_path"] = run_producer_agent(prompt, mode="draft")
    print(S["paper_edit_json_path"])
    return S["paper_edit_json_path"]


def producer_draft2(feedback):
    """Cells 15-17 -- hand the Producer feedback and get a revised draft.
    Those cells were three commented-out copies of the same call with different
    text; pass the text in instead.

        producer_draft2("The 3d Recon is too fast make it twice as long")
    """
    from claude_agent.producer_agent_execution import (
        run_producer_agent, build_producer_prompt_draft2)
    prompt = build_producer_prompt_draft2(
        feedback,
        _need("analysis_2d_for_decisions", "call gps()"),
        _need("producer_flags", "call set_flags()"),
    )
    S["paper_edit_json_path"] = run_producer_agent(prompt, mode="draft2")
    print(S["paper_edit_json_path"])
    return S["paper_edit_json_path"]


def paper_edit(mode=None):
    """The paper edit path, from S if the kernel still has it, off disk otherwise.
    This is what makes a restart cheap -- nothing below needs the Producer re-run."""
    from render_paper_edit import paper_edit_path
    config_if_needed()
    if mode is not None:
        return paper_edit_path(mode=mode)
    if "paper_edit_json_path" not in S:
        S["paper_edit_json_path"] = paper_edit_path()
    print(S["paper_edit_json_path"])
    return S["paper_edit_json_path"]


def load_producer_flags(mode=None):
    """The Producer's derived flags, read back off disk (<stem>_flags.json, written
    by draft_tidy_up). This is what makes build_jobs() work after a restart --
    without it flags are the operator defaults, RECON_CAM_POSES is False, and you
    get 0 recon jobs. No agent run, no render."""
    p = Path(paper_edit(mode=mode))
    flags_file = p.with_name(p.stem + "_flags.json")
    if not flags_file.exists():
        print(f"no {flags_file.name} yet -- run draft_tidy_up() once")
        return None
    producer_output_flags = json.loads(flags_file.read_text(encoding="utf-8"))
    _need("flags", "call set_flags()").update(producer_output_flags)
    S["producer_output_flags"] = producer_output_flags
    print(f"=== Derived flags (from {flags_file.name}) ===")
    print(json.dumps(producer_output_flags, indent=2))
    return producer_output_flags


def draft_tidy_up():
    """Cell 19 -- regenerate the HTML/flags after hand-editing the paper edit.
    Safe to re-run on its own to correct mistakes."""
    import importlib
    import render_paper_edit
    importlib.reload(render_paper_edit)
    from render_paper_edit import render_and_save, derive_and_save_flags

    p = paper_edit()
    html_path = render_and_save(p)
    print(f"\n=== Rendered HTML ===\n  {html_path}")

    flags_path = derive_and_save_flags(
        p, _need("producer_flags", "call set_flags()").keys(),
        fps=_need("fps", "call config()"),
        gps_signal=_need("analysis_2d_for_decisions", "call gps()").get("GPS_signal"),
    )
    producer_output_flags = json.loads(flags_path.read_text(encoding="utf-8"))
    S["flags"].update(producer_output_flags)  # Producer owns MENU_KEYS; rest stay operator-set
    S["producer_output_flags"] = producer_output_flags
    print(f"\n=== Derived flags (from {flags_path.name}) ===")
    print(json.dumps(producer_output_flags, indent=2))
    return producer_output_flags


# ---------------------------------------------------------------------------
# TRACKING
# ---------------------------------------------------------------------------
def tracking_sam3():
    """Cell 21 -- SAM3 tracking pipeline."""
    from run_models.A_ROBOFLOW_SAM3 import track_subject_sam3
    if not _need("flags", "call set_flags()")["TRACKING"]:
        print("Cell disabled")
        return None
    p = paper_edit()
    print("using this paper edit", p)
    a2d = _need("analysis_2d_for_decisions", "call gps()")
    a2d.update(track_subject_sam3(S["video_path"], p))
    S["tracker"] = "SAM3"
    pprint(a2d)
    return a2d


def tracking_sam3_manual(prompt="person"):
    """Cell 24 -- SAM3 manual controls."""
    from run_models.A_ROBOFLOW_SAM3 import run_sam3_manual
    config_if_needed()
    a2d = _need("analysis_2d_for_decisions", "call gps()")
    a2d.update(run_sam3_manual(S["video_path"], prompt))
    S["tracker"] = "SAM3"
    print(a2d)
    return a2d


def tracking_recovery(case_name=None, experiment=None, pass_num=None, gps_step=True):
    """Cell 26 -- rebuild the state a finished tracking run would have left, from
    the mask folders on disk instead of re-running SAM3. Bare, it reuses the
    configured case and pass. Not restored: reid_match(), a Claude call never cached."""
    from A_Config import pass_num as active_pass
    from D_2d_analysis import analysis_2d_tracking_recovery
    # Resolved before config() gets a chance to clear S on a case change. Not
    # defaulted to 1: PASS picks the paper edit, so forcing it recovers pass 1's
    # tracking against pass 3's edit without saying so.
    if pass_num is None:
        pass_num = S.get("PASS") or active_pass()
    if case_name is None:
        config_if_needed()
        case_name, experiment = S["case_name"], S["experiment"]
    print(f"recovering {case_name}/{experiment} PASS {pass_num}")
    restore_pre_track(case_name, experiment, pass_num, gps_step=False)
    if gps_step and "analysis_2d_for_decisions" not in S:
        gps()      # re-reads the whole video, so keep S's copy if it has one
    S.setdefault("analysis_2d_for_decisions", {})
    S["analysis_2d_for_decisions"].update(analysis_2d_tracking_recovery(paper_edit()))
    pprint(S["analysis_2d_for_decisions"])
    return S["analysis_2d_for_decisions"]


# ---------------------------------------------------------------------------
# RE-ID
# ---------------------------------------------------------------------------
def reid_match():
    """Cell 29 -- match Gemini's people to SAM tracks via masked crops (Claude API,
    so this is NOT part of restore(); re-run it only if you need person-A style ids)."""
    from A_Config import source_video_path
    from Two2D.B_video_processing import video_dims
    from Two2D.AX_gem_SAM_matcher import gem_person_targets_lookup
    from Two2D.AY_claude_crop_matcher import write_masked_crops, match_tracks_to_descriptors
    from D_2d_analysis import analysis_2d_gemvsSAM

    p = paper_edit()
    _, _, fps = video_dims(source_video_path())
    S["fps"] = fps

    crops = write_masked_crops()                    # eyeball these first
    S["crops"] = crops
    match_results = match_tracks_to_descriptors(p, crops_by_track=crops, fps=fps)
    S["match_results"] = match_results

    a2d = _need("analysis_2d_for_decisions", "call gps()")
    a2d.update(analysis_2d_gemvsSAM(
        a2d,
        gem_person_targets=gem_person_targets_lookup(paper_edit_json_path=p),  # beat_ids only
        match_results=match_results,                                           # crops, not box IoU
    ))
    return a2d


def mask_check_maps():
    """Cell 31 -- mask checker, all in one.
    MAP beats (frames mode): masks exist only on the ~200 recon frames, spread over
    the whole window. video_overlay_edit would cut the SOURCE video first-to-last
    masked frame -- a 4-minute clip that's ~99% unmasked, plus it raises on
    _sam_id_shots. mask_check_map_beats builds the video from the recon frames
    instead, so every frame carries its mask. Zero-arg: resolves its own masks +
    recon frames, no cell order."""
    from Two2D.W_video_editor import mask_check_map_beats
    config_if_needed()
    out = list(mask_check_map_beats())
    for p in out:
        print(p)
    return out


def mask_check_overlays():
    """Cell 32 -- mask checker (overlays) over the source video, one per mask dir."""
    from Two2D.W_video_editor import video_overlay_edit
    config_if_needed()
    print(S["video_path"])
    print("md = ", S["masks_dir"])
    out = []
    for mask in (p for p in S["masks_dir"].iterdir() if p.is_dir()):
        print(mask)
        out.append(video_overlay_edit(S["video_path"], mask, name=f"dancing{mask.name}"))
    return out


# ---------------------------------------------------------------------------
# RECON -- jobs and frame selection
# ---------------------------------------------------------------------------
def build_jobs():
    """Cells 35 + 36 -- one job per (beat, recon kind) the Producer asked for.

    Every beat that asks for a recon gets its own frame range, frames folder and
    recon folder, so several MAP and several EVENT recons live side by side.
    Max one of each kind per beat -- two recons of a kind means two beats.
    Each job dict is added to by the steps below; nothing here binds a single
    global `recon`. Cheap (reads the paper edit), so restore() calls it."""
    from render_paper_edit import build_recon_requests
    p = paper_edit()
    if "producer_output_flags" not in S:
        load_producer_flags()      # otherwise the operator defaults give 0 jobs
    flags = _need("flags", "call set_flags()")
    S["recon_jobs"] = [
        j for j in build_recon_requests(p)
        if flags["RECON_CAM_POSES" if j["kind"] == "MAP" else "RECON_3D"]
    ]
    print(f"{len(S['recon_jobs'])} recon job(s)")
    for job in S["recon_jobs"]:
        print(f"  {job['beat_key']:<10} {job['kind']:<6} "
              f"frames {job['start_frame']}-{job['end_frame']}  -> {job['recon_dir']}")
    return S["recon_jobs"]


def jobs_of(kind=None, archetype=None, flag=None, beat_key=None):
    """The lookup every step below uses instead of a single global recon variable.

    kind      -- which SOLVE this is: "MAP" (RECON_CAM_POSES, the camera track a
                 map is built from) or "EVENT" (RECON_3D, a moment of action).
    archetype -- what the BEAT is for: "MAP" for a beat that renders a map.
    These two are not the same thing -- a non-MAP beat can ask for either solve --
    so map rendering selects on archetype and physics work selects on kind.
    flag      -- something in the beat's own requested_flags, e.g. "BEV_MAP"."""
    return [j for j in _need("recon_jobs", "call build_jobs()")
            if (kind is None or j["kind"] == kind)
            and (archetype is None or j["archetype"] == archetype)
            and (flag is None or flag in (j["beat"].get("requested_flags") or []))
            and (beat_key is None or j["beat_key"] == beat_key)]


def frame_selection(capacity=200, min_sharp=0.0):
    """Cell 37 -- one frame set per recon job. Each beat's own window is thinned to
    `capacity` frames; raise min_sharp after inspecting selection_log.json."""
    from Two2D.B_video_processing import image_sequencer
    if not _need("flags", "call set_flags()")["FRAME_SELECTION"]:
        print("Cell disabled")
        return None
    fps = _need("fps", "call config()")
    window = fps / 2     # search window for sharpness
    out = {}
    for job in _need("recon_jobs", "call build_jobs()"):
        span = job["end_frame"] - job["start_frame"]
        skip = (span + capacity - 1) // capacity    # frames to skip between samples
        interval = skip / fps                       # image_sequencer wants seconds
        print(f"{job['beat_key']} {job['kind']}: {span} frames, interval {skip}")
        results = image_sequencer(
            video_path=str(S["video_path"]),
            output_dir=str(job["frames_dir"]),
            interval_sec=interval,
            search_window=window,
            min_sharpness=min_sharp,
            start_frame=job["start_frame"],
            end_frame=job["end_frame"],
        )
        out[job["beat_key"]] = results
        print(f"  {len(results)} frames written to {job['frames_dir']}")
    return out


def manual_frame_selection(start_frame=0, end_frame=7581, capacity=650, min_sharp=0.0):
    """Cell 39 -- same capacity/interval maths, one hand-picked window, straight
    into frames_for_cam_poses_dir()."""
    from A_Config import frames_for_cam_poses_dir
    from Two2D.B_video_processing import image_sequencer
    config_if_needed()
    fps = S["fps"]
    span = end_frame - start_frame
    skip = (span + capacity - 1) // capacity
    output_dir = str(frames_for_cam_poses_dir())
    results = image_sequencer(
        video_path=str(S["video_path"]),
        output_dir=output_dir,
        interval_sec=skip / fps,
        search_window=fps / 2,
        min_sharpness=min_sharp,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    print(output_dir)
    return results


# ---------------------------------------------------------------------------
# VGGT OMEGA
# ---------------------------------------------------------------------------
def cuda_clear():
    """Cells 41-43 -- free the GPU and report what's left. Run this first after a
    CUDA OOM crash, before re-running the solve."""
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    print("allocated GiB", torch.cuda.memory_allocated() / 2**30,
          "reserved GiB", torch.cuda.memory_reserved() / 2**30)
    print("autocast enabled:", torch.is_autocast_enabled())


def vggt_omega(rerun_existing=False):
    """Cell 44 -- one solve per recon job.

    Several jobs means several full solves, so a re-run skips the ones already on
    disk rather than repeating every pass. rerun_existing=True forces a re-solve."""
    import torch
    from run_models.E_VGGT_omega import run_vggt_omega
    gc.collect()
    torch.cuda.empty_cache()

    checkpoint = (project_root / "Models" / "vggt-omega" / "checkpoints"
                  / "VGGT-Omega-1B-512" / "vggt_omega_1b_512.pt")

    for job in _need("recon_jobs", "call build_jobs()"):
        if not rerun_existing and (job["recon_dir"] / "predictions.npz").exists():
            print(f"skip {job['beat_key']} {job['kind']} -- predictions.npz already there")
            continue
        print(f"solving {job['beat_key']} {job['kind']} from {job['frames_dir']}")
        run_vggt_omega(
            image_dir=str(job["frames_dir"]),
            output_dir=job["recon_dir"],
            checkpoint_path=str(checkpoint),
            vggt_omega_dir=str(project_root / "Models" / "vggt-omega"),
            image_resolution=512,
            conf_thres=20.0,
            max_points=0,
            show_cam=True,
        )
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# POST RECON
# ---------------------------------------------------------------------------
def _PRP(recon, frames):
    """Cell 46 helper -- camera centres in model space, per frame."""
    from Thr3D.F_post_recon_processing import positions_to_frame_dict
    frame_keys = sorted(recon.frame_to_row, key=recon.frame_to_row.get)
    rows = [recon.frame_to_row[f] for f in frame_keys]
    extrinsic = recon.preds["extrinsic"][rows].cpu().numpy()   # (N,3,4) model-to-cam
    R = extrinsic[:, :3, :3]
    t = extrinsic[:, :3, 3]
    cam_poses = -np.einsum("nji,nj->ni", R, t)                 # cam centre, model space
    return positions_to_frame_dict(cam_poses, frames)


def load_recons():
    """Cell 46 -- load each job's OWN predictions.npz from its own recon_dir.
    This is the step that puts the solves back after a restart; no GPU work."""
    from Thr3D.F_post_recon_processing import Reconstruction
    for job in _need("recon_jobs", "call build_jobs()"):
        npz = job["recon_dir"] / "predictions.npz"
        if not npz.exists():
            # not solved yet -- skip rather than kill the whole restore
            print(f"{job['beat_key']} {job['kind']}: no predictions.npz "
                  f"-- run vggt_omega()")
            continue
        job["recon"] = Reconstruction.load(
            job["frames_dir"], job["recon_dir"] / "predictions.npz", FRAME_NAME_FMT)
        job["cam_poses_model"] = _PRP(job["recon"], job["frames_dir"])
        print(f"{job['beat_key']} {job['kind']}: {len(job['cam_poses_model'])} camera poses")
    return S["recon_jobs"]


def _solved(job):
    """Is this job's reconstruction loaded? Steps below skip the ones that aren't
    rather than dying on KeyError('recon') halfway through a multi-job run."""
    if "recon" not in job:
        load_recons()
    if "recon" not in job:
        print(f"{job['beat_key']} {job['kind']}: skipped, no reconstruction")
        return False
    return True


def transform_poses_gps():
    """Cell 47 -- calibrate each recon's camera track against GPS.

    Every recon calibrates on its own: its own R/s/t and its own lat_0/lon_0 origin.
    Everything lands in the same real-world frame, so lat/lons from different beats
    ARE comparable (unlike the raw model-space positions). Source is anything but
    "RS_path" so the poses dict is used directly rather than read as a RealityScan
    csv; the string doubles as the alignment chart's label."""
    from Q_Metric_georeferencing import model_to_GPS_calibrated_locations
    a2d = _need("analysis_2d_for_decisions", "call gps()")
    if a2d["GPS_signal"] != "yes":
        print("Cell disabled")
        return None
    for job in _need("recon_jobs", "call build_jobs()"):
        if "cam_poses_model" not in job:
            load_recons()
        if "cam_poses_model" not in job:
            continue                       # unsolved, load_recons() already said so
        (job["cam_poses_real"], job["cam_latlons"], job["R_cw"], job["s_cw"],
         job["t_cw"], job["lat_0"], job["lon_0"]) = model_to_GPS_calibrated_locations(
            S["csv_path_GPS"], job["cam_poses_model"], Source=f"VGGT_O {job['beat_key']}")
        print(f"{job['beat_key']}: {len(job['cam_latlons'])} lat/lons, "
              f"origin {job['lat_0']}, {job['lon_0']}")
    return S["recon_jobs"]


def export_plys():
    """Cell 49 -- model space -> ply, one folder per recon.

    Each recon's clouds go next to its own predictions.npz (recon_dir/ply) instead
    of a shared VGGT_O_output_dir/MAP|EVENT folder, so two beats can't overwrite
    each other. n.b. this translates to the beginning of the camera track solve,
    then re-translates to that recon's own cam 0 space."""
    for job in _need("recon_jobs", "call build_jobs()"):
        if not _solved(job):
            continue
        ext = job["recon"].preds["extrinsic"][0].clone()
        job["ext"] = ext                 # this recon's cam0 view, reused by BEV/projection
        job["ply_dir"] = job["recon_dir"] / "ply"
        job["recon"].VGGT_O_preds_to_ply_export(job["ply_dir"],
                                                R=ext[:, 0:3], t=ext[:, 3], s=1)
        print(f"{job['beat_key']} {job['kind']} -> {job['ply_dir']}")
    return S["recon_jobs"]


def visualise_plys():
    """Cell 50 -- PLY visualisation, one per recon."""
    import importlib
    import F_cut3r_vis
    importlib.reload(F_cut3r_vis)
    for job in _need("recon_jobs", "call build_jobs()"):
        print(f"{job['beat_key']} {job['kind']}: {job['recon_dir']}")
        F_cut3r_vis.visualise_cut3r(job["recon_dir"])


def depth_confidence_stats():
    """Cell 53 -- depth confidence statistics, per recon.

    add_to_report rows are overwrite-by-key, so each recon's numbers carry its own
    beat key -- otherwise only the last job's would survive in the CSV."""
    from C_CSV_report import add_to_report
    for job in _need("recon_jobs", "call build_jobs()"):
        if not _solved(job):
            continue
        conf = job["recon"].preds["depth_conf"].cpu().numpy()
        job["depth_conf"] = conf
        print(f"--- {job['beat_key']} {job['kind']} ---")
        print("min, max confidence", conf.min(), conf.max())
        print("frames, y, x", conf.shape)
        print("confidence percentiles", np.percentile(conf, [5, 25, 50, 75, 95]))
        tag = f"{job['beat_key']} {job['kind']}"
        add_to_report({
            f"Depth Confidence (min) [{tag}]": conf.min(),
            f"Depth Confidence (max) [{tag}]": conf.max(),
            f"Depth Confidence (mean) [{tag}]": conf.mean(),
            f"Depth Confidence percentiles [{tag}]": np.percentile(conf, [5, 25, 50, 75, 95]),
        })
    return S["recon_jobs"]


# ---------------------------------------------------------------------------
# subject 3d positions
# ---------------------------------------------------------------------------
def subject_positions():
    """Cell 56 -- multi-subject positioning, GPS or not.

    Positions come out in each recon's OWN arbitrary model space (its own scale and
    origin), so they stay on the job. Two jobs' positions are not comparable and
    must not be merged -- only the GPS path below puts them in a shared frame.
    A subject whose masks never land in this recon's frames simply drops out."""
    import cv2
    from Q_subject_analysis import (find_mask_centroids_in_model_space,
                                    subject_centroids_to_metric_and_gps_space)
    cv2.setLogLevel(0)

    a2d = _need("analysis_2d_for_decisions", "call gps() then tracking_recovery()")
    has_gps = a2d["GPS_signal"] == "yes"

    for job in _need("recon_jobs", "call build_jobs()"):
        if not _solved(job):
            continue
        if has_gps:
            R_mw, s_mw, t_mw = job["R_cw"], job["s_cw"], job["t_cw"]

        subjects_positions, subjects_depth_std, subjects_confidence = {}, {}, {}
        subjects_real_pos, subjects_latlon, subjects_confidence_gps = {}, {}, {}

        for subject_slug, entry in a2d.items():
            if not isinstance(entry, dict) or not entry["Subject_frames_present"]:
                continue
            track_dirs = entry.get("masks_dirs") or [entry["masks_dir"]]
            merged_SAMS, merged_std, merged_confidence = {}, {}, {}
            merged_real, merged_confidence_gps = {}, {}
            merged_latlon = []

            for track_dir in track_dirs:
                track_dir = Path(track_dir)
                centroids, frame_keys, depth_std_model, weight_sums_R = \
                    find_mask_centroids_in_model_space(
                        job["recon"], masks_dir=track_dir, RA=8, analyse=False)
                for fk, c, s, w in zip(frame_keys, centroids, depth_std_model, weight_sums_R):
                    if np.isfinite(np.asarray(c)).all():   # subject positions can be NaN
                        merged_SAMS[fk] = c
                        merged_std[fk] = s
                        merged_confidence[fk] = w

                if has_gps:
                    sub_latlon, sub_pos, sub_pos_real_dict, sub_conf_dict = \
                        subject_centroids_to_metric_and_gps_space(
                            centroids, frame_keys, depth_std_model, weight_sums_R,
                            lat_0=job["lat_0"], lon_0=job["lon_0"],
                            R_mw=R_mw, s_mw=s_mw, t_mw=t_mw)
                    for fk, p in sub_pos_real_dict.items():
                        if np.isfinite(np.asarray(p)).all():
                            merged_real[fk] = p
                    merged_confidence_gps.update(sub_conf_dict)
                    merged_latlon += list(sub_latlon)

            if merged_SAMS:
                subjects_positions[subject_slug] = merged_SAMS
                subjects_depth_std[subject_slug] = merged_std
                subjects_confidence[subject_slug] = merged_confidence
            if merged_real:
                subjects_real_pos[subject_slug] = merged_real
                subjects_latlon[subject_slug] = sorted(merged_latlon)
                subjects_confidence_gps[subject_slug] = merged_confidence_gps

        job["subjects_positions"] = subjects_positions
        job["subjects_depth_std"] = subjects_depth_std
        job["subjects_confidence"] = subjects_confidence
        job["subjects_real_pos"] = subjects_real_pos
        job["subjects_latlon"] = subjects_latlon
        job["subjects_confidence_GPS"] = subjects_confidence_gps
        print(f"{job['beat_key']} {job['kind']}: {list(subjects_positions)}")
    return S["recon_jobs"]


def show_jobs():
    """Cells 57, 58, 59, 61, 80 -- the inspection prints, in one place."""
    for job in _need("recon_jobs", "call build_jobs()"):
        print(f"--- {job['beat_key']} {job['kind']} {job.get('flag')} "
              f"{job['start_frame']}-{job['end_frame']} ---")
        pprint(job.get("subjects_real_pos") or job.get("subjects_positions"))


def show_a2d():
    """Cells 23, 30, 54, 72, 78 -- pprint analysis_2d_for_decisions."""
    pprint(_need("analysis_2d_for_decisions", "call gps()"))
    return S["analysis_2d_for_decisions"]


def plot_positions_2d():
    """Cell 60 -- one figure per recon. Axes are that recon's own space, so plotting
    two on one set of axes would be meaningless."""
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    markers = ["o", "^", "s", "D", "v", "P"]

    for job in _need("recon_jobs", "call build_jobs()"):
        subj = job.get("subjects_real_pos") or job.get("subjects_positions") or {}
        if not subj:
            print(f"{job['beat_key']} {job['kind']}: no subject positions")
            continue

        fig, ax = plt.subplots()
        allf = np.concatenate([np.array(sorted(d)) for d in subj.values()])
        handles = []
        # Letter-named (gem-matched, e.g. person-A) subjects plot before numeric raw
        # ones -- markers is short, so anything past len(markers) gets dropped by zip
        # below, and letter-named subjects matter more for the report.
        ordered_items = sorted(subj.items(), key=lambda kv: kv[0].rsplit('-', 1)[-1].isdigit())
        for (slug, d), mk in zip(ordered_items, markers):
            frames = np.array(sorted(d))
            pts = np.array([d[k] for k in frames])
            sc = ax.scatter(pts[:, 0], pts[:, 2], c=frames, cmap="viridis", marker=mk,
                            vmin=allf.min(), vmax=allf.max(), s=18)   # colour = frame number
            handles.append(Line2D([0], [0], marker=mk, ls="", label=slug))

        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("frame", color="white")
        cb.ax.yaxis.set_tick_params(color="white", labelcolor="white")
        ax.legend(handles=handles)
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_title(f"Subject position, {job['beat_key']} {job['kind']} (color = time)")


def metrics_3d():
    """Cell 62 -- speed/direction and meetings, per recon.

    Both are distance maths, so they run inside one recon's space. A requested pair
    is only measurable in a recon that actually saw both; pairs that split across
    two recons are reported as skipped rather than silently wrong."""
    from Q_Metric_georeferencing import speed_direction
    from Q_subject_analysis import meeting_calculator
    from render_paper_edit import build_meeting_requests

    fps = _need("fps", "call config()")
    meeting_pairs = build_meeting_requests(paper_edit())   # [["person-1","person-2"], ...]
    S["meeting_pairs"] = meeting_pairs

    for job in _need("recon_jobs", "call build_jobs()"):
        if not _solved(job):
            continue
        subjects = job.get("subjects_real_pos") or job.get("subjects_positions") or {}
        job["group_metrics"] = speed_direction({"camera": job["cam_poses_model"], **subjects}, fps)

        meetings = {}
        subjects_distance_to_camera = {}
        for A_dict_entry, B_dict_entry in meeting_pairs:
            B_is_camera = B_dict_entry == "camera"
            if A_dict_entry not in subjects or (not B_is_camera and B_dict_entry not in subjects):
                print(f"{job['beat_key']}: skipping {A_dict_entry} vs {B_dict_entry} "
                      f"-- not both present in this recon")
                continue
            A_real_dict = subjects[A_dict_entry]
            B_real_dict = job["cam_poses_model"] if B_is_camera else subjects[B_dict_entry]

            if B_is_camera:
                meeting_report, distance_series = meeting_calculator(
                    A_real_dict, A_dict_entry, B_real_dict, B_dict_entry,
                    B_is_camera=B_is_camera, return_series=True)
                subjects_distance_to_camera[A_dict_entry] = distance_series
            else:
                meeting_report = meeting_calculator(
                    A_real_dict, A_dict_entry, B_real_dict, B_dict_entry,
                    B_is_camera=B_is_camera)
            meetings[(A_dict_entry, B_dict_entry)] = meeting_report
        job["meetings"] = meetings
        job["subjects_distance_to_camera"] = subjects_distance_to_camera
        print(f"--- {job['beat_key']} {job['kind']} ---")
        print(meetings)
    return S["recon_jobs"]


# ---------------------------------------------------------------------------
# TEMP / WIP -- depth intersection + touching test
# ---------------------------------------------------------------------------
def _subject_mask_dirs(slug):
    e = _need("analysis_2d_for_decisions", "call gps()")[slug]
    return [Path(d) for d in (e.get("masks_dirs") or [e["masks_dir"]])]


def depth_intersection_overlays(A_slug="person-1", B_slug="the_large_sword-1", B_dir=None):
    """Cell 65 -- TEMP. Mask overlay sequence: A (green) vs B (magenta).
    B_dir overrides B's mask folder (the old cell hard-coded the sword as person-3
    of another case); leave it None to use B_slug's own dirs."""
    import cv2
    from Two2D.W_video_editor import apply_mask_overlay, _load_mask_bool
    from A_Config import assets_dir, source_video_path

    a2d = _need("analysis_2d_for_decisions", "call gps()")
    A_dir = _subject_mask_dirs(A_slug)[0]
    B_dir = Path(B_dir) if B_dir else _subject_mask_dirs(B_slug)[0]
    S["A_dir"], S["B_dir"] = A_dir, B_dir

    out_dir = assets_dir() / f"contact_check_{A_slug}_vs_{B_slug}"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(set(a2d[A_slug]["Subject_frames_present"]) |
                    set(a2d[B_slug]["Subject_frames_present"]))
    cap = cv2.VideoCapture(str(source_video_path()))
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if not ret:
            continue
        for d, colour in ((A_dir, (0, 200, 0)), (B_dir, (200, 0, 200))):
            m = _load_mask_bool(d, f)
            if m is not None:
                frame = apply_mask_overlay(frame, m, color=colour)
        cv2.imwrite(str(out_dir / f"{f:04d}.png"), frame)
    cap.release()
    print(out_dir, len(frames), "frames")
    return out_dir


def touching_test(job_index=0, contact_frame=386):
    """Cell 67 -- WIP. There's no rule yet for which recon a contact test belongs to,
    so pick the job by hand (see build_jobs() output for what's available)."""
    from matplotlib import pyplot as plt
    from Q_subject_analysis import contact_frames
    job = _need("recon_jobs", "call build_jobs()")[job_index]
    res = contact_frames(job["recon"],
                         _need("A_dir", "call depth_intersection_overlays() first"),
                         S["B_dir"])
    r = res[contact_frame]
    plt.plot(list(r["A_profile"]), list(r["A_profile"].values()), "o-", label="A")
    plt.plot(list(r["B_profile"]), list(r["B_profile"].values()), "o-", label="B")
    plt.xlabel("depthels from seam")
    plt.ylabel("depth")
    plt.legend()
    print(r["depth_gap"], r["fit_resid"], r["contact"])
    return res


# ---------------------------------------------------------------------------
# PRODUCER 2
# ---------------------------------------------------------------------------
def find_cut_points():
    """Cells 71 + 76 -- find the cut points (run again after the revision)."""
    import Two2D.W_video_editor
    config_if_needed()
    return Two2D.W_video_editor.find_all_cut_points(
        S["video_path"],
        _need("analysis_2d_for_decisions", "call gps()"),
        paper_edit(),
    )


def producer_revision():
    """Cell 73 -- run the Producer again over what the pipeline has since learned."""
    from producer_agent_execution import run_producer_agent, build_producer_prompt_revision
    prompt = build_producer_prompt_revision(
        _need("Question", "call question()"),
        _need("analysis_2d_for_decisions", "call gps()"),
        _need("producer_flags", "call set_flags()"),
        paper_edit(),
    )
    S["paper_edit_json_path"] = run_producer_agent(prompt)
    print(S["paper_edit_json_path"])
    return S["paper_edit_json_path"]


def match_beat_ids():
    """Cell 75 -- prevent Producer mistakes: reconcile revision beat ids."""
    from render_paper_edit import match_revision_beat_ids
    config_if_needed()
    return match_revision_beat_ids()


def revised_beats():
    """The revised paper edit's beats, keyed by beat_key.

    The revision pass can edit tracked_subject after build_jobs() ran (that reads the
    draft), so the render steps re-read the revision here rather than trusting
    job["beat"]. Cells 81, 83 and 88 each had their own copy of this."""
    return {b["beat_id"][0]: b
            for b in json.loads(paper_edit(mode="revision").read_text(encoding="utf-8"))["beats"]}


def _wanted_subjects(beat):
    """The subject slugs THIS beat's own (revised) tracked_subject named -- not every
    track the recon happens to have positions for."""
    from run_models.A_ROBOFLOW_SAM3 import _slugify_subject
    return {entry["gem_person_id"] if isinstance(entry, dict) else _slugify_subject(entry)
            for entry in (beat.get("tracked_subject") or [])}


# ---------------------------------------------------------------------------
# RENDERING -- projection
# ---------------------------------------------------------------------------
def projection(margin=0.3, extra_distance_margin=0.0, conf_percentile=30, splat_radius=3):
    """Cell 81 -- projection mapping, one sequence per EVENT recon.

    Knobs are in raw VGGT-O model units, no metric/GPS needed:
      margin                -- fraction of the full-mask-point box, on every side
      extra_distance_margin -- extra pull-back so off-center corners don't clip
      conf_percentile       -- drop masked pixels below this depth-confidence
                               percentile (guards against a misprojected frame
                               blowing the box out)
      splat_radius          -- splat radius
    """
    from Q_subject_analysis import (find_mask_centroids_in_model_space,
                                    auto_camera_position, look_at_rotation,
                                    subject_oriented_box)
    from P_projection_mapping import (solve_auto_camera_extrinsic,
                                      projection_mapping_sequence, mean_camera_extrinsic)
    from Two2D.B_video_processing import available_frames

    beats_by_key = revised_beats()
    projection_beats = [b for b in beats_by_key.values() if b["archetype"] == "PROJECTION"]
    if not projection_beats:
        print("no PROJECTION beats in this revision")
        return None

    a2d = _need("analysis_2d_for_decisions", "call gps()")
    fps = _need("fps", "call config()")
    out = []

    for job in jobs_of(kind="EVENT"):
        if not _solved(job):
            continue
        all_positions = job.get("subjects_real_pos") or job.get("subjects_positions") or {}
        beat = beats_by_key.get(job["beat_key"], job["beat"])
        wanted = _wanted_subjects(beat)
        print("Subjects to set camera", wanted)

        beat_name = job["beat_ids"][0]   # same as W_video_editor's beat["beat_id"][0]

        masks_dirs = list(dict.fromkeys(
            d
            for key, entry in (a2d or {}).items()
            if isinstance(entry, dict)
            and job["beat_key"] in (entry.get("beat_ids") or [])
            and (key in wanted or key.rsplit("-", 1)[0] in wanted)
            for d in ([entry["masks_dir"]] if entry.get("masks_dir")
                      else entry.get("masks_dirs") or [])
        ))
        print(masks_dirs)

        extrinsics_np = job["recon"].preds["extrinsic"].cpu().numpy()
        conf_vggt = job["recon"].preds["depth_conf"].cpu().numpy()
        ct_vggt = np.percentile(conf_vggt, conf_percentile)

        # Tracking can come back with nothing: no masks_dir for the wanted subject at
        # all, or dirs whose masks are empty on every frame. Collect whatever we do
        # get, per mask dir, and decide afterwards whether there's enough to aim with.
        mask_points_all, weights_all = [], []
        for masks_dir in masks_dirs:
            print(masks_dir)
            _, _, _, _, weights, mask_points = find_mask_centroids_in_model_space(
                job["recon"], masks_dir, autocam=True, conf_thresh=ct_vggt)
            if mask_points is not None and len(mask_points):
                mask_points_all.append(np.asarray(mask_points))
            weights_all.append(np.asarray(weights, dtype=float))
        # one point cloud and one per-frame weight vector across all subjects
        mask_points_total = (np.concatenate(mask_points_all, axis=0)
                             if mask_points_all else np.empty((0, 3)))
        weights_total = (np.max(np.stack(weights_all), axis=0)
                         if weights_all else np.zeros(len(extrinsics_np)))
        # enough to aim at a subject? need real (non-NaN) points AND at least one frame
        # the subject was actually visible in -- both are what the look-at solve needs.
        tracked_ok = ((~np.isnan(mask_points_total).any(axis=1)).sum() >= 2
                      and np.any(weights_total > 0))

        job["extra_indices"] = available_frames(job["frames_dir"], FRAME_NAME_FMT)
        extra_indices = job["extra_indices"]
        print(f"{job['beat_key']} {job['kind']}: {len(extra_indices)} frames in recon")
        print(extra_indices)

        if tracked_ok:
            # Position from the weighted average of where the real cameras stood
            # (a robust signal).
            down_refs = extrinsics_np[:, 1, :3]   # each frame's own down axis, model space
            up_model = -(down_refs[weights_total > 0]
                         * weights_total[weights_total > 0, None]).sum(axis=0)
            up_model /= np.linalg.norm(up_model)

            C_avg = auto_camera_position(extrinsics_np, weights_total)
            # One oriented box, used for BOTH the framing solve and the debug draw.
            # Fitting the raw points instead let the box's corners (extreme in all 3
            # axes at once) stick out past every real point and shoot off-screen.
            obb_corners = subject_oriented_box(mask_points_total, margin=margin, percentile=95)
            box_center = obb_corners.mean(axis=0)
            R_auto_c2w = look_at_rotation(C_avg, box_center, up_model)

            new_view = solve_auto_camera_extrinsic(
                obb_corners, box_center, R_auto_c2w,
                job["recon"].preds["intrinsic"][0].cpu().numpy(),
                job["recon"].preds["depth"].shape[1:],
                extra_distance_margin=extra_distance_margin,
                dtype=job["recon"].preds["extrinsic"].dtype,
            )
            print("R_auto orthonormal check:",
                  np.allclose(R_auto_c2w @ R_auto_c2w.T, np.eye(3), atol=1e-4),
                  "det:", np.linalg.det(R_auto_c2w))
            print("C_avg (model units):", C_avg, "box_center:", box_center)
            print("weights: min/max/mean/#nonzero:", weights_total.min(), weights_total.max(),
                  weights_total.mean(), (weights_total > 0).sum())
        else:
            # FALLBACK -- tracking gave us nothing to aim at, so frame on the mean of
            # the real cameras instead of on the subject. The shot still gets made;
            # it's just "where the shoot was pointed", not "where the subject was".
            print(f"{job['beat_key']}: no usable subject masks "
                  f"({len(masks_dirs)} mask dir(s), {len(mask_points_total)} points) "
                  "-- falling back to the mean camera pose")
            new_view = mean_camera_extrinsic(
                extrinsics_np, pull_back=extra_distance_margin,
                dtype=job["recon"].preds["extrinsic"].dtype)

        out.append(projection_mapping_sequence(
            recon=job["recon"],
            extra_indices=extra_indices,   # one output frame per source frame
            new_view=new_view,
            confidence_threshold=ct_vggt,
            use_lowres_images=False,
            splat_radius=splat_radius,
            frame_rate=fps,
            out_dir_name=f"projection_frames_{job['beat_key']}",
        ))
    return out


# ---------------------------------------------------------------------------
# RENDERING -- mapping
# ---------------------------------------------------------------------------
_DEFAULT_PALETTE = ["#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22"]


def animated_map(gif=True, mp4=None):
    """Cell 83 -- one animated map per GOOGLE_MAP beat.

    Each writes its own image sequence to assets/map_frames_<beat_key> (the standard
    hand-off to the editor); gif/mp4 are optional extra exports of the same frames.
    Duration comes from the beat we pass in, so two MAP beats get two different-length
    animations instead of sharing one."""
    import Rendering.R_map_animator
    config_if_needed()
    S["flags"]["GOOGLE_MAP"] = True
    beats_by_key = revised_beats()

    for job in jobs_of(archetype="MAP", flag="GOOGLE_MAP"):
        beat = beats_by_key.get(job["beat_key"], job["beat"])
        wanted = _wanted_subjects(beat)
        subject_traces = [
            {"coords": latlons, "trail_color": _DEFAULT_PALETTE[i % len(_DEFAULT_PALETTE)]}
            for i, (slug, latlons) in enumerate(job.get("subjects_latlon", {}).items())
            if slug in wanted
        ]
        Rendering.R_map_animator.main(
            traces=[{"coords": job["cam_latlons"], "trail_color": "#3498db"},  # cameras
                    *subject_traces],
            api_key=S["api_key"],
            beat=beat,
            gif=gif,
            mp4=mp4,
        )


def times_of_day(fps_override=30.07):
    """Cell 84 -- get times of day into the report."""
    from C_CSV_report import attach_times_of_day
    config_if_needed()
    S["report"] = attach_times_of_day(S["video_path"], fps_override)
    return S["report"]


def bev_setup():
    """Cell 85 -- the frames each recon actually holds, per job. Cheap; restore()
    calls it because the BEV steps below index into extra_indices."""
    from Two2D.B_video_processing import available_frames
    for job in _need("recon_jobs", "call build_jobs()"):
        job["extra_indices"] = available_frames(job["frames_dir"], FRAME_NAME_FMT)
        print(f"{job['beat_key']} {job['kind']}: {len(job['extra_indices'])} frames in recon")
    return S["recon_jobs"]


def bev_hires(conf_percentile=60, splat_radius=1, margin_frac=0.05):
    """Cell 86 -- one BEV still per BEV_MAP beat.

    BEV_tile_render_PM's default out_path is asset_name-based (one file for the whole
    case), so a beat-keyed path is passed in. That also skips its own
    add_to_asset_list, so the beat-keyed asset row is logged here instead.

    n.b. the notebook cell took its confidence threshold from `conf`, a global left
    behind by the depth-stats cell -- i.e. the LAST job's confidence, applied to every
    job. Here each job uses its own."""
    from A_Config import assets_dir, asset_name, to_report_path
    from C_CSV_report import add_to_asset_list
    from P_projection_mapping import BEV_tile_render_PM

    for job in jobs_of(archetype="MAP", flag="BEV_MAP"):
        if not _solved(job):
            continue
        if "extra_indices" not in job:
            bev_setup()
        conf = job.get("depth_conf")
        if conf is None:
            conf = job["recon"].preds["depth_conf"].cpu().numpy()
            job["depth_conf"] = conf
        out_path = assets_dir() / f"{asset_name()}_BEV_tile_render_PM_{job['beat_key']}.png"
        BEV, job["new_view"], job["ortho_params"], job["BEV_path"] = BEV_tile_render_PM(
            job["recon"],
            job["extra_indices"],
            masks_dir=None,
            splat_radius=splat_radius,
            confidence_threshold=np.percentile(conf, conf_percentile),
            margin_frac=margin_frac,
            out_path=out_path,
        )
        add_to_asset_list({f"BEV_tile_render_PM_{job['beat_key']}": to_report_path(job["BEV_path"])})
        print(f"{job['beat_key']}: {job['BEV_path']}")
    return S["recon_jobs"]


def bev_trace_overlay():
    """Cell 88 -- animate the camera (position + heading frustum) over the
    BEV_tile_render_PM still. One animation per BEV_MAP beat, each over its own BEV
    still; n_frames comes from the beat we pass in (duration_seconds * fps), and
    frames land in bev_trace_frames_<beat_key>."""
    from P_trace_overlayer import BEV_trace_overlay as _overlay
    beats_by_key = revised_beats()

    for job in jobs_of(archetype="MAP", flag="BEV_MAP"):
        if not _solved(job):
            continue
        all_positions = job.get("subjects_real_pos") or job.get("subjects_positions") or {}
        beat = beats_by_key.get(job["beat_key"], job["beat"])
        wanted = _wanted_subjects(beat)
        subject_positions = {k: v for k, v in all_positions.items() if k in wanted}
        _overlay(
            job["recon"],
            job["extra_indices"],
            job["new_view"],
            job["ortho_params"],
            subject_positions=subject_positions,
            bev_png_path=job["BEV_path"],
            beat=beat,
        )


# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------
def assemble_movie():
    """Cell 90 -- make the edited multimedia movie."""
    import Two2D.W_video_editor
    config_if_needed()
    return Two2D.W_video_editor.assemble_paper_edit(
        S["video_path"],
        _need("analysis_2d_for_decisions", "call gps()"),
        recon_jobs=_need("recon_jobs", "call build_jobs()"),
    )


def final_report(ttl="6000s"):
    """Cell 91 -- cache the rough cut with Gemini and write the final report."""
    from A_Config import assets_dir, asset_name
    from V_text_summary import gemini_final_report, gemini_cache_video
    config_if_needed()
    PASS = _need("PASS", "call passer()")
    vid_path = assets_dir() / f"{asset_name()}_{PASS}_rough_cut.mp4"
    S["cache"] = gemini_cache_video(vid_path, assets_dir(), ttl=ttl)
    return gemini_final_report()


def populate_pptx(ppt_template="report"):
    """Cell 94 -- populate PowerPoint from CSV. The notebook cell relied on a
    `ppt_template` global that was never assigned; pass the template name in."""
    import importlib
    import W_populate_pptx
    importlib.reload(W_populate_pptx)
    config_if_needed()

    report_template = project_root / "src" / "Powerpoint" / f"{ppt_template}.pptx"
    ppt_assets = S["case_dir"] / "090_Assets"
    ppt_csv = ppt_assets / f"{S['case_name']}.csv"
    ppt_output = S["case_dir"] / "100_Powerpoint" / f"{S['case_name']}.pptx"
    S.update(ppt_csv=ppt_csv, ppt_output=ppt_output)
    return W_populate_pptx.populate(str(report_template))


# ---------------------------------------------------------------------------
# on-demand resolution + restart
# ---------------------------------------------------------------------------
# Only things that are cheap AND safe to re-derive go here. Anything that costs an
# API call or a GPU solve is deliberately absent, so _need() raises and names the
# step to run instead of quietly spending money.
_PROVIDERS = {
    "video_path": config, "fps": config, "case_name": config, "case_dir": config,
    "masks_dir": config, "api_key": config, "csv_path_GPS": config,
    "flags": set_flags, "producer_flags": producer_flags,
    "paper_edit_json_path": paper_edit,
    "recon_jobs": build_jobs,
}


def restore(case_name ,experiment,pass_num=1, gps_step=True, tracking=True):
    """After a crash or a kernel restart: back to a working state without redoing
    any Gemini / ASR / SAM3 / Producer / VGGT work.

    Re-derives config and paths, flags, PASS, the paper edit path, the recon jobs,
    each job's loaded reconstruction and camera poses, the GPS calibration, and the
    per-job frame lists.

    Not restored, because they cost money or a GPU:
      reid_match()          -- re-run only if you need person-A style ids back
      subject_positions()   -- cheap-ish but mask-heavy; call it if you need
                               subject 3D positions, metrics_3d() or the BEV traces
    """
    config(case_name ,experiment)
    set_flags()
    passer(pass_num)
    question()
    paper_edit()
    load_producer_flags()
    if gps_step:
        gps()
    if tracking:
        tracking_recovery()
        reid_match()
    build_jobs()
    load_recons()
    transform_poses_gps()
    subject_positions()
    bev_setup()
    print("\n--- restored ---")
    state()
    return S

def restore_pre_track(case_name ,experiment,pass_num=1, gps_step=True, tracking=True):
    """After a crash or a kernel restart: back to a working state without redoing
    any Gemini / ASR / SAM3 / Producer / VGGT work.

    Re-derives config and paths, flags, PASS, the paper edit path, the recon jobs,
    each job's loaded reconstruction and camera poses, the GPS calibration, and the
    per-job frame lists.

    Not restored, because they cost money or a GPU:
      reid_match()          -- re-run only if you need person-A style ids back
      subject_positions()   -- cheap-ish but mask-heavy; call it if you need
                               subject 3D positions, metrics_3d() or the BEV traces
    """
    config(case_name ,experiment)
    set_flags()
    passer(pass_num)
    question()
    paper_edit()
    load_producer_flags()
    if gps_step:
        gps()
    return S

def restore_pre_recon(case_name ,experiment,pass_num=1, gps_step=True, tracking=True):
    """After a crash or a kernel restart: back to a working state without redoing
    any Gemini / ASR / SAM3 / Producer / VGGT work.

    Re-derives config and paths, flags, PASS, the paper edit path, the recon jobs,
    each job's loaded reconstruction and camera poses, the GPS calibration, and the
    per-job frame lists.

    Not restored, because they cost money or a GPU:
      reid_match()          -- re-run only if you need person-A style ids back
      subject_positions()   -- cheap-ish but mask-heavy; call it if you need
                               subject 3D positions, metrics_3d() or the BEV traces
    """
    config(case_name ,experiment)
    set_flags()
    passer(pass_num)
    question()
    paper_edit()
    load_producer_flags()
    if gps_step:
        gps()
    if tracking:
        tracking_recovery()
        reid_match()
    build_jobs()
    
    return S