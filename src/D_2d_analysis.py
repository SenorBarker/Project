

from pathlib import Path
import json
import re
import subprocess
import cv2
import numpy as np
import pandas as pd
from A_Config import report_path, has_gps_data
from C_CSV_report import add_to_report
from Q_GPS_processing import android_movie_GPS

#2D analysis
#CSV 
def frames_present(masks_path):
    '''INPUT  : path to a directory of per-frame mask PNGs (filename = frame number)
       OUTPUT : list of frame indices where the mask is non-empty
              :  '''
    frames_list = []

    for mask_file in sorted(Path(masks_path).glob("*.png")):
        frame = int(mask_file.stem)
        image = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if np.any(image):
            frames_list.append(frame)
    
    if not (frames_list):
        print(f"no masks in {masks_path}")
        return frames_list, None
    #total time subject was featured (ignoring gaps)
    subject_duration = np.max(frames_list)- np.min(frames_list)

    return frames_list, subject_duration


def list_track_id_dirs(masks_root, subject_slug):
    '''Every candidate tracked-instance folder for one subject, directly
    under masks_root (see A_ROBOFLOW_SAM3.py's layout, sam3_masks_dir()/
    <subject_slug>-<track_id>/ -- flat, no per-subject parent folder).
    [] if masks_root doesn't exist or holds no matching folders (both mean
    "no candidates", not an error).

    No filtering here -- for now every candidate is used as-is (see
    assemble_paper_edit). FUTURE: this is also what will enumerate the
    candidates analysis_2d_from_masks reports per beat; the Producer's
    revision pass will review that per-candidate breakdown and write back
    which instance folder names it actually wants for the beat, and
    assemble_paper_edit will resolve just those named folders instead of
    taking this function's full list -- that selection field doesn't exist
    in the schema yet, so it isn't wired up here.'''
    masks_root = Path(masks_root)
    if not masks_root.is_dir():
        return []
    return sorted(p for p in masks_root.iterdir() if p.is_dir() and p.name.startswith(f"{subject_slug}-"))


'''HELPERS THAT COULD BE ELSEWHERE'''


def is_3D_possible(frames_list, recon_tool):
    #populate this with real data]
    frame_reqs = {
        "VGGT_O" : {"min" : 3, "max":80},
        "Reality_scan" : {"min" : 50, "max":100000000}          
                 }
    minimum_fr = frame_reqs[recon_tool]["min"]
    maximum_fr = frame_reqs[recon_tool]["max"]
    if len(frames_list) < minimum_fr:
        print(f"insufficient frames for recon with {recon_tool}")
        return
    else:
        print(f"sufficient frames for recon with {recon_tool}")
    if len(frames_list) < maximum_fr:
        print(f"all frames can be used in {recon_tool}")
    else:print (f"too many frames for single {recon_tool} reconstruction")

    #overlap tests
    #enough overlap to make recon? 
    #which frames shoudl I use?

    #movie check
    contiguous_frames = contiguous_durations(frames_list)
    max_duration = max(x[2] for x in contiguous_frames)
    if max_duration < 60:
        print(f"too few contiguous frames for movie")


    #montage check
    #need a scene-based check so that the recon creates a single unit of space


    frames_for_recon = frames_list
    return frames_for_recon

#checks for making a video - do we have enough frames to even bother?
def contiguous_durations(frames, tolerance):
    '''INPUT frames list
    output - list of tuples (first, last, duration)
     '''
    frames = sorted(frames)
    if not frames:
        return []
    groups = []
    start = prev = frames[0]

    for f in frames[1:]:
        if f <= prev + tolerance+1:
                prev = f
        else:
            groups.append((start, prev))
            start = prev = f
    groups.append((start, prev))
    durations = [(s, e, e - s + 1) for s, e in groups]
    longest_idx = max(range(len(durations)), key=lambda i: durations[i][2])
    return durations, longest_idx

def GPS_test(video_path, api_key):
   lat,lon = android_movie_GPS(video_path, api_key)
   if lat is not None and lon is not None:
       return "yes"
   else:
       return "no"
   

def analysis_2D(video_path, api_key):
    '''
    #the commands I will run to analyse the video itself
    OUTPUTS
    analysis_2d_for_CSV :  appends the report CSV with metrics I think the report needs
    analysis_2d_for_decisions : dict that is used to decide what to do next in analysis
    '''
    # GPS data on disk, not the video's own tag -- one tag can't align anything, and the
    # data is often hand-measured (Google Earth) rather than carried by the footage
    GPS_sig   = "yes" if has_gps_data() else "no"

    analysis_2d_for_decisions = {
            "GPS_signal"            : GPS_sig,
                    }

    return analysis_2d_for_decisions

def mask_analysis(masks_dir, report_prefix=None):
    '''Disk-based frame/contiguity analysis for a masks folder already on
    disk -- rebuilds the same shape A_ROBOFLOW_SAM3.track_subject_sam3
    computes in-process, without re-running SAM3 (expensive); also the
    primary path for a tracker that doesn't build this itself (e.g.
    A_YOLO_seg). report_prefix (e.g. "beat03") scopes the CSV rows so
    multiple beats don't overwrite each other's keys -- omit for a
    single-subject/whole-video call.

    No detections -> returns a "no detections" entry (all Nones/empty)
    instead of raising, same as track_subject_sam3's per-beat handling.'''
    fps = 30 #obtained from image sequencer

    frames_list, subject_duration_frames = frames_present(masks_dir)

    if not frames_list:
        print(f"no masks in {masks_dir}")
        return {
            "Subject_frames_present": [],
            "subject_first_frame": None,
            "subject_last_frame": None,
            "subject_duration_frames": None,
            "continuous frame sequences": [],
            "best_seq_idx": None,
            "masks_dir": str(masks_dir),
        }

    subject_start = min(frames_list)
    subject_end  = max(frames_list)

    seqs, best_idx        = contiguous_durations(frames_list, tolerance =15)

    prefix = f"{report_prefix}_" if report_prefix else ""
    analysis_2d_for_CSV = {
        # count/range, not a position in the video -- must NOT end in "_frame" or
        # attach_times_of_day will wrongly treat it as a frame index and convert
        # it into a clock time instead of a duration
        f"{prefix}subject_duration_frame_count": subject_duration_frames,
        f"{prefix}{{{{subject_duration_s}}}}"   : subject_duration_frames / fps,
        f"{prefix}first_frame"          : subject_start,
        f"{prefix}last_frame"           : subject_end,
    }
    analysis_2d_for_decisions = {
        "Subject_frames_present": frames_list,
        "subject_first_frame"   : subject_start,
        "subject_last_frame"    : subject_end,
        "subject_duration_frames": subject_duration_frames,
        "continuous frame sequences" : seqs,
        "best_seq_idx"           : best_idx,
        "masks_dir"              : str(masks_dir),
        }
    #print new rows each item
    print("\n".join(f"{k}: {v}" for k, v in analysis_2d_for_decisions.items()))
    add_to_report(analysis_2d_for_CSV)

    return analysis_2d_for_decisions


def analysis_2d_from_masks(paper_edit_json_path):
    '''Runs analysis on every tracked mask folder on disk at
    sam3_masks_dir()/<subject_slug>-<track_id>/ -- masks are stored per
    instance, flat, possibly spanning several beats' merged windows.
    Each instance's full tracked range is reported as-is -- this runs before
    find_all_cut_points, so no beat has resolved frames to filter against.

    The live tracker (A_ROBOFLOW_SAM3.track_subject_sam3) knows exactly which
    request produced each track_id, but that result only ever lives in memory
    -- disk only keeps the masks themselves and detections.csv, neither of
    which records beat_ids. So beat_ids here is approximated by frame overlap:
    a gem_person_id dict entry names "person" (per build_tracking_requests),
    so several beats can share one subject_slug, and matching by name alone
    would blanket-credit every such beat to every person track. Instead each
    beat's tracking_windows are converted to frame ranges and only credited to
    an instance whose own frame range actually overlaps -- a plain range
    check, not exact frame matching, since a beat's window and its masks were
    never going to line up frame-for-frame anyway (tracking dropout, edge
    frames). An instance with no masks has nothing to test overlap against,
    so it gets no beat_ids -- it still gets a full entry here, just with no
    beat attributed.'''
    from A_Config import sam3_masks_dir, source_video_path
    from run_models.A_ROBOFLOW_SAM3 import _slugify_subject
    from render_paper_edit import tracking_windows, beat_key
    from Two2D.B_video_processing import video_fps

    data = json.loads(Path(paper_edit_json_path).read_text(encoding="utf-8"))
    masks_root = sam3_masks_dir()
    fps = video_fps(source_video_path())

    # subject_slug -> [(beat_key, beat_id_list, start_frame, end_frame), ...] --
    # cheap in-memory pass, no disk touched yet.
    windows_by_slug = {}
    for beat in data["beats"]:
        if not beat.get("tracked_subject"):
            continue
        key = beat_key(beat)
        subjects = dict.fromkeys(
            "person" if isinstance(subject, dict) else subject
            for subject in beat["tracked_subject"]
        )
        subject_slugs = list(map(_slugify_subject, subjects))
        for start_s, end_s in tracking_windows(beat, data["beats"]):
            start_frame, end_frame = round(start_s * fps), round(end_s * fps)
            for subject_slug in subject_slugs:
                windows_by_slug.setdefault(subject_slug, []).append(
                    (key, beat["beat_id"], start_frame, end_frame)
                )

    analysis_2d_for_decisions = {}
    track_dirs = sorted(p for p in masks_root.iterdir() if p.is_dir() and not p.name.startswith("_"))
    for track_dir in track_dirs:
        instance_key = track_dir.name
        subject_slug = instance_key.rsplit("-", 1)[0]
        frames_list = sorted(frames_present(track_dir)[0])
        frame_span = (frames_list[0], frames_list[-1]) if frames_list else None

        matching = [
            (beat_key_, beat_ids_)
            for beat_key_, beat_ids_, start_frame, end_frame in windows_by_slug.get(subject_slug, [])
            if frame_span is not None and start_frame <= frame_span[1] and frame_span[0] <= end_frame
        ]
        beat_ids = sorted({beat_id for _, ids in matching for beat_id in ids})

        if not frames_list:
            print(f"no masks for {instance_key}")
            entry = {
                "Subject_frames_present": [],
                "subject_first_frame": None,
                "subject_last_frame": None,
                "subject_duration_frames": None,
                "continuous frame sequences": [],
                "best_seq_idx": None,
                "masks_dir": str(track_dir),
                "beat_ids": beat_ids,
            }
        else:
            seqs, best_idx = contiguous_durations(frames_list, tolerance=15)
            entry = {
                "Subject_frames_present": frames_list,
                "subject_first_frame": min(frames_list),
                "subject_last_frame": max(frames_list),
                "subject_duration_frames": max(frames_list) - min(frames_list),
                "continuous frame sequences": seqs,
                "best_seq_idx": best_idx,
                "masks_dir": str(track_dir),
                "beat_ids": beat_ids,
            }

        analysis_2d_for_decisions[instance_key] = entry
        for beat_key_, _ in matching:
            add_to_report({
                f"{beat_key_}_{instance_key}_first_frame": entry["subject_first_frame"],
                f"{beat_key_}_{instance_key}_last_frame": entry["subject_last_frame"],
            })

    return analysis_2d_for_decisions


def analysis_2d_gemvsSAM(gem_person_targets=None, SAM_dets=None, match_results=None, tolerance=15):
    """gem_person_id -> analysis_2d_for_decisions-shaped entry, built
    directly from detections.csv: union every frame across all of a
    person's matched track_ids (one to many -- separated beats can put
    the same person in different spans, where track_id numbering
    restarts, so several track_ids can legitimately belong to one
    gem_person_id).

    Returns GEM_SAM_matches, kept separate from analysis_2d_for_decisions
    (analysis_2d_from_masks) on purpose: a track_id that hasn't been
    matched to a gem_person_id here is still a tracked instance with real
    mask data -- untagged is not the same as untracked, so that data must
    not be dropped just because it's absent from this gem-id-keyed view."""
    from Two2D.AX_gem_SAM_matcher import gem_person_targets_lookup, load_SAM_dets, match_gem_people_to_sam
    from A_Config import sam3_masks_dir

    if gem_person_targets is None:
        from A_Config import source_video_path
        from Two2D.B_video_processing import video_dims
        _, _, fps = video_dims(source_video_path())
        gem_person_targets = gem_person_targets_lookup(fps)
    if SAM_dets is None:
        SAM_dets = load_SAM_dets()
    if match_results is None:
        match_results = match_gem_people_to_sam(gem_person_targets=gem_person_targets, SAM_dets=SAM_dets)

    person_SAM_dets = [d for d in SAM_dets if d["subject"] == "person"]
    masks_root = sam3_masks_dir()

    GEM_SAM_matches = {}
    for gem_person_id, track_ids in match_results.items():
        frames_present_set = sorted({
            int(float(d["frame_idx"])) for d in person_SAM_dets if int(d["track_id"]) in track_ids
        })
        beat_ids = gem_person_targets[gem_person_id]["beat_ids"]
        # track_id is the same number the mask folder is named with (person-<track_id>)
        masks_dirs = sorted(str(masks_root / f"person-{n:02d}") for n in track_ids)

        if not frames_present_set:
            GEM_SAM_matches[gem_person_id] = {
                "Subject_frames_present": [],
                "subject_first_frame": None,
                "subject_last_frame": None,
                "subject_duration_frames": None,
                "continuous frame sequences": [],
                "best_seq_idx": None,
                "masks_dirs": [],
                "beat_ids": beat_ids,
            }
            add_to_report({
                f"{gem_person_id}_first_frame": None,
                f"{gem_person_id}_last_frame": None,
            })
            continue

        seqs, best_idx = contiguous_durations(frames_present_set, tolerance)
        GEM_SAM_matches[gem_person_id] = {
            "Subject_frames_present": frames_present_set,
            "subject_first_frame": min(frames_present_set),
            "subject_last_frame": max(frames_present_set),
            "subject_duration_frames": max(frames_present_set) - min(frames_present_set),
            "continuous frame sequences": seqs,
            "best_seq_idx": best_idx,
            "masks_dirs": masks_dirs,
            "beat_ids": beat_ids,
        }
        add_to_report({
            f"{gem_person_id}_first_frame": min(frames_present_set),
            f"{gem_person_id}_last_frame": max(frames_present_set),
        })

    return GEM_SAM_matches