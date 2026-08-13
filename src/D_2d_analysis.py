

from pathlib import Path
import json
import re
import subprocess
import cv2
import numpy as np
import pandas as pd
from A_Config import report_path
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
    <subject_slug>-<instance_num>/ -- flat, no per-subject parent folder).
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
    GPS_sig   = GPS_test(video_path, api_key)

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
        f"{prefix}subject_first_frame"          : subject_start,
        f"{prefix}subject_last_frame"           : subject_end,
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
    '''Walks every beat with tracked_subject set and rebuilds
    analysis_2d_for_decisions -- a flat report keyed by tracked instance
    ("{subject_slug}-{instance_num}", matching A_ROBOFLOW_SAM3.py's
    _run_sam3_tracking output shape exactly) -- from masks already on disk
    at sam3_masks_dir()/<subject_slug>-<instance_num>/ -- masks are stored
    per instance now, flat, not nested under a beat or subject-label
    folder, and possibly spanning several beats' merged windows.
    fixed_frames beats filter each instance down to its own
    [start_frame, end_frame]; auto_select beats report each instance's full
    tracked range as-is, since their window is still just a coarse hint,
    not a resolved cut, and no frame number exists for it yet.

    A beat can list several tracked_subject labels, and each label can
    resolve to several tracked instances -- every one of those instances
    gets its own top-level entry here, same as the live tracking path. Each
    entry's "beat_ids" is the union of every beat_id whose tracked_subject
    resolved to that instance, for quick beat lookup without a separate
    join.'''
    from A_Config import sam3_masks_dir
    from run_models.A_ROBOFLOW_SAM3 import _slugify_subject

    data = json.loads(Path(paper_edit_json_path).read_text(encoding="utf-8"))
    analysis_2d_for_decisions = {}
    for beat in data["beats"]:
        if not beat.get("tracked_subject"):
            continue
        beat_key = beat["beat_id"][0]
        fixed = beat.get("cut_mode") == "fixed_frames"

        for subject in beat["tracked_subject"]:
            subject_slug = _slugify_subject(subject)
            track_dirs = list_track_id_dirs(sam3_masks_dir(), subject_slug)

            for track_dir in track_dirs:
                instance_key = track_dir.name
                all_frames = frames_present(track_dir)[0]
                if fixed:
                    frames_list = sorted(f for f in all_frames if beat["start_frame"] <= f <= beat["end_frame"])
                else:
                    frames_list = sorted(all_frames)

                # An instance's folder can be revisited across several beats that all
                # tracked the same subject -- union beat_ids onto whatever's already
                # there instead of losing earlier beats' ids on a later overwrite.
                prior_beat_ids = set(analysis_2d_for_decisions.get(instance_key, {}).get("beat_ids", []))
                beat_ids = sorted(prior_beat_ids | set(beat["beat_id"]))

                if not frames_list:
                    print(f"no masks for {beat_key}/{instance_key}")
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
                add_to_report({
                    f"{beat_key}_{instance_key}_subject_first_frame": entry["subject_first_frame"],
                    f"{beat_key}_{instance_key}_subject_last_frame": entry["subject_last_frame"],
                })

    return analysis_2d_for_decisions


def analysis_2d_gemvsSAM(gem_person_targets=None, SAM_dets=None, match_results=None, tolerance=15):
    """gem_person_id -> analysis_2d_for_decisions-shaped entry, built
    directly from detections.csv: union every frame across all of a
    person's matched tracker_ids (one to many -- separated beats can put
    the same person in different spans, where tracker_ids are different, so several tracker_ids can legitimately belong to one
    gem_person_id)."""
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
    person_mask_dirs = list_track_id_dirs(sam3_masks_dir(), "person")

    analysis_2d_for_decisions = {}
    for gem_person_id, tracker_ids in match_results.items():
        frames_present_set = sorted({
            int(float(d["frame_idx"])) for d in person_SAM_dets if d["tracker_id"] in tracker_ids
        })
        beat_ids = gem_person_targets[gem_person_id]["beat_ids"]
        # tracker_id doesn't map to a folder name (it's per-span-local) -- match by frame membership instead
        masks_dirs = sorted(
            str(d) for d in person_mask_dirs if set(frames_present(d)[0]) & set(frames_present_set)
        )

        if not frames_present_set:
            analysis_2d_for_decisions[gem_person_id] = {
                "Subject_frames_present": [],
                "subject_first_frame": None,
                "subject_last_frame": None,
                "subject_duration_frames": None,
                "continuous frame sequences": [],
                "best_seq_idx": None,
                "masks_dirs": [],
                "beat_ids": beat_ids,
            }
            continue

        seqs, best_idx = contiguous_durations(frames_present_set, tolerance)
        analysis_2d_for_decisions[gem_person_id] = {
            "Subject_frames_present": frames_present_set,
            "subject_first_frame": min(frames_present_set),
            "subject_last_frame": max(frames_present_set),
            "subject_duration_frames": max(frames_present_set) - min(frames_present_set),
            "continuous frame sequences": seqs,
            "best_seq_idx": best_idx,
            "masks_dirs": masks_dirs,
            "beat_ids": beat_ids,
        }

    return analysis_2d_for_decisions