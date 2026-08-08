

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


def list_track_id_dirs(beat_mask_dir):
    '''Every candidate track_id subfolder under a beat's mask directory
    (see A_ROBOFLOW_SAM3.py's layout, sam3_masks_dir()/<beat_id>/
    <subject>-<track_id>/). [] if the directory doesn't exist or holds none
    (both mean "no candidates", not an error).

    No filtering here -- for now every candidate is used as-is (see
    assemble_paper_edit). FUTURE: this is also what will enumerate the
    candidates analysis_2d_from_masks reports per beat; the Producer's
    revision pass will review that per-candidate breakdown and write back
    which track_id folder names it actually wants for the beat, and
    assemble_paper_edit will resolve just those named folders instead of
    taking this function's full list -- that selection field doesn't exist
    in the schema yet, so it isn't wired up here.'''
    beat_mask_dir = Path(beat_mask_dir)
    if not beat_mask_dir.is_dir():
        return []
    return sorted(p for p in beat_mask_dir.iterdir() if p.is_dir())


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
    '''The "dict maker" -- knows which beats to check. Walks every beat with
    tracked_subject set and rebuilds analysis_2d_for_decisions (keyed by
    beat_id -- specifically beat["beat_id"][0], matching the plain-string
    keys A_ROBOFLOW_SAM3._run_sam3_tracking's return dict uses; see that
    module for why beat_id, not order) from the per-beat masks already on
    disk at sam3_masks_dir()/<beat_id>/<subject>-<track_id>/ -- no SAM3 API
    call, just re-reading files that are already there. Unions masks across
    every beat_id in the beat's beat_id list (more than one after a
    revision-pass merge), and every candidate track_id folder counts
    (list_track_id_dirs, no curation yet), same as the write side.'''
    from A_Config import sam3_masks_dir

    data = json.loads(Path(paper_edit_json_path).read_text(encoding="utf-8"))
    analysis_2d_for_decisions = {}
    for beat in data["beats"]:
        if not beat.get("tracked_subject"):
            continue
        beat_key = beat["beat_id"][0]
        track_dirs = [d for bid in beat["beat_id"] for d in list_track_id_dirs(sam3_masks_dir() / bid)]
        frames_list = sorted({f for d in track_dirs for f in frames_present(d)[0]})

        if not frames_list:
            print(f"no masks for {beat_key}")
            analysis_2d_for_decisions[beat_key] = {
                "Subject_frames_present": [],
                "subject_first_frame": None,
                "subject_last_frame": None,
                "subject_duration_frames": None,
                "continuous frame sequences": [],
                "best_seq_idx": None,
                "masks_dir": str(sam3_masks_dir() / beat_key),
            }
            continue

        seqs, best_idx = contiguous_durations(frames_list, tolerance=15)
        entry = {
            "Subject_frames_present": frames_list,
            "subject_first_frame": min(frames_list),
            "subject_last_frame": max(frames_list),
            "subject_duration_frames": max(frames_list) - min(frames_list),
            "continuous frame sequences": seqs,
            "best_seq_idx": best_idx,
            "masks_dir": str(sam3_masks_dir() / beat_key),
        }
        analysis_2d_for_decisions[beat_key] = entry
        add_to_report({
            f"{beat_key}_subject_first_frame": entry["subject_first_frame"],
            f"{beat_key}_subject_last_frame": entry["subject_last_frame"],
        })

    return analysis_2d_for_decisions