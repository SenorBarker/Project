

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
    beat "order", same shape A_ROBOFLOW_SAM3.track_subject_sam3 returns) from
    the per-beat masks already on disk at sam3_masks_dir()/beat{order:02d}/
    -- no SAM3 API call, just re-reading files that are already there.'''
    from A_Config import sam3_masks_dir

    data = json.loads(Path(paper_edit_json_path).read_text(encoding="utf-8"))
    analysis_2d_for_decisions = {}
    for beat in data["beats"]:
        if not beat.get("tracked_subject"):
            continue
        order = beat["order"]
        masks_dir = sam3_masks_dir() / f"beat{order:02d}"
        analysis_2d_for_decisions[order] = mask_analysis(masks_dir, report_prefix=f"beat{order:02d}")

    return analysis_2d_for_decisions