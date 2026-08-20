#!/usr/bin/env python3
"""
select_frames.py — Sharp frame selector for video-to-photogrammetry workflows.

Samples a video at a regular interval and picks the sharpest frame within a
search window around each sample point, discarding motion-blurred frames.

Requires: opencv-python, numpy
Optional: tqdm  (pip install tqdm)

Usage:
    python select_frames.py input.mp4 output/ --interval 0.5 --window 7
    python select_frames.py clip.mp4 frames/ --interval 1.0 --window 15 --ext png
"""

import cv2
import json
import argparse
import re
import numpy as np
from pathlib import Path
import pandas as pd
import subprocess
try:
    from tqdm import tqdm
    def _progress(it, desc, total=None):
        return tqdm(it, desc=desc, total=total, unit="fr")
except ImportError:
    def _progress(it, desc, total=None):
        print(f"{desc}...")
        return it
from A_Config import report_path, to_report_path

def _blur_score(gray: np.ndarray) -> float:
    """Laplacian variance — higher means sharper edges."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def video_dims(video_path: str) -> tuple[int, int, float]:
    """Returns (width, height, fps)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    vid_w, vid_h, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return vid_w, vid_h, fps


def image_sequencer(
    video_path: str,
    output_dir: str,
    res: tuple[int, int] | None = None,
    interval_sec: float | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
    search_window: int = 7,
    min_sharpness: float = 0.0,
    outrank_margin: float = 0.1,
    score_scale: float = 0.25,
    output_ext: str = "jpg",
    jpeg_quality: int = 95,
) -> list[dict]:
    """
    Select a sharp frame near each sample point in a video, biased toward the
    sample point itself rather than always taking the single sharpest frame
    in the window.

    Parameters
    ----------
    res             Optional (width, height) to resize saved frames to.
    interval_sec    Optional target time between selected frames. If omitted,
                    every frame is kept.
    start_frame     First frame (inclusive) of the range to sample from.
    end_frame       Last frame (exclusive) of the range to sample from. Defaults
                    to the end of the video.
    search_window   Search ±N frames around each sample point for a sharper frame.
    min_sharpness   Skip output frames whose best sharpness score is below this.
                    Score scale depends on --score-scale; leave at 0 to keep all.
    outrank_margin  Search ratchets outward from the sample point one frame at a
                    time; a candidate only takes over as the winner if it beats the
                    CURRENT winner by this fraction (0.1 = needs to be >10% sharper),
                    so moving further from the center requires progressively bigger
                    improvements. This avoids both (a) jumping to a frame that's only
                    marginally sharper than a much closer near-miss just because it
                    happens to be the single sharpest in the whole window, and
                    (b) staying on the center frame over noise-level differences --
                    while still moving away from a genuinely bad center frame, since
                    nearby frames will usually clear the margin against it quickly.
    score_scale     Resize factor for blur scoring (0.25 = quarter resolution, 16× faster).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration     = total_frames / fps

    start_frame = max(0, start_frame)
    end_frame   = total_frames if end_frame is None else min(total_frames, end_frame)
    if start_frame >= end_frame:
        raise ValueError(f"start_frame ({start_frame}) must be before end_frame ({end_frame})")

    if interval_sec is None:
        interval_frames = 1
        sample_points = list(range(start_frame, end_frame))
        effective_window = 0
        print("Sampling: every frame (interval not set)")
    else:
        interval_frames = max(1, round(fps * interval_sec))
        sample_points = list(range(start_frame, end_frame, interval_frames))

        # Cap search window to half the interval so adjacent windows never overlap
        max_window = max(1, interval_frames // 2)
        if search_window > max_window:
            print(f"Window  : clamped from ±{search_window} to ±{max_window} (half interval)")
            search_window = max_window
        effective_window = int(search_window)

    print(f"Video   : {Path(video_path).name}")
    print(f"          {fps:.3g} fps · {total_frames} frames · {duration:.1f}s")
    if start_frame != 0 or end_frame != total_frames:
        print(f"Range   : frames {start_frame}-{end_frame} ({end_frame - start_frame} frames)")
    if interval_sec is not None:
        print(f"Sampling: every {interval_frames} frames ({interval_sec}s) → {len(sample_points)} candidates")
    print(f"Window  : ±{effective_window} frames  |  scoring at {score_scale:.0%} resolution")
    if res is not None:
        print(f"Output  : resizing saved frames to {res[0]}x{res[1]}")

    # ── Pass 1: sequential read, compute sharpness scores ───────────────────
    # Only store floats — no frame data kept in memory. Indexed relative to
    # start_frame so we only decode the requested sub-range of the video.
    scores: list[float] = []

    seek_exact(cap, start_frame)
    for _ in _progress(range(start_frame, end_frame), "Pass 1: scoring", total=end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            scores.append(-1.0)
            continue
        small = cv2.resize(frame, None, fx=score_scale, fy=score_scale,
                           interpolation=cv2.INTER_AREA)
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        scores.append(_blur_score(gray))

    # ── Pass 2: select best frame index per sample point ────────────────────
    # selection: (sample_point, best_frame_index, best_score)
    selections: list[tuple[int, int, float]] = []

    for sp in sample_points:
        def eligible_score(fi):
            """None if unreadable or below min_sharpness -- ineligible either way."""
            s = scores[fi - start_frame]
            return s if s >= min_sharpness else None

        # Ratchet outward from the center one step at a time, among eligible (>=
        # min_sharpness) frames only: a candidate only takes over as the winner if it
        # beats the CURRENT winner by outrank_margin, not the original center. This stops
        # a frame that's only marginally sharper than a near-miss from winning just
        # because it happens to be the global best far away -- each step further out
        # needs a progressively bigger improvement to justify it.
        lo = max(start_frame, sp - effective_window)
        hi = min(end_frame - 1, sp + effective_window)

        best_idx, best_score = sp, eligible_score(sp)

        for d in range(1, effective_window + 1):
            step_candidates = []
            if sp - d >= lo:
                s = eligible_score(sp - d)
                if s is not None:
                    step_candidates.append((sp - d, s))
            if sp + d <= hi:
                s = eligible_score(sp + d)
                if s is not None:
                    step_candidates.append((sp + d, s))
            if not step_candidates:
                continue
            step_idx, step_score = max(step_candidates, key=lambda x: x[1])
            if best_score is None or step_score > best_score * (1 + outrank_margin):
                best_idx, best_score = step_idx, step_score

        if best_score is None:
            continue
        selections.append((sp, best_idx, best_score))

    n_shifted = sum(1 for sp, fi, _ in selections if fi != sp)
    n_skipped = len(sample_points) - len(selections)
    print(f"Selected: {len(selections)} frames  "
          f"({n_shifted} shifted from sample point, {n_skipped} skipped below threshold)")

    # ── Pass 3: sequential read, write out the chosen frames ────────────────
    # Deliberately NOT seeking individually per selected frame here -- on this
    # (variable frame rate) source, cap.set(POS_FRAMES, N) + read() can land one
    # frame off from a true sequential decode to N (verified: returns frame N+1's
    # pixels while still being labeled N, at some positions but not others). A
    # single sequential pass, like Pass 1 already does, doesn't have that failure
    # mode -- it's the same approach track_subject_masks uses for its masks.
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted = {fi: (sp, score) for sp, fi, score in selections}
    results: list[dict] = []

    seek_exact(cap, start_frame)
    for frame_idx in _progress(range(start_frame, end_frame), "Pass 3: writing", total=end_frame - start_frame):
        ret, frame = cap.read()
        if frame_idx not in wanted:
            continue
        if not ret:
            print(f"  Warning: could not read frame {frame_idx}, skipping.")
            continue
        sp, score = wanted[frame_idx]
        fi = frame_idx

        if res is not None:
            frame = cv2.resize(frame, res, interpolation=cv2.INTER_AREA)

        name = f"{fi:04d}.{output_ext}"
        path = out_dir / name

        if output_ext in ("jpg", "jpeg"):
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        else:
            cv2.imwrite(str(path), frame)

        results.append({
            "file":         name,
            "source_frame": fi,
            "sample_point": sp,
            "offset":       fi - sp,         # negative = pulled earlier, positive = later
            "sharpness":    round(score, 2),  # relative to score_scale
        })

    # Save a log alongside the frames for auditing / debugging
    log_path = out_dir / "selection_log.json"
    log_path.write_text(json.dumps({
        "video":          video_path,
        "fps":            fps,
        "interval_sec":   interval_sec,
        "start_frame":    start_frame,
        "end_frame":      end_frame,
        "search_window":  search_window,
        "score_scale":    score_scale,
        "min_sharpness":  min_sharpness,
        "outrank_margin": outrank_margin,
        "frames":         results,
    }, indent=2))
    print(f"Log     : {log_path}")

    cap.release()

    if results:
        sh = [r["sharpness"] for r in results]
        print(f"Sharpness (at {score_scale:.0%} res): "
              f"min {min(sh):.0f} · mean {np.mean(sh):.0f} · max {max(sh):.0f}")

    return results, fps


def main() -> None:
    p = argparse.ArgumentParser(
        description="Select sharp frames from video for photogrammetry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("video",  help="Input video file")
    p.add_argument("output", help="Output directory for selected frames")

    p.add_argument("--interval", type=float, default=None, metavar="SEC",
                   help="Target interval between selected frames (seconds). Omit to keep every frame.")
    p.add_argument("--start-frame", type=int, default=0, metavar="N",
                   help="First frame (inclusive) of the range to sample from")
    p.add_argument("--end-frame", type=int, default=None, metavar="N",
                   help="Last frame (exclusive) of the range to sample from. Defaults to end of video.")
    p.add_argument("--window", type=int, default=7, metavar="N",
                   help="Search ±N frames around each sample point for the sharpest frame")
    p.add_argument("--min-sharpness", type=float, default=0.0, metavar="SCORE",
                   help="Drop frames whose best score is below this (0 = keep all). "
                        "Score is Laplacian variance at --score-scale resolution; "
                        "run once and inspect selection_log.json to calibrate.")
    p.add_argument("--outrank-margin", type=float, default=0.1, metavar="FRACTION",
                   help="Keep the sample point's own frame unless another frame in the "
                        "window is sharper by more than this fraction (0.1 = >10%% sharper "
                        "to outrank the center)")
    p.add_argument("--score-scale", type=float, default=0.25, metavar="0-1",
                   help="Downsample factor for blur scoring (smaller = faster scoring, "
                        "affects --min-sharpness scale)")
    p.add_argument("--ext", default="jpg", choices=["jpg", "png", "tiff"],
                   help="Output image format")
    p.add_argument("--quality", type=int, default=95, metavar="0-100",
                   help="JPEG quality (ignored for png/tiff)")

    args = p.parse_args()

    select_frames(
        video_path     = args.video,
        output_dir     = args.output,
        res            = None,
        interval_sec   = args.interval,
        start_frame     = args.start_frame,
        end_frame       = args.end_frame,
        search_window   = args.window,
        min_sharpness   = args.min_sharpness,
        outrank_margin  = args.outrank_margin,
        score_scale     = args.score_scale,
        output_ext      = args.ext,
        jpeg_quality    = args.quality,
    )


# Backwards-compatible notebook name.
select_frames = image_sequencer

#GET START TIMES FFROM VIDEOS - LOTS OF CODE FOR NOT MUCH
def _probe_format(video_path):
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_entries", "format=duration:format_tags=creation_time", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(probe.stdout)["format"]


def video_creation_time(video_path):
    '''INPUT  : path to a video file
       OUTPUT : (timestamp, is_start) -- timestamp is the container's embedded
                 creation_time (pandas Timestamp, UTC). is_start tells you whether
                 that timestamp marks the recording's start or its end: many
                 camera/phone MP4 muxers stamp creation_time (and the file's mtime)
                 at the moment the file is finalised, i.e. when recording STOPS, not
                 when it started -- so this can't be assumed, it has to be checked.
                 Found by comparing creation_time, and creation_time + duration,
                 against the file's mtime and seeing which one it lines up with.'''
    info = _probe_format(video_path)
    creation_time = info.get("tags", {}).get("creation_time")
    if creation_time is None:
        raise ValueError(f"No creation_time metadata in {video_path}")
    timestamp = pd.Timestamp(creation_time)#from ff probe
    duration = pd.Timedelta(seconds=float(info["duration"])) #from the metadata
    mtime = pd.Timestamp(Path(video_path).stat().st_mtime, unit='s', tz='UTC')#last modified time

    dist_if_start = abs((timestamp + duration) - mtime)
    dist_if_end = abs(timestamp - mtime)
    is_start = dist_if_start < dist_if_end

    return timestamp, is_start


VID_FILENAME_RE = re.compile(r"VID_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})(\d{3})")

def video_start_time(video_path):
    '''INPUT  : path to a video file
       OUTPUT : pandas Timestamp -- the recording's actual start time (local,
                as set on the recording device).
                Parsed straight from the filename when it follows this phone's
                VID_YYYYMMDD_HHMMSSfff naming (confirmed against video_creation_time
                to encode the true start, unlike creation_time/mtime which can drift
                with edits/copies). Falls back to the ffprobe heuristic otherwise.'''
    match = VID_FILENAME_RE.search(Path(video_path).stem)
    #if the filename has a start time, use this
    if match:
        year, month, day, hour, minute, second, millis = (int(g) for g in match.groups())
        return pd.Timestamp(year, month, day, hour, minute, second, millis * 1000)
    #if not use metadata, which may be the staart or the end
    timestamp, is_start = video_creation_time(video_path)
    if is_start:
        return timestamp

    duration = pd.Timedelta(seconds=float(_probe_format(video_path)["duration"]))
    return timestamp - duration

def video_fps(video_path):
    '''INPUT  : path to a video file
       OUTPUT : float -- frames per second, read from the video stream's r_frame_rate
                (ffprobe reports this as a "num/den" fraction, e.g. "30000/1001").'''
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    r_frame_rate = json.loads(probe.stdout)["streams"][0]["r_frame_rate"]
    num, den = r_frame_rate.split("/")
    return float(num) / float(den)


def frames_at_indices(video_path, frame_indices):
    '''INPUT  : path to a video file, iterable of frame indices
       OUTPUT : dict {frame_idx: ndarray} -- same single sequential pass as
                frames_at_times, but addressed by frame number instead of time.
                No cap.set(POS_FRAMES, N): it lands on the preceding keyframe,
                pairing frame N's index with frame N-k's pixels.'''
    wanted = set(frame_indices)
    cap = cv2.VideoCapture(str(video_path))
    frames, idx = {}, 0
    while wanted:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in wanted:
            frames[idx] = frame.copy()
            wanted.discard(idx)
        idx += 1
    cap.release()
    return frames


def seek_exact(cap, frame_idx):
    '''INPUT  : an open VideoCapture, the frame index wanted
       OUTPUT : bool -- cap left so the NEXT read() returns exactly frame_idx.
                For contiguous range readers; use frames_at_indices for scattered
                frames. Rewind + grab() counts exactly, cap.set(POS_FRAMES) doesn't.'''
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(frame_idx):
        if not cap.grab():
            return False
    return True


def frames_at_times(video_path, target_times_s):
    '''INPUT  : path to a video file, list of target times in seconds (e.g. one
                per subject)
       OUTPUT : (list[ndarray], list[int]) -- one frame per target, the frame whose
                own decoded timestamp (CAP_PROP_POS_MSEC) is nearest that target,
                and the frame index each one actually came from. Use that index to
                label the frame: recomputing it as start_s * fps is a guess that
                disagrees with the frame you were handed.
                Single sequential cap.read() pass, same principle as
                image_sequencer's Pass 3 -- no cap.set(POS_FRAMES, N)
                mid-stream (unreliable on VFR).'''
    cap = cv2.VideoCapture(str(video_path))
    best_frame = [None] * len(target_times_s)
    best_idx = [None] * len(target_times_s)
    best_dt = [float("inf")] * len(target_times_s)
    frame_idx = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        t_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        for i, target in enumerate(target_times_s):
            dt = abs(t_s - target)
            if dt < best_dt[i]:
                best_dt[i] = dt
                best_frame[i] = frame.copy()
                best_idx[i] = frame_idx
    cap.release()
    return best_frame, best_idx


if __name__ == "__main__":
    main()



#--------------frames won't be contigous, this lists them in order 
def available_frames(frames_dir, name_fmt):
    """Sorted list of absolute frame numbers that actually exist as files in
    frames_dir. image_sequencer's output is sparse (sharpest frame per sample
    point, not every Nth frame) -- never assume a contiguous range here."""
    suffix = Path(name_fmt).suffix
    return sorted(int(p.stem) for p in Path(frames_dir).glob(f"*{suffix}"))