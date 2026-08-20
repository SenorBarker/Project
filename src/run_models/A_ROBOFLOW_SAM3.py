"""
Open-vocabulary mask tracking via Roboflow's SAM3-prompted-video-tracker
workflow -- a companion to A_YOLO_seg.track_subject_masks_from_hints for
subjects that aren't in YOLO's fixed class list (e.g. "gun"). Same
detections-shaped input, same {"results", "subject_start", "subject_end"}
output, same per-frame MASK_NAME_FMT convention -- so it's a drop-in
alternative tracker, not a separate pipeline.

Confirmed against a live run: this workflow serializes each detection's mask
as a "points" polygon outline (bbox + points), never RLE -- see
decode_mask_from_detection.
"""

import bisect
import logging
import os
import re
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from inference_sdk import InferenceHTTPClient
from inference_sdk.webrtc import VideoFileSource, StreamConfig
# NB: A_LOCAL_SAM3 is deliberately NOT imported here. It imports make_span_clip
# and friends back out of this module, so a module-level import either way round
# is circular. _dispatch_sam3_tracking imports it inside the function instead,
# which also keeps torch off the import path when running against Roboflow.

load_dotenv(Path(__file__).parent.parent / ".env")

# Quiet the WebRTC SDK's own INFO-level connection-state logging (ICE/connection
# state changes) -- purely noise for this use case, not something we act on.
logging.getLogger("inference_sdk").setLevel(logging.WARNING)

WORKSPACE = "david-barker-25-ucl-ac-uk"
WORKFLOW = "sam3-prompted-video-tracker-1784979867494"
MASK_NAME_FMT = "{:04d}.png"  # matches A_YOLO_seg / W_video_editor's convention

# Which SAM3 backend track_subject_sam3/run_sam3_manual dispatch to: the local
# on-GPU model in Models/sam3 (A_LOCAL_SAM3) or Roboflow's hosted workflow.
# Everything below this line is the Roboflow path and is left intact as the
# fallback -- set this False to go straight back to it, nothing else changes.
USE_LOCAL_SAM3 = True

# Roboflow's serverless WebRTC worker-init endpoint is flaky (read timeouts,
# intermittent 500s) independent of region/plan -- retry session setup rather
# than failing the whole beat on one bad allocation attempt.
SAM3_SESSION_MAX_ATTEMPTS = 5
SAM3_SESSION_RETRY_BACKOFF_S = 5

# Candidate keys for a detection's per-object tracker ID, most-specific
# first -- the workflow's response schema isn't documented client-side, so
# this is a best-effort ordered guess rather than a confirmed field name.
# If extract_track_id() raises, its error prints the real keys from a live
# detection -- add the correct one here.
TRACK_ID_KEYS = ("tracker_id", "track_id", "trackerId", "trackId", "id")


def find_preceding_keyframe_s(video_path, target_s, initial_window_s=20.0, max_window_s=300.0):
    """
    Finds the pts_time (seconds) of the nearest keyframe at or before target_s
    in the source video, by asking ffprobe for frame types in a window ending
    at target_s. This is the timestamp ffmpeg's `-ss` (input seeking) +
    `-c copy` will actually snap to, since stream copy can't cut mid-GOP.

    Widens the search window and retries if no keyframe is found (e.g. an
    unusually long GOP), up to max_window_s, and falls back to 0.0 for
    targets near the start of the file.
    """
    window_s = initial_window_s
    while True:
        probe_start = max(0.0, target_s - window_s)
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                # Request both field names: ffmpeg <5 only has pkt_pts_time,
                # ffmpeg >=5 renamed it to pts_time. Whichever name ffprobe
                # doesn't recognize is DROPPED from the row entirely (not
                # blanked to "N/A"), so the column count varies by version --
                # parse generically below instead of assuming a fixed shape.
                "-show_entries", "frame=pts_time,pkt_pts_time,pict_type",
                "-of", "csv=p=0",
                "-read_intervals", f"{probe_start}%{target_s + 0.5}",
                str(video_path),
            ],
            capture_output=True, text=True, check=True,
        )

        keyframe_times = []
        for line in result.stdout.strip().splitlines():
            parts = line.split(",")
            if len(parts) < 2:
                continue  # skip a malformed/incomplete ffprobe row rather than crash
            pict_type = parts[-1]
            actual_pts = next((p for p in parts[:-1] if p != "N/A"), None)
            if actual_pts is not None and pict_type == "I" and float(actual_pts) <= target_s:
                keyframe_times.append(float(actual_pts))

        if keyframe_times:
            return max(keyframe_times)

        if probe_start <= 0.0:
            return 0.0  # target is near the start of the file, no earlier keyframe exists

        if window_s >= max_window_s:
            raise RuntimeError(
                f"No keyframe found at or before {target_s:.3f}s within {max_window_s:.0f}s window"
            )

        window_s *= 2


_FRAME_PTS_CACHE = {}


def frame_pts_times(video_path):
    """
    Every video frame's presentation timestamp (seconds), in frame order --
    i.e. frame N of a sequential cap.read() pass has pts frame_pts_times()[N].

    Read from the container's packets (demux only, no decode: ~1s for a
    4-minute file) and cached per (path, mtime, size).
    """
    video_path = Path(video_path)
    stat = video_path.stat()
    key = (str(video_path.resolve()), stat.st_mtime_ns, stat.st_size)
    if key not in _FRAME_PTS_CACHE:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "packet=pts_time", "-of", "csv=p=0", str(video_path)],
            capture_output=True, text=True, check=True,
        )
        # B-frames make packet order != presentation order, hence the sort.
        _FRAME_PTS_CACHE[key] = sorted(
            float(field)
            for field in result.stdout.replace(",", "\n").split()
            if field not in ("", "N/A")
        )
    return _FRAME_PTS_CACHE[key]


def frame_index_at_time(video_path, t_s):
    """
    The 0-based frame number a sequential decode of video_path reaches at
    timestamp t_s -- i.e. the index W_video_editor's overlay loop and the
    MASK_NAME_FMT mask filenames both count in.

    Counts real frames rather than multiplying by an fps, because phone
    footage is routinely VFR: this file reports r_frame_rate 30 but
    avg_frame_rate 29.9726, and neither converts times to frame numbers
    correctly (measured error up to 8 frames, varying along the file --
    which lands a mask on the wrong frame entirely).
    """
    pts = frame_pts_times(video_path)
    return max(0, bisect.bisect_left(pts, t_s - 1e-6))


def make_span_clip(video_path, out_path, start_s, end_s):
    """
    Creates a temporary video containing only [start_s, end_s], using ffmpeg
    stream copy (-c copy) -- no decode/re-encode, so no quality loss, unlike
    writing frames back out through cv2.VideoWriter. Stream copy can only cut
    on a keyframe boundary, so we explicitly resolve the actual keyframe
    ffmpeg will snap to (find_preceding_keyframe_s) and use THAT as the true
    clip start -- rather than assuming the clip starts exactly at start_s,
    which would silently offset every downstream frame number.

    That keyframe time is turned into a frame number by counting frames
    (frame_index_at_time), NOT by multiplying by fps -- see that function:
    time * fps put every mask up to 8 frames away from the frame it was
    traced on.

    Returns:
      original_start_frame, fps, width, height
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    actual_start_s = find_preceding_keyframe_s(video_path, start_s)
    duration_s = end_s - actual_start_s

    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-hide_banner",
            "-ss", str(actual_start_s),
            "-i", str(video_path),
            "-t", str(duration_s),
            "-c", "copy",
            str(out_path),
        ],
        check=True,
    )

    start_frame = frame_index_at_time(video_path, actual_start_s)

    return start_frame, fps, width, height


def unwrap_predictions(raw):
    """
    Handles common workflow output shapes.
    Returns a list of detection dicts.
    """
    if raw is None:
        return []

    if isinstance(raw, list):
        return raw

    if isinstance(raw, dict):
        for key in ["predictions", "detections", "instances", "tracked_detections"]:
            val = raw.get(key)
            if isinstance(val, list):
                return val

        # Some serializers put detections under nested output fields.
        if "value" in raw and isinstance(raw["value"], list):
            return raw["value"]

    return []


def decode_mask_from_detection(det, image_shape):
    """
    Decodes a detection's mask from its "points" polygon outline -- confirmed
    by an actual run against the sam3-prompted-video-tracker workflow to be
    the only format it returns (bbox + points, no RLE/"mask" field at all).

    Returns uint8 mask, or None if the detection has fewer than 3 points.
    """
    points = det.get("points")
    if points is None:
        return None

    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    poly = [[int(round(p["x"])), int(round(p["y"]))] for p in points]

    if len(poly) < 3:
        return None

    cv2.fillPoly(mask, [np.array(poly, dtype=np.int32)], 255)
    return mask


def extract_track_id(det):
    """
    Pulls a detection's per-object tracker ID by trying TRACK_ID_KEYS in
    order. Raises with the detection's actual keys if none match, so a bad
    guess fails loudly with what's needed to fix it, rather than silently
    grouping unrelated detections together.
    """
    for key in TRACK_ID_KEYS:
        if key in det:
            return det[key]
    raise RuntimeError(
        f"No track-ID field found on detection (tried {TRACK_ID_KEYS}); "
        f"actual keys were {sorted(det.keys())}. Update TRACK_ID_KEYS."
    )


def _slugify_subject(subject):
    """Filesystem-safe folder-name fragment for a subject query (e.g. "the
    red car" -> "red_car"). Used both as run_sam3_manual's beat_id (no
    paper_edit lifecycle, so the subject string is the only stable-enough
    identity available) and as the per-subject prefix on track_id folder
    names within a beat -- two different subject sessions in the same beat
    can independently hand out raw_track_id=0, and prefixing with the
    subject keeps their folders from colliding without a separate directory
    level (see A_Config.sam3_masks_dir()/<beat_id>/<subject>-<track_id>/)."""
    slug = re.sub(r"[^a-z0-9]+", "_", subject.strip().lower()).strip("_")
    return slug or "subject"


def _dispatch_sam3_tracking(*args, **kwargs):
    """
    Routes to whichever backend USE_LOCAL_SAM3 selects. Both sides take the
    same arguments and return the same analysis_2d_for_decisions shape, so
    callers of track_subject_sam3/run_sam3_manual don't change either way.
    The local import keeps torch/SAM3 off the import path when running
    against Roboflow.
    """
    if USE_LOCAL_SAM3:
        try:
            # How the rest of src/ imports this package (see D_2d_analysis,
            # Two2D/W_video_editor); the notebook only puts src/ on sys.path.
            from run_models.A_LOCAL_SAM3 import _run_sam3_tracking_local
        except ModuleNotFoundError:
            # How the spike scripts import it, running from inside run_models/.
            from A_LOCAL_SAM3 import _run_sam3_tracking_local
        return _run_sam3_tracking_local(*args, **kwargs)
    return _run_sam3_tracking(*args, **kwargs)


def track_subject_sam3(video_path, threshold=0.5,
                        requested_region="us", requested_plan="webrtc-gpu-large"):
    import json
    import sys
    from collections import defaultdict
    from A_Config import REPO_ROOT, case_name, agent_p_output_dir
    print("here")
    sys.path.insert(0, str(Path(REPO_ROOT) / "src" / "claude_agent"))
    #-------------------get data----------------
    print("accessing data")
    from render_paper_edit import build_tracking_requests

    paper_edit_json_path = agent_p_output_dir() / f"{case_name()}_paper_edit_draft.json"

    flags_path = paper_edit_json_path.with_name(
        paper_edit_json_path.stem.replace("_paper_edit", "") + "_flags.json"
    )
    current_flags = json.loads(flags_path.read_text(encoding="utf-8"))
    

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    tracking_requests = build_tracking_requests(paper_edit_json_path, fps)
    
    tracking_requests_by_subject = defaultdict(list)
    print(tracking_requests_by_subject)
    for request in tracking_requests:
        tracking_requests_by_subject[request["subject"]].append(request)

    return _dispatch_sam3_tracking(video_path, tracking_requests_by_subject, threshold, requested_region, requested_plan)


def run_sam3_manual(video_path, subject, start_frame=None, end_frame=None,
                     threshold=0.5, requested_region="us", requested_plan="webrtc-gpu-large"):
    """
    Manual version of this: ad-hoc SAM3 run against a single video file for
    one subject string -- no paper edit / case involved. start_frame/end_frame
    optionally restrict the run to a sub-range (inclusive); default is the
    whole video. Runs through the same tracking core track_subject_sam3
    uses (including the make_span_clip keyframe-aligned cut), so
    masks/report/return shape are identical.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    start_frame = 0 if start_frame is None else start_frame
    end_frame = total_frames - 1 if end_frame is None else end_frame

    start_s = start_frame / fps
    end_s = (end_frame + 1) / fps  # end_frame is inclusive; make_span_clip's end_s is not

    tracking_requests_by_subject = {subject: [{"subject": subject, "start_s": start_s, "end_s": end_s}]}

    return _dispatch_sam3_tracking(video_path, tracking_requests_by_subject, threshold, requested_region, requested_plan)


def _run_sam3_tracking(video_path, tracking_requests_by_subject, threshold, requested_region, requested_plan):
    """
    Shared tracking core for track_subject_sam3 and run_sam3_manual. Given
    {subject: [{"subject","start_s","end_s"}, ...]}, cuts a subclip per
    span, streams each through SAM3, writes masks, and returns
    analysis_2d_for_decisions keyed by tracked instance ("{subject_slug}-
    {track_id}", e.g. "police-1", "police-2") -- a subject label can
    resolve to multiple distinct tracked instances, and each gets its own
    flat masks_dir folder and its own dict entry, not merged into one
    per-label entry (see build_tracking_requests/D_2d_analysis).
    """
    from A_Config import sam3_masks_dir
    from D_2d_analysis import contiguous_durations

    print("initialising tracking")
    api_key = os.environ["ROBOFLOW_API_KEY"]
    client = InferenceHTTPClient.init(api_url="https://serverless.roboflow.com", api_key=api_key)

    masks_root = sam3_masks_dir()
    masks_root.mkdir(parents=True, exist_ok=True)

    analysis_2d_for_decisions = {}
    raw_detection_rows = []

    for subject, requests in tracking_requests_by_subject.items():
        subject_slug = _slugify_subject(subject)
        frames_by_instance = {}  # global track_id -> [frame_idx, ...]
        beat_ids_by_instance = {}  # global track_id -> set(beat_id)
        subject_beat_ids = set()  # union across all this subject's requests, for the no-detection case
        next_track_id = [1]  # mutable cell so on_data's closure can bump it

        for request in requests:
            span_start_s = request["start_s"]
            span_end_s = request["end_s"]
            request_beat_ids = request.get("beat_ids", [])
            subject_beat_ids.update(request_beat_ids)
            tmp_clip = masks_root / f"_span_with_handle_{subject_slug}.mp4"
            start_frame, fps, width, height = make_span_clip(video_path, tmp_clip, span_start_s, span_end_s)
            print(f"[{subject_slug}] requested {span_start_s}-{span_end_s}s -> keyframe-aligned start frame {start_frame}")

            source = VideoFileSource(str(tmp_clip), realtime_processing=False)
            config = StreamConfig(
                stream_output=[],
                data_output=["predictions"],
                realtime_processing=False,
                requested_plan=requested_plan,
                requested_region=requested_region,
                workflow_parameters={"class_names": subject, "threshold": threshold},
            )
            # track_id from SAM3 is scoped to this one streaming session, so it
            # restarts at 1 for every new span -- map it to the subject's
            # globally-continuing instance number instead, or span 2's "instance 1"
            # would silently overwrite span 1's "instance 1" on disk.
            local_to_global = {}
            for attempt in range(1, SAM3_SESSION_MAX_ATTEMPTS + 1):
                attempt_frames = {}  # global track_id -> [frame_idx, ...]
                try:
                    with client.webrtc.stream(
                        source=source, workflow=WORKFLOW, workspace=WORKSPACE,
                        image_input="image", config=config,
                    ) as session:

                        @session.on_data()
                        def on_data(data, metadata, start_frame=start_frame, width=width, height=height,
                                    masks_root=masks_root, attempt_frames=attempt_frames,
                                    subject_slug=subject_slug, local_to_global=local_to_global,
                                    next_track_id=next_track_id,
                                    raw_detection_rows=raw_detection_rows, _debug_count=[0]):
                            frame_idx = start_frame + int(metadata.frame_id) - 1
                            dets = unwrap_predictions(data.get("predictions"))
                            if _debug_count[0] < 3:
                                print(f"[DEBUG frame {frame_idx}] raw data: {data}")
                            else:
                                print(f"[DEBUG frame {frame_idx}] raw predictions count: {len(dets)}")
                            _debug_count[0] += 1
                            n_instances = 0
                            for d in dets:
                                raw_detection_rows.append({"frame_idx": frame_idx, "subject": subject_slug, **d})

                                mask = decode_mask_from_detection(d, image_shape=(height, width))
                                if mask is not None:
                                    local_id = extract_track_id(d)
                                    if local_id not in local_to_global:
                                        local_to_global[local_id] = next_track_id[0]
                                        next_track_id[0] += 1
                                    track_id = local_to_global[local_id]

                                    track_id_dir = masks_root / f"{subject_slug}-{track_id:02d}"
                                    track_id_dir.mkdir(parents=True, exist_ok=True)
                                    cv2.imwrite(str(track_id_dir / MASK_NAME_FMT.format(frame_idx)), mask)
                                    attempt_frames.setdefault(track_id, []).append(frame_idx)
                                    n_instances += 1

                        session.run()
                except Exception as e:
                    if attempt == SAM3_SESSION_MAX_ATTEMPTS:
                        raise
                    print(f"[{subject_slug}] SAM3 session attempt {attempt}/{SAM3_SESSION_MAX_ATTEMPTS} "
                          f"failed ({e!r}); retrying in {SAM3_SESSION_RETRY_BACKOFF_S}s")
                    time.sleep(SAM3_SESSION_RETRY_BACKOFF_S)
                    continue
                else:
                    for track_id, frames in attempt_frames.items():
                        frames_by_instance.setdefault(track_id, []).extend(frames)
                        beat_ids_by_instance.setdefault(track_id, set()).update(request_beat_ids)
                    break
            tmp_clip.unlink(missing_ok=True)

        if frames_by_instance:
            for track_id, instance_frames in frames_by_instance.items():
                instance_key = f"{subject_slug}-{track_id:02d}"
                seqs, best_idx = contiguous_durations(instance_frames, tolerance=15)
                entry = {
                    "Subject_frames_present": instance_frames,
                    "subject_first_frame": min(instance_frames),
                    "subject_last_frame": max(instance_frames),
                    "subject_duration_frames": max(instance_frames) - min(instance_frames),
                    "continuous frame sequences": seqs,
                    "best_seq_idx": best_idx,
                    "masks_dir": str(masks_root / instance_key),
                    "beat_ids": sorted(beat_ids_by_instance.get(track_id, set())),
                }
                analysis_2d_for_decisions[instance_key] = entry
        else:
            print(f"[{subject_slug}] no detections across any requested span")
            entry = {
                "Subject_frames_present": [],
                "subject_first_frame": None,
                "subject_last_frame": None,
                "continuous frame sequences": [],
                "best_seq_idx": None,
                "masks_dir": str(masks_root / subject_slug),
                "beat_ids": sorted(subject_beat_ids),
            }
            analysis_2d_for_decisions[subject_slug] = entry

    if raw_detection_rows:
        pd.DataFrame(raw_detection_rows).to_csv(masks_root / "detections.csv", index=False)

    if not any(v["Subject_frames_present"] for v in analysis_2d_for_decisions.values()):
        raise RuntimeError("SAM3 tracker found no detections across any beat")

    return analysis_2d_for_decisions
