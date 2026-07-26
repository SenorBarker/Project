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

import logging
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

from inference_sdk import InferenceHTTPClient
from inference_sdk.webrtc import VideoFileSource, StreamConfig

load_dotenv(Path(__file__).parent.parent / ".env")

# Quiet the WebRTC SDK's own INFO-level connection-state logging (ICE/connection
# state changes) -- purely noise for this use case, not something we act on.
logging.getLogger("inference_sdk").setLevel(logging.WARNING)

WORKSPACE = "david-barker-25-ucl-ac-uk"
WORKFLOW = "sam3-prompted-video-tracker-1784979867494"
MASK_NAME_FMT = "{:04d}.png"  # matches A_YOLO_seg / W_video_editor's convention


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


def make_span_clip(video_path, out_path, start_s, end_s):
    """
    Creates a temporary video containing only [start_s, end_s], using ffmpeg
    stream copy (-c copy) -- no decode/re-encode, so no quality loss, unlike
    writing frames back out through cv2.VideoWriter. Stream copy can only cut
    on a keyframe boundary, so we explicitly resolve the actual keyframe
    ffmpeg will snap to (find_preceding_keyframe_s) and use THAT as the true
    clip start -- rather than assuming the clip starts exactly at start_s,
    which would silently offset every downstream frame number.

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

    start_frame = max(0, int(round(actual_start_s * fps)))

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


def track_subject_sam3(video_path, threshold=0.5,
                        requested_region="us", requested_plan="webrtc-gpu-large"):
    """
    Self-contained: loads the current case's paper edit + derived flags
    fresh from disk (not from in-memory notebook variables)

    Reads the DRAFT paper edit, not the final one

    Writes one MASK_NAME_FMT-named PNG per frame with a detected mask into
    A_Config.sam3_masks_dir()/beat{order:02d}/ -- a separate subfolder per
    beat, since a beat can have multiple spans and different beats must not
    share mask frame numbers.

    Returns None (and prints "Cell disabled") if MASK_OVERLAYS wasn't
    requested. Otherwise returns analysis_2d_for_decisions: a dict keyed by
    beat "order", each value built directly from the frames this run just
    wrote for that beat (no re-reading masks off disk) --
    {order: {"Subject_frames_present", "subject_first_frame",
             "subject_last_frame", "subject_duration_frames",
             "continuous frame sequences", "best_seq_idx", "masks_dir"}}.
    A beat with zero detections across all its spans still gets an entry
    (all Nones/empty), rather than aborting the whole run.
    """
    import json
    import sys
    from collections import defaultdict
    from A_Config import REPO_ROOT, case_dir, case_name, sam3_masks_dir
    from C_CSV_report import add_to_report
    from D_2d_analysis import contiguous_durations

    sys.path.insert(0, str(Path(REPO_ROOT) / "src" / "claude_agent"))
    #-------------------get data----------------
    print("accessing data")
    from render_paper_edit import build_tracking_detections

    paper_edit_json_path = case_dir() / "012_agent_p_output" / f"{case_name()}_paper_edit_draft.json"

    flags_path = paper_edit_json_path.with_name(
        paper_edit_json_path.stem.replace("_paper_edit", "") + "_flags.json"
    )
    current_flags = json.loads(flags_path.read_text(encoding="utf-8"))

    if not current_flags["MASK_OVERLAYS"]:
        print("Cell disabled")
        return None

    detections = build_tracking_detections(paper_edit_json_path)
    detections_by_beat = defaultdict(list)
    for det in detections:
        detections_by_beat[det["order"]].append(det)

    #--------do tracking-------------
    print("initialising tracking")
    api_key = os.environ["ROBOFLOW_API_KEY"]
    client = InferenceHTTPClient.init(api_url="https://serverless.roboflow.com", api_key=api_key)

    masks_root = sam3_masks_dir()
    masks_root.mkdir(parents=True, exist_ok=True)

    analysis_2d_for_decisions = {}

    for order, beat_detections in detections_by_beat.items():
        beat_mask_dir = masks_root / f"beat{order:02d}"
        beat_mask_dir.mkdir(parents=True, exist_ok=True)
        beat_frames = []

        for det in beat_detections:
            subject = det["subject"]
            start_s = det["start_s"]
            end_s = det["end_s"]
            print(f"[beat {order}] making subclip {subject}")
            tmp_clip = beat_mask_dir / "_span_with_handle.mp4"
            start_frame, fps, width, height = make_span_clip(video_path, tmp_clip, start_s, end_s)
            print(f"[beat {order}] requested {start_s}-{end_s}s -> keyframe-aligned start frame {start_frame}")

            source = VideoFileSource(str(tmp_clip), realtime_processing=False)
            config = StreamConfig(
                stream_output=[],
                data_output=["predictions"],
                realtime_processing=False,
                requested_plan=requested_plan,
                requested_region=requested_region,
                workflow_parameters={"class_names": subject, "threshold": threshold},
            )
            # Context manager guarantees close() runs even if session init/run fails
            # partway through -- without it, a failed/interrupted run (e.g. during
            # repeated debugging reruns) can leave a stale session/allocation open
            # server-side instead of being torn down.
            with client.webrtc.stream(
                source=source, workflow=WORKFLOW, workspace=WORKSPACE,
                image_input="image", config=config,
            ) as session:

                @session.on_data()
                def on_data(data, metadata, start_frame=start_frame, width=width, height=height,
                            beat_mask_dir=beat_mask_dir, beat_frames=beat_frames):
                    frame_idx = start_frame + int(metadata.frame_id) - 1
                    dets = unwrap_predictions(data.get("predictions"))
                    #turn detections into actual masks
                    n_instances = 0
                    for d in dets:
                        mask = decode_mask_from_detection(d, image_shape=(height, width))
                        if mask is not None:
                            cv2.imwrite(str(beat_mask_dir / MASK_NAME_FMT.format(frame_idx)), mask)
                            n_instances += 1

                    if n_instances:
                        beat_frames.append(frame_idx)

                session.run()
            tmp_clip.unlink(missing_ok=True)

        if beat_frames:
            seqs, best_idx = contiguous_durations(beat_frames, tolerance=15)
            entry = {
                "Subject_frames_present": beat_frames,
                "subject_first_frame": min(beat_frames),
                "subject_last_frame": max(beat_frames),
                "subject_duration_frames": max(beat_frames) - min(beat_frames),
                "continuous frame sequences": seqs,
                "best_seq_idx": best_idx,
                "masks_dir": str(beat_mask_dir),
            }
        else:
            print(f"[beat {order}] no detections across any requested span")
            entry = {
                "Subject_frames_present": [],
                "subject_first_frame": None,
                "subject_last_frame": None,
                "continuous frame sequences": [],
                "best_seq_idx": None,
                "masks_dir": str(beat_mask_dir),
            }

        analysis_2d_for_decisions[order] = entry
        add_to_report({
            f"beat{order:02d}_subject_first_frame": entry["subject_first_frame"],
            f"beat{order:02d}_subject_last_frame": entry["subject_last_frame"],
        })

    if not any(v["Subject_frames_present"] for v in analysis_2d_for_decisions.values()):
        raise RuntimeError("SAM3 tracker found no detections across any beat")

    return analysis_2d_for_decisions
