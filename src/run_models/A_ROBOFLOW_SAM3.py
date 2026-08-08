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

load_dotenv(Path(__file__).parent.parent / ".env")

# Quiet the WebRTC SDK's own INFO-level connection-state logging (ICE/connection
# state changes) -- purely noise for this use case, not something we act on.
logging.getLogger("inference_sdk").setLevel(logging.WARNING)

WORKSPACE = "david-barker-25-ucl-ac-uk"
WORKFLOW = "sam3-prompted-video-tracker-1784979867494"
MASK_NAME_FMT = "{:04d}.png"  # matches A_YOLO_seg / W_video_editor's convention

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


def track_subject_sam3(video_path, threshold=0.5,
                        requested_region="us", requested_plan="webrtc-gpu-large"):
    """
    Pipeline version of this Self-contained: 
    
    loads the current case's paper edit + derived flags from disk 

    Reads the DRAFT paper edit, not the final one

    Writes one MASK_NAME_FMT-named PNG per detection into
    A_Config.sam3_masks_dir()/<beat_id>/<subject>-<track_id>/ -- one flat
    folder per beat, with a subfolder per (subject, tracked-object ID) pair,
    never merged with another pair's masks, so multiple simultaneous
    detections (e.g. two people in frame, or two different subjects
    requested in the same beat) don't overwrite each other. Filenames are
    the absolute source-video frame number (start_frame + metadata.frame_id
    - 1, see below). `beat_id` (not `order`) scopes the folder because
    "beat order" isn't a stable identity: the Producer's revision pass can
    renumber/merge beats, which orphaned already-generated mask folders
    under the old beat{order:02d}/ scheme (see git history). The
    subject-prefixed track_id folder name exists because raw SAM3 track IDs
    reset per subject-query session, so two subjects in one beat could
    otherwise collide on the same raw ID. Also writes every raw detection
    record (unfiltered) to sam3_masks_dir()/detections.csv.

    Returns None (and prints "Cell disabled") if MASK_OVERLAYS wasn't
    requested. Otherwise returns analysis_2d_for_decisions: a dict keyed by
    `beat_id` (stable, not `order`), each value built directly from the
    frames this run just wrote for that beat (no re-reading masks off disk)
    -- {beat_id: {"Subject_frames_present", "subject_first_frame",
             "subject_last_frame", "subject_duration_frames",
             "continuous frame sequences", "best_seq_idx", "masks_dir"}}.
    A beat with zero detections across all its spans still gets an entry
    (all Nones/empty), rather than aborting the whole run.
    """
    import json
    import sys
    from collections import defaultdict
    from A_Config import REPO_ROOT, case_dir, case_name

    sys.path.insert(0, str(Path(REPO_ROOT) / "src" / "claude_agent"))
    #-------------------get data----------------
    print("accessing data")
    from render_paper_edit import build_tracking_requests

    paper_edit_json_path = case_dir() / "012_agent_p_output" / f"{case_name()}_paper_edit_draft.json"

    flags_path = paper_edit_json_path.with_name(
        paper_edit_json_path.stem.replace("_paper_edit", "") + "_flags.json"
    )
    current_flags = json.loads(flags_path.read_text(encoding="utf-8"))

    if not current_flags["MASK_OVERLAYS"]:
        print("Cell disabled")
        return None

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    tracking_requests = build_tracking_requests(paper_edit_json_path, fps)
    tracking_requests_by_beat = defaultdict(list)
    for request in tracking_requests:
        tracking_requests_by_beat[request["beat_id"]].append(request)

    return _run_sam3_tracking(video_path, tracking_requests_by_beat, threshold, requested_region, requested_plan)


def run_sam3_manual(video_path, subject, start_frame=None, end_frame=None,
                     threshold=0.5, requested_region="us", requested_plan="webrtc-gpu-large"):
    """
    Manual version of this: ad-hoc SAM3 run against a single video file for
    one subject string -- no paper edit / case involved. start_frame/end_frame
    optionally restrict the run to a sub-range (inclusive); default is the
    whole video. Builds a single-span tracking_requests_by_beat, keyed by a
    subject slug (no paper_edit lifecycle here, so there's no beat_id --
    the subject string is the only stable-enough identity), and runs it
    through the same tracking core track_subject_sam3 uses (including the
    make_span_clip keyframe-aligned cut), so masks/report/return shape are
    identical.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    start_frame = 0 if start_frame is None else start_frame
    end_frame = total_frames - 1 if end_frame is None else end_frame

    start_s = start_frame / fps
    end_s = (end_frame + 1) / fps  # end_frame is inclusive; make_span_clip's end_s is not

    beat_id = _slugify_subject(subject)
    tracking_requests_by_beat = {beat_id: [{"subject": subject, "start_s": start_s, "end_s": end_s}]}

    return _run_sam3_tracking(video_path, tracking_requests_by_beat, threshold, requested_region, requested_plan)


def _run_sam3_tracking(video_path, tracking_requests_by_beat, threshold, requested_region, requested_plan):
    """
    Shared tracking core for track_subject_sam3 and run_sam3_manual: given a
    video and a {beat_id: [{"subject", "start_s", "end_s"}, ...]} dict of
    tracking *requests* (queries to search for -- not detections, which is
    what SAM3 actually finds and writes), cuts a subclip per span, streams
    each through the SAM3 workflow, writes masks, and builds/reports the
    same analysis_2d_for_decisions shape either caller returns (keyed by
    beat_id, not order -- see track_subject_sam3's docstring for why).
    """
    from A_Config import sam3_masks_dir
    from C_CSV_report import add_to_report
    from D_2d_analysis import contiguous_durations

    #--------do tracking-------------
    print("initialising tracking")
    api_key = os.environ["ROBOFLOW_API_KEY"]
    client = InferenceHTTPClient.init(api_url="https://serverless.roboflow.com", api_key=api_key)

    masks_root = sam3_masks_dir()
    masks_root.mkdir(parents=True, exist_ok=True)

    analysis_2d_for_decisions = {}
    raw_detection_rows = []

    for beat_id, beat_requests in tracking_requests_by_beat.items():
        beat_mask_dir = masks_root / beat_id
        beat_mask_dir.mkdir(parents=True, exist_ok=True)
        beat_frames = []

        # Every subject in a beat shares that beat's own time range (see
        # build_tracking_requests) -- cut the source span once per beat, not
        # once per subject, so N subjects in one beat don't redundantly
        # re-cut (and immediately discard) the identical clip N times.
        span_start_s = beat_requests[0]["start_s"]
        span_end_s = beat_requests[0]["end_s"]
        if any(r["start_s"] != span_start_s or r["end_s"] != span_end_s for r in beat_requests):
            raise ValueError(
                f"[{beat_id}] subjects disagree on time span "
                f"({[(r['subject'], r['start_s'], r['end_s']) for r in beat_requests]}) -- "
                "expected every subject in a beat to share the beat's own span."
            )
        tmp_clip = beat_mask_dir / "_span_with_handle.mp4"
        start_frame, fps, width, height = make_span_clip(video_path, tmp_clip, span_start_s, span_end_s)
        print(f"[{beat_id}] requested {span_start_s}-{span_end_s}s -> keyframe-aligned start frame {start_frame}")

        for request in beat_requests:
            subject = request["subject"]
            subject_slug = _slugify_subject(subject)
            print("SAM seeks subject:", subject)

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
            #
            # Session init against Roboflow's serverless WebRTC endpoint is flaky
            # (read timeouts, intermittent 500s) regardless of region/plan, so
            # retry the whole session rather than failing the beat on one bad
            # attempt. Frames are only committed to beat_frames after a fully
            # successful attempt, so a retry can't double-count frames from a
            # partially-completed prior attempt.
            for attempt in range(1, SAM3_SESSION_MAX_ATTEMPTS + 1):
                attempt_frames = []
                try:
                    with client.webrtc.stream(
                        source=source, workflow=WORKFLOW, workspace=WORKSPACE,
                        image_input="image", config=config,
                    ) as session:

                        @session.on_data()
                        def on_data(data, metadata, start_frame=start_frame, width=width, height=height,
                                    beat_mask_dir=beat_mask_dir, attempt_frames=attempt_frames,
                                    beat_id=beat_id, subject_slug=subject_slug,
                                    raw_detection_rows=raw_detection_rows, _debug_count=[0]):
                            frame_idx = start_frame + int(metadata.frame_id) - 1
                            dets = unwrap_predictions(data.get("predictions"))
                            # DEBUG: full raw payload for the first 3 frames, one-line
                            # summary thereafter -- to see whether the server is
                            # returning empty predictions vs an error/warning embedded
                            # in the response, without flooding stdout for the whole clip.
                            if _debug_count[0] < 3:
                                print(f"[DEBUG frame {frame_idx}] raw data: {data}")
                            else:
                                print(f"[DEBUG frame {frame_idx}] raw predictions count: {len(dets)}")
                            _debug_count[0] += 1
                            #turn detections into actual masks -- one file per detection,
                            #in its own subject+track-ID-named folder, never merged with
                            #another detection or another subject's session
                            n_instances = 0
                            for d in dets:
                                raw_detection_rows.append({"frame_idx": frame_idx, "beat_id": beat_id, "subject": subject_slug, **d})

                                mask = decode_mask_from_detection(d, image_shape=(height, width))
                                if mask is not None:
                                    track_id_dir = beat_mask_dir / f"{subject_slug}-{extract_track_id(d)}"
                                    track_id_dir.mkdir(parents=True, exist_ok=True)
                                    cv2.imwrite(str(track_id_dir / MASK_NAME_FMT.format(frame_idx)), mask)
                                    n_instances += 1

                            if n_instances:
                                attempt_frames.append(frame_idx)

                        session.run()
                except Exception as e:
                    if attempt == SAM3_SESSION_MAX_ATTEMPTS:
                        raise
                    print(f"[{beat_id}] SAM3 session attempt {attempt}/{SAM3_SESSION_MAX_ATTEMPTS} "
                          f"failed ({e!r}); retrying in {SAM3_SESSION_RETRY_BACKOFF_S}s")
                    time.sleep(SAM3_SESSION_RETRY_BACKOFF_S)
                    continue
                else:
                    beat_frames.extend(attempt_frames)
                    break
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
            print(f"[{beat_id}] no detections across any requested span")
            entry = {
                "Subject_frames_present": [],
                "subject_first_frame": None,
                "subject_last_frame": None,
                "continuous frame sequences": [],
                "best_seq_idx": None,
                "masks_dir": str(beat_mask_dir),
            }

        analysis_2d_for_decisions[beat_id] = entry
        add_to_report({
            f"{beat_id}_subject_first_frame": entry["subject_first_frame"],
            f"{beat_id}_subject_last_frame": entry["subject_last_frame"],
        })

    if raw_detection_rows:
        pd.DataFrame(raw_detection_rows).to_csv(masks_root / "detections.csv", index=False)

    if not any(v["Subject_frames_present"] for v in analysis_2d_for_decisions.values()):
        raise RuntimeError("SAM3 tracker found no detections across any beat")

    return analysis_2d_for_decisions
