"""
Track one or more subject classes across a frame range of a video using YOLO
instance segmentation + tracking, and write one combined binary mask per frame.

Mask filenames use the absolute source frame number, not a re-indexed count --
so mask N lines up directly with source/extracted frame N, even when
start_frame > 0.
"""

from pathlib import Path
import math
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from Two2D.B_video_processing import seek_exact

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "Models/yolo_checkpoints/yolo26l-seg.pt"
MASK_NAME_FMT = "{:04d}.png"


def _get_subject_ids(model, subjects):
    name_to_id = {name: idx for idx, name in model.names.items()}
    missing = [s for s in subjects if s not in name_to_id]
    if missing:
        raise ValueError(f"Unknown class name(s): {missing}. Available: {sorted(name_to_id)}")
    return [name_to_id[s] for s in subjects]


def _detect_in_frame(model, cap, frame_idx, subject_ids, h, w, conf):
    if not seek_exact(cap, frame_idx):
        return False
    ret, frame = cap.read()
    if not ret:
        return False
    r = model.predict(frame, classes=subject_ids, conf=conf,
                      imgsz=(h, w), verbose=False)[0]
    return r.masks is not None and len(r.masks.data) > 0

#we can't trust Gemini's timecodes, so this searches wider. it wastes time, but it's needed
def _expand_bounds(model, cap, total_frames, subject_ids, h, w,
                   hint_start, hint_end, miss_threshold, conf):
    subject_start = hint_start
    misses = 0
    print("Seeking subject start frame")
    for idx in range(hint_start - 1, -1, -1):
        if _detect_in_frame(model, cap, idx, subject_ids, h, w, conf):
            subject_start = idx
            misses = 0
        else:
            misses += 1
            if misses >= miss_threshold:
                break

    subject_end = hint_end
    misses = 0
    print("Seeking subject end frame")
    for idx in range(hint_end + 1, total_frames):
        if _detect_in_frame(model, cap, idx, subject_ids, h, w, conf):
            subject_end = idx
            misses = 0
        else:
            misses += 1
            if misses >= miss_threshold:
                break

    print(f"Subject frames found: {subject_start} → {subject_end}")
    return subject_start, subject_end


def _run_tracking(model, cap, out_dir, subject_ids, h, w,
                  start_frame, end_frame, tracker, conf):
    model.predictor = None  # reset tracker state for each segment
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seek_exact(cap, start_frame)



    results = []
    for frame_idx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            print(f"  Warning: could not read frame {frame_idx}, stopping.")
            break

        r = model.track(
            frame, persist=True, tracker=tracker, classes=subject_ids, conf=conf,
            retina_masks=True, imgsz=(h, w), verbose=False,
        )[0]

        if r.masks is not None and len(r.masks.data):
            n_instances = len(r.masks.data)
            mask = (r.masks.data.cpu().numpy() > 0.5).any(axis=0).astype(np.uint8)
        else:
            n_instances = 0
            mask = np.zeros((h, w), dtype=np.uint8)

        cv2.imwrite(str(out_dir / MASK_NAME_FMT.format(frame_idx)), mask * 255)
        results.append({"frame": frame_idx, "n_instances": n_instances})

    return results


def track_subject_masks(
    video_path: str,
    output_dir: str,
    subjects: str | list[str],
    start_frame: int = 0,
    end_frame: int | None = None,
    model_path: str = MODEL_PATH,
    tracker: str = "bytetrack.yaml",
    conf: float = 0.25,
) -> list[dict]:
    """
    Parameters
    ----------
    subjects    Class name, or list of class names (e.g. "person", ["person", "cat"]) --
                only instances of these classes are tracked/masked.
    start_frame First frame (inclusive) to process.
    end_frame   Last frame (exclusive) to process. Defaults to end of video.
    conf        Minimum detection confidence (0-1). Lower this if a real instance
                is being missed; ultralytics' own default is 0.25.

    All matching instances in a frame are combined into a single binary mask
    (union) -- per-instance/track separation is not kept for now.
    """
    if isinstance(subjects, str):
        subjects = [subjects]

    model = YOLO(model_path)
    model.to(DEVICE)
    subject_ids = _get_subject_ids(model, subjects)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    h = math.ceil(h / 32) * 32 # make it match YOLO's stride 
    w = math.ceil(w / 32) * 32


    start_frame = max(0, start_frame)
    end_frame = total_frames if end_frame is None else min(total_frames, end_frame)
    if start_frame >= end_frame:
        raise ValueError(f"start_frame ({start_frame}) must be before end_frame ({end_frame})")

    results = _run_tracking(model, cap, output_dir, subject_ids, h, w,
                            start_frame, end_frame, tracker, conf)
    cap.release()
    print(f"Wrote {len(results)} masks to {output_dir}")
    return results


def track_subject_masks_from_hints(
    video_path: str,
    output_dir: str,
    detections: list[dict],
    miss_threshold: int = 30,
    model_path: str = MODEL_PATH,
    tracker: str = "bytetrack.yaml",
    conf: float = 0.25,
) -> list[dict]:
    """
    Expands Gemini-provided detection hints to find true frame boundaries,
    then runs full tracking on each expanded range.

    Parameters
    ----------
    detections      List of dicts with 'subject', 'start_s' and 'end_s' (integer seconds),
                    as returned by the Gemini query wrapper / subject_selector.
    miss_threshold  Consecutive missed detections before the boundary search stops.
                    Must be larger than the longest gap in the subject's appearance.
    """
    model = YOLO(model_path)
    model.to(DEVICE)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    all_results = []
    subject_start, subject_end = None, None
    for det in detections: # this does separate detections each time
        subject = det["subject"].lower().removeprefix("the ").removeprefix("a ").removeprefix("an ").strip()
        subject_ids = _get_subject_ids(model, [subject])
        hint_start = int(det["start_s"] * fps)
        hint_end   = int(det["end_s"]   * fps)

        subject_start, subject_end = _expand_bounds(
            model, cap, total_frames, subject_ids, h, w,
            hint_start, hint_end, miss_threshold, conf,
        )
        print(f"  Hint [{hint_start}-{hint_end}] → subject [{subject_start}-{subject_end}]")

        results = _run_tracking(model, cap, output_dir, subject_ids, h, w,
                                subject_start, subject_end + 1, tracker, conf)
        all_results.extend(results)

    cap.release()
    print(f"Wrote {len(all_results)} masks total to {output_dir}")
    return {"results": all_results, "subject_start": subject_start, "subject_end": subject_end}


if __name__ == "__main__":
    track_subject_masks(
        video_path="path/to/video.mp4",
        output_dir="path/to/masks",
        subjects=["person"],
        start_frame=0,
        end_frame=None,
    )
