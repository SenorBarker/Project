"""
Spike / feasibility test for LOCAL SAM3 (Models/sam3) -- confirms the
checkout actually loads on this box's GPU and tracks a text-prompted subject
through an mp4, before any of that gets wired into the real pipeline
(A_LOCAL_SAM3.py / A_ROBOFLOW_SAM3.py).

Answers the questions the README can't: does the checkpoint load under this
env's torch 2.6 / numpy 2 (both below what the README asks for), does the
cv2 video loader eat make_span_clip's output directly, does an 848M-param
model plus a decoded clip fit in the 3090's 24GB, and are the tracker's
object IDs stable frame to frame.

Not imported by anything else. Run directly:

    /opt/conda/envs/Msc2/bin/python spike_sam3_local.py

Reuses make_span_clip from A_ROBOFLOW_SAM3.py so the clip-cutting behavior
under test matches production exactly.
"""

import time
from pathlib import Path

import cv2
import numpy as np

from A_ROBOFLOW_SAM3 import make_span_clip

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CASE_NAME = "304_met_multi_4"
VIDEO_PATH = REPO_ROOT / "Data" / CASE_NAME / "010_source" / "04.mp4"

SUBJECT = "person"
SPAN_START_S = 0.0
SPAN_LEN_S = 5.0  # keep it short -- this is a "does it run at all" check, not a benchmark
THRESHOLD = 0.5

# Write masks for only a handful of frames: enough to eyeball that they're
# real silhouettes and not noise, without dumping a folder of PNGs to review.
N_MASKS_TO_SAVE = 3

OUT_DIR = REPO_ROOT / "Data" / CASE_NAME / "015_SAM3_masks" / "_spike_local"


def main():
    from sam3.model_builder import build_sam3_video_predictor

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_clip = OUT_DIR / "_spike_span.mp4"

    start_frame, fps, width, height = make_span_clip(
        VIDEO_PATH, tmp_clip, SPAN_START_S, SPAN_START_S + SPAN_LEN_S
    )
    print(
        f"clip: requested {SPAN_START_S}-{SPAN_START_S + SPAN_LEN_S}s -> "
        f"keyframe-aligned start frame {start_frame}, {width}x{height} @ {fps}fps"
    )

    t0 = time.time()
    predictor = build_sam3_video_predictor()
    print(f"model loaded in {time.time() - t0:.1f}s")

    session_id = None
    try:
        response = predictor.handle_request(
            request=dict(type="start_session", resource_path=str(tmp_clip))
        )
        session_id = response["session_id"]
        print(f"session {session_id} started")

        t0 = time.time()
        response = predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text=SUBJECT,
                output_prob_thresh=THRESHOLD,
            )
        )
        prompt_out = response["outputs"]
        print(
            f"prompt {SUBJECT!r} on frame 0 in {time.time() - t0:.1f}s -> "
            f"obj_ids={np.asarray(prompt_out['out_obj_ids']).tolist()}"
        )

        # What the pipeline will actually consume: per-frame obj IDs + binary
        # masks. Track IDs staying put across frames is the thing that makes
        # this usable as a tracker rather than a per-frame detector.
        t0 = time.time()
        n_frames = 0
        saved = 0
        ids_per_frame = []
        for response in predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
                propagation_direction="forward",
                start_frame_index=0,
                output_prob_thresh=THRESHOLD,
            )
        ):
            clip_frame_idx = response["frame_index"]
            outputs = response["outputs"]
            obj_ids = np.asarray(outputs["out_obj_ids"]).tolist()
            masks = np.asarray(outputs["out_binary_masks"])
            areas = [int(m.sum()) for m in masks]
            ids_per_frame.append(tuple(obj_ids))
            n_frames += 1

            # propagate_in_video's frame_index is 0-based, unlike the Roboflow
            # workflow's 1-based metadata.frame_id -- so absolute frame is a
            # plain add, with no -1 correction. Getting this wrong offsets
            # every mask filename by one, so print it here to check against
            # the source video.
            abs_frame_idx = start_frame + clip_frame_idx
            print(f"clip_frame={clip_frame_idx} abs_frame={abs_frame_idx} obj_ids={obj_ids} areas={areas}")

            if saved < N_MASKS_TO_SAVE and len(masks) > 0:
                mask_u8 = (masks[0].astype(np.uint8) * 255)
                cv2.imwrite(str(OUT_DIR / f"{abs_frame_idx:04d}.png"), mask_u8)
                saved += 1

        elapsed = time.time() - t0
        print(f"\npropagated {n_frames} frames in {elapsed:.1f}s ({n_frames / max(elapsed, 1e-6):.2f} fps)")
        print(f"wrote {saved} sample masks to {OUT_DIR}")

        if not n_frames:
            print("NO FRAMES PROPAGATED -- session or prompt failed, check above for errors")
        elif not any(ids for ids in ids_per_frame):
            print(f"NO DETECTIONS for {SUBJECT!r} -- try a different subject or a lower threshold")
        else:
            distinct = sorted({i for ids in ids_per_frame for i in ids})
            n_with_dets = sum(1 for ids in ids_per_frame if ids)
            print(
                f"distinct obj_ids across the span: {distinct} "
                f"({n_with_dets}/{n_frames} frames had at least one detection)"
            )
    finally:
        if session_id is not None:
            predictor.handle_request(request=dict(type="close_session", session_id=session_id))
        tmp_clip.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
