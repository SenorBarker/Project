"""
Spike / feasibility test for Roboflow's visually-prompted SAM3 workflow
(sam3-timed-box-seed-tracker) -- confirms a box seed from people.json
actually locks onto and tracks the intended person, before any of that
gets wired into the real pipeline (A_ROBOFLOW_SAM3.py / render_paper_edit.py).

Not imported by anything else. Run directly:

    python spike_sam3_box_seed.py

Reuses find_preceding_keyframe_s/make_span_clip from A_ROBOFLOW_SAM3.py so
the clip-cutting behavior under test matches production exactly.
"""

import json
import os
from pathlib import Path

import cv2
from dotenv import load_dotenv

from inference_sdk import InferenceHTTPClient
from inference_sdk.webrtc import VideoFileSource, StreamConfig

from A_ROBOFLOW_SAM3 import find_preceding_keyframe_s, make_span_clip

load_dotenv(Path(__file__).parent.parent / ".env")

WORKSPACE = "david-barker-25-ucl-ac-uk"
WORKFLOW_SEEDED = "sam3-timed-box-seed-tracker-1786563524729"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VIDEO_PATH = REPO_ROOT / "Data" / "304_met_multi_4" / "010_source" / "04.mp4"
PEOPLE_JSON_PATH = (
    REPO_ROOT / "Data" / "304_met_multi_4" / "012_Gemini_outputs" / "304_met_multi_4_people.json"
)

SPAN_PADDING_S = 5.0  # how far past the seeded appearance's start_s to track

# A_gemini_v02.py's own convention: box_2d is reported "at start_s" but is
# actually sampled at the middle of that labeled second (start_s + 0.5s),
# not the literal start_s instant -- see draw_point_overlay's target_times_s.
SEED_TIME_OFFSET_S = 0.5


def box_2d_to_corners(box_2d, width, height):
    """people.json's box_2d is [y0, x0, y1, x1] normalized 0-1000 (same
    convention as A_gemini_v02.get_box/draw_point_overlay). The seeded
    workflow wants pixel-space "corners": [x1, y1, x2, y2]."""
    y0, x0, y1, x1 = [int(v / 1000 * d) for v, d in zip(box_2d, (height, width, height, width))]
    return [x0, y0, x1, y1]


def main():
    people = json.loads(PEOPLE_JSON_PATH.read_text())
    person = people[1]
    appearance = person["appearances"][0]
    person_id = person["person_id"]

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    span_start_s = float(appearance["start_s"])
    span_end_s = span_start_s + SPAN_PADDING_S
    corners = box_2d_to_corners(appearance["box_2d"], width, height)
    seed_time_s = span_start_s + SEED_TIME_OFFSET_S
    # Pass an explicit absolute frame number instead of time_s -- avoids
    # relying on Timed_Box_Seeds' own time_s-to-frame conversion (which
    # depends on it reading the same fps we do) and removes a second,
    # redundant floating-point conversion step from the picture entirely.
    seed_frame_number = round(seed_time_s * fps)

    print(
        f"seeding {person_id} ({person['descriptor']!r}) at t={seed_time_s}s "
        f"(absolute frame {seed_frame_number}, fps={fps}), corners={corners}"
    )

    tmp_clip = Path(__file__).parent / "_spike_span.mp4"
    clip_start_s = find_preceding_keyframe_s(VIDEO_PATH, span_start_s)
    make_span_clip(VIDEO_PATH, tmp_clip, span_start_s, span_end_s)

    seed_schedule = [
        {
            "frame_number": seed_frame_number,
            "absolute_frame": True,
            "corners": corners,
            "subject_id": person_id,
            # give it slack against off-by-one frame-counter mismatches
            # between our target-frame math and however the stream numbers
            # frames.
            "frame_tolerance": 3,
        }
    ]

    api_key = os.environ["ROBOFLOW_API_KEY"]
    client = InferenceHTTPClient.init(api_url="https://serverless.roboflow.com", api_key=api_key)

    source = VideoFileSource(str(tmp_clip), realtime_processing=False)
    config = StreamConfig(
        stream_output=[],
        data_output=["predictions", "seed_boxes", "frame_number"],
        realtime_processing=False,
        requested_plan="webrtc-gpu-large",
        requested_region="us",
        workflow_parameters={
            "seed_schedule": seed_schedule,
            "clip_start_s": clip_start_s,
            "mask_threshold": 0.0,
        },
    )

    seen_frames = []

    try:
        with client.webrtc.stream(
            source=source, workflow=WORKFLOW_SEEDED, workspace=WORKSPACE,
            image_input="image", config=config,
        ) as session:

            @session.on_data()
            def on_data(data, metadata):
                if data is None:
                    return  # transient startup callback before any predictions exist
                frame_number = data.get("frame_number")
                seen_frames.append(frame_number)
                print(
                    f"clip_frame={frame_number} "
                    f"predictions={data.get('predictions')} "
                    f"seed_boxes={data.get('seed_boxes')}"
                )

            session.run()
    finally:
        tmp_clip.unlink(missing_ok=True)

    print(f"\ntotal frames received: {len(seen_frames)}")
    if not seen_frames:
        print("NO DATA RECEIVED -- workflow parameters may be wrong, check above for errors")


if __name__ == "__main__":
    main()
