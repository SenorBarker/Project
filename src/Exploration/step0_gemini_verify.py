"""
Throwaway step-0 verification -- run this against a real case, do NOT wire into
the pipeline yet. Confirms the design before Q_gemini_seed_masks work starts:

  (a) does the <machine> JSON block convention work cleanly for a "list every
      distinct person + appearance windows" query against a cached video --
      and can that SAME call also return each appearance's box_2d directly,
      so nothing downstream has to re-identify a person from a lossy
      descriptor string in a second call?
  (b) box-only detection (no mask) against the cached video + a timestamp --
      confirmed fast (~6s) vs asking for an inline base64 mask (~4min, because
      the model has to generate the image as text output token-by-token).
      SAM3's point-prompt refinement only needs a point, not a full mask, so
      we derive a center point from the box ourselves in plain code instead
      of asking the LLM to render an image.

Results get dumped to 080_experiments/GEMMASKTEST for review -- nothing here
touches query_dir/, sam3_masks_dir(), or any real pipeline output.
"""
import os
import json
import cv2
import numpy as np

from A_Config import set_case, case_dir
from Two2D.B_video_processing import frames_at_times
from google import genai
from google.genai import types

# ---- EDIT THESE ----
CASE_NAME = "304_met_multi_4"          # pick a case with multiple similar people in shot
VIDEO_PATH = None                       # e.g. case_dir()/"010_source"/"04.mp4" -- set after set_case
TEST_TIMESTAMP_S = 10                   # a timestamp you know has a person visible, for part (b)
# ---------------------

print("=== step0_gemini_verify.py: BOX-ONLY VERSION (no mask requested) ===")

set_case(CASE_NAME)
out_dir = case_dir() / "080_experiments" / "GEMMASKTEST"
out_dir.mkdir(parents=True, exist_ok=True)

if VIDEO_PATH is None:
    src_dir = case_dir() / "010_source" / "burnin"
    candidates = list(src_dir.glob("*.mp4"))
    assert candidates, f"no .mp4 found in {src_dir}, set VIDEO_PATH manually"
    VIDEO_PATH = candidates[0]
    print(f"using video: {VIDEO_PATH}")

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

# --- upload + cache the video once ---
print("uploading video...")
myfile = client.files.upload(file=str(VIDEO_PATH))
while myfile.state.name != "ACTIVE":
    import time; time.sleep(5)
    myfile = client.files.get(name=myfile.name)

cache = client.caches.create(
    model="gemini-3.5-flash",
    config=types.CreateCachedContentConfig(contents=[myfile], ttl="600s"),
)
cache_name = cache.name
print(f"cache created: {cache_name}")
(out_dir / "cache_name.txt").write_text(cache_name)

# --- part (a): people + appearance windows + box_2d, all in one <machine> block ---
PEOPLE_SUFFIX = """
After your response, append a <machine> block with this exact JSON:
<machine>
{"people": [{"person_id": "...", "descriptor": "...", "appearances": [{"start_s": 0, "end_s": 0, "box_2d": [0, 0, 0, 0]}]}]}
</machine>
Give every distinct person who appears anywhere in the video a stable person_id
(short, e.g. "person-1", "person-2") -- the same person must keep the same person_id
across every appearance, even in separate, non-contiguous time windows. "descriptor"
is a short human-readable description (role, clothing, position) to help a human
tell people apart, but is not used to identify them programmatically -- person_id is
the only identity key. List every time window (start_s, end_s, integer seconds) that
person is visible, across the whole video, as separate entries in "appearances". For
each appearance also give "box_2d": that person's 2D bounding box (normalized 0-1000,
[y0,x0,y1,x1]) at start_s of that window. Box only, no mask.
Separately, return a list of the numbers you see in the bottom left corner - bright green numbers. every one you see
"""

print("\n--- part (a): people query ---")
people_response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents="Identify every distinct person visible in this video." + PEOPLE_SUFFIX,
    config=types.GenerateContentConfig(cached_content=cache_name),
)
raw_people = people_response.text
print(raw_people)
(out_dir / "people_raw_response.txt").write_text(raw_people)

human_text, _, machine_block = raw_people.partition("<machine>")
json_str = machine_block.partition("</machine>")[0].strip()
try:
    people = json.loads(json_str)["people"]
    (out_dir / "people.json").write_text(json.dumps(people, indent=2))
    print(f"\nparsed OK -- {len(people)} people found")
except Exception as e:
    print(f"\nPARSE FAILED: {e!r}")
    people = []


def get_box(det):
    """Model has been observed emitting either 'box_2d' or 'box' as the key --
    handle both rather than assuming one."""
    return det.get("box_2d") or det.get("box")


def draw_point_overlay(dets, native_frame, out_path):
    """No mask decoding at all -- just draws the box and its center point
    (the actual SAM3 point-prompt seed) on the frame for visual inspection."""
    h, w = native_frame.shape[:2]
    overlay = native_frame.copy()
    for det in dets:
        box = get_box(det)
        if box is None:
            print(f"  detection has no box_2d/box field: {det}")
            continue
        y0, x0, y1, x1 = [int(v / 1000 * d) for v, d in zip(box, (h, w, h, w))]
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.circle(overlay, (cx, cy), 8, (0, 255, 0), -1)
        print(f"  box={box} -> center point=({cx},{cy})")
    cv2.imwrite(str(out_path), overlay)
    print(f"  overlay saved: {out_path}")


# --- one frame per person, straight from part (a)'s box_2d -- no second
#     Gemini call, no re-identifying anyone from their descriptor text. ---
if people:
    print("\n--- per-person overlays from part (a) box_2d ---")
    N = 25
    #get 25 frames per person, 
    target_times_s = [person["appearances"][0]["start_s"] + i / N
                       for person in people for i in range(N)]
    all_frames = frames_at_times(VIDEO_PATH, target_times_s)
    for p_idx, person in enumerate(people):
        person_id = person["person_id"]
        box = get_box(person["appearances"][0])
        if box is None:
            print(f"  [{person_id}] first appearance has no box_2d -- skipping")
            continue
        person_dir = out_dir / person_id
        person_dir.mkdir(parents=True, exist_ok=True)
        #grab 25 frames per person
        for i, frame in enumerate(all_frames[p_idx * N:(p_idx + 1) * N]):
            draw_point_overlay([{"box_2d": box, "label": person_id}], frame,
                                person_dir / f"f{i:02d}.png")

# --- part (b): box-only detection via cache + timestamp (no image upload, no mask) ---
print("\n--- part (b): box-only detection via cache + timestamp ---")
seg_prompt_cached = (
    f"Give the bounding box for every distinct person visible at approximately "
    f"{TEST_TIMESTAMP_S} seconds into the video. Output a JSON list where each "
    f'entry contains the 2D bounding box in "box_2d" (normalized 0-1000, '
    f'[y0,x0,y1,x1]) and a short "label". Do not include a mask -- box only.'
)
try:
    seg_response_cached = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=seg_prompt_cached,
        config=types.GenerateContentConfig(
            cached_content=cache_name,
            response_mime_type="application/json",
        ),
    )
    print("cache+timestamp call SUCCEEDED:")
    print(seg_response_cached.text)
    (out_dir / "seg_cached_raw_response.txt").write_text(seg_response_cached.text)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(TEST_TIMESTAMP_S * fps))
    ret, native_frame = cap.read()
    cap.release()
    if ret:
        dets_cached = json.loads(seg_response_cached.text)
        draw_point_overlay(dets_cached, native_frame, out_dir / "overlay_cached_points.png")
except Exception as e:
    print(f"cache+timestamp call FAILED: {e!r}")

print(f"\n\nAll results saved under: {out_dir}")
print("Send back: overlay_cached_points.png (and raw response .txt files if anything looks off)")
