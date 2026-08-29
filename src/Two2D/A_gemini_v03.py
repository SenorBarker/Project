import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
import json
import string
import dotenv
import cv2

from Two2D.B_video_processing import frames_at_times, video_fps

load_dotenv(Path(__file__).parent.parent / ".env")
from A_Config import asset_name , assets_dir


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
    cv2.imwrite(str(out_path), overlay)


PEOPLE_SUFFIX = """
After your response, append a <machine> block with this exact JSON:
<machine>
{"people": [{"person_id": "...", "descriptor": "...", "appearances": [{"start_s": 0, "end_s": 0, "box_2d": [0, 0, 0, 0]}]}]}
</machine>
Give every distinct person who appears anywhere in the video a stable person_id
(Starting "person-A", "person-B") -- the same person must keep the same person_id
across every appearance, even in separate, non-contiguous time windows. "descriptor"
is a short human-readable description (distinguishing features: appearance, hair, skin, clothing, position, role) to help a human
tell people apart. Each frame, ask: 'Is there someone here?' 
Then 'Is this person the same as an existing one?' 
If new, create a new person_id and add an entry to "appearances" with the time window 
(start_s integer seconds). If a person stops being present, then add an entry to "appearances" with the time window 
(end_s integer seconds). If the person reappears later, add a new entry to "appearances", for the CORRECT person, 
with the new time  (start_s). If you get to the end of the video and a person is still present, 
add an entry to "appearances" with the time window (end_s) as the last second of the video.
For each appearance start_s also give "box_2d": that person's 2D bounding box (normalized 0-1000,
[y0,x0,y1,x1]). Box only, no mask.
"""


def _deterministic_config(cache_name):
    """temperature=0 -- these calls feed a forensic/evidence pipeline and are
    treated as fact downstream, so re-running the same video must converge on
    the same answer instead of resampling a different (possibly incomplete)
    account each time."""
    return types.GenerateContentConfig(cached_content=cache_name, temperature=0)


def _strip_code_fence(text):
    """The Pro models wrap the <machine> JSON in a ```json fence even though the
    prompt asks for bare JSON (Flash did not) -- strip the fence before parsing
    rather than relying on the model obeying the format instruction."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rpartition("```")[0] if "```" in text else text
    return text.strip()


def _log_response_meta(response, label):
    """Diagnostic: surface what the API reports about a generation that the
    pipeline currently throws away -- finish_reason/safety_ratings tell you
    if content was blocked/softened by safety filtering rather than just
    printing the (possibly already-modified) response.text."""
    try:
        feedback = getattr(response, "prompt_feedback", None)
        print(f"[gemini-diag] {label}: prompt_feedback={feedback}")
        for i, cand in enumerate(getattr(response, "candidates", []) or []):
            print(f"[gemini-diag] {label}: candidate[{i}].finish_reason={cand.finish_reason}")
            print(f"[gemini-diag] {label}: candidate[{i}].safety_ratings={cand.safety_ratings}")
    except Exception as e:
        print(f"[gemini-diag] {label}: failed to inspect response metadata: {e}")


def gemini_vid_to_text(video_path, case_name, query_dir):
    """Get Gemini's summary/objects/transcript/places/people for a video. Reuses the
    remote cache from a previous call if it's still live (checked via
    caches.get); otherwise uploads the video and creates a fresh cache."""
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    cache_name_path = query_dir / f"{case_name}_cache.txt"
    cache_name = None
    if cache_name_path.exists():
        try:
            # push the expiry back out rather than just checking it exists -- a
            # reused cache with two minutes left on it dies partway through the
            # five queries below, which surfaces as a 403 on a later query.
            client.caches.update(
                name=cache_name_path.read_text().strip(),
                config=types.UpdateCachedContentConfig(ttl="3600s"),
            )
            cache_name = cache_name_path.read_text().strip()
        except Exception:
            cache_name = None  # expired/gone -- fall through and re-cache below

    if cache_name is None:
        print("re_caching video")
                # Upload + wait (once)
        myfile = client.files.upload(file=video_path)
        while myfile.state.name != "ACTIVE":
            import time; time.sleep(5)
            myfile = client.files.get(name=myfile.name)

        # Create the cache for 1 hour, just to get the queries done. 10 mins was
        # enough for Flash but the Pro models spend long enough thinking that the
        # five queries below overrun it and the later ones 403 with
        # "CachedContent not found". Storage is ~$4.50/1M cached tokens per hour.
        cache = client.caches.create(
            model="gemini-3.1-pro-preview",
            config=types.CreateCachedContentConfig(
                contents=[myfile],
                ttl="3600s", #cache time
            )
        )
        cache_name = cache.name
        query_dir.mkdir(parents=True, exist_ok=True)
        cache_name_path.write_text(cache_name)#store path to cached video

    # Query as many times as you like — no reprocessing
    FORMAT_SUFFIX = " Format the response as multiple short paragraphs separated by line breaks, not one continuous block of text."

    response = client.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents="Give a detailed Summary of this video. provide start and end times. Intervals can be no coarser than 1 minute ." + FORMAT_SUFFIX,
        config=_deterministic_config(cache_name)
    )
    print(response.text)
    _log_response_meta(response, "summary")

    response2 = client.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents=f"summary of the video:{response.text} \n\nQuestion:What objects, pertinent to the summary are visible and when? Provde times. don't subdivide the same object into multiple times unless there is a long gap" + FORMAT_SUFFIX,
        config=_deterministic_config(cache_name)
    )
    print(response2.text)
    _log_response_meta(response2, "objects")


    response3 = client.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents= F"Give a full human voice audio transcript of this video."
        f" make sure you report exactly what the person says, vocalisations should be described e.g, animals noises shoule be miaow, or woof"
        f"**MM:SS** [Speaker]: transcript\n\n"
        f" if the same person is talking and there is less than 2s pause, this is one entry not 2",
        config=_deterministic_config(cache_name)
    )
    print(response3.text)
    _log_response_meta(response3, "transcript")

    places = client.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents= F"in 2 words, per location, describe the locations in the video. be precise."
            f"word 1 is location category, word 2 is precise location e.g house kitchen / restaurant kitchen / parking-lot apartments /parking-lot multistory "
            f"**MM:SS** : location\n\n"
            ,
            config=_deterministic_config(cache_name)
        )
    print(places.text)
    _log_response_meta(places, "places")

    # --- people + per-appearance box_2d, from the same cached video ---
    people_response = client.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents="Identify every distinct person visible in this video." + PEOPLE_SUFFIX,
        config=_deterministic_config(cache_name),
    )
    print(people_response.text)
    _log_response_meta(people_response, "people")
    people_raw = people_response.text
    _, _, people_machine_block = people_raw.partition("<machine>")
    people_json_str = _strip_code_fence(people_machine_block.partition("</machine>")[0])
    people = json.loads(people_json_str)["people"]
    # Prompt asks for "person-A", "person-B"... already, but the model isn't
    # trusted to get the exact format/order right -- reassign here regardless,
    # by first appearance, so it's never ambiguous with SAM's own numeric
    # "person-01", "person-02" track-slug convention (see A_ROBOFLOW_SAM3.py).
    def _first_appearance_s(person):
        starts = [a["start_s"] for a in person.get("appearances", []) if "start_s" in a]
        return min(starts) if starts else float("inf")

    def _letter_suffix(i):
        '''0-indexed -> A, B, ... Z, AA, AB, ... (base-26, letters not digits).'''
        letters = string.ascii_uppercase
        s, i = "", i + 1
        while i > 0:
            i, r = divmod(i - 1, 26)
            s = letters[r] + s
        return s

    people = sorted(people, key=_first_appearance_s)
    for i, person in enumerate(people):
        person["person_id"] = f"person-{_letter_suffix(i)}"

    # one debug overlay per appearance (not just per person) -- a person can
    # have multiple, possibly non-contiguous, appearance windows and every one
    # of them needs its own seed for downstream use.
    seed_index = [(person["person_id"], a_idx, appearance)
                  for person in people
                  for a_idx, appearance in enumerate(person["appearances"])]
    # sample at the middle of each appearance's labeled second (start_s +
    # fps/2 frames, i.e. start_s + 0.5s), not the literal start_s instant.
    target_times_s = [appearance["start_s"] + 0.5 for _, _, appearance in seed_index]
    frames, frame_indices = frames_at_times(video_path, target_times_s)

    masks_dir = query_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    for (person_id, a_idx, appearance), frame, frame_idx in zip(seed_index, frames, frame_indices):
        box = get_box(appearance)
        if box is None:
            print(f"  [{person_id}] appearance {a_idx} has no box_2d -- skipping")
            continue
        # frame_idx is the frame this image actually came from, reported by
        # frames_at_times -- 0-indexed, so it lines up with the numbers burned in
        # by the companion ffmpeg drawtext=text='%{n}' verification video.
        draw_point_overlay([{"box_2d": box, "label": person_id}], frame,
                            masks_dir / f"{person_id}_a{a_idx}_f{frame_idx:05d}.png")

     # --- SAVE (run once after getting response3) ---
    query_dir.mkdir(parents=True, exist_ok=True)
    (query_dir / f"{case_name}_description.txt").write_text(response.text)
    (query_dir / f"{case_name}_objects.txt").write_text(response2.text)
    (query_dir / f"{case_name}_transcript_full.txt").write_text(response3.text)
    (query_dir / f"{case_name}_places.txt").write_text(places.text)
    (query_dir / f"{case_name}_people.json").write_text(json.dumps(people, indent=2))
    return

def query_from_description(operator_prompt, description_path, objects_path, case_dir):
    description = Path(description_path).read_text()
    objects = Path(objects_path).read_text()
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    MACHINE_SUFFIX = """
    Filter the objects list and return objects that are directly related to the operator_prompt. this is objects and people, not just objects.
    Format your response as multiple short paragraphs separated by line breaks — do not
    write it as one continuous block of text.

    After your response, append a <machine> block with this exact JSON:
    <machine>
    {"detections": [{"subject": "...", "start_s": 0, "end_s": 0}]}
    </machine>
    "subject" must be a bare generic object category only (e.g. "cat", "dog", "person", "car") —
    no adjectives, descriptors, or articles (write "cat", not "stray cat" or "the cat").
    Use integer seconds only. One entry per detected subject appearance."""

    augmented = (
    f"The following is a detailed description of a video:\n\n"
    f"{description}\n\n"
    f"Detected objects and their appearances:\n\n"
    f"{objects}\n\n"
    f"Filter criterion (do not answer this as a question — use it only to decide which "
    f"objects/people are relevant): {operator_prompt}\n\n"
    f"{MACHINE_SUFFIX}"
)


    response = client.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents=augmented,
        config=types.GenerateContentConfig(temperature=0)
    )

    raw = response.text
    human_text, _, machine_block = raw.partition("<machine>")
    json_str = _strip_code_fence(machine_block.partition("</machine>")[0]) #converts the response to machine readable format
    detections = json.loads(json_str)["detections"]
    #saving
    out_dir = case_dir / "012_Gemini_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "human_out.txt").write_text(human_text.strip())
    (out_dir / "detections.json").write_text(json.dumps(detections, indent=2))
    print(human_text)
    return human_text.strip(), detections

#generic command to cache this
def gemini_cache_video(video_path, query_dir, ttl="600s"):
    """Upload a video and create a cache for it, persisting the cache name to
    disk so it can be reused across separate calls/processes until the TTL expires."""
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    myfile = client.files.upload(file=video_path)
    while myfile.state.name != "ACTIVE":
        import time; time.sleep(5)
        myfile = client.files.get(name=myfile.name)

    cache = client.caches.create(
        model="gemini-3.1-pro-preview",
        config=types.CreateCachedContentConfig(
            contents=[myfile],
            ttl=ttl,
        )
    )

    query_dir.mkdir(parents=True, exist_ok=True)
    (query_dir / f"{asset_name()}_cache_name.txt").write_text(cache.name)
    return cache.name


def gemini_query_CSV_cached(prompt, asset_name, CSV_path, query_dir, extend_ttl=None):
    """Query a previously created cache (see gemini_cache_video) for asset_name.
    Pass extend_ttl (e.g. "600s") to refresh the cache's expiry before querying."""
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    cache_name = (query_dir / f"{asset_name()}_cache_name.txt").read_text().strip()


    if extend_ttl is not None:
        client.caches.update(
            name=cache_name,
            config=types.UpdateCachedContentConfig(ttl=extend_ttl),
        )
    with open(CSV_path) as f:
        csv_text = f.read()

    FORMAT_PREFIX = "Format your response as multiple short paragraphs separated by line breaks, not one continuous block of text.\n\n"

    response = client.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents=FORMAT_PREFIX + prompt,
        config=types.GenerateContentConfig(cached_content=cache_name)
    )
    print(response.text)


    return response.text
