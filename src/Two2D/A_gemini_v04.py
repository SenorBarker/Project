import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
import json
import dotenv
import cv2

from Two2D.B_video_processing import frames_at_times, video_fps, _probe_format

load_dotenv(Path(__file__).parent.parent / ".env")
from A_Config import asset_name , assets_dir
VIDEO_WINDOWS = 1  # how many clips to split the coverage-sensitive queries into.
# Asked about a 33-minute video whole, Flash answers for roughly the first half and
# stops with finish_reason=STOP -- it is not truncated, it believes it has finished.
# Clipping the part it receives is the only way to make "the rest of the video" not
# exist. Applies to the summary and transcript; objects/places/people still see the
# whole video via the cache.

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
Reply with ONLY the <machine> block below and nothing else -- no prose, no
preamble, no written account of the video. The prose is never saved by the
pipeline, and generating it spends the output-token budget the JSON needs:
a long enough prose answer is truncated before the block is ever reached,
leaving nothing to parse.
Emit exactly this JSON:
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


MODEL = "gemini-3.5-flash"  # single source of truth: a cache is bound to the
# model that created it, so a stray second model id here means either a 400
# ("Model used by GenerateContent request ... and CachedContent ... has to be
# the same") or a silent re-upload on every run.


def _machine_json(raw, label):
    """The JSON out of a response, however the model chose to wrap it.

    The prompts ask for a <machine>...</machine> block and usually get one, but
    the wrapper is the model's choice and it varies run to run -- the people
    query came back fenced as ```json instead, and partitioning on "<machine>"
    turned a perfectly good payload into an empty string and a JSONDecodeError
    pointing at column 1. The content was never the problem, so all three
    wrappings are accepted: the block, a fence, or bare JSON.

    Returns (human_text, parsed). human_text is whatever preceded the JSON,
    which some callers save alongside it."""
    human_text, _, after = raw.partition("<machine>")
    if after:
        return human_text, json.loads(after.partition("</machine>")[0].strip())

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(
            f"no JSON found in the {label} response; the raw text is saved "
            f"next to the other outputs. It starts: {raw[:200]!r}"
        )
    return raw[:start], json.loads(raw[start:end + 1])


def _deterministic_config(cache_name):
    """temperature=0 -- these calls feed a forensic/evidence pipeline and are
    treated as fact downstream, so re-running the same video must converge on
    the same answer instead of resampling a different (possibly incomplete)
    account each time."""
    return types.GenerateContentConfig(cached_content=cache_name, temperature=0)


def _extend_cache(client, cache_name, ttl="600s"):
    """Push the cache's expiry out after a query finishes. `ttl` is input-only and
    is converted at creation into a fixed expireTime -- referencing the cache in a
    generate_content call reads it but does not renew it, so on a long video the
    queries outlive the cache and the later ones fail with a 403 "CachedContent
    not found". Each call therefore buys the next one a fresh window.

    Deliberately unguarded: a failure here means the cache is already gone or
    unreachable, and stopping on that is clearer than printing a diagnostic that
    scrolls past and letting the next query throw a 403 instead."""
    client.caches.update(
        name=cache_name,
        config=types.UpdateCachedContentConfig(ttl=ttl),
    )


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





def _mmss(seconds):
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


def _window_bounds(duration_s, n):
    step = duration_s / n
    return [(i * step, min((i + 1) * step, duration_s)) for i in range(n)]


def _windowed_query(client, file_ref, duration_s, prompt, label,
                    n=VIDEO_WINDOWS, temperature=0, cache_name=None):
    """Ask `prompt` of one clip at a time and join the answers.

    A clipped part and cached content are mutually exclusive in one request, so
    these calls reference the uploaded file directly and forgo the cache discount
    -- each clip is 1/n of the video, so n of them costs about one uncached pass.

    Timestamps are the thing to watch: the model's clock starts at the clip, so
    each window is told which part of the full recording it is and asked for
    whole-recording times. If the second window comes back numbered from 00:00,
    that instruction was ignored and the times need shifting by start_s instead."""
    out = []
    for i, (start_s, end_s) in enumerate(_window_bounds(duration_s, n), start=1):
        clip = types.Part(
            file_data=types.FileData(file_uri=file_ref["uri"],
                                     mime_type=file_ref["mime_type"]),
            video_metadata=types.VideoMetadata(start_offset=f"{int(start_s)}s",
                                               end_offset=f"{int(end_s)}s"),
        )
        framing = (
            f"This clip is part {i} of {n} of a longer recording: it covers "
            f"{_mmss(start_s)} to {_mmss(end_s)} of the full recording. Every "
            f"timestamp you write must be a time in the FULL recording, so the "
            f"first moment of this clip is {_mmss(start_s)}, not 00:00. Cover the "
            f"clip all the way to its end.\n\n"
        )
        response = client.models.generate_content(
            model=MODEL,
            contents=[clip, framing + prompt],
            config=types.GenerateContentConfig(temperature=temperature),
        )
        print(response.text, flush=True)
        _log_response_meta(response, f"{label} [{i}/{n}]")
        # the windowed calls don't touch the cache, but they take minutes, and the
        # cached queries later in the run still need it alive
        if cache_name is not None:
            _extend_cache(client, cache_name)
        out.append(f"--- {_mmss(start_s)}-{_mmss(end_s)} ---\n{response.text or ''}")
    return "\n\n".join(out)


def gemini_vid_to_text(video_path, case_name, query_dir):
    """Get Gemini's summary/objects/transcript/places/people for a video. Reuses the
    remote cache from a previous call if it's still live (checked via
    caches.get); otherwise uploads the video and creates a fresh cache."""
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    cache_name_path = query_dir / f"{case_name}_cache.txt"
    cache_name = None
    if cache_name_path.exists():
        try:
            cached = client.caches.get(name=cache_name_path.read_text().strip())
            # A cache belongs to the model that created it, and this file is
            # shared with the other gemini modules -- reusing a cache another
            # model built fails the whole run with a 400 on the first query, so
            # treat a model mismatch exactly like an expired cache.
            if (cached.model or "").split("/")[-1] != MODEL:
                print(f"cache belongs to {cached.model}, not {MODEL} -- re-caching")
                cache_name = None
            else:
                cache_name = cache_name_path.read_text().strip()
        except Exception:
            cache_name = None  # expired/gone -- fall through and re-cache below

    # The uploaded file is now kept as well as the cache: windowed queries need a
    # file part they can clip, which a cache cannot give them. Uploads live 48h, so
    # a same-day re-run reuses both. Cached-only runs used to discard this handle.
    file_ref_path = query_dir / f"{case_name}_file.json"
    file_ref = None
    if file_ref_path.exists():
        try:
            ref = json.loads(file_ref_path.read_text())
            client.files.get(name=ref["name"])  # 404s once the 48h window lapses
            file_ref = ref
        except Exception:
            file_ref = None

    if file_ref is None:
        print("uploading video", flush=True)
        myfile = client.files.upload(file=video_path)
        while myfile.state.name != "ACTIVE":
            import time; time.sleep(5)
            myfile = client.files.get(name=myfile.name)
        file_ref = {"name": myfile.name, "uri": myfile.uri, "mime_type": myfile.mime_type}
        query_dir.mkdir(parents=True, exist_ok=True)
        file_ref_path.write_text(json.dumps(file_ref, indent=2))
    else:
        myfile = client.files.get(name=file_ref["name"])

    if cache_name is None:
        print("re_caching video")
        # Create the cache for 10 mins, just to get the queries done -- every query
        # extends it, so this only has to outlast one call at a time
        cache = client.caches.create(
            model=MODEL,
            config=types.CreateCachedContentConfig(
                contents=[myfile],
                ttl="600s", #cache time
            )
        )
        cache_name = cache.name
        query_dir.mkdir(parents=True, exist_ok=True)
        cache_name_path.write_text(cache_name)#store path to cached video

    duration_s = float(_probe_format(video_path)["duration"])
    print(f"[gemini-diag] video duration {_mmss(duration_s)}, "
          f"{VIDEO_WINDOWS} windows for summary/transcript", flush=True)

    # Query as many times as you like — no reprocessing
    FORMAT_SUFFIX = " Format the response as multiple short paragraphs separated by line breaks, not one continuous block of text."

    query_dir.mkdir(parents=True, exist_ok=True)

    def _save(suffix, text):
        """Write each answer the moment it lands. Holding all the writes to the end
        of the function meant a single bad parse (the people JSON) threw away four
        good queries that had already been paid for."""
        out = query_dir / f"{case_name}_{suffix}"
        out.write_text(text)
        print(f"[gemini-diag] saved {out.name}", flush=True)

    summary_text = _windowed_query(
        client, file_ref, duration_s,
        prompt="Give a detailed Summary of this clip. Break it into moments: important events that have a clear narrative start and end. " \
        "Every moment needs its own line with a start and stop time. When a moment starts, before you know what is happening note the start, Then when it " \
        "ends, note the end. Once it ends, the next moment begins" 
        "Keep moments together, but if nothing important is happening, summarisse in 30s intervals, but if a moment starts mid interval, START A NEW ONE." + FORMAT_SUFFIX,
        #prompt="Give a detailed Summary of this clip "
        #"Intervals can be no coarser than 10 seconds. provide start and end times each time" + FORMAT_SUFFIX,
                
        label="summary",
        cache_name=cache_name,
    )
    _save("description.txt", summary_text)

    response2 = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=f"summary of the video:{summary_text} \n\nQuestion:What objects, pertinent to the summary are visible and when? Provide times. don't subdivide the same object into multiple times unless there is a long gap" + FORMAT_SUFFIX,
        config=_deterministic_config(cache_name)
    )
    print(response2.text, flush=True)
    _log_response_meta(response2, "objects")
    _extend_cache(client, cache_name)
    _save("objects.txt", response2.text)


    transcript_text = _windowed_query(
        client, file_ref, duration_s,
        prompt="IF there is audio, Give a full human voice audio transcript of this clip."
               " make sure you report exactly what the person says, vocalisations should be described e.g, animals noises shoule be miaow, or woof"
               "**MM:SS** [Speaker]: transcript\n\n"
               " if the same person is talking and there is less than 2s pause, this is one entry not 2",
        label="transcript",
        cache_name=cache_name,
    )
    _save("transcript_full.txt", transcript_text)

    places = client.models.generate_content(
            model="gemini-3.5-flash",
            contents= F"in 2 words, per location, describe the locations in the video. be precise."
            f"word 1 is location category, word 2 is precise location e.g house kitchen / restaurant kitchen / parking-lot apartments /parking-lot multistory "
            f"**MM:SS** : location\n\n"
            ,
            config=_deterministic_config(cache_name)
        )
    print(places.text, flush=True)
    _log_response_meta(places, "places")
    _extend_cache(client, cache_name)
    _save("places.txt", places.text)

    # --- people + per-appearance box_2d, from the same cached video ---
    people_response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents="Identify every distinct person visible in this video." + PEOPLE_SUFFIX,
        config=_deterministic_config(cache_name),
    )
    print(people_response.text, flush=True)
    _log_response_meta(people_response, "people")
    # saved before the parse, so a response that won't parse is still on disk to
    # look at instead of being lost with the traceback
    _save("people_raw.txt", people_response.text or "")
    # last query of the run, but still worth extending -- it leaves the cache with
    # a fresh window for an immediate re-run, which is the common case in a notebook
    _extend_cache(client, cache_name)
    people_raw = people_response.text
    people = _machine_json(people_raw, "people")[1]["people"]

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

    # --- SAVE ---
    # the four text answers are already on disk -- each was written by _save the
    # moment its query returned. Only the parsed people JSON is left, and it waits
    # until here because the debug overlays above can still reveal a bad box.
    _save("people.json", json.dumps(people, indent=2))
    return

#---DEAD---CODE-----WAS A FAILED TRACKING ATTEMPT
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
        model="gemini-3.5-flash",
        contents=augmented,
        config=types.GenerateContentConfig(temperature=0)
    )

    raw = response.text
    human_text, machine = _machine_json(raw, "detections")
    detections = machine["detections"]
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
        model="gemini-3.5-flash",
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
        model="gemini-3.5-flash",
        contents=FORMAT_PREFIX + prompt,
        config=types.GenerateContentConfig(cached_content=cache_name, temperature=0)
    )
    print(response.text)


    return response.text
