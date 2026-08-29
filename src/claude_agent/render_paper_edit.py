"""Renders a Producer paper-edit JSON (see .claude/agents/producer.md's
Output structure section) into the fixed HTML document. Pure code, no LLM —
formatting is deterministic and identical across every run."""
import json
from pathlib import Path

from A_Config import MASK_SEARCH_HANDLE_S

# Frames at which SAM3 offloads video+state to host RAM and gets unreliable
# (see tracking_windows/_merge_spans below). A merge that would push a span
# past this is refused rather than handed to the tracker.
SAM3_MAX_TRACK_FRAMES = 650

# The closed set of 8 from CONSTITUTION.md section 0, verbatim -- anything a beat
# carries that isn't a key here is an invalid archetype and gets raised as an error
# by _beat_flags. INTERVIEW is the pre-SOUNDBITE name, kept only so older paper
# edits still render and still validate.
ARCHETYPE_LABELS = {
    "ESTABLISHER": "Opening shot",
    "EVENT_TRIGGER": "Event — trigger",
    "EVENT_ACTION": "Event — action",
    "AFTERMATH": "Aftermath",
    "SOUNDBITE": "Soundbite",
    "MAP": "Map",
    "METRIC": "Metric",
    "PROJECTION_MAP": "Projection map",
    "INTERVIEW": "Interview",  # legacy alias for SOUNDBITE
}

_STYLE = """
  body { font-family: Georgia, serif; max-width: 900px; margin: 2rem auto; color: #1a1a1a; line-height: 1.5; }
  h1, h2 { font-family: Arial, sans-serif; }
  .flag { color: #b00; font-weight: bold; }
  .judgment { color: #806000; font-weight: bold; }
  table { border-collapse: collapse; width: 100%; margin-bottom: 2rem; }
  th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: left; vertical-align: top; font-size: 0.92rem; }
  th { background: #f0f0f0; }
  .beat { border: 1px solid #ccc; padding: 1rem; margin-bottom: 1rem; }
  .field { margin: 0.3rem 0; }
  .label { font-weight: bold; }
  .rationale { font-style: italic; }
  .rejected { color: #555; }
"""


def _esc(s):
    if s is None:
        return ""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _toc_row(beat):
    label = ARCHETYPE_LABELS.get(beat["archetype"], beat["archetype"])
    source_or_request = beat.get("source") or beat.get("asset_request") or ""
    return (
        f"<tr><td>{beat['order']}</td><td>{_esc(label)}</td>"
        f"<td>{_esc(beat['segment_type'])}</td><td>{_esc(source_or_request)}</td>"
        f"<td>{beat['duration_seconds']}</td></tr>"
    )


def _beat_block(beat):
    label = ARCHETYPE_LABELS.get(beat["archetype"], beat["archetype"])
    lines = [
        f'<div class="beat">',
        f'<div class="field"><span class="label">Beat {beat["order"]} — {_esc(label)}</span> '
        f'| order {beat["order"]} | segment_type: {_esc(beat["segment_type"])} '
        f'| duration_seconds: {beat["duration_seconds"]}</div>',
    ]
    if beat.get("source"):
        lines.append(f'<div class="field"><span class="label">Source:</span> {_esc(beat["source"])}</div>')
    if beat.get("auto_select_resolved"):
        frame_range = f'{beat.get("start_frame")}–{beat.get("end_frame")}'
        lead_in = beat.get("lead_in_seconds")
        lead_in_txt = f' | lead_in_seconds: {lead_in}' if lead_in else ""
        lines.append(f'<div class="field"><span class="label">Frames:</span> {_esc(frame_range)}{_esc(lead_in_txt)}</div>')
    elif beat.get("segment_type") == "real":
        window = f'{beat.get("search_window_start_seconds")}s–{beat.get("search_window_end_seconds")}s'
        lines.append(f'<div class="field"><span class="label">Search window:</span> {_esc(window)} (unresolved)</div>')
    if beat.get("asset_request"):
        lines.append(f'<div class="field"><span class="label">Asset request:</span> {_esc(beat["asset_request"])}</div>')
    if beat.get("requested_flags"):
        lines.append(f'<div class="field"><span class="label">Flags requested:</span> {_esc(", ".join(beat["requested_flags"]))}</div>')
    if beat.get("quote"):
        lines.append(f'<div class="field"><span class="label">Quote:</span> "{_esc(beat["quote"])}"</div>')
    if beat.get("rationale"):
        lines.append(f'<div class="field rationale"><span class="label">Rationale:</span> {_esc(beat["rationale"])}</div>')
    if beat.get("rejected_alternative"):
        lines.append(f'<div class="field rejected"><span class="label">Rejected alternative:</span> {_esc(beat["rejected_alternative"])}</div>')
    if beat.get("flag"):
        css = "judgment" if "JUDGMENT" in beat["flag"] else "flag"
        lines.append(f'<div class="field"><span class="{css}">{_esc(beat["flag"])}</span></div>')
    lines.append("</div>")
    return "\n".join(lines)


def render_html(data: dict) -> str:
    beats = sorted(data["beats"], key=lambda b: b["order"])
    toc_rows = "\n".join(_toc_row(b) for b in beats)
    beat_blocks = "\n\n".join(_beat_block(b) for b in beats)
    title = f'{_esc(data.get("case_name", ""))} — Paper Edit'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>{_STYLE}</style>
</head>
<body>

<h1>{title}</h1>
<p><strong>Brief:</strong> {_esc(data.get("brief", ""))}</p>

<h2>Table of Contents</h2>
<table>
<tr><th>#</th><th>Content type</th><th>Segment type</th><th>Source / asset request</th><th>Duration (s)</th></tr>
{toc_rows}
</table>

<h2>Paper Edit</h2>

{beat_blocks}

</body>
</html>
"""


# Every file this pipeline writes into agent_p_output_dir() (paper edit,
# HTML, flags, feedback) shares one stem: "<asset_name>_<pass>_<kind>". This
# is the ONLY place that stem gets built -- every writer and every reader
# elsewhere in the codebase must go through paper_edit_path/paper_edit_stem
# rather than re-deriving the filename itself, or renaming producer_agent_execution's
# pass-numbering scheme (or fixing a naming bug here) silently stops applying
# to whichever caller still hardcodes the old pattern.
_KIND_FOR_MODE = {"draft": "draft", "draft2": "draft", "revision": "revision"}


def paper_edit_stem(mode="draft", pass_n=None):
    '''"<asset_name>_<pass>_<draft|revision>" -- the shared stem for a pass's
    paper-edit JSON and every file derived from it. pass_n defaults to
    A_Config.pass_num() (bump set_pass(...) in the notebook to point this at
    a later pass); pass an explicit pass_n to reach back to an older one
    (e.g. pass_num() - 1 for "the previous pass's draft").'''
    from A_Config import asset_name, pass_num
    kind = _KIND_FOR_MODE.get(mode, mode)
    p = pass_num() if pass_n is None else pass_n
    return f"{asset_name()}_{p}_{kind}"


def paper_edit_path(mode="draft", pass_n=None):
    '''Current case's paper-edit JSON on disk, from A_Config -- so every
    function here is callable with no path after a kernel restart, when the
    value run_producer_agent returned is gone. "draft" is the pre-revision
    file, anything else the revised one.'''
    from A_Config import agent_p_output_dir
    return agent_p_output_dir() / f"{paper_edit_stem(mode, pass_n)}.json"


def flags_path(mode="draft", pass_n=None):
    from A_Config import agent_p_output_dir
    return agent_p_output_dir() / f"{paper_edit_stem(mode, pass_n)}_flags.json"


def feedback_path(mode="draft", pass_n=None):
    from A_Config import agent_p_output_dir
    return agent_p_output_dir() / f"{paper_edit_stem(mode, pass_n)}_feedback.json"


def render_and_save(paper_edit_json_path: str | Path = None) -> Path:
    """Reads a Producer-written paper-edit JSON and writes the matching HTML
    next to it, same stem. Returns the html path."""
    json_path = Path(paper_edit_json_path or paper_edit_path())
    data = json.loads(json_path.read_text(encoding="utf-8"))
    html = render_html(data)
    html_path = json_path.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")
    return html_path


# Recon kind per requested flag. RECON_CAM_POSES is the camera-track solve behind
# MAP assets; RECON_3D is the solve of a moment of action, for physics and novel
# views (see producer.md's PRODUCER FLAGS).
RECON_FLAG_KINDS = {"RECON_CAM_POSES": "MAP", "RECON_3D": "EVENT"}


def wants_projection_map(beat: dict) -> bool:
    """Is this beat a projection map? PROJECTION_MAP is BOTH an archetype (the
    closed set in CONSTITUTION.md section 0, which is how W_video_editor decides
    what clip to build) and a menu flag the notebook gates its rendering cell on,
    so the Producer can legitimately express it either way. Every test in this
    file goes through here rather than looking in one field only -- checking
    requested_flags alone silently dropped every PROJECTION_MAP-archetype beat,
    flag off and no RECON_3D."""
    return (beat.get("archetype") == "PROJECTION_MAP"
            or "PROJECTION_MAP" in (beat.get("requested_flags") or []))


def beat_recon_flags(beat: dict) -> set[str]:
    """Which recon flags ONE beat actually needs """

    requested = set(beat.get("requested_flags") or [])
    has_window = (beat.get("search_window_start_seconds") is not None
                  and beat.get("search_window_end_seconds") is not None)

    if beat.get("archetype") == "MAP" and has_window:
        return {"RECON_CAM_POSES"}

    flags = requested & set(RECON_FLAG_KINDS)
    if wants_projection_map(beat) and has_window:
        flags.add("RECON_3D")
    return flags


def _beat_flags(beat: dict, fps: float, gps_signal: str | None, errors: list[str], notes: list[str]) -> set[str]:
    """Runs every rule against ONE beat, returns the flags that beat
    actually needs. Ignores beat["requested_flags"] for any key we have a
    rule for -- that field is LLM output and is not trusted, except where
    noted below (missing data we can't check ourselves). Wrong/missing
    flags go to `notes`, missing data Producer must supply goes to
    `errors` -- neither is raised, so every beat gets checked in one pass."""
    flags = set()
    requested = beat.get("requested_flags") or []
    beat_id = beat.get("beat_id")

    # Archetype drives real dispatch downstream (W_video_editor picks the clip
    # kind off it), so a value outside the closed set doesn't fail here -- it
    # just quietly builds nothing. Catch it as something Producer must fix.
    if beat.get("archetype") not in ARCHETYPE_LABELS:
        errors.append(
            f"Beat {beat_id} has archetype {beat.get('archetype')!r}, which is not "
            f"one of {sorted(ARCHETYPE_LABELS)} -- Producer must use the closed set "
            f"in CONSTITUTION.md section 0, verbatim."
        )

    # TRACKING -- beat asks for a subject to be tracked.
    if beat.get("tracked_subject"):
        flags.add("TRACKING")  # CHECK
        if "TRACKING" not in requested:
            notes.append(f"Beat {beat_id}: added TRACKING (Producer didn't request it).")

    if beat.get("archetype") == "MAP":
        # A map places subjects/objects in the field of view, so it always
        # needs them tracked -- not conditional on the LLM remembering to
        # ask for it (see producer.md: MAP requires TRACKING).
        if not beat.get("tracked_subject"):
            errors.append(
                f"MAP beat {beat_id} needs TRACKING but has no "
                f"tracked_subject -- Producer must list what to track."
            )
            if "TRACKING" in requested:
                flags.add("TRACKING")  # can't verify, but don't override what was right
        else:
            flags.add("TRACKING")  # CHECK
            if "TRACKING" not in requested:
                notes.append(f"Beat {beat_id}: added TRACKING (Producer didn't request it).")

        if beat.get("search_window_start_seconds") is None or beat.get("search_window_end_seconds") is None:
            errors.append(
                f"MAP beat {beat_id} needs a recon but has no "
                f"search_window_start_seconds/search_window_end_seconds -- Producer must set them."
            )

        # GOOGLE_MAP vs BEV_MAP -- GPS_signal from analysis_2d_for_decisions.
        if gps_signal is not None:
            right_flag = "GOOGLE_MAP" if gps_signal == "yes" else "BEV_MAP"
            flags.add(right_flag)  # CHECK
            if right_flag not in requested:
                notes.append(f"Beat {beat_id}: added {right_flag} (Producer didn't request it).")
        else:
            for f in ("GOOGLE_MAP", "BEV_MAP"):
                if f in requested:
                    flags.add(f)

    # PROJECTION_MAP -- no rule yet on when it's needed, trust the LLM on
    # that. But it's a novel view of a RECON_3D, so it always needs one --
    # beat_recon_flags below is what actually adds that RECON_3D.
    # wants_projection_map, not `in requested`: the archetype is the Producer's
    # usual way of asking for one (see CONSTITUTION.md section 0).
    if wants_projection_map(beat):
        if beat.get("archetype") == "MAP":
            # A MAP beat only ever carries its RECON_CAM_POSES, so there'd be no
            # RECON_3D here for the projection to be a novel view of. Not something
            # we can correct -- the Producer has to split it into its own beat.
            errors.append(
                f"MAP beat {beat_id} requests PROJECTION_MAP, but a MAP beat only "
                f"carries RECON_CAM_POSES and a projection map needs a RECON_3D -- "
                f"Producer must move PROJECTION_MAP to its own beat."
            )
        else:
            flags.add("PROJECTION_MAP")
            if beat.get("search_window_start_seconds") is None or beat.get("search_window_end_seconds") is None:
                errors.append(
                    f"Beat {beat_id} needs RECON_3D (for PROJECTION_MAP) but has no "
                    f"search_window_start_seconds/search_window_end_seconds -- Producer must set them."
                )

    # RECON_CAM_POSES / RECON_3D -- one shared decision, so the flags this file
    # derives and the recon jobs build_recon_requests hands the notebook can
    # never disagree about which recon a beat gets.
    for f in beat_recon_flags(beat):
        flags.add(f)
        if f not in requested:
            notes.append(f"Beat {beat_id}: added {f} (Producer didn't request it).")

    # MASK_OVERLAYS -- no rule yet, trust the LLM. A mask overlay always
    # needs a tracked subject to draw the mask from, so it switches TRACKING
    # on too, regardless of whether the beat separately requested it.
    if "MASK_OVERLAYS" in requested:
        if not beat.get("tracked_subject"):
            errors.append(
                f"Beat {beat_id} requests MASK_OVERLAYS but has "
                f"no tracked_subject -- Producer must list what to track."
            )
        else:
            flags.add("MASK_OVERLAYS")
            flags.add("TRACKING")  # CHECK
            if "TRACKING" not in requested:
                notes.append(f"Beat {beat_id}: added TRACKING (Producer didn't request it).")

    # MULTICAM -- out of scope per producer.md, never on.

    return flags


def derive_flags(paper_edit_json_path: str | Path = None, available_flag_keys=None, fps: float = None, gps_signal: str = None) -> tuple[dict, list[str], list[str]]:
    """For each beat, checks its flags against the rules in _beat_flags and
    corrects any that are wrong. Final dict is the union across all beats.
    Collects every beat's problems before raising, so Producer gets the
    full list to fix in one revision pass instead of one-at-a-time."""
    data = json.loads(Path(paper_edit_json_path or paper_edit_path()).read_text(encoding="utf-8"))
    beats = data["beats"]

    errors: list[str] = []
    notes: list[str] = []
    on = set()
    for beat in beats:
        on |= _beat_flags(beat, fps, gps_signal, errors, notes)

    if notes:
        print("Paper edit flag corrections (feed back to Producer):\n" + "\n".join(f"- {n}" for n in notes))

    if errors:
        print("Paper edit flag problems (feed back to Producer):\n" + "\n".join(f"- {e}" for e in errors))

    flags = {k: (k in on) for k in available_flag_keys}
    return flags, errors, notes


def derive_and_save_flags(paper_edit_json_path: str | Path = None, available_flag_keys=None, fps: float = None, gps_signal: str = None) -> Path:
    """Writes "<stem>_flags.json" and "<stem>_feedback.json" (errors + notes
    from derive_flags) next to the given paper edit -- "<stem>_feedback.json"
    is the file to read from when building a MODE: revision re-draft prompt
    for Producer. Both derive their name from json_path's own stem (not
    A_Config.pass_num()) so they always match the pass that was actually
    read, even if pass_num() has since moved on."""
    json_path = Path(paper_edit_json_path or paper_edit_path())
    flags, errors, notes = derive_flags(json_path, available_flag_keys, fps, gps_signal)
    flags_file = json_path.with_name(json_path.stem + "_flags.json")
    flags_file.write_text(json.dumps(flags, indent=2), encoding="utf-8")

    feedback_file = json_path.with_name(json_path.stem + "_feedback.json")
    feedback_file.write_text(json.dumps({"errors": errors, "notes": notes}, indent=2), encoding="utf-8")

    return flags_file


def _beat_window(beat: dict) -> tuple[float, float] | None:
    """Beat's search window, or None if the Producer set no window. A real beat gets
    MASK_SEARCH_HANDLE_S either side (start clamped at 0): the Producer only guesses
    when the subject appears, and the pad gives SAM3 room to find the real onset.
    Synthetic beats aren't cut against an onset, so they get the window as written."""
    start_s = beat.get("search_window_start_seconds")
    end_s = beat.get("search_window_end_seconds")
    if start_s is None or end_s is None:
        return None
    pad_s = MASK_SEARCH_HANDLE_S if beat.get("segment_type") == "real" else 0.0
    return max(0.0, start_s - pad_s), end_s + pad_s


def tracking_windows(beat: dict, beats: list[dict], fps: float = None) -> list[tuple[float, float]]:
    """Real beat -> its own window. MAP beat: own window if <=SAM3_MAX_TRACK_FRAMES;
    else [] if real tracked beats fully cover it; else chunk own window into
    <=SAM3_MAX_TRACK_FRAMES pieces. fps (avg, approximate -- see video_fps) is only
    used for the frame-count->duration check; read from the video if not given.
    Every window comes from _beat_window, which is where the real-beat pad lives."""
    if beat.get("archetype") != "MAP":
        window = _beat_window(beat)
        return [] if window is None else [window]

    map_window = _beat_window(beat)
    if map_window is None:
        return []
    map_start, map_end = map_window

    if fps is None:
        from A_Config import source_video_path
        from Two2D.B_video_processing import video_fps
        fps = video_fps(source_video_path())
    max_span_seconds = SAM3_MAX_TRACK_FRAMES / fps

    if map_end - map_start <= max_span_seconds:
        return [(map_start, map_end)]

    covering = sorted(
        w
        for w in (
            _beat_window(b)
            for b in beats
            if b.get("segment_type") == "real" and b.get("tracked_subject")
        )
        if w is not None and w[1] > map_start and w[0] < map_end
    )
    if _fully_covers(covering, map_start, map_end):
        return []

    chunks = []
    t = map_start
    while t < map_end:
        chunks.append((t, min(map_end, t + max_span_seconds)))
        t += max_span_seconds
    return chunks


def _fully_covers(spans: list[tuple[float, float]], start: float, end: float) -> bool:
    """True if sorted (s, e) spans leave no gap across [start, end]."""
    cur = start
    for s, e in spans:
        if s > cur:
            return False
        cur = max(cur, e)
    return cur >= end


def build_tracking_requests(paper_edit_json_path: str | Path = None, fps: float = None,
                             video_path: str | Path = None) -> list[dict]:
    """Merges tracked_subject spans per subject into list[{"subject","start_s","end_s","beat_ids"}].
    Spans are the Producer's search windows, MASK_SEARCH_HANDLE_S-padded on real beats
    (see tracking_windows) -- tracking runs before find_all_cut_points, so no beat
    has resolved frames yet.
    A gem_person_id dict entry groups under subject="person" (one class scan, not per-person).
    Which windows a beat contributes comes from tracking_windows -- see there for the
    MAP-beat cases (own window / covered by others / chunked)."""
    if not video_path:
        from A_Config import source_video_path
        video_path = source_video_path()
    if not fps:
        # Only read the video when the caller didn't already have fps -- keeps this
        # callable (and testable) without the cv2/ffmpeg stack B_video_processing pulls in.
        from Two2D.B_video_processing import video_fps
        fps = video_fps(video_path)
    # fps is only used for this frame-count -> duration threshold, not for any
    # seconds->frame conversion below (those use real per-frame timestamps
    # via frame_indices_at_times instead, since a single fps misrepresents
    # the true local rate on VFR sources).
    max_span_seconds = SAM3_MAX_TRACK_FRAMES / fps

    data = json.loads(Path(paper_edit_json_path or paper_edit_path()).read_text(encoding="utf-8"))
    spans_by_subject = {}

    # open existing detections csv if it exists and group frame indices for each thing
    from A_Config import sam3_masks_dir
    from run_models.A_ROBOFLOW_SAM3 import _slugify_subject
    detections_dict = {}
    detections_path = sam3_masks_dir() / "detections.csv"
    if detections_path.exists():
        import pandas as pd
        detections = pd.read_csv(detections_path)
        detections_dict = {
            slug: set(sub["frame_idx"]) for slug, sub in detections.groupby("subject")
        }

    # How many of each subject the whole video asks for, which is what a frame is
    # allowed to hold (plus leeway) before the tracker calls the prompt bad. Counted
    # across every beat -- a subject named on one beat is still in shot during the
    # spans that do get tracked. "person" is a class scan, so its count is the whole
    # named cast rather than any one beat's few.
    # This applies to MAP beats too: a MAP beat's tracked_subject names specific
    # things (the tree someone hid behind), not a class of them, so a frame holding
    # four trees still means the bare-noun prompt failed to isolate the one meant.
    cast = set()
    expected_by_subject = {}
    for beat in data["beats"]:
        for entry in beat.get("tracked_subject") or []:
            if isinstance(entry, dict):
                cast.add(entry.get("gem_person_id"))
            else:
                expected_by_subject[entry] = expected_by_subject.get(entry, 0) + 1
    if cast:
        expected_by_subject["person"] = len(cast)

    for beat in data["beats"]:
        tracked = beat.get("tracked_subject")
        if not tracked:
            continue
        windows = tracking_windows(beat, data["beats"], fps=fps)
        # beat_ids is this beat's own, even for a MAP beat searching event windows --
        # the masks belong to the map, and tagging them with the event beats would
        # give those beats a mask overlay in assemble_paper_edit that they never
        # requested.
        beat_ids = tuple(beat.get("beat_id") or [])
        for start_s, end_s in windows:
            for tracked_subject_entry in tracked:
                subject = "person" if isinstance(tracked_subject_entry, dict) else tracked_subject_entry
                spans_by_subject.setdefault(subject, []).append((start_s, end_s, beat_ids))

    # Pass 1: resolve every subject's merged spans (cheap, in-memory) before
    # touching the video at all.
    pending = []  # (subject, start_s, end_s, beat_ids)
    for subject, spans in spans_by_subject.items():
        # set() because the eight gem_person_id entries all collapse to subject
        # "person" and would otherwise queue the same span eight times -- _merge_spans
        # used to absorb that silently. _merge_spans itself caps any merge at
        # max_span_seconds, so this can't chain a MAP beat's own chunks (or anything
        # else) back into an oversized span.
        for start_s, end_s, beat_ids in _merge_spans(sorted(set(spans)), max_span_seconds):
            pending.append((subject, start_s, end_s, beat_ids))

    # One batched real-timestamp->frame lookup for every span, instead of
    # round(seconds*fps) per span -- inaccurate on VFR sources.
    from Two2D.B_video_processing import frame_indices_at_times
    flat_targets = [v for p in pending for v in (p[1], p[2])]
    flat_frames = frame_indices_at_times(video_path, flat_targets) if flat_targets else []

    requests = []
    for i, (subject, start_s, end_s, beat_ids) in enumerate(pending):
        start_frame, end_frame = flat_frames[2 * i], flat_frames[2 * i + 1]
        span_frames = set(range(start_frame, end_frame + 1))
        # Skip only if every frame in this span is already tracked -- a partial
        # overlap (e.g. a widened window) means new frames still need tracking.
        if span_frames <= detections_dict.get(_slugify_subject(subject), set()):
            continue
        requests.append({"subject": subject, "start_s": start_s, "end_s": end_s,
                          "beat_ids": sorted(beat_ids),
                          "expected_count": expected_by_subject.get(subject)})
    print(requests)
    return requests


def build_meeting_requests(paper_edit_json_path: str | Path = None) -> list[tuple[str, str]]:
    """Unique subject pairs the Producer asked to have compared, from each beat's
    "meeting_requests": [["person-1", "camera"], ...]. "camera" is a reserved name meaning
    the camera track rather than a tracked subject. Order within a pair doesn't matter,
    so (a,b) and (b,a) collapse to one -- the report shouldn't carry both."""
    data  = json.loads(Path(paper_edit_json_path or paper_edit_path()).read_text(encoding="utf-8"))
    pairs = []
    for beat in data["beats"]:
        for a, b in beat.get("meeting_requests") or []:
            if (a, b) not in pairs and (b, a) not in pairs:
                pairs.append((a, b))
    return pairs


def _merge_spans(spans: list[tuple[float, float, tuple[str, ...]]],
                  max_span_seconds: float | None = None) -> list[tuple[float, float, set]]:
    """Merges overlapping/touching (start_s, end_s, beat_ids) spans into
    their union, unioning beat_ids along with the time range.

    A merge that would produce a span longer than max_span_seconds is skipped --
    the incoming span starts a new group instead of extending the last one. Fewer,
    longer spans are cheaper to track, but not past the point where SAM3 itself
    breaks (SAM3_MAX_TRACK_FRAMES); two spans that would fuse into a too-long job
    are kept separate rather than handed to the tracker oversized."""
    merged = []
    for start_s, end_s, beat_ids in sorted(spans, key=lambda s: (s[0], s[1])):
        fits = (max_span_seconds is None
                or max(merged[-1][1], end_s) - merged[-1][0] <= max_span_seconds) if merged else False
        if merged and start_s <= merged[-1][1] and fits:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end_s), merged[-1][2] | set(beat_ids))
        else:
            merged.append((start_s, end_s, set(beat_ids)))
    return merged


def beat_key(beat: dict) -> str:
    '''The one string that names a beat's outputs on disk -- its first beat_id.
    Same convention as D_2d_analysis.analysis_2d_from_masks and
    W_video_editor.assemble_paper_edit, so a beat's masks, frames, recon and
    map frames all sit under the same name. Merged beats keep the first ID;
    the full list travels beside it as beat_ids.'''
    return beat["beat_id"][0]


def build_recon_requests(paper_edit_json_path=None, video_path=None) -> list[dict]:
    '''One entry per (beat, recon kind) the Producer asked for, in beat order.

    A beat may need both RECON_CAM_POSES and RECON_3D, so it yields up to two
    requests -- never two of the same kind. Each beat's window is its own: windows
    are never merged or deduplicated across beats, so two beats four minutes apart
    give two separate recons rather than one four-minute one.

    Which recons a beat needs comes from beat_recon_flags, not raw requested_flags,
    so these jobs always match the flags derive_flags wrote -- a MAP beat whose
    requested flag was corrected still gets the recon it actually needs.

    Each request carries the frames_dir to sample into and the recon_dir to solve
    into, both already beat-keyed, so callers never build those paths themselves.'''
    from A_Config import (frames_for_cam_poses_dir, frames_for_recon_dir,
                          recon_for_MAP_dir, recon_for_EVENT_dir)

    json_path = Path(paper_edit_json_path or paper_edit_path())
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not video_path:
        # Only read the video when the caller didn't already have it -- keeps this
        # callable (and testable) without the cv2/ffmpeg stack B_video_processing pulls in.
        from A_Config import source_video_path
        video_path = source_video_path()

    dirs = {"MAP":   (frames_for_cam_poses_dir, recon_for_MAP_dir),
            "EVENT": (frames_for_recon_dir,     recon_for_EVENT_dir)}

    # Pass 1: collect every beat/flag's window (cheap, in-memory) before
    # touching the video at all.
    pending = []  # (beat, flag, kind, start_s, end_s)
    for beat in sorted(data["beats"], key=lambda b: b["order"]):
        needed = beat_recon_flags(beat)
        for flag in sorted(needed):
            kind = RECON_FLAG_KINDS[flag]
            start_s = beat.get("search_window_start_seconds")
            end_s = beat.get("search_window_end_seconds")
            if start_s is None or end_s is None:
                raise ValueError(
                    f"Beat {beat.get('beat_id')} needs {flag} but has no "
                    f"search_window_start_seconds/search_window_end_seconds in {json_path.name} "
                    f"-- Producer must set them."
                )
            pending.append((beat, flag, kind, start_s, end_s))

    # One batched real-timestamp->frame lookup for every beat/flag, instead of
    # round(seconds*fps) per beat -- inaccurate on VFR sources; these become
    # the actual recon sampling frame bounds.
    from Two2D.B_video_processing import frame_indices_at_times
    flat_targets = [v for p in pending for v in (p[3], p[4])]
    flat_frames = frame_indices_at_times(video_path, flat_targets) if flat_targets else []

    requests = []
    for i, (beat, flag, kind, start_s, end_s) in enumerate(pending):
        start_frame, end_frame = flat_frames[2 * i], flat_frames[2 * i + 1]
        key = beat_key(beat)
        frames_dir_fn, recon_dir_fn = dirs[kind]
        requests.append({
            "beat_key":    key,
            "beat_ids":    list(beat["beat_id"]),
            "order":       beat["order"],
            "archetype":   beat.get("archetype"),
            "kind":        kind,
            "flag":        flag,
            "start_frame": start_frame,
            "end_frame":   end_frame,
            "frames_dir":  frames_dir_fn(key),
            "recon_dir":   recon_dir_fn(key),
            # The beat itself, so renderers that need more of it than the key
            # (R_map_animator/P_trace_overlayer want duration_seconds and
            # requested_flags) don't have to re-open the paper edit to find it.
            "beat":        beat,
        })
        print(f"{key} {kind} ({flag}): {start_s}-{end_s}s "
              f"-> frames {start_frame}-{end_frame}")

    return requests
