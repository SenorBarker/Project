"""Renders a Producer paper-edit JSON (see .claude/agents/producer.md's
Output structure section) into the fixed HTML document. Pure code, no LLM —
formatting is deterministic and identical across every run."""
import json
from pathlib import Path

# Frames at which SAM3 offloads video+state to host RAM and gets unreliable
# (see tracking_windows/_merge_spans below). A merge that would push a span
# past this is refused rather than handed to the tracker.
SAM3_MAX_TRACK_FRAMES = 650

ARCHETYPE_LABELS = {
    "ESTABLISHER": "Opening shot",
    "EVENT_TRIGGER": "Event — trigger",
    "EVENT_ACTION": "Event — action",
    "AFTERMATH": "Aftermath",
    "INTERVIEW": "Interview",
    "MAP": "Map",
    "METRIC": "Metric",
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


def paper_edit_path(mode="draft"):
    '''Current case's paper-edit JSON on disk, from A_Config -- so every
    function here is callable with no path after a kernel restart, when the
    value run_producer_agent returned is gone. "draft" is the pre-revision
    file, anything else the revised one.'''
    from A_Config import agent_p_output_dir, case_name
    suffix = "_paper_edit_draft.json" if mode == "draft" else "_paper_edit.json"
    return agent_p_output_dir() / f"{case_name()}{suffix}"


def render_and_save(paper_edit_json_path: str | Path = None) -> Path:
    """Reads a Producer-written <case_name>_paper_edit.json and writes the
    matching <case_name>_paper_edit.html next to it. Returns the html path."""
    json_path = Path(paper_edit_json_path or paper_edit_path())
    data = json.loads(json_path.read_text(encoding="utf-8"))
    html = render_html(data)
    html_path = json_path.with_name(json_path.stem.replace("_paper_edit", "") + "_paper_edit.html")
    html_path.write_text(html, encoding="utf-8")
    return html_path


# Recon kind per requested flag. RECON_CAM_POSES is the camera-track solve behind
# MAP assets; RECON_3D is the solve of a moment of action, for physics and novel
# views (see producer.md's AVAILABLE FLAGS).
RECON_FLAG_KINDS = {"RECON_CAM_POSES": "MAP", "RECON_3D": "EVENT"}


def beat_recon_flags(beat: dict) -> set[str]:
    """Which recon flags ONE beat actually needs -- the single place that decision
    is made, used both by _beat_flags (to derive the global flags file) and by
    build_recon_requests (to hand the notebook its jobs), so the two can't drift.

    Same doctrine as the rest of _beat_flags: correct what we have a rule for,
    trust requested_flags where we don't.
    This returns RECON flags only. A beat's other flags -- TRACKING (which a MAP
    beat always needs), GOOGLE_MAP/BEV_MAP, MASK_OVERLAYS -- are decided in
    _beat_flags and are not affected by anything here.

    - MAP beat: RECON_CAM_POSES as its one and only recon, overriding whatever was
      requested. A map is built off the camera-track solve, and a MAP beat never
      carries an EVENT recon -- not even for a PROJECTION_MAP, which belongs on its
      own beat (_beat_flags raises that as an error for the Producer to fix).
      Window length doesn't enter into it: a short MAP beat still gets a MAP recon.
    - PROJECTION_MAP on any other beat is a novel view of a RECON_3D, so it needs one.
    - Anything else: no rule, so the beat's own request stands."""
    requested = set(beat.get("requested_flags") or [])
    has_window = (beat.get("search_window_start_seconds") is not None
                  and beat.get("search_window_end_seconds") is not None)

    if beat.get("archetype") == "MAP" and has_window:
        return {"RECON_CAM_POSES"}

    flags = requested & set(RECON_FLAG_KINDS)
    if "PROJECTION_MAP" in requested and has_window:
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
    if "PROJECTION_MAP" in requested:
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
    """Writes <case_name>_flags.json as before. Also writes
    <case_name>_assistant_feedback.json (errors + notes from derive_flags) --
    that's the file to read from when building a MODE: revision re-draft
    prompt for Producer."""
    json_path = Path(paper_edit_json_path or paper_edit_path())
    flags, errors, notes = derive_flags(json_path, available_flag_keys, fps, gps_signal)
    flags_path = json_path.with_name(json_path.stem.replace("_paper_edit", "") + "_flags.json")
    flags_path.write_text(json.dumps(flags, indent=2), encoding="utf-8")

    feedback_path = json_path.with_name(json_path.stem.replace("_paper_edit", "") + "_ast_fb.json")
    feedback_path.write_text(json.dumps({"errors": errors, "notes": notes}, indent=2), encoding="utf-8")

    return flags_path


def tracking_windows(beat: dict, beats: list[dict]) -> list[tuple[float, float]]:
    """The time windows this beat's tracked_subject should actually be searched over.

    A real beat searches its own window -- short, consecutive footage, which is the
    only shape SAM3 tracks well.

    A MAP beat normally searches nothing. Its window is recon-scale (7700 frames on
    a 4-minute map), and SAM3 cannot track a sampled subset of that: tried at 1.3s
    and at 0.4s frame spacing, and both produced masks oscillating between empty and
    whole-frame, because propagation conditions on ~6 previous frames that at those
    gaps are unrelated images. Its subjects are expected to be covered by the event
    beats instead.

    BUT if no real beat tracks anything, the Producer has left the map's subjects
    with nowhere else to be found -- so sweep up the event footage: every real
    beat's window inside the MAP's span, merged. That is derived here rather than
    demanded of the Producer, and it deliberately does not put those subjects onto
    the event beats themselves (see build_tracking_requests on beat_ids), so no
    beat gains a mask overlay it never asked for.

    Used by build_tracking_requests and AY_claude_crop_matcher.descriptor_targets;
    they held separate copies of this rule and drifted, which cost a debugging
    session when one was fixed and the other silently kept returning nothing."""
    if beat.get("archetype") != "MAP":
        start_s = beat.get("search_window_start_seconds")
        end_s = beat.get("search_window_end_seconds")
        return [] if start_s is None or end_s is None else [(start_s, end_s)]

    if any(b.get("tracked_subject") for b in beats if b.get("archetype") != "MAP"):
        return []   # event beats cover it; the normal path

    map_start = beat.get("search_window_start_seconds")
    map_end = beat.get("search_window_end_seconds")
    spans = [
        (b["search_window_start_seconds"], b["search_window_end_seconds"], ())
        for b in beats
        if b.get("segment_type") == "real"
        and b.get("search_window_start_seconds") is not None
        and b.get("search_window_end_seconds") is not None
        and (map_start is None or b["search_window_end_seconds"] > map_start)
        and (map_end is None or b["search_window_start_seconds"] < map_end)
    ]
    if not spans:
        raise ValueError(
            f"MAP beat {beat.get('beat_id')} is the only beat tracking anything, but "
            f"there are no real beats inside its window to search -- nothing to sweep."
        )
    # Deliberately NOT merged. Merging touching windows is right when a Producer
    # names a subject across two adjacent beats, but the sweep hands every subject
    # every window, so adjacent beats chain: (15,33)+(33,65) becomes a single
    # 1500-frame span, well past the 650 at which SAM3 offloads video+state to host
    # RAM. Kept as the Producer's own beat windows, the worst case here is 960.
    return [(start_s, end_s) for start_s, end_s, _ in sorted(spans)]


def build_tracking_requests(paper_edit_json_path: str | Path = None, fps: float = None) -> list[dict]:
    """Merges tracked_subject spans per subject into list[{"subject","start_s","end_s","beat_ids"}].
    Spans are the Producer's search windows -- tracking runs before find_all_cut_points,
    so no beat has resolved frames yet.
    A gem_person_id dict entry groups under subject="person" (one class scan, not per-person).
    Which windows a beat contributes comes from tracking_windows -- normally its own,
    but a MAP beat sweeps the event footage when nothing else tracks anything."""
    if not fps:
        # Only read the video when the caller didn't already have fps -- keeps this
        # callable (and testable) without the cv2/ffmpeg stack B_video_processing pulls in.
        from A_Config import source_video_path
        from Two2D.B_video_processing import video_fps
        fps = video_fps(source_video_path())
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

    swept = False
    for beat in data["beats"]:
        tracked = beat.get("tracked_subject")
        if not tracked:
            continue
        windows = tracking_windows(beat, data["beats"])
        if windows and beat.get("archetype") == "MAP":
            swept = True
        # beat_ids is this beat's own, even for a MAP beat searching event windows --
        # the masks belong to the map, and tagging them with the event beats would
        # give those beats a mask overlay in assemble_paper_edit that they never
        # requested.
        beat_ids = tuple(beat.get("beat_id") or [])
        for start_s, end_s in windows:
            for tracked_subject_entry in tracked:
                subject = "person" if isinstance(tracked_subject_entry, dict) else tracked_subject_entry
                spans_by_subject.setdefault(subject, []).append((start_s, end_s, beat_ids))

    requests = []
    for subject, spans in spans_by_subject.items():
        # Merging touching spans is right when the Producer named a subject across
        # adjacent beats. It is wrong for a swept MAP beat, where every subject gets
        # every window and adjacent beats would chain into spans far past the 650
        # frames at which SAM3 offloads -- so a sweep keeps the beat windows as-is.
        # (When the sweep fires it is the only source of spans, so this is all-or-nothing.)
        # set() because the eight gem_person_id entries all collapse to subject
        # "person" and would otherwise queue the same span eight times -- _merge_spans
        # used to absorb that silently.
        for start_s, end_s, beat_ids in (sorted(set(spans)) if swept
                                          else _merge_spans(spans, max_span_seconds)):
            # Check tracking hasn't already been done
            start_frame, end_frame = round(start_s * fps), round(end_s * fps)
            span_range = range(start_frame, end_frame + 1)
            if detections_dict.get(_slugify_subject(subject), set()).intersection(span_range):
                continue
            requests.append({"subject": subject, "start_s": start_s, "end_s": end_s,
                              "beat_ids": sorted(beat_ids),
                              "expected_count": expected_by_subject.get(subject)})
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


def build_recon_requests(paper_edit_json_path=None, fps=None) -> list[dict]:
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
    if not fps:
        # Only read the video when the caller didn't already have fps -- keeps this
        # callable (and testable) without the cv2/ffmpeg stack B_video_processing pulls in.
        from A_Config import source_video_path
        from Two2D.B_video_processing import video_fps
        fps = video_fps(source_video_path())

    dirs = {"MAP":   (frames_for_cam_poses_dir, recon_for_MAP_dir),
            "EVENT": (frames_for_recon_dir,     recon_for_EVENT_dir)}

    requests = []
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
            key = beat_key(beat)
            frames_dir_fn, recon_dir_fn = dirs[kind]
            requests.append({
                "beat_key":    key,
                "beat_ids":    list(beat["beat_id"]),
                "order":       beat["order"],
                "archetype":   beat.get("archetype"),
                "kind":        kind,
                "flag":        flag,
                "start_frame": round(start_s * fps),
                "end_frame":   round(end_s * fps),
                "frames_dir":  frames_dir_fn(key),
                "recon_dir":   recon_dir_fn(key),
                # The beat itself, so renderers that need more of it than the key
                # (R_map_animator/P_trace_overlayer want duration_seconds and
                # requested_flags) don't have to re-open the paper edit to find it.
                "beat":        beat,
            })
            print(f"{key} {kind} ({flag}): {start_s}-{end_s}s @ {fps}fps "
                  f"-> frames {requests[-1]['start_frame']}-{requests[-1]['end_frame']}")

    return requests
