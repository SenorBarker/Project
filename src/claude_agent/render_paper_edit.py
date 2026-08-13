"""Renders a Producer paper-edit JSON (see .claude/agents/producer.md's
Output structure section) into the fixed HTML document. Pure code, no LLM —
formatting is deterministic and identical across every run."""
import json
from pathlib import Path

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
    if beat.get("cut_mode") == "fixed_frames" or beat.get("auto_select_resolved"):
        frame_range = f'{beat.get("start_frame")}–{beat.get("end_frame")}'
        lead_in = beat.get("lead_in_seconds")
        lead_in_txt = f' | lead_in_seconds: {lead_in}' if lead_in else ""
        lines.append(f'<div class="field"><span class="label">Frames:</span> {_esc(frame_range)}{_esc(lead_in_txt)}</div>')
    elif beat.get("cut_mode") == "auto_select":
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


def render_and_save(paper_edit_json_path: str | Path) -> Path:
    """Reads a Producer-written <case_name>_paper_edit.json and writes the
    matching <case_name>_paper_edit.html next to it. Returns the html path."""
    json_path = Path(paper_edit_json_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    html = render_html(data)
    html_path = json_path.with_name(json_path.stem.replace("_paper_edit", "") + "_paper_edit.html")
    html_path.write_text(html, encoding="utf-8")
    return html_path


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

        window_start = beat.get("search_window_start_seconds")
        window_end = beat.get("search_window_end_seconds")
        if window_start is None or window_end is None:
            errors.append(
                f"MAP beat {beat_id} needs a recon but has no "
                f"search_window_start_seconds/search_window_end_seconds -- Producer must set them."
            )
            for f in ("RECON_CAM_POSES", "RECON_3D"):
                if f in requested:
                    flags.add(f)  # can't verify, but don't override what was right
        else:
            # RECON_CAM_POSES vs RECON_3D -- source footage duration,
            # >1min vs <=1min. NOT the map beat's own duration_seconds.
            right_flag = "RECON_CAM_POSES" if (window_end - window_start) > 60 else "RECON_3D"
            flags.add(right_flag)  # CHECK
            if right_flag not in requested:
                notes.append(f"Beat {beat_id}: added {right_flag} (Producer didn't request it).")

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
    # that. But it's a novel view of a RECON_3D, so it always needs one.
    if "PROJECTION_MAP" in requested:
        flags.add("PROJECTION_MAP")
        if beat.get("search_window_start_seconds") is None or beat.get("search_window_end_seconds") is None:
            errors.append(
                f"Beat {beat_id} needs RECON_3D (for PROJECTION_MAP) but has no "
                f"search_window_start_seconds/search_window_end_seconds -- Producer must set them."
            )
            if "RECON_3D" in requested:
                flags.add("RECON_3D")  # can't verify, but don't override what was right
        else:
            flags.add("RECON_3D")  # CHECK
            if "RECON_3D" not in requested:
                notes.append(f"Beat {beat_id}: added RECON_3D (Producer didn't request it).")

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


def derive_flags(paper_edit_json_path: str | Path, available_flag_keys, fps: float, gps_signal: str = None) -> tuple[dict, list[str], list[str]]:
    """For each beat, checks its flags against the rules in _beat_flags and
    corrects any that are wrong. Final dict is the union across all beats.
    Collects every beat's problems before raising, so Producer gets the
    full list to fix in one revision pass instead of one-at-a-time."""
    data = json.loads(Path(paper_edit_json_path).read_text(encoding="utf-8"))
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


def derive_and_save_flags(paper_edit_json_path: str | Path, available_flag_keys, fps: float, gps_signal: str = None) -> Path:
    """Writes <case_name>_flags.json as before. Also writes
    <case_name>_assistant_feedback.json (errors + notes from derive_flags) --
    that's the file to read from when building a MODE: revision re-draft
    prompt for Producer."""
    json_path = Path(paper_edit_json_path)
    flags, errors, notes = derive_flags(json_path, available_flag_keys, fps, gps_signal)
    flags_path = json_path.with_name(json_path.stem.replace("_paper_edit", "") + "_flags.json")
    flags_path.write_text(json.dumps(flags, indent=2), encoding="utf-8")

    feedback_path = json_path.with_name(json_path.stem.replace("_paper_edit", "") + "_ast_fb.json")
    feedback_path.write_text(json.dumps({"errors": errors, "notes": notes}, indent=2), encoding="utf-8")

    return flags_path


def build_tracking_requests(paper_edit_json_path: str | Path, fps: float) -> list[dict]:
    """Merges tracked_subject spans per subject into list[{"subject","start_s","end_s","beat_ids"}].
    A gem_person_id dict entry groups under subject="person" (one class scan, not per-person)."""
    data = json.loads(Path(paper_edit_json_path).read_text(encoding="utf-8"))
    spans_by_subject = {}

    for beat in data["beats"]:
        tracked = beat.get("tracked_subject")
        if not tracked:
            continue
        if beat.get("cut_mode") == "fixed_frames":
            start_s = beat["start_frame"] / fps
            end_s = beat["end_frame"] / fps
        else:
            start_s = beat["search_window_start_seconds"]
            end_s = beat["search_window_end_seconds"]
        beat_ids = tuple(beat.get("beat_id") or [])
        for tracked_subject_entry in tracked:
            subject = "person" if isinstance(tracked_subject_entry, dict) else tracked_subject_entry
            spans_by_subject.setdefault(subject, []).append((start_s, end_s, beat_ids))

    requests = []
    for subject, spans in spans_by_subject.items():
        for start_s, end_s, beat_ids in _merge_spans(spans):
            requests.append({"subject": subject, "start_s": start_s, "end_s": end_s,
                              "beat_ids": sorted(beat_ids)})
    return requests


def _merge_spans(spans: list[tuple[float, float, tuple[str, ...]]]) -> list[tuple[float, float, set]]:
    """Merges overlapping/touching (start_s, end_s, beat_ids) spans into
    their union, unioning beat_ids along with the time range."""
    merged = []
    for start_s, end_s, beat_ids in sorted(spans, key=lambda s: (s[0], s[1])):
        if merged and start_s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end_s), merged[-1][2] | set(beat_ids))
        else:
            merged.append((start_s, end_s, set(beat_ids)))
    return merged
