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


def derive_flags(paper_edit_json_path: str | Path, available_flag_keys) -> dict:
    """Builds the flags dict straight from Producer's own paper-edit JSON —
    unions every beat's requested_flags against the full key set. No model
    call: Producer already wrote this data once, in requested_flags; this
    is a lookup, not a second document to generate."""
    data = json.loads(Path(paper_edit_json_path).read_text(encoding="utf-8"))
    requested = set()
    for beat in data["beats"]:
        for flag in beat.get("requested_flags") or []:
            requested.add(flag)
    return {k: (k in requested) for k in available_flag_keys}


def derive_and_save_flags(paper_edit_json_path: str | Path, available_flag_keys) -> Path:
    json_path = Path(paper_edit_json_path)
    flags = derive_flags(json_path, available_flag_keys)
    flags_path = json_path.with_name(json_path.stem.replace("_paper_edit", "") + "_flags.json")
    flags_path.write_text(json.dumps(flags, indent=2), encoding="utf-8")
    return flags_path


def build_tracking_requests(paper_edit_json_path: str | Path, fps: float) -> list[dict]:
    """Collects every beat's tracked_subject list into a tracking-*request*-
    shaped list[{"beat_id","subject","start_s","end_s"}] for
    A_YOLO_seg.track_subject_masks_from_hints or
    A_ROBOFLOW_SAM3.track_subject_sam3 -- one dict per requested subject.
    Deliberately not called "detections": these are queries asking a
    tracker to go find something, not results a tracker already found (see
    e.g. A_ROBOFLOW_SAM3's `dets`/`raw_detection_rows`, which are the actual
    detections this feeds into). Each subject in a beat's tracked_subject
    list shares that beat's own time range (no per-subject sub-spans).
    Beats reach this function in draft form (before find_all_cut_points
    resolves cut points), so fixed_frames beats convert start_frame/
    end_frame via fps and auto_select beats fall back to their (still
    coarse) search window. beat_id is drawn from beat["beat_id"][0] -- at
    draft time this list is always single-element; multi-element beat_id
    lists only appear later, after a revision-pass merge, and are a
    read-time (not write-time) concern."""
    data = json.loads(Path(paper_edit_json_path).read_text(encoding="utf-8"))
    requests = []
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
        beat_id = beat["beat_id"][0]
        for subject in tracked:
            requests.append({
                "beat_id": beat_id,
                "subject": subject,
                "start_s": start_s,
                "end_s": end_s,
            })
    return requests
