"""Matches GEMINI'S people.json person_id -> SAM3 detection by box overlap (not seeding)."""

import ast
import csv
import json
from pathlib import Path


def gem_box_to_xyxy(gem_box, width, height):
    """people.json's box_2d is [y0, x0, y1, x1] normalized 0-1000 (same
    convention as A_gemini_v02.get_box/draw_point_overlay)."""
    y0, x0, y1, x1 = [int(v / 1000 * d) for v, d in zip(gem_box, (height, width, height, width))]
    return (x0, y0, x1, y1)


def sam_det_to_xyxy(sam_det):
    """Roboflow's standard detection shape is center-based (x, y, width,
    height); fall back to the bounding box of the mask polygon ("points")
    if those fields aren't present."""
    if all(k in sam_det and sam_det[k] not in (None, "") for k in ("x", "y", "width", "height")):
        x, y, w, h = (float(sam_det["x"]), float(sam_det["y"]), float(sam_det["width"]), float(sam_det["height"]))
        return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)

    points = sam_det.get("points")
    if points:
        if isinstance(points, str):
            points = ast.literal_eval(points)
        xs = [p["x"] for p in points]
        ys = [p["y"] for p in points]
        return (min(xs), min(ys), max(xs), max(ys))

    return None


def iou_xyxy(box_a, box_b):
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b

    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter_w, inter_h = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    return inter_area / (area_a + area_b - inter_area)


def load_SAM_dets(raw_detection_rows=None, SAM_dets_path=None):
    """Prefer already-in-memory rows (same pipeline call); fall back to detections.csv."""
    if raw_detection_rows is not None:
        return raw_detection_rows
    if SAM_dets_path is None:
        import A_Config
        SAM_dets_path = A_Config.sam3_masks_dir() / "detections.csv"
    with open(SAM_dets_path, newline="") as f:
        return list(csv.DictReader(f))


def gem_person_targets_lookup(fps, paper_edit_json_path=None, gem_people_json_path=None):
    """gem_person_id -> {appearance_targets: [{box_2d, target_frame}, ...], beat_ids}.
    Matching-candidates only -- the eventual analysis_2d_for_decisions entry
    is still built from frames_by_instance once matching resolves a tracker_id."""
    if paper_edit_json_path is None or gem_people_json_path is None:
        import A_Config
        if paper_edit_json_path is None:
            paper_edit_json_path = A_Config.case_dir() / "012_agent_p_output" / f"{A_Config.case_name()}_paper_edit_draft.json"
        if gem_people_json_path is None:
            gem_people_json_path = A_Config.query_dir() / f"{A_Config.case_name()}_people.json"

    paper_edit = json.loads(Path(paper_edit_json_path).read_text(encoding="utf-8"))
    gem_people = json.loads(Path(gem_people_json_path).read_text(encoding="utf-8"))

    gem_person_targets = {}  # return value, built up below

    for beat in paper_edit["beats"]:
        producer_tracking_requests = beat.get("tracked_subject") or []
        if not producer_tracking_requests:
            continue

        for request in producer_tracking_requests:
            if not isinstance(request, dict):
                continue
            gem_person_id = request["gem_person_id"]
            gem_person = next(p for p in gem_people if p["person_id"] == gem_person_id)

            appearance_targets = [
                {"box_2d": a["box_2d"], "target_frame": round((a["start_s"] + 0.5) * fps)}
                for a in gem_person["appearances"]
            ]

            if gem_person_id not in gem_person_targets:
                gem_person_targets[gem_person_id] = {"appearance_targets": appearance_targets, "beat_ids": set()}
            gem_person_targets[gem_person_id]["beat_ids"].update(beat.get("beat_id") or [])

    for target in gem_person_targets.values():
        target["beat_ids"] = sorted(target["beat_ids"])

    return gem_person_targets


def match_gem_people_to_sam(vid_w=None, vid_h=None, fps=None, gem_person_targets=None, SAM_dets=None, iou_threshold=0.1):
    """gem_person_id -> sorted list of matched sam_tracker_id (one to many --
    different appearances can land in different spans, where SAM3's
    tracker_id numbering restarts, so the same person legitimately matches
    more than one tracker_id across the whole case)."""
    if vid_w is None or vid_h is None or fps is None:
        import A_Config
        from Two2D.B_video_processing import video_dims
        vid_w, vid_h, fps = video_dims(A_Config.source_video_path())

    if gem_person_targets is None:
        gem_person_targets = gem_person_targets_lookup(fps)
    if SAM_dets is None:
        SAM_dets = load_SAM_dets()

    results = {}
    for gem_person_id, target in gem_person_targets.items():
        matched_tracker_ids = set()
        for appearance_target in target["appearance_targets"]:
            target_frame = appearance_target["target_frame"]
            sam_fr_dets = [d for d in SAM_dets if int(float(d["frame_idx"])) == target_frame]
            target_box = gem_box_to_xyxy(appearance_target["box_2d"], vid_w, vid_h)

            best_tracker_id, best_iou = None, 0.0
            for sam_det in sam_fr_dets:
                sam_box = sam_det_to_xyxy(sam_det)
                if sam_box is None:
                    continue
                iou = iou_xyxy(target_box, sam_box)
                if iou > best_iou:
                    best_tracker_id, best_iou = sam_det["tracker_id"], iou
            if best_iou >= iou_threshold:
                matched_tracker_ids.add(best_tracker_id)

        results[gem_person_id] = sorted(matched_tracker_ids)

    return results
