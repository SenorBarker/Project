"""Matches GEMINI'S people.json person_id -> SAM3 detection by box overlap (not seeding)."""

import csv
import json
from pathlib import Path


def gem_box_to_xyxy(gem_box, width, height):
    """people.json's box_2d is [y0, x0, y1, x1] normalized 0-1000 (same
    convention as A_gemini_v02.get_box/draw_point_overlay)."""
    y0, x0, y1, x1 = [int(v / 1000 * d) for v, d in zip(gem_box, (height, width, height, width))]
    return (x0, y0, x1, y1)


def sam_det_to_xyxy(sam_det, vid_w, vid_h):
    """A_LOCAL_SAM3's detections.csv box_xywh is corner-based (x, y = top-left),
    normalized 0-1 (sam3_video_inference.py divides by W_video/H_video before
    returning out_boxes_xywh -- A_LOCAL_SAM3.py writes that straight to CSV,
    never converted back to pixels)."""
    x, y, w, h = (float(sam_det["x"]) * vid_w, float(sam_det["y"]) * vid_h,
                  float(sam_det["w"]) * vid_w, float(sam_det["h"]) * vid_h)
    return (x, y, x + w, y + h)


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
    if not Path(SAM_dets_path).exists():
        print(f"No SAM3 tracking found at {SAM_dets_path} -- continuing with no SAM dets.")
        return []
    with open(SAM_dets_path, newline="") as f:
        return list(csv.DictReader(f))


def gem_person_targets_lookup(video_path=None, paper_edit_json_path=None, gem_people_json_path=None):
    """gem_person_id -> {appearance_targets: [{box_2d, target_frame}, ...], beat_ids}.
    Matching-candidates only -- the eventual analysis_2d_for_decisions entry
    is still built from frames_by_instance once matching resolves a track_id."""
    if paper_edit_json_path is None or gem_people_json_path is None:
        import A_Config
        if paper_edit_json_path is None:
            from render_paper_edit import paper_edit_path
            paper_edit_json_path = paper_edit_path("draft")
        if gem_people_json_path is None:
            gem_people_json_path = A_Config.query_dir() / f"{A_Config.case_name()}_people.json"
        
    paper_edit = json.loads(Path(paper_edit_json_path).read_text(encoding="utf-8"))
    gem_people = json.loads(Path(gem_people_json_path).read_text(encoding="utf-8"))

    # Pass 1: walk beats -> tracking requests (cheap, in-memory) to find every
    # distinct gem_person_id referenced and which beats reference it --
    # appearance_targets only need building once per person (same
    # gem_person["appearances"] regardless of which beat named them), same as
    # the original per-gem_person_id-first-seen behavior.
    person_obj_by_id = {}
    beat_ids_by_person = {}
    for beat in paper_edit["beats"]:
        producer_tracking_requests = beat.get("tracked_subject") or []
        if not producer_tracking_requests:
            continue

        for request in producer_tracking_requests:
            if not isinstance(request, dict):
                continue
            gem_person_id = request["gem_person_id"]
            if gem_person_id not in person_obj_by_id:
                person_obj_by_id[gem_person_id] = next(p for p in gem_people if p["person_id"] == gem_person_id)
            beat_ids_by_person.setdefault(gem_person_id, set()).update(beat.get("beat_id") or [])

       # box_2d is gone from people.json: A_gemini_v04's PEOPLE_SCHEMA dropped it
    # deliberately (box-IoU matching superseded by AY_claude_crop_matcher's
    # crops). Nothing left to look up per appearance -- beat_ids is the only
    # field analysis_2d_gemvsSAM reads off this now.
    return {gem_person_id: {"beat_ids": sorted(beat_ids_by_person.get(gem_person_id, set()))}
            for gem_person_id in person_obj_by_id}


    return gem_person_targets


def match_gem_people_to_sam(vid_w=None, vid_h=None, video_path=None, gem_person_targets=None, SAM_dets=None, iou_threshold=0.1):
    """gem_person_id -> sorted list of matched track_id (one to many --
    different appearances can land in different spans, where SAM3's
    track_id numbering restarts, so the same person legitimately
    matches more than one track_id across the whole case)."""
    if vid_w is None or vid_h is None or video_path is None:
        import A_Config
        from Two2D.B_video_processing import video_dims
        video_path = video_path or A_Config.source_video_path()
        vid_w, vid_h, _ = video_dims(video_path)

    if gem_person_targets is None:
        gem_person_targets = gem_person_targets_lookup(video_path)
    if SAM_dets is None:
        SAM_dets = load_SAM_dets()

    results = {}
    for gem_person_id, target in gem_person_targets.items():
        matched_track_ids = set()
        for appearance_target in target["appearance_targets"]:
            target_frame = appearance_target["target_frame"]
            sam_fr_dets = [d for d in SAM_dets if int(float(d["frame_idx"])) == target_frame]
            target_box = gem_box_to_xyxy(appearance_target["box_2d"], vid_w, vid_h)

            best_track_id, best_iou = None, 0.0
            for sam_det in sam_fr_dets:
                iou = iou_xyxy(target_box, sam_det_to_xyxy(sam_det, vid_w, vid_h))
                if iou > best_iou:
                    best_track_id, best_iou = int(sam_det["track_id"]), iou
            if best_iou >= iou_threshold:
                matched_track_ids.add(best_track_id)

        results[gem_person_id] = sorted(matched_track_ids)
        if not results[gem_person_id]:
            # silent here means the person vanishes from analysis_2d_for_decisions
            # entirely (empty masks_dirs, empty Subject_frames_present) -- say so.
            print(f"No SAM track matched {gem_person_id} -- check iou_threshold={iou_threshold}.")

    return results
