"""
Local (on-GPU) SAM3 mask tracking -- a drop-in replacement for
A_ROBOFLOW_SAM3._run_sam3_tracking's hosted-workflow call, using the SAM3
checkout in Models/sam3 instead of Roboflow's serverless WebRTC endpoint.

Same input ({subject: [{"subject","start_s","end_s","beat_ids"}, ...]}), same
{instance_key: entry} output, same sam3_masks_dir()/<subject_slug>-<n>/{:04d}.png
mask layout on disk -- so D_2d_analysis and Two2D/W_video_editor read it
unchanged. The Roboflow path stays intact in A_ROBOFLOW_SAM3.py; flipping
USE_LOCAL_SAM3 there switches between them.

Three things genuinely differ from the hosted path, all in this module's favour:

  * Masks come back as true binary masks, not polygon outlines -- the hosted
    workflow only ever serialized "points" (see decode_mask_from_detection),
    which capped mask fidelity at whatever the polygon captured.
  * Object IDs are real integers from the tracker (out_obj_ids), so none of
    A_ROBOFLOW_SAM3's TRACK_ID_KEYS guesswork is needed.
  * No session retry loop. Roboflow's worker-init was flaky enough to need 5
    attempts with backoff; a local model either loads or doesn't, so failures
    raise immediately rather than being retried into a longer failure.

Expect instance COUNTS to differ from a Roboflow run of the same clip: local
SAM3 text prompting is exhaustive open-vocabulary detection, while the hosted
workflow wrapped its own tracker heuristics, and `threshold` there is not
guaranteed to be the same scale as output_prob_thresh here.
"""

import gc
import shutil
import time

import cv2
import numpy as np
import pandas as pd
import torch

try:
    # How the rest of src/ imports this package (see D_2d_analysis,
    # Two2D/W_video_editor); the notebook only puts src/ on sys.path.
    from run_models.A_ROBOFLOW_SAM3 import MASK_NAME_FMT, _slugify_subject, make_span_clip
except ModuleNotFoundError:
    # How the spike scripts import it, running from inside run_models/.
    from A_ROBOFLOW_SAM3 import MASK_NAME_FMT, _slugify_subject, make_span_clip

# Built once per process and reused across every subject and span -- the
# checkpoint is 848M params and takes tens of seconds to load, so rebuilding
# it per span would dominate the run.
_PREDICTOR = None

# Past this span length, SAM3's all-on-GPU default OOMs a 24GB card late in
# propagation (683 frames died at 681); offload video+state to host RAM instead.
OFFLOAD_ABOVE_N_FRAMES = 650

# A prompt implies how many things it should match. Track ids accumulating over a
# span is just re-identification churn, but one frame holding far more instances
# than were asked for (1 bush requested, 17 found) means the prompt is wrong --
# and tracking them all exhausts the GPU, since per-frame state scales with
# instance count.
INSTANCE_COUNT_LEEWAY = 2


class ObjectTooDense(RuntimeError):
    """A single frame held more instances than the prompt asked for."""


def get_predictor():
    """
    Returns the process-wide SAM3 video predictor, building it on first use.

    Zero-arg: resolves its own checkpoint (downloaded from HuggingFace into
    the HF cache on first call -- facebook/sam3 is a gated repo, so this
    needs `hf auth login` or HF_TOKEN to have been set up once) and its own
    device.
    """
    global _PREDICTOR
    if _PREDICTOR is None:
        from sam3.model_builder import build_sam3_video_predictor

        t0 = time.time()
        _PREDICTOR = build_sam3_video_predictor()
        print(f"[local SAM3] model loaded in {time.time() - t0:.1f}s")
    return _PREDICTOR


def release_predictor():
    """Drop the model off the GPU (~4.7GB) so the next stage has a clean card."""
    global _PREDICTOR
    if _PREDICTOR is None:
        return
    try:
        _PREDICTOR.shutdown()
    except Exception as exc:
        print(f"[local SAM3] shutdown failed, freeing anyway: {exc}")
    _PREDICTOR = None
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[local SAM3] released -- {torch.cuda.memory_allocated()/1024**3:.2f}GiB still allocated")


def _track_span(predictor, clip_path, subject, threshold, expected_count=None):
    """
    Runs one already-cut clip through SAM3 with a text prompt, yielding
    (clip_frame_idx, obj_id, binary_mask, box_xywh) per detection.

    clip_frame_idx is 0-based within the clip -- unlike the Roboflow
    workflow's 1-based metadata.frame_id, so callers add it straight to the
    clip's absolute start frame with no -1 correction.

    Raises ObjectTooDense if any one frame holds more than
    expected_count + INSTANCE_COUNT_LEEWAY instances (no limit when
    expected_count is None).
    """
    max_instances = (
        None if expected_count is None else expected_count + INSTANCE_COUNT_LEEWAY
    )
    cap = cv2.VideoCapture(str(clip_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    offload = n_frames > OFFLOAD_ABOVE_N_FRAMES
    if offload:
        print(f"[local SAM3] {n_frames} frames -> offloading video+state to CPU")

    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=str(clip_path),
            offload_video_to_cpu=offload,
            offload_state_to_cpu=offload,
        )
    )
    session_id = response["session_id"]
    try:
        predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text=subject,
                output_prob_thresh=threshold,
            )
        )
        for response in predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
                propagation_direction="forward",
                start_frame_index=0,
                output_prob_thresh=threshold,
            )
        ):
            sam_outputs = response["outputs"]
            sam_obj_ids = np.asarray(sam_outputs["out_obj_ids"])
            # Checked before anything is yielded, so a doomed span writes no masks.
            if max_instances is not None and len(sam_obj_ids) > max_instances:
                raise ObjectTooDense(
                    f"'{subject}' expected ~{expected_count}, but frame "
                    f"{response['frame_index']} held {len(sam_obj_ids)}"
                )
            sam_masks = np.asarray(sam_outputs["out_binary_masks"])
            sam_boxes = np.asarray(sam_outputs["out_boxes_xywh"])
            # per-frame mask quality; out_probs is fixed at first detection, and
            # some of SAM3's output paths don't carry the per-frame one at all
            sam_probs = np.asarray(
                sam_outputs.get("out_tracker_probs",
                                sam_outputs.get("out_sam2_probs", sam_outputs["out_probs"]))
            )
            for i, sam_obj_id in enumerate(sam_obj_ids.tolist()):
                yield (response["frame_index"], sam_obj_id, sam_masks[i],
                       sam_boxes[i], float(sam_probs[i]))
    finally:
        predictor.handle_request(
            request=dict(type="close_session", session_id=session_id)
        )


def _write_id_shots(best, clip_path, masks_root, subject_slug, start_frame):
    """One crop per track, cut from the frame SAM3 scored highest, while the span
    clip is still on disk."""
    if not best:
        return
    from Two2D.B_video_processing import frames_at_indices

    out_dir = masks_root / "_sam_id_shots"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = frames_at_indices(clip_path, [clip_frame for _, clip_frame, _ in best.values()])

    for track_id, (prob, clip_frame, mask) in best.items():
        frame = frames.get(clip_frame)
        if frame is None:
            continue
        ys, xs = np.nonzero(mask)
        crop = (frame * (mask[:, :, None] > 0))[ys.min():ys.max(), xs.min():xs.max()]
        name = f"{subject_slug}-{track_id:02d}_f{start_frame + clip_frame:05d}.png"
        cv2.imwrite(str(out_dir / name), crop)


def _run_sam3_tracking_local(video_path, tracking_requests_by_subject, threshold,
                             *_roboflow_args, **_roboflow_kwargs):
    """
    Local counterpart to A_ROBOFLOW_SAM3._run_sam3_tracking, with the same
    signature and the same return value. The hosted path's requested_region /
    requested_plan arguments are accepted and ignored so callers don't have to
    know which backend they're on.
    """
    from A_Config import sam3_masks_dir
    from D_2d_analysis import contiguous_durations

    print("initialising tracking (local SAM3)")
    predictor = get_predictor()

    masks_root = sam3_masks_dir()
    masks_root.mkdir(parents=True, exist_ok=True)

    analysis_2d_for_decisions = {}
    raw_detection_rows = []

    for subject, requests in tracking_requests_by_subject.items():
        subject_slug = _slugify_subject(subject)
        frames_by_instance = {}  # global track_id -> [frame_idx, ...]
        beat_ids_by_instance = {}  # global track_id -> set(beat_id)
        subject_beat_ids = set()  # union across all this subject's requests, for the no-detection case
        subject_too_dense = []  # ObjectTooDense reasons, for the no-detection entry
        next_track_id = 1

        for request in requests:
            span_start_s = request["start_s"]
            span_end_s = request["end_s"]
            request_beat_ids = request.get("beat_ids", [])
            subject_beat_ids.update(request_beat_ids)

            tmp_clip = masks_root / f"_span_with_handle_{subject_slug}.mp4"
            start_frame, fps, width, height = make_span_clip(video_path, tmp_clip, span_start_s, span_end_s)
            print(f"[{subject_slug}] requested {span_start_s}-{span_end_s}s -> keyframe-aligned start frame {start_frame}")

            # obj_id from SAM3 is scoped to this one session, so it restarts
            # for every new span -- map it to the subject's globally-continuing
            # instance number instead, or span 2's "instance 1" would silently
            # overwrite span 1's "instance 1" on disk.
            local_to_global = {}
            span_frames = {}  # global track_id -> [frame_idx, ...]
            best = {}  # global track_id -> (prob, clip_frame_idx, mask)
            rows_before_span = len(raw_detection_rows)
            try:
                for clip_frame_idx, sam_obj_id, sam_mask, sam_box, sam_prob in _track_span(
                    predictor, tmp_clip, subject, threshold, request.get("expected_count")
                ):
                    frame_idx = start_frame + clip_frame_idx

                    if sam_obj_id not in local_to_global:
                        local_to_global[sam_obj_id] = next_track_id
                        next_track_id += 1
                    track_id = local_to_global[sam_obj_id]

                    track_id_dir = masks_root / f"{subject_slug}-{track_id:02d}"
                    track_id_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(
                        str(track_id_dir / MASK_NAME_FMT.format(frame_idx)),
                        sam_mask.astype(np.uint8) * 255,
                    )
                    span_frames.setdefault(track_id, []).append(frame_idx)

                    # SAM3's own confidence for this masklet on this frame -- keep
                    # the frame it was surest about as the track's ID shot
                    if sam_prob > best.get(track_id, (0.0,))[0]:
                        best[track_id] = (sam_prob, clip_frame_idx, sam_mask)

                    x, y, w, h = [float(v) for v in np.asarray(sam_box).reshape(-1)[:4]]
                    raw_detection_rows.append({
                        "frame_idx": frame_idx, "subject": subject_slug,
                        "obj_id": sam_obj_id, "track_id": track_id,
                        "x": x, "y": y, "w": w, "h": h,
                    })
            except ObjectTooDense as too_dense:
                # Leave nothing half-written behind: this span's track ids are
                # its own (next_track_id only ever moves forward), so their dirs
                # and rows can go without touching an earlier span's output.
                print(f"[{subject_slug}] OBJECT TOO DENSE -- {too_dense}; skipping span")
                subject_too_dense.append(str(too_dense))
                for track_id in local_to_global.values():
                    shutil.rmtree(masks_root / f"{subject_slug}-{track_id:02d}", ignore_errors=True)
                del raw_detection_rows[rows_before_span:]
                span_frames.clear()
                best.clear()
            else:
                _write_id_shots(best, tmp_clip, masks_root, subject_slug, start_frame)
            finally:
                tmp_clip.unlink(missing_ok=True)
                # predictor is a process-wide singleton, so every span's freed
                # session tensors otherwise stay in torch's cache and stack up.
                gc.collect()
                torch.cuda.empty_cache()

            for track_id, frames in span_frames.items():
                frames_by_instance.setdefault(track_id, []).extend(frames)
                beat_ids_by_instance.setdefault(track_id, set()).update(request_beat_ids)

        if frames_by_instance:
            for track_id, instance_frames in frames_by_instance.items():
                instance_key = f"{subject_slug}-{track_id:02d}"
                seqs, best_idx = contiguous_durations(instance_frames, tolerance=15)
                entry = {
                    "Subject_frames_present": instance_frames,
                    "subject_first_frame": min(instance_frames),
                    "subject_last_frame": max(instance_frames),
                    "subject_duration_frames": max(instance_frames) - min(instance_frames),
                    "continuous frame sequences": seqs,
                    "best_seq_idx": best_idx,
                    "masks_dir": str(masks_root / instance_key),
                    "beat_ids": sorted(beat_ids_by_instance.get(track_id, set())),
                }
                analysis_2d_for_decisions[instance_key] = entry
        else:
            if subject_too_dense:
                print(f"[{subject_slug}] not tracked -- every span was too dense")
            else:
                print(f"[{subject_slug}] no detections across any requested span")
            entry = {
                "Subject_frames_present": [],
                "subject_first_frame": None,
                "subject_last_frame": None,
                "continuous frame sequences": [],
                "best_seq_idx": None,
                "masks_dir": str(masks_root / subject_slug),
                "beat_ids": sorted(subject_beat_ids),
                "error": "OBJECT_TOO_DENSE" if subject_too_dense else None,
                "error_detail": subject_too_dense or None,
            }
            analysis_2d_for_decisions[subject_slug] = entry

    release_predictor()  # tracking is done; don't hold the card for the next stage

    if raw_detection_rows:
        pd.DataFrame(raw_detection_rows).to_csv(masks_root / "detections.csv", index=False)

    # Two different failures used to share one message, and "found no detections"
    # sends you looking at the footage when the real answer is that nothing was
    # ever asked for.
    if not tracking_requests_by_subject:
        raise RuntimeError(
            "SAM3 tracker had nothing to run: no beat produced a tracking request. "
            "Check that some beat has a tracked_subject -- see "
            "render_paper_edit.build_tracking_requests/tracking_windows."
        )

    if not any(v["Subject_frames_present"] for v in analysis_2d_for_decisions.values()):
        raise RuntimeError(
            f"SAM3 tracker ran {sum(len(r) for r in tracking_requests_by_subject.values())} "
            f"request(s) for {sorted(tracking_requests_by_subject)} but found no detections "
            f"in any of them."
        )

    return analysis_2d_for_decisions
