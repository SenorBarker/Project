import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from C_CSV_report import add_to_asset_list
from A_Config import assets_dir, asset_name, case_dir, case_name, agent_p_output_dir, report_path, to_report_path, sam3_masks_dir, length_units
from Two2D.B_video_processing import video_fps
from run_models.A_ROBOFLOW_SAM3 import _slugify_subject
from P_projection_mapping import reproject

MASK_NAME_FMT = "{:04d}.png"
CUT_FADE_S = 0.015  # audio ramp on both ends of every clip -- kills cut-point pops

#-------TIMELINE-----------


def video_edit(video_path, start, end, out_path, handles = 1):
    '''Cuts [start, end] (frame numbers) out of video_path and re-exports it to
    out_path, re-encoding for a frame-accurate cut (a fast stream-copy would
    only snap to the nearest keyframe, not the exact requested frame).
    handles is extra padding, in seconds, added onto both the start and the
    end before cutting (e.g. handles=2 -> 2s earlier, 2s later).
    Also appends the output path to the asset list CSV.'''
    fps = video_fps(video_path)
    start_s = max(0.0, start / fps - handles)
    end_s   = end / fps + handles

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path),
         "-ss", f"{start_s:.3f}", "-to", f"{end_s:.3f}",
         "-c:v", "libx264", "-c:a", "aac", str(out_path)],
        check=True,
    )

    add_to_asset_list({"edited_video": to_report_path(out_path)})
    return out_path


def mask_frame_range(masks_dir):
    '''(first_frame, last_frame) that masks_dir actually holds masks for, read
    off the MASK_NAME_FMT-named PNG filenames -- so callers never have to
    hard-code a track's span. Raises if the folder holds no frame-numbered
    PNGs (e.g. it's _id_crops, or the track wrote nothing).'''
    frames = sorted(int(p.stem) for p in Path(masks_dir).glob("*.png") if p.stem.isdigit())
    if not frames:
        raise ValueError(f"No frame-numbered mask PNGs in {masks_dir}.")
    return frames[0], frames[-1]


def _load_mask_bool(mask_dir, frame_idx):
    '''None if mask_dir is not set or no mask file exists for this frame.'''
    if mask_dir is None:
        return None
    path = os.path.join(mask_dir, MASK_NAME_FMT.format(frame_idx))
    if not os.path.exists(path):
        return None
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        print(f"Warning: could not decode mask {path} -- treating frame {frame_idx} as unmasked.")
        return None
    return m > 0


def apply_mask_overlay(frame_bgr, mask_bool, color=(0, 200, 0), alpha=0.4):
    '''Blend color into frame_bgr wherever mask_bool is True. Green is
    channel-symmetric so this works the same whether frame_bgr is BGR or RGB.'''
    out = frame_bgr.copy()
    out[mask_bool] = (out[mask_bool] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return out


#colour per track, cycled -- distinct enough to tell fragments apart at a glance
_TRACK_COLORS = [(0, 200, 0), (200, 0, 200), (0, 165, 255), (255, 200, 0),
                 (0, 0, 255), (255, 0, 128), (0, 255, 255), (128, 0, 255)]


#-------SCREEN LABELS (position/speed/heading/distance/confidence overlay)-----------

def project_to_pixel(recon, frame_idx, point_xyz, frame_w, frame_h):
    '''Model-space (x,y,z) -> pixel (u,v) in this recon frame's own source-video
    frame, via that frame's own extrinsic/intrinsic. None if the point is behind
    the camera, has no matching recon row, or lands outside the frame.'''
    row = recon.frame_to_row.get(frame_idx)
    if row is None or point_xyz is None or not np.isfinite(np.asarray(point_xyz)).all():
        return None
    intrinsic = recon.preds["intrinsic"][row]
    extrinsic = recon.preds["extrinsic"][row]
    point = torch.as_tensor(np.asarray(point_xyz), dtype=extrinsic.dtype, device=extrinsic.device)
    u, v, _z, valid = reproject(point, intrinsic, extrinsic)
    u, v = float(u), float(v)
    if not bool(valid) or not (0 <= u < frame_w and 0 <= v < frame_h):
        return None
    return u, v


def draw_subject_label(frame, u, v, subject_slug, label, color=(255, 255, 255)):
    '''Small multi-line text block (position/speed/heading/distance/confidence)
    next to (u, v), on a translucent background so it stays legible over any
    video content.'''
    lines = [subject_slug]
    pos = label.get("position")
    if pos is not None and np.isfinite(np.asarray(pos)).all():
        lines.append(f"pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
    speed = label.get("speed_kmh")
    if speed is not None and np.isfinite(speed):
        lines.append(f"speed: {speed:.1f} km/h")
    heading = label.get("heading")
    if heading is not None and np.isfinite(heading):
        lines.append(f"heading: {heading:.0f} deg")
    dist = label.get("dist_cam")
    if dist is not None and np.isfinite(dist):
        lines.append(f"dist: {dist:.1f} {length_units()}")
    conf = label.get("confidence")
    if conf is not None and np.isfinite(conf):
        lines.append(f"conf: {conf:.0f}")

    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
    line_h = 18
    x, y = int(u) + 10, int(v)
    box_w = max(cv2.getTextSize(l, font, scale, thick)[0][0] for l in lines) + 8
    box_h = line_h * len(lines) + 6

    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 4, y - 12), (x - 4 + box_w, y - 12 + box_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + i * line_h), font, scale, color, thick, cv2.LINE_AA)
    return frame


def build_frame_label_lookup(recon_jobs):
    '''frame_idx -> [(subject_slug, label_dict, recon), ...], merged across every
    recon job's screen_labels. Frame-indexed rather than beat-indexed: a job's
    frame range isn't tied to the one beat that requested its reconstruction, and
    plenty of beats (plain "real" beats especially) have no job of their own at
    all -- so any beat's footage looks itself up by absolute frame number instead
    of trying to find "its" job.

    Collects job["screen_labels"] on demand (build_screen_labels) for any job
    that doesn't already have it, so callers (assemble_paper_edit,
    render_screen_labels_video) don't need a separate notebook step to
    populate it first.'''
    from Q_subject_analysis import build_screen_labels

    lookup = {}
    for job in recon_jobs:
        recon = job.get("recon")
        if recon is None:
            continue
        if "screen_labels" not in job:
            build_screen_labels(job)
        screen_labels = job.get("screen_labels") or {}
        for subject_slug, by_frame in screen_labels.items():
            # screen_labels is sparse (recon frames only) -- hold each label
            # forward to just before the next solved frame, else it only hits
            # on one frame out of many and flashes.
            frames = sorted(by_frame)
            for i, frame_idx in enumerate(frames):
                label = by_frame[frame_idx]
                next_frame = frames[i + 1] if i + 1 < len(frames) else frame_idx + 1
                for f in range(frame_idx, next_frame):
                    # source_frame_idx (not f) is what recon.frame_to_row has a
                    # row for -- held frames have no row of their own.
                    lookup.setdefault(f, []).append((subject_slug, label, recon, frame_idx))
    return lookup


def _draw_frame_labels(frame, frame_idx, frame_label_lookup):
    if not frame_label_lookup:
        return frame
    fh, fw = frame.shape[:2]
    for subject_slug, label, recon, source_frame_idx in frame_label_lookup.get(frame_idx, []):
        proj = project_to_pixel(recon, source_frame_idx, label.get("position"), fw, fh)
        if proj is not None:
            frame = draw_subject_label(frame, proj[0], proj[1], subject_slug, label)
    return frame


def frames_overlay_check(frames_dir, masks_dirs, name=None, review_fps=6, color=None):
    '''Mask-check video for a MAP beat's SPARSE masks, built from the recon's own
    frames instead of the source video.

    video_overlay_edit is the wrong tool for these: it cuts the source video from
    the first to the last masked frame, but frames mode only writes masks on the
    ~200 frames the recon solved, spread over the whole window. A track with 78
    masks across 7268 frames comes out as a 4-minute video that is ~99% unmasked --
    a flicker, not a track. Here every output frame is a recon frame, so every frame
    carries whatever masks it has and the track is actually readable.

    No audio: the frames are ~1.3s apart in source time, so there is no continuous
    audio that belongs to them. review_fps is a viewing speed, not real time.

    masks_dirs: one track dir, or several (all of a subject's fragmented tracks) --
    each gets its own colour, so one video shows how a subject broke up rather than
    one video per fragment. Dirs holding no frame-numbered PNGs (e.g. _sam_id_shots)
    are skipped rather than raising.

    Returns the mp4 path.'''
    from Two2D.B_video_processing import available_frames
    from A_Config import FRAME_NAME_FMT

    frames_dir = Path(frames_dir)
    if isinstance(masks_dirs, (str, Path)):
        masks_dirs = [masks_dirs]
    # Keep only dirs that actually hold {:04d}.png masks -- _sam_id_shots names its
    # crops <track>_f<frame>.png, which is what makes mask_frame_range raise.
    track_dirs = [d for d in (Path(m) for m in masks_dirs)
                  if d.is_dir() and any(p.stem.isdigit() for p in d.glob("*.png"))]
    if not track_dirs:
        raise ValueError(f"No frame-numbered mask PNGs in any of {list(masks_dirs)}.")

    frames = available_frames(frames_dir, FRAME_NAME_FMT)
    if not frames:
        raise ValueError(f"No {FRAME_NAME_FMT}-named frames in {frames_dir}.")

    name = name or asset_name()
    tmp_dir = assets_dir() / f"_{name}_overlay_frames"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    masked_counts = {d.name: 0 for d in track_dirs}
    try:
        for out_idx, frame_number in enumerate(frames):
            img = cv2.imread(str(frames_dir / FRAME_NAME_FMT.format(frame_number)))
            if img is None:
                continue
            for track_idx, track_dir in enumerate(track_dirs):
                mask = _load_mask_bool(track_dir, frame_number)
                if mask is None:
                    continue
                if mask.shape[:2] != img.shape[:2]:   # masks are written at model res
                    mask = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]),
                                      interpolation=cv2.INTER_NEAREST_EXACT) > 0
                img = apply_mask_overlay(
                    img, mask, color=color or _TRACK_COLORS[track_idx % len(_TRACK_COLORS)])
                masked_counts[track_dir.name] += 1
            # stamp the real source frame number -- the only way to tell where you are,
            # since playback time means nothing on a sparse sequence
            cv2.putText(img, f"frame {frame_number}", (12, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(tmp_dir / f"frame_{out_idx:04d}.png"), img)

        out_path = assets_dir() / f"{name}_mask_check.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-hide_banner",
             "-framerate", str(review_fps), "-i", str(tmp_dir / "frame_%04d.png"),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path)],
            check=True,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    total = sum(masked_counts.values())
    print(f"{out_path.name}: {len(frames)} recon frames, {total} masks "
          f"({total / len(frames):.0%} of frames carry one)")
    for track_name, n in sorted(masked_counts.items()):
        print(f"    {track_name:<28} {n:>4} masks")

    add_to_asset_list({f"mask_check_{name}": to_report_path(out_path)})
    return out_path


def mask_check_map_beats(review_fps=6):
    '''Mask-check every MAP beat's tracks for the current case -- zero-arg, resolves
    its own masks and recon frames from A_Config + the paper edit, so it does not
    depend on any notebook cell having run first.

    One video per (MAP beat, subject) rather than per track: a subject's fragmented
    tracks share a video, each in its own colour, so you see how it broke up instead
    of hunting through one clip per fragment. Returns the list of mp4 paths.'''
    import collections
    from A_Config import sam3_masks_dir
    from render_paper_edit import build_recon_requests

    masks_dir = sam3_masks_dir()
    by_subject = collections.defaultdict(list)
    for d in sorted(p for p in masks_dir.iterdir() if p.is_dir()):
        by_subject[d.name.rsplit("-", 1)[0]].append(d)

    out_paths = []
    for job in build_recon_requests():
        if job["archetype"] != "MAP":
            continue
        for subject, track_dirs in by_subject.items():
            try:
                out_paths.append(frames_overlay_check(
                    job["frames_dir"], track_dirs,
                    name=f"{job['beat_key']}_{subject}", review_fps=review_fps))
            except ValueError as exc:
                print(f"skipped {subject}: {exc}")   # e.g. _sam_id_shots -- no numbered PNGs
    return out_paths


def video_overlay_edit(
        video_path,
        masks_dir,
        start = None,
        end = None,
        mask_start = None,
        mask_end = None,
        handles=1,
        color=(0, 200, 0),
        alpha=0.4,
        name=None):
    '''Makes mask overlay video: compressed, shareable/reviewable mp4 with real audio,

    burns a mask-highlight overlay onto any frame in [start, end] (frame
    numbers, the video range) that both has a matching MASK_NAME_FMT-named
    PNG in masks_dir AND falls within [mask_start, mask_end]. 

    cv2 can't touch audio, so this is done in two passes: frames are written
    to a silent temp video, then muxed with the original video's audio
    (re-clipped to the same range and re-encoded to AAC, same as video_edit)
    in one final ffmpeg pass. Also writes a thumbnail and appends both to
    the asset list CSV.

    name distinguishes the output filename (defaults to asset_name(), the
    original single-output behavior) -- pass a distinct name per call (e.g.
    a track_id) so repeated calls against different masks_dir don't
    overwrite each other's output.

    start/end/mask_start/mask_end all default to the range masks_dir itself
    covers (mask_frame_range), so the everyday "show me this track" call is
    just video_overlay_edit(video_path, track_dir, name=...). Pass them
    explicitly to cut a wider video range than the masks, or to use only part
    of a track's masks.'''
    name = name or asset_name()
    if mask_start is None or mask_end is None:
        first_masked, last_masked = mask_frame_range(masks_dir)
        mask_start = first_masked if mask_start is None else mask_start
        mask_end   = last_masked  if mask_end   is None else mask_end
    start = mask_start if start is None else start
    end   = mask_end   if end   is None else end

    fps = video_fps(video_path)
    start_s = max(0.0, start / fps - handles)
    end_s   = end / fps + handles

    handles_frames = round(handles * fps)
    start_frame = max(0, start - handles_frames)
    end_frame   = end + handles_frames  # inclusive

    cap = cv2.VideoCapture(str(video_path))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp_silent = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_silent.close()
    tmp_silent_path = tmp_silent.name

    try:
        writer = cv2.VideoWriter(tmp_silent_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))

        frame_idx = 0
        written = 0
        first_frame = None
        while frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                if frame_idx < start_frame:
                    cap.release()
                    writer.release()
                    raise ValueError(
                        f"{video_path} ended at frame {frame_idx}, before requested start {start_frame}."
                    )
                print(f"Warning: video ended at frame {frame_idx}, before requested end {end_frame} "
                      f"-- output will be shorter than requested.")
                break
            #frames sent to have mask overlay here
            if frame_idx >= start_frame:
                mask = None
                if mask_start <= frame_idx <= mask_end:
                    mask = _load_mask_bool(masks_dir, frame_idx)
                if mask is not None:
                    frame = apply_mask_overlay(frame, mask, color=color, alpha=alpha)
                if first_frame is None:
                    first_frame = frame.copy()
                writer.write(frame)
                written += 1
            frame_idx += 1

        cap.release()
        writer.release()

        if written == 0:
            raise ValueError(f"No frames written for [{start_frame}, {end_frame}] in {video_path}.")
        outpath = assets_dir() / f"{name}_overlay_video.mp4"
        outname = f"{name}_overlay_video.mp4"
        subprocess.run(
            ["ffmpeg", "-y",
             "-i", tmp_silent_path,
             "-ss", f"{start_s:.3f}", "-to", f"{end_s:.3f}", "-i", str(video_path),
             "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "libx264", "-c:a", "aac", str(outpath)],
            check=True,
        )

        thumb_path = assets_dir() / f"{name}_overlay_thumb.png"
        thumb_name = f"{name}_overlay_thumb.png"
        cv2.imwrite(str(thumb_path), first_frame)
    finally:
        os.remove(tmp_silent_path)

    add_to_asset_list({"overlay_mp4": outname, "overlay_mp4_thumb": thumb_name})
    return outpath


def render_screen_labels_video(video_path, job, name=None):
    '''Manual/ad hoc: burns one job's screen_labels (see build_screen_labels)
    straight onto video_path, no paper_edit.json or beats involved -- for
    eyeballing the label overlay on a single job without running the full
    assemble_paper_edit pipeline (which is the normal path once recon_jobs is
    passed into it).

    Covers only the span of frames that job's subjects actually have data
    for (min..max of every subject's frame keys), same shape as
    video_overlay_edit but for labels instead of a mask.'''
    name = name or asset_name()
    frame_label_lookup = build_frame_label_lookup([job])
    if not frame_label_lookup:
        raise ValueError(f"job {job.get('beat_key')!r} has no screen_labels to render.")

    start_frame = min(frame_label_lookup)
    end_frame   = max(frame_label_lookup)

    fps = video_fps(video_path)
    start_s = start_frame / fps
    end_s   = (end_frame + 1) / fps

    cap = cv2.VideoCapture(str(video_path))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp_silent = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_silent.close()
    tmp_silent_path = tmp_silent.name

    try:
        writer = cv2.VideoWriter(tmp_silent_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))

        frame_idx = 0
        written = 0
        while frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx >= start_frame:
                frame = _draw_frame_labels(frame, frame_idx, frame_label_lookup)
                writer.write(frame)
                written += 1
            frame_idx += 1

        cap.release()
        writer.release()

        if written == 0:
            raise ValueError(f"No frames written for {name} screen labels [{start_frame}, {end_frame}].")

        outpath = assets_dir() / f"{name}_screen_labels.mp4"
        outname = f"{name}_screen_labels.mp4"
        subprocess.run(
            ["ffmpeg", "-y",
             "-i", tmp_silent_path,
             "-ss", f"{start_s:.3f}", "-to", f"{end_s:.3f}", "-i", str(video_path),
             "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "libx264", "-c:a", "aac", str(outpath)],
            check=True,
        )
    finally:
        os.remove(tmp_silent_path)

    add_to_asset_list({"screen_labels_mp4": to_report_path(outpath)})
    return outpath


def mask_compositor(
        video_path,
        start,
        end,
        masks_dir,
        mask_start,
        mask_end,
        handles=1,
        color=(0, 200, 0),
        alpha=0.4):
    '''Burns a mask-highlight overlay onto any frame in [start, end] (frame
    numbers, the video range) that both has a matching MASK_NAME_FMT-named
    PNG in masks_dir AND falls within [mask_start, mask_end]. masks_dir may
    hold masks for a wider range than wanted here (e.g. other detections in
    the same sequence) -- mask_start/mask_end scope which of those files
    actually get used, independent of the video's start/end. Frames with no
    mask, or outside [mask_start, mask_end], pass through unchanged.
    masks_dir is taken as-is -- caller resolves which folder to point at
    (e.g. sam3_masks_dir() / beat_id / "<subject>-<track_id>"), same as
    _mask_overlay_mezzanine_clip.

    Writes every frame in the (handle-padded) range out as a PNG sequence
    under assets_dir()/overlay_frames,  appends the frames
    folder to the asset list CSV.'''
    fps = video_fps(video_path)

    handles_frames = round(handles * fps)
    start_frame = max(0, start - handles_frames)
    end_frame   = end + handles_frames  # inclusive

    cap = cv2.VideoCapture(str(video_path))

    out_dir = assets_dir() / "overlay_frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    written = 0
    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            if frame_idx < start_frame:
                cap.release()
                raise ValueError(
                    f"{video_path} ended at frame {frame_idx}, before requested start {start_frame}."
                )
            print(f"Warning: video ended at frame {frame_idx}, before requested end {end_frame} "
                  f"-- output will be shorter than requested.")
            break
        #frames sent to have mask overlay here
        if frame_idx >= start_frame:
            mask = None
            if mask_start <= frame_idx <= mask_end:
                mask = _load_mask_bool(masks_dir, frame_idx)
            if mask is not None:
                frame = apply_mask_overlay(frame, mask, color=color, alpha=alpha)
            cv2.imwrite(str(out_dir / f"frame_{frame_idx}.png"), frame)
            written += 1
        frame_idx += 1

    cap.release()

    if written == 0:
        raise ValueError(f"No frames written for [{start_frame}, {end_frame}] in {video_path}.")

    add_to_asset_list({"overlay_frames_dir": to_report_path(out_dir)})
    return


def _mask_overlay_mezzanine_clip(video_path, fps, start_frame, end_frame, masks_dirs, out_dir, name,
                                  frame_label_lookup=None):
    '''Lossless extraction of a mask-overlay clip for an explicit frame
    range. Burns every masks_dirs entry's mask onto each frame (composited
    together if more than one -- e.g. several distinct subjects tracked in
    the same beat), filename keyed by absolute source-video frame number --
    see A_ROBOFLOW_SAM3.track_subject_sam3 -- then encodes losslessly with
    the source video's own audio for that range -- not the silent
    placeholder track _map_mezzanine_clip uses, since this is real audio,
    just with an overlaid picture. No beat/track lookup here -- the caller
    decides what masks_dirs contains and what name to use, so this works
    equally for a paper-edit beat (assemble_paper_edit) or an ad hoc mask
    folder from a notebook. masks_dirs may be empty -- e.g. a "real" beat with
    no mask overlay of its own, extracted through here only because
    frame_label_lookup has screen labels to burn onto it.

    frame_label_lookup, if given (see build_frame_label_lookup), draws
    position/speed/heading/distance/confidence text next to every subject
    with data for that frame, regardless of whether that subject has a mask.

    cv2/OpenCV's FFV1 VideoWriter support is unreliable across builds, so
    frames are written to a temp PNG sequence first (same approach as
    R_map_animator/_map_mezzanine_clip) and ffmpeg does the actual lossless
    encode from that sequence.'''
    start_s = start_frame / fps
    end_s   = (end_frame + 1) / fps

    tmp_frames_dir = out_dir / f"_{name}_mask_frames"
    tmp_frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    written = 0
    try:
        while frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                if frame_idx < start_frame:
                    raise ValueError(
                        f"{video_path} ended at frame {frame_idx}, before {name}'s "
                        f"start {start_frame}."
                    )
                break
            if frame_idx >= start_frame:
                for masks_dir in masks_dirs:
                    mask = _load_mask_bool(masks_dir, frame_idx)
                    if mask is not None:
                        frame = apply_mask_overlay(frame, mask)
                frame = _draw_frame_labels(frame, frame_idx, frame_label_lookup)
                cv2.imwrite(str(tmp_frames_dir / f"frame_{written:04d}.png"), frame)
                written += 1
            frame_idx += 1
    finally:
        cap.release()

    if written == 0:
        shutil.rmtree(tmp_frames_dir)
        raise ValueError(f"No frames written for {name} [{start_frame}, {end_frame}].")

    out_path = out_dir / f"{name}_mask.mkv"
    subprocess.run(
        ["ffmpeg", "-y",
         "-framerate", str(fps), "-i", str(tmp_frames_dir / "frame_%04d.png"),
         "-ss", f"{start_s:.6f}", "-to", f"{end_s:.6f}", "-i", str(video_path),
         "-map", "0:v:0", "-map", "1:a:0",
         "-c:v", "ffv1", "-c:a", "pcm_s16le", str(out_path)],
        check=True,
    )

    shutil.rmtree(tmp_frames_dir)
    return out_path


#-------PAPER-EDIT ASSEMBLER-----------
# Rough-cut assembler for the Producer's paper_edit.json. Three beat shapes
# are wired up -- what this test needs:
#   - segment_type == "real": start_frame/end_frame cut from video_path.
#   - a tracked_subject beat (segment_type "synthetic" but real footage --
#     archetype can't be trusted to identify this, see producer.md): the
#     resolved start_frame/end_frame with its SAM3 mask burned in.
#   - segment_type == "synthetic" and archetype == "MAP": the image sequence
#     at assets_dir()/map_frames (see R_map_animator).
# Assumes exactly one video and one map per case -- no camera/asset-id
# lookup yet. Any other beat shape raises rather than guessing at an asset
# location that isn't wired up.
#
# Each beat is first extracted to a lossless mezzanine clip (FFV1 + PCM --
# real losslessness, not just visually-lossless ProRes-style compression),
# using fast input-seeking so a late beat doesn't force decoding the whole
# source video from frame 0. Only the final concat step re-encodes, once,
# straight to the lossy delivery mp4 -- so every frame that ends up in the
# output has been through exactly one lossy compression pass, same as a
# normal NLE export. Mezzanine clips are deleted only once the final mp4 is
# confirmed written (exists, non-empty); left in place on any failure so a
# retry doesn't have to re-extract everything and there's something to
# inspect.

def _real_mezzanine_clip(beat, video_path, fps, out_dir):
    '''Fast-seek + frame-accurate lossless extraction of a real segment.
    -ss before -i seeks near the target via the nearest keyframe (fast --
    doesn't decode from frame 0); ffmpeg's accurate-seek default then
    decodes forward to the exact frame. -to as an input option is an
    absolute position in the source timeline, same clock as -ss.'''
    start_s = beat["start_frame"] / fps
    end_s   = beat["end_frame"] / fps
    out_path = out_dir / f"beat{beat['order']:02d}_real.mkv"
    subprocess.run(
        ["ffmpeg", "-y",
         "-ss", f"{start_s:.6f}", "-to", f"{end_s:.6f}", "-i", str(video_path),
         "-avoid_negative_ts", "make_zero",
         "-c:v", "ffv1", "-c:a", "pcm_s16le", str(out_path)],
        check=True,
    )
    return out_path


def _map_mezzanine_clip(beat, type, target_w, target_h, target_fps, out_dir):
    '''Turns the map image sequence into a lossless clip scaled/padded (not
    stretched) to match the main footage, with a silent PCM audio track so
    it concatenates cleanly alongside the real clips' real audio.

    target_fps is also the rate the frames were rendered at -- R_map_animator
    (GOOGLE_MAP) / P_trace_overlayer (BEV_MAP) derive their fps from this same
    source video, so there's no resample here, just a straight read at the
    rate the frames already are.

    Each renderer suffixes its frames dir with the beat it rendered for, so a
    paper edit with two MAP beats gets two distinct animations. Falls back to the
    unsuffixed folder for cases whose assets predate that.'''
    if type == 'GOOGLE_MAP':
        stem = "map_frames"
    elif type == 'BEV_MAP':
        stem = "bev_trace_frames"
    else:
        raise ValueError(f"Unknown map type {type!r} -- expected 'GOOGLE_MAP' or 'BEV_MAP'.")

    beat_frames_dir = assets_dir() / f"{stem}_{beat['beat_id'][0]}"
    frames_dir = beat_frames_dir if beat_frames_dir.exists() else assets_dir() / stem
    if not frames_dir.exists():
        raise FileNotFoundError(
            f"Beat {beat['beat_id']} wants a {type} clip but neither {beat_frames_dir} "
            f"nor {assets_dir() / stem} exists -- render the map for this beat first."
        )

    out_path = out_dir / f"beat{beat['order']:02d}_map.mkv"
    subprocess.run(
        ["ffmpeg", "-y",
         "-framerate", str(target_fps), "-i", str(frames_dir / "frame_%04d.png"),
         "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
         "-shortest",
         "-vf", f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
                f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2",
         "-c:v", "ffv1", "-c:a", "pcm_s16le", str(out_path)],
        check=True,
    )
    return out_path


def _audio_duration(clip_path):
    '''Length of a mezzanine clip's audio stream, for anchoring its fade-out.
    FFV1/mkv often reports N/A per-stream, so fall back to the container.'''
    for entries in ("stream=duration", "format=duration"):
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-select_streams", "a:0", "-show_entries", entries, str(clip_path)],
            capture_output=True, text=True, check=True,
        )
        info = json.loads(probe.stdout)
        value = (info.get("streams") or [{}])[0].get("duration") if "stream" in entries \
                else info.get("format", {}).get("duration")
        if value not in (None, "N/A"):
            return float(value)
    raise ValueError(f"No audio duration for {clip_path}.")


def _concat_mezzanine_clips(clip_paths, out_path, fade_s=CUT_FADE_S):
    '''Single final lossy encode -- reads every lossless mezzanine clip and
    concatenates them via the concat filter (re-encoding once; not the
    stream-copy concat demuxer, which requires byte-identical stream params
    rather than just matching codec/resolution/fps) straight into the
    delivery mp4.

    Each clip's audio gets a fade_s ramp in and out first. Cutting in silence
    (find_true_audio_span) makes a boundary quiet, not zero -- room tone still
    steps discontinuously at the join, and a MAP beat's anullsrc drops to hard
    digital silence -- so without the ramp the join can pop.'''
    inputs = []
    for p in clip_paths:
        inputs += ["-i", str(p)]

    fades, streams = [], []
    for i, p in enumerate(clip_paths):
        dur = _audio_duration(p)
        d = min(fade_s, dur / 3)  # keep in+out clear of each other on a very short beat
        if d > 0:
            fades.append(f"[{i}:a:0]afade=t=in:st=0:d={d:.4f},"
                         f"afade=t=out:st={dur - d:.4f}:d={d:.4f}[a{i}]")
            streams.append(f"[{i}:v:0][a{i}]")
        else:
            streams.append(f"[{i}:v:0][{i}:a:0]")

    concat = f"{''.join(streams)}concat=n={len(clip_paths)}:v=1:a=1[outv][outa]"
    filter_complex = ";".join([*fades, concat])
    subprocess.run(
        ["ffmpeg", "-y", *inputs,
         "-filter_complex", filter_complex,
         "-map", "[outv]", "-map", "[outa]",
         "-c:v", "libx264", "-c:a", "aac", str(out_path)],
        check=True,
    )


def assemble_paper_edit(video_path, analysis_2d_for_decisions=None, recon_jobs=None, draw_screen_labels=True):
    '''Walks the current case's paper_edit.json beats in order, extracts
    each beat to a lossless mezzanine clip, then concatenates all of them
    with a single final lossy encode into
    assets_dir()/<asset_name>_rough_cut.mp4. See module comment above for
    what's wired up and the mezzanine/cleanup rationale.
    analysis_2d_for_decisions supplies tracked_subject beats' mask dirs.

    recon_jobs (draw_screen_labels=True, the default) supplies every job's
    screen_labels for the position/speed/heading/distance/confidence overlay
    (see build_screen_labels) -- merged frame-indexed via
    build_frame_label_lookup so it's not tied to beat/job matching (see that
    function's docstring for why). Pass recon_jobs=None or
    draw_screen_labels=False to skip the overlay and get the plain rough cut.
    A beat only takes the slower per-frame extraction path if it actually has
    labels (or a mask) to burn in; a plain "real" beat with no label data in
    its frame range still takes the fast direct-cut path.'''
    paper_edit_path = agent_p_output_dir() / f"{case_name()}_paper_edit.json"
    paper_edit = json.loads(paper_edit_path.read_text(encoding="utf-8"))
    beats = sorted(paper_edit["beats"], key=lambda b: b["order"])

    fps = video_fps(video_path)
    cap = cv2.VideoCapture(str(video_path))
    target_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    target_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    mezzanine_dir = assets_dir() / "_rough_cut_mezzanine"
    mezzanine_dir.mkdir(parents=True, exist_ok=True)

    frame_label_lookup = (
        build_frame_label_lookup(recon_jobs) if recon_jobs and draw_screen_labels else {}
    )

    clip_paths = []
    for beat in beats:
        if beat["segment_type"] == "real":
            beat_name = beat["beat_id"][0]
            beat_has_labels = any(
                f in frame_label_lookup for f in range(beat["start_frame"], beat["end_frame"] + 1)
            )
            if beat.get("tracked_subject") and "MASK_OVERLAYS" in (beat.get("requested_flags") or []):
                print(f"making overlay for beat {beat_name}")
                # Select this beat's instances out of analysis_2d_for_decisions
                # instead of re-deriving from the subject label -- gem_person_id
                # entries have no label to slugify. Matched on both beat_ids and the
                # beat's current tracked_subject, so a subject the Producer dropped
                # in the revision stays out even if analysis was built from the draft.
                # from_masks/track_subject_sam3 write masks_dir, gemvsSAM masks_dirs.
                keys = {s["gem_person_id"] if isinstance(s, dict) else _slugify_subject(s)
                        for s in beat["tracked_subject"]}
                masks_dirs = list(dict.fromkeys(
                    d
                    for key, entry in (analysis_2d_for_decisions or {}).items()
                    if isinstance(entry, dict)
                    and beat_name in (entry.get("beat_ids") or [])
                    and (key in keys or key.rsplit("-", 1)[0] in keys)
                    for d in ([entry["masks_dir"]] if entry.get("masks_dir") else entry.get("masks_dirs") or [])
                ))
                clip_paths.append(_mask_overlay_mezzanine_clip(
                    video_path, fps, beat["start_frame"], beat["end_frame"],
                    masks_dirs, mezzanine_dir, name=beat_name,
                    frame_label_lookup=frame_label_lookup))
            elif beat_has_labels:
                # no mask overlay for this beat, but there IS screen-label data
                # covering some of its frames -- worth the slower per-frame path
                clip_paths.append(_mask_overlay_mezzanine_clip(
                    video_path, fps, beat["start_frame"], beat["end_frame"],
                    [], mezzanine_dir, name=beat_name,
                    frame_label_lookup=frame_label_lookup))
            else:
                clip_paths.append(_real_mezzanine_clip(beat, video_path, fps, mezzanine_dir))
        
        elif beat["segment_type"] == "synthetic" and beat["archetype"] == "MAP":
            type = next(f for f in beat["requested_flags"] if f in ("GOOGLE_MAP", "BEV_MAP"))
            clip_paths.append(_map_mezzanine_clip(beat, type, target_w, target_h, fps, mezzanine_dir))
        else:
            raise NotImplementedError(
                f"Beat {beat['order']} ({beat['segment_type']}/{beat['archetype']}) "
                "has no assembler wired up yet -- only real, mask-overlay, and MAP beats are handled."
            )

    out_path = assets_dir() / f"{asset_name()}_rough_cut.mp4"
    _concat_mezzanine_clips(clip_paths, out_path)

    if not (out_path.exists() and out_path.stat().st_size > 0):
        raise RuntimeError(
            f"{out_path} missing or empty after concat -- mezzanine clips left in "
            f"{mezzanine_dir} for inspection."
        )
    shutil.rmtree(mezzanine_dir)

    add_to_asset_list({"rough_cut_mp4": to_report_path(out_path)})
    return out_path


#-------CUT-POINT FINDER (auto_select beats)-----------
# Resolves a Producer beat's rough search_window_start/end_seconds into real
# start_frame/end_frame, audio-first. The Producer's window is trusted for
# CONTENT (the quote it brackets is the right one) but not for exact timing
# (its bounds come from the transcript, sampled at 1Hz). silencedetect never
# picks the edit itself -- it only tells us where it's SAFE to cut (silent),
# so we never land a cut on a voice, mid-word or otherwise.

def find_true_audio_span(video_path, window_start_s, window_end_s,
                          analysis_margin_s=3, noise_db=-20, min_silence_s=0.3,
                          handle_s=1.5):
    '''Finds the true in/out cut points for a beat, never landing on voice.

    Two forward-only ffmpeg scans  Each scan looks
    up to 0.999s INSIDE the window  and analysis_margin_s
    OUTSIDE the window (untrusted content -- needs enough room for the
    pre/post-roll cut itself).

    True onset/offset = where voice actually starts/stops, a cut must never land inside [onset, offset].

    handle_in/handle_out = the recommended target cut point, handle_s back
    from onset/offset but clamped to the nearest silence boundary --

    Returns (onset, offset, handle_in, handle_out).'''
    #00: either 0 or the window-margin
    #01 window start and 1s
    #10 1s before end asked for
    #11 end + 3s 
    scan_ranges = [
        (max(0.0, window_start_s - analysis_margin_s), window_start_s + 0.999),
        (window_end_s - 0.999, window_end_s + analysis_margin_s),
    ]

    silences = []
    for scan_lo, scan_hi in scan_ranges:
        scan_len = scan_hi - scan_lo
        result = subprocess.run(
            ["ffmpeg", "-i", str(video_path), "-vn",
             "-af", f"atrim=start={scan_lo}:end={scan_hi},asetpts=PTS-STARTPTS,"
                    f"silencedetect=noise={noise_db}dB:d={min_silence_s}",
             "-f", "null", "-"],
            capture_output=True, text=True,
        )
        stderr = result.stderr

        starts = [float(m) for m in re.findall(r"silence_start:\s*([0-9.]+)", stderr)]
        ends   = [float(m) for m in re.findall(r"silence_end:\s*([0-9.]+)\s*\|", stderr)]
        if len(starts) > len(ends):
            ends.append(scan_len)  # scan window ended while still silent
        silences.extend((scan_lo + s, scan_lo + e) for s, e in zip(starts, ends))
    silences.sort()

    # True onset: walk forward from the start scan's outer edge through any
    # silence that covers it -- voice actually begins where that run of
    # silence ends.
    onset = scan_ranges[0][0]
    for s, e in silences:
        if s <= onset:
            onset = max(onset, e)
        else:
            break

    # True offset: mirror, walking backward from the end scan's outer edge.
    offset = scan_ranges[1][1]
    for s, e in reversed(silences):
        if e >= offset:
            offset = min(offset, s)
        else:
            break

    # Cut-in: pre-roll back from onset, but never past the silence
    # immediately preceding it -- if voice resumes inside that pre-roll
    # window, cut right after it ends instead of over it.
    prev_noise_end = next((s for s, e in silences if e == onset), None)
    handle_in = max(prev_noise_end, onset - handle_s) if prev_noise_end is not None else onset

    # Handle-out: mirror.
    next_noise_start = next((e for s, e in silences if s == offset), None)
    handle_out = min(next_noise_start, offset + handle_s) if next_noise_start is not None else offset

    return onset, offset, handle_in, handle_out

#currently garbage its' an AV comparer for cut points, but back-to-front
def find_cut_point_in_zone(zone_lo, zone_hi, boundary_frames, prefer_near):
    '''Nearest subject-visibility-boundary frame inside [zone_lo, zone_hi] to
    prefer_near, or None if no boundary frame falls in the zone.'''
    candidates = [f for f in boundary_frames if zone_lo <= f <= zone_hi]
    if not candidates:
        return None
    return min(candidates, key=lambda f: abs(f - prefer_near))


def find_cut_points(beat, video_path, fps, analysis_2d_for_decisions=None,
                     handle_s=1.5,
                     analysis_margin_s=2.5, noise_db=-20, min_silence_s=0.3):
    '''Finds this beat's actual in/out frames from its rough search window,
    audio-first (see module comment). Mutates and returns beat; no-op on
    synthetic beats -- they have no footage to cut.'''
    if beat.get("segment_type") != "real":
        return beat

    #if we have masks
    beat_analysis = (analysis_2d_for_decisions or {}).get(beat["beat_id"][0]) #check for beat_id/beat and don't crash
    if beat_analysis and beat_analysis["subject_first_frame"] is not None \
            and beat_analysis["subject_last_frame"] is not None:
        window_start_s = beat_analysis["subject_first_frame"] / fps
        window_end_s = beat_analysis["subject_last_frame"] / fps
    #if we don't have masks
    else:
        window_start_s = beat["search_window_start_seconds"]
        window_end_s = beat["search_window_end_seconds"]

    # No mask bounds AND no search window (e.g. a tracked_subject beat SAM3
    # found nothing for) -- nothing to resolve from. Leave start_frame/
    # end_frame unset rather than crash; the Producer's revision pass
    # already treats "no range came back" as a drop/substitute case.
    if window_start_s is None or window_end_s is None:
        return beat

    onset_s, offset_s, handle_in_s, handle_out_s = find_true_audio_span(
        video_path, window_start_s, window_end_s,
        analysis_margin_s=analysis_margin_s, noise_db=noise_db, min_silence_s=min_silence_s,
        handle_s=handle_s,
    )

    # Hard, unconditional no-go bound -- the final cut can never land inside
    # the true speech span, before any visual zone logic runs.
    onset_frame  = math.floor(onset_s * fps)
    offset_frame = math.ceil(offset_s * fps)

    # Target zones: between the audio-recommended handle point and the
    # no-go boundary 
    head_lo = max(0, math.floor(handle_in_s * fps))
    head_hi = onset_frame
    tail_lo = offset_frame
    tail_hi = math.ceil(handle_out_s * fps)


    '''Garbage, but could be useful - leaving in for the moment. it's back to front right now mind you.
    # Reuse analysis_2D's already-computed "continuous frame sequences"
    # rather than re-reading/re-decoding every mask PNG a second time.
    boundary_frames = []
    if analysis_2d_for_decisions is not None:
        seqs = analysis_2d_for_decisions["continuous frame sequences"]
        boundary_frames = sorted({f for s, e, _ in seqs for f in (s, e)})  
    # Prefer a real visibility boundary inside the safe zone; fall back to
    # the audio-recommended handle point itself when nothing visual applies.
    start_frame = find_cut_point_in_zone(head_lo, head_hi, boundary_frames, prefer_near=head_lo)
    if start_frame is None:
        start_frame = head_lo
    end_frame = find_cut_point_in_zone(tail_lo, tail_hi, boundary_frames, prefer_near=tail_hi)
    if end_frame is None:
        end_frame = tail_hi
    '''
    #temp variables until we get masks and vut points working properly 
    start_frame = head_lo
    end_frame = tail_hi

    beat["start_frame"] = start_frame
    beat["end_frame"] = end_frame
    beat["auto_select_resolved"] = True
    beat["auto_select_audio_span_seconds"] = [round(onset_s, 2), round(offset_s, 2)]
    beat["auto_select_original_search_window_seconds"] = [window_start_s, window_end_s]
    return beat


def find_all_cut_points(video_path=None, analysis_2d_for_decisions=None,paper_edit_json_path=None, **kwargs):
    '''Walks the current case's paper_edit_draft.json (this runs between the
    producer's draft and revision passes, before the revision file exists),
    finds real in/out frames for every auto_select beat, and rewrites the
    same file in place -- run this before the producer's revision pass, which
    expects auto_select beats already resolved.'''
    paper_edit_path = agent_p_output_dir() / f"{case_name()}_paper_edit_draft.json"
    #paper_edit = json.loads(paper_edit_path.read_text(encoding="utf-8"))
    # new for multiple drafts

    paper_edit = json.loads(paper_edit_json_path.read_text(encoding="utf-8"))

    if video_path is None:
        video_path = next((case_dir() / "010_source").glob("*.mp4"))
    fps = video_fps(video_path)

    for beat in paper_edit["beats"]:
        find_cut_points(beat, video_path, fps,
                         analysis_2d_for_decisions=analysis_2d_for_decisions, **kwargs)

    paper_edit_json_path.write_text(json.dumps(paper_edit, indent=2), encoding="utf-8")
    return paper_edit_path