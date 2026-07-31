import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from C_CSV_report import add_to_report
from A_Config import assets_dir, asset_name, case_dir, report_path, to_report_path, sam3_masks_dir
from B_video_processing import video_fps

MASK_NAME_FMT = "{:04d}.png"

#-------TIMELINE-----------


def video_edit(video_path, start, end, out_path, handles = 1):
    '''Cuts [start, end] (frame numbers) out of video_path and re-exports it to
    out_path, re-encoding for a frame-accurate cut (a fast stream-copy would
    only snap to the nearest keyframe, not the exact requested frame).
    handles is extra padding, in seconds, added onto both the start and the
    end before cutting (e.g. handles=2 -> 2s earlier, 2s later).
    Also appends the output path to the report CSV.'''
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

    add_to_report({"edited_video": to_report_path(out_path)})
    return out_path


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


def mask_compositor(
        video_path,
        start,
        end,
        tracker,
        mask_start,
        mask_end,
        handles=1,
        color=(0, 200, 0),
        alpha=0.4):
    '''Burns a mask-highlight overlay onto any frame in [start, end] (frame
    numbers, the video range) that both has a matching MASK_NAME_FMT-named
    PNG in mask_dir AND falls within [mask_start, mask_end]. mask_dir may
    hold masks for a wider range than wanted here (e.g. other detections in
    the same sequence) -- mask_start/mask_end scope which of those files
    actually get used, independent of the video's start/end. Frames with no
    mask, or outside [mask_start, mask_end], pass through unchanged.

    Writes every frame in the (handle-padded) range out as a PNG sequence
    under assets_dir()/overlay_frames,  appends the frames
    folder to the report CSV.'''
    fps = video_fps(video_path)
    if tracker == "SAM3":
          masks_dir = sam3_masks_dir()

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

    add_to_report({"overlay_frames_dir": to_report_path(out_dir)})
    return


def _mask_overlay_mezzanine_clip(beat, video_path, fps, out_dir):
    '''Lossless extraction of a mask-overlay beat -- a tracked_subject beat
    (segment_type "synthetic", but real footage, not a generated asset; see
    producer.md's "Segment types & duration" exception) whose start_frame/
    end_frame have already been resolved by find_cut_points. Burns the
    SAM3 mask (sam3_masks_dir()/, filename keyed by absolute source-video
    frame number -- see A_ROBOFLOW_SAM3.track_subject_sam3) onto each frame, then encodes
    losslessly with the source video's own audio for that range -- not the
    silent placeholder track _map_mezzanine_clip uses, since this is real
    audio, just with an overlaid picture.

    cv2/OpenCV's FFV1 VideoWriter support is unreliable across builds, so
    frames are written to a temp PNG sequence first (same approach as
    R_map_animator/_map_mezzanine_clip) and ffmpeg does the actual lossless
    encode from that sequence.'''
    start_frame = beat["start_frame"]
    end_frame   = beat["end_frame"]
    start_s = start_frame / fps
    end_s   = (end_frame + 1) / fps

    # masks_dir = sam3_masks_dir() / f"beat{beat['order']:02d}"
    masks_dir = sam3_masks_dir()

    tmp_frames_dir = out_dir / f"_beat{beat['order']:02d}_mask_frames"
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
                        f"{video_path} ended at frame {frame_idx}, before beat {beat['order']}'s "
                        f"start {start_frame}."
                    )
                break
            if frame_idx >= start_frame:
                mask = _load_mask_bool(masks_dir, frame_idx)
                if mask is not None:
                    frame = apply_mask_overlay(frame, mask)
                cv2.imwrite(str(tmp_frames_dir / f"frame_{written:04d}.png"), frame)
                written += 1
            frame_idx += 1
    finally:
        cap.release()

    if written == 0:
        shutil.rmtree(tmp_frames_dir)
        raise ValueError(f"No frames written for beat {beat['order']} [{start_frame}, {end_frame}].")

    out_path = out_dir / f"beat{beat['order']:02d}_mask.mkv"
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


def _map_mezzanine_clip(beat, target_w, target_h, target_fps, out_dir):
    '''Turns the map image sequence into a lossless clip scaled/padded (not
    stretched) to match the main footage, with a silent PCM audio track so
    it concatenates cleanly alongside the real clips' real audio.

    target_fps is also the rate map_frames/ was rendered at -- R_map_animator
    derives its fps from this same source video, so there's no resample
    here, just a straight read at the rate the frames already are.'''
    frames_dir = assets_dir() / "map_frames"
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


def _concat_mezzanine_clips(clip_paths, out_path):
    '''Single final lossy encode -- reads every lossless mezzanine clip and
    concatenates them via the concat filter (re-encoding once; not the
    stream-copy concat demuxer, which requires byte-identical stream params
    rather than just matching codec/resolution/fps) straight into the
    delivery mp4.'''
    inputs = []
    for p in clip_paths:
        inputs += ["-i", str(p)]
    filter_parts = "".join(f"[{i}:v:0][{i}:a:0]" for i in range(len(clip_paths)))
    filter_complex = f"{filter_parts}concat=n={len(clip_paths)}:v=1:a=1[outv][outa]"
    subprocess.run(
        ["ffmpeg", "-y", *inputs,
         "-filter_complex", filter_complex,
         "-map", "[outv]", "-map", "[outa]",
         "-c:v", "libx264", "-c:a", "aac", str(out_path)],
        check=True,
    )


def assemble_paper_edit(video_path):
    '''Walks the current case's paper_edit.json beats in order, extracts
    each beat to a lossless mezzanine clip, then concatenates all of them
    with a single final lossy encode into
    assets_dir()/<asset_name>_rough_cut.mp4. See module comment above for
    what's wired up and the mezzanine/cleanup rationale.'''
    paper_edit_path = case_dir() / "012_agent_p_output" / f"{case_dir().name}_paper_edit.json"
    paper_edit = json.loads(paper_edit_path.read_text(encoding="utf-8"))
    beats = sorted(paper_edit["beats"], key=lambda b: b["order"])

    fps = video_fps(video_path)
    cap = cv2.VideoCapture(str(video_path))
    target_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    target_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    mezzanine_dir = assets_dir() / "_rough_cut_mezzanine"
    mezzanine_dir.mkdir(parents=True, exist_ok=True)

    clip_paths = []
    for beat in beats:
        if beat["segment_type"] == "real":
            if beat.get("tracked_subject"):
                print(f"making overlay for beat {beat}")
                clip_paths.append(_mask_overlay_mezzanine_clip(beat, video_path, fps, mezzanine_dir))
            else:
                clip_paths.append(_real_mezzanine_clip(beat, video_path, fps, mezzanine_dir))
        
        elif beat["segment_type"] == "synthetic" and beat["archetype"] == "MAP":
            clip_paths.append(_map_mezzanine_clip(beat, target_w, target_h, fps, mezzanine_dir))
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

    add_to_report({"rough_cut_mp4": to_report_path(out_path)})
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
                     analysis_margin_s=1.5, noise_db=-20, min_silence_s=0.3):
    '''Finds this beat's actual in/out frames from its rough search window,
    audio-first (see module comment). Mutates and returns beat; no-op if
    beat isn't  auto_select -- other beats are someone else's job.'''
    if not (beat.get("cut_mode") == "auto_select"):
        return beat

  
    #if we have masks
    beat_analysis = (analysis_2d_for_decisions or {}).get(beat["order"]) #check for order/beat and don't crash
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

    beat["cut_mode"] = "fixed_frames"
    beat["start_frame"] = start_frame
    beat["end_frame"] = end_frame
    beat["auto_select_resolved"] = True
    beat["auto_select_audio_span_seconds"] = [round(onset_s, 2), round(offset_s, 2)]
    beat["auto_select_original_search_window_seconds"] = [window_start_s, window_end_s]
    return beat


def find_all_cut_points(video_path=None, analysis_2d_for_decisions=None, **kwargs):
    '''Walks the current case's paper_edit_draft.json (this runs between the
    producer's draft and revision passes, before the revision file exists),
    finds real in/out frames for every auto_select beat, and rewrites the
    same file in place -- run this before the producer's revision pass, which
    expects auto_select beats already resolved.'''
    paper_edit_path = case_dir() / "012_agent_p_output" / f"{case_dir().name}_paper_edit_draft.json"
    paper_edit = json.loads(paper_edit_path.read_text(encoding="utf-8"))

    if video_path is None:
        video_path = next((case_dir() / "010_source").glob("*.mp4"))
    fps = video_fps(video_path)

    for beat in paper_edit["beats"]:
        find_cut_points(beat, video_path, fps,
                         analysis_2d_for_decisions=analysis_2d_for_decisions, **kwargs)

    paper_edit_path.write_text(json.dumps(paper_edit, indent=2), encoding="utf-8")
    return paper_edit_path