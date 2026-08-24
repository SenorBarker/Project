"""
Identity matching by masked crop instead of by Gemini box IoU.

AX_gem_SAM_matcher matches a gem_person_id to a SAM track_id by IoU against
people.json's box_2d. Those boxes are written by the model as text in one
whole-video pass, not measured off frames, so on 402_Hide_n_seek they are
degenerate (repeated coordinates, boxes pinned to the frame edge, appearance
times past the end of the video) and the IoU match is noise.

Here the only inputs are things that ARE measured: SAM's own masks, the
Producer's descriptor, and the beat's frame range. Each track is rendered as
a crop with everything except the subject blacked out -- no second person, no
background, nothing to be ambiguous about -- and the model is only ever asked
"does this image match this description", never asked for a coordinate or a
timecode.

Returns the same {gem_person_id: [track_id, ...]} shape as
match_gem_people_to_sam, so analysis_2d_gemvsSAM and everything downstream of
it read it unchanged.
"""

import asyncio
import json
import random
import sys
import threading
from pathlib import Path

import cv2
import numpy as np


def descriptor_targets(paper_edit_json_path,descriptor_targets=None):
    """gem_person_id -> {"descriptor": str, "windows": [(start_s, end_s), ...]}.

    Windows come from render_paper_edit.tracking_windows -- the same helper the
    tracker builds its spans from, so a person is matched over exactly the windows
    they were tracked over. This file used to keep its own copy of that rule (skip
    MAP beats outright) and the two drifted: with a paper edit whose subjects sit
    only on the MAP beat, targets came back empty and every track reported NONE.

    A MAP beat still contributes nothing in the normal case. It contributes only
    when no event beat tracks anything, and then its windows are the swept event
    spans rather than the whole location section -- which answers the original
    objection to letting MAP beats in at all (a recon-scale window would let a
    track from anywhere in the clip answer for a person seen in one 6s beat).
    """
    from render_paper_edit import tracking_windows

    if paper_edit_json_path is None:
        import A_Config
        paper_edit_json_path = A_Config.agent_p_output_dir() / f"{A_Config.case_name()}_paper_edit_draft.json"

    paper_edit = json.loads(Path(paper_edit_json_path).read_text(encoding="utf-8"))

    targets = {}
    for beat in paper_edit["beats"]:
        windows = tracking_windows(beat, paper_edit["beats"])
        if not windows:
            continue
        for entry in beat.get("tracked_subject") or []:
            if not isinstance(entry, dict):
                continue
            target = targets.setdefault(
                entry["gem_person_id"], {"descriptor": entry.get("descriptor", ""), "windows": []}
            )
            target["windows"].extend(windows)
    return targets


def _mask_frames(track_dir):
    """(frame_idx, mask_path) for every mask this track wrote, frame-ordered."""
    frames = []
    for mask_path in track_dir.glob("*.png"):
        try:
            frames.append((int(mask_path.stem), mask_path))
        except ValueError:
            continue
    return sorted(frames)


def _isolate(frame, mask, pad=0.12, max_side=640):
    """Subject on black, cropped to the mask's bbox with a little padding, and
    capped at max_side -- plenty to recognise clothing on, and small enough
    that several crops fit in one SDK message."""
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()

    pad_y, pad_x = int((y1 - y0) * pad), int((x1 - x0) * pad)
    y0, y1 = max(0, y0 - pad_y), min(frame.shape[0], y1 + pad_y)
    x0, x1 = max(0, x0 - pad_x), min(frame.shape[1], x1 + pad_x)

    isolated = (frame * (mask[:, :, None] > 0))[y0:y1, x0:x1]

    scale = max_side / max(isolated.shape[:2])
    if scale < 1:
        isolated = cv2.resize(isolated, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return isolated


def _save_8bit(bgr, out_path, colors=256):
    """256-colour palette PNG. Clothing colour is what the ID turns on, so this
    quantizes rather than going grayscale."""
    from PIL import Image

    Image.fromarray(bgr[:, :, ::-1]).convert(
        "P", palette=Image.ADAPTIVE, colors=colors
    ).save(out_path, optimize=True)


def write_masked_crops(masks_root=None, video_path=None, out_dir=None,
                       per_track=10, min_mask_px=1500, subject_slug="person"):
    """One masked crop per track (the frames where the mask is biggest, so the
    clearest view of the subject). Returns {track_key: [crop_path, ...]}.

    subject_slug keeps this to the person tracks -- the Producer also asks for
    hiding places ("the barbecue grill", "the large hedge"), and those folders
    sit alongside the person ones. Pass None to crop every track.

    Runs with no API access -- use it on its own to eyeball the crops before
    spending a model call on them.
    """
    if masks_root is None or video_path is None:
        import A_Config
        masks_root = masks_root or A_Config.sam3_masks_dir()
        video_path = video_path or A_Config.source_video_path()
    masks_root = Path(masks_root)
    out_dir = Path(out_dir) if out_dir else masks_root / "_id_crops"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Which frames to fetch, decided from the masks alone.
    wanted = {}  # frame_idx -> [(track_name, mask_path), ...]
    scored_count = {}
    span = {}  # track_name -> (first_frame, last_frame)
    for track_dir in sorted(p for p in masks_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        if subject_slug is not None and track_dir.name.rsplit("-", 1)[0] != subject_slug:
            continue
        frames = _mask_frames(track_dir)
        scored_count[track_dir.name] = len(frames)
        span[track_dir.name] = (frames[0][0], frames[-1][0]) if frames else (None, None)
        step = max(1, len(frames) // per_track)
        for frame_idx, mask_path in frames[::step][:per_track]:
            wanted.setdefault(frame_idx, []).append((track_dir.name, mask_path))

    from Two2D.B_video_processing import frames_at_indices

    frames = frames_at_indices(video_path, wanted)

    crops_by_track = {}
    too_small = {}   # track_name -> masks skipped for being under min_mask_px
    for frame_idx, mask_jobs in sorted(wanted.items()):
        frame = frames.get(frame_idx)
        if frame is None:
            print(f"could not read frame {frame_idx} -- skipping its crop(s)")
            continue
        for track_name, mask_path in mask_jobs:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask.shape[:2] != frame.shape[:2]:
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            # min_mask_px was a parameter that only ever appeared in the closing
            # message -- nothing enforced it, so an empty mask reached _isolate and
            # np.nonzero(...).min() raised "zero-size array to reduction operation".
            # SAM3 does emit near-empty masks when a track degrades, so this is a
            # real case, not a corrupt file.
            if int((mask > 0).sum()) < min_mask_px:
                too_small[track_name] = too_small.get(track_name, 0) + 1
                continue
            crop_path = out_dir / f"{track_name}_f{frame_idx:05d}.png"
            _save_8bit(_isolate(frame, mask), crop_path)
            track = crops_by_track.setdefault(
                track_name,
                {"crops": [], "first_frame": span[track_name][0], "last_frame": span[track_name][1]},
            )
            track["crops"].append(crop_path)

    for track_name, n_scored in scored_count.items():
        n_crops = len(crops_by_track.get(track_name, {}).get("crops", ()))
        skipped = too_small.get(track_name, 0)
        note = f" ({skipped} under {min_mask_px}px, skipped)" if skipped else ""
        print(f"[{track_name}] {n_crops} crop(s) from {n_scored} masked frames{note}"
              if n_crops else
              f"[{track_name}] no crop -- {n_scored} masked frames, "
              f"{skipped} of the {min(n_scored, per_track)} sampled were under {min_mask_px}px")

    return crops_by_track


def _match_prompt(track_key, crop_paths, candidates, targets):
    """One track's crops against every descriptor whose window it falls in."""
    lines = [
        f"These images are all one tracked subject ({track_key}), isolated on black",
        "(everything except that subject is masked out). Read them, then say which",
        "of the people described below this subject is, if any:",
        "",
    ]
    lines += [f"  {crop_path}" for crop_path in crop_paths]
    lines.append("")
    for gem_person_id in candidates:
        lines.append(f"    {gem_person_id}: {targets[gem_person_id]['descriptor']}")
    lines += [
        "",
        "Most tracked subjects are NOT on this list -- bystanders and passers-by",
        "get tracked too, and the list only names the few people of interest. NONE",
        "is the expected answer; give an id only when the description clearly fits",
        "what you can see. Judge on the visible subject only -- clothing, hair,",
        "apparent age, build.",
        'Reply with nothing but the id, e.g. "person-2", or NONE.',
    ]
    return "\n".join(lines)


def _ask_sync(prompt, cwd, model):
    """Same trick as run_producer_agent: the SDK spawns the `claude` CLI as a
    subprocess, and Jupyter's kernel already owns an event loop, so run it on
    its own thread with a fresh loop."""
    result, error = {}, {}

    def _runner():
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result["value"] = loop.run_until_complete(_ask(prompt, cwd, model))
        except Exception as e:
            error["value"] = e
        finally:
            loop.close()

    # the CLI intermittently returns is_error with no error text; retry it
    for attempt in range(3):
        result.clear(); error.clear()
        t = threading.Thread(target=_runner)
        t.start()
        t.join()
        if "value" not in error:
            return result["value"]
        print(f"  call failed ({error['value']}) -- attempt {attempt + 1}/3")
    raise error["value"]


async def _ask(prompt, cwd, model):
    from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, TextBlock, query

    reply = ""
    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            cwd=cwd,
            model=model,
            tools=["Read"],  # Read alone -- it only ever looks at the crops
            system_prompt="You identify people in masked image crops. Answer only with the JSON list asked for.",
            # a Read'd image comes back base64 in one message, and several of
            # them blow through the SDK's 1MB default
            max_buffer_size=32 * 1024 * 1024,
        ),
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    reply += block.text
    return reply


def match_tracks_to_descriptors(paper_edit_json_path,crops_by_track=None, targets=None, masks_root=None, fps=None,
                                model="claude-sonnet-5", delete_crops=True):
    """gem_person_id -> [track_id, ...]. One call per track ("which of these people
    is this, or NONE"), so two people can't claim one mask. Crops deleted as used.
    """
    import A_Config

    masks_root = Path(masks_root or A_Config.sam3_masks_dir())
    targets = targets or descriptor_targets(paper_edit_json_path)
    crops_by_track = crops_by_track or write_masked_crops(masks_root=masks_root)

    results = {gem_person_id: [] for gem_person_id in targets}
    unmatched = []
    taken = {}  # gem_person_id -> track already given it
    for track_key, track in crops_by_track.items():
        if track_key.rsplit("-", 1)[0] != "person":
            continue

        # someone already identified while this track was on screen is someone else
        candidates = []
        for gem_person_id in targets:
            other = taken.get(gem_person_id)
            if other and track["first_frame"] <= other["last_frame"] \
                    and other["first_frame"] <= track["last_frame"]:
                continue
            candidates.append(gem_person_id)
        if not candidates:
            # Two very different reasons land here. With no targets at all nothing
            # can ever be a candidate, and reporting that as co-visibility sends you
            # hunting through track overlaps for a problem that is really an empty
            # descriptor_targets() -- which is exactly what happened once already.
            if not targets:
                print(f"{track_key} -> NONE (no descriptor targets: no beat lists a "
                      f"gem_person_id the tracker could match against)")
            else:
                print(f"{track_key} -> NONE (every person already on screen with it)")
            unmatched.append(track_key)
            continue
        try:
            reply = _ask_sync(
                _match_prompt(track_key, track["crops"], candidates, targets),
                cwd=A_Config.REPO_ROOT, model=model,
            )
        except Exception as exc:
            print(f"{track_key} -> SKIPPED, {exc}")
            unmatched.append(track_key)
            continue
        # One id or none: a track is one subject, so it can answer for at most
        # one person. Longest id first, so "person-2" can't shadow "person-20".
        matched = next(
            (g for g in sorted(candidates, key=len, reverse=True) if g in reply), None
        )
        if matched:
            results[matched].append(int(track_key.rsplit("-", 1)[1]))
            taken[matched] = track
        else:
            unmatched.append(track_key)
        print(f"{track_key} -> {matched or 'NONE'}")

#         if delete_crops:
#             for crop_path in crop_paths:
#                 Path(crop_path).unlink(missing_ok=True)

    for gem_person_id, track_ids in results.items():
        track_ids.sort()
        print(f"{gem_person_id} ({targets[gem_person_id]['descriptor']}) -> {track_ids or 'NO MATCH'}")
    if unmatched:
        print(f"{len(unmatched)} track(s) matched nobody: {', '.join(unmatched)}")

    return results
