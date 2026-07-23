import glob
import re
import sys
from pathlib import Path

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".heic", ".heif")
_SHARD_NAME_RE = re.compile(r"predictions_(\d+)_(\d+)\.npz")


def discover_vggt_omega_shards(output_dir):
    """Sorted list of (start, end, npz_path) for every predictions_{start}_{end}.npz
    shard in output_dir -- i.e. what run_vggt_omega_shards_batched wrote -- so
    callers can load a multi-part recon without hardcoding shard names/counts.
    start/end are int frame indices, matching the frame_range each shard was
    run with (see Reconstruction.load's frame_range)."""
    shards = []
    for npz_path in Path(output_dir).glob("predictions_*_*.npz"):
        m = _SHARD_NAME_RE.fullmatch(npz_path.name)
        if m:
            shards.append((int(m.group(1)), int(m.group(2)), npz_path))
    return sorted(shards)


def run_vggt_omega_shards_batched(image_dir, output_dir, checkpoint_path, vggt_omega_shards_dir,
                                   devices=None, image_resolution=512, conf_thres=20.0,
                                   max_points=0, show_cam=True, batch_size=100):
    """
    Run VGGT-Omega over every frame in image_dir in batches of batch_size,
    so a single shared frame folder can be reconstructed as several
    manageable VGGT-Omega runs instead of one over all frames. Consecutive
    batches share their one boundary frame (each batch after the first
    starts on the previous batch's last frame) so the two independent
    reconstructions can later be stitched into one coordinate frame using
    that shared frame's pose/depth as a known correspondence.
    Each batch is saved as its own {output_dir}/predictions_{start:04d}_{end:04d}.npz
    (+ matching .glb), via run_vggt_omega_shards' frame_range/output_name.

    Args:
        image_dir              : path-like — directory of input images (all frames)
        output_dir              : path-like — where each batch's .npz/.glb land
        checkpoint_path         : path-like — vggt_omega checkpoint
        vggt_omega_shards_dir   : path-like — root of the vggt-omega-shards repo
        devices                 : list[str] of CUDA devices, e.g. ["cuda:0", "cuda:1"].
                                  Defaults to all visible GPUs.
        image_resolution        : int   — input resolution
        conf_thres              : float — percentile confidence threshold for the GLB
        max_points              : int   — cap on GLB point count; 0 disables decimation
        show_cam                : bool  — draw camera frustums in the GLB
        batch_size               : int   — frames per batch (last batch may be smaller)

    Returns:
        list of Path to each batch's saved predictions_{start}_{end}.npz
    """
    image_names = sorted(glob.glob(str(Path(image_dir) / "*")))
    image_names = [p for p in image_names if p.lower().endswith(_IMG_EXTS)]
    n = len(image_names)

    batches = []
    start = 0
    while start < n:
        end = min(start + batch_size, n)
        batches.append((start, end))
        if end >= n:
            break
        start = end - 1  # -1 so the next batch starts on this batch's last frame (1-frame overlap for boundary stitching)
    print(f"VGGT_O will run {len(batches)} batches of up to {batch_size} frames ({n} frames total)")

    npz_paths = []
    for batch_id, (start, end) in enumerate(batches):
        print(f"[{batch_id + 1}/{len(batches)}] frames {start}:{end}")
        npz_paths.append(run_vggt_omega_shards(
            image_dir              = image_dir,
            output_dir              = output_dir,
            checkpoint_path         = checkpoint_path,
            vggt_omega_shards_dir   = vggt_omega_shards_dir,
            devices                 = devices,
            image_resolution        = image_resolution,
            conf_thres               = conf_thres,
            max_points               = max_points,
            show_cam                 = show_cam,
            frame_range              = (start, end),
            output_name              = f"predictions_{start:04d}_{end:04d}",
        ))
    return npz_paths


def run_vggt_omega_shards(image_dir, output_dir, checkpoint_path, vggt_omega_shards_dir,
                           devices=None, image_resolution=512, conf_thres=20.0,
                           max_points=0, show_cam=True, frame_range=None,
                           output_name="predictions"):
    """
    Thin wrapper around Models/vggt-omega-shards/run_sharded.run_sharded -
    the actual multi-GPU sharding logic lives there, not here, since it
    reimplements VGGT-Omega's internal Aggregator.forward (not just calling
    its public API), so it stays sandboxed alongside its own copy of the
    model rather than living in src/ with the rest of the pipeline glue.

    Args:
        image_dir              : path-like — directory of input images
        output_dir              : path-like — where {output_name}.npz / .glb land
        checkpoint_path         : path-like — vggt_omega checkpoint
        vggt_omega_shards_dir   : path-like — root of the vggt-omega-shards repo
        devices                 : list[str] of CUDA devices, e.g. ["cuda:0", "cuda:1"].
                                  Defaults to all visible GPUs.
        image_resolution        : int   — input resolution
        conf_thres              : float — percentile confidence threshold for the GLB
        max_points              : int   — cap on GLB point count; 0 disables decimation
        show_cam                : bool  — draw camera frustums in the GLB
        frame_range             : optional (start_idx, end_idx) slice into the
                                  sorted directory listing of image_dir - lets
                                  successive runs pull chunks (e.g. (0, 100),
                                  (100, 200), ...) out of one shared frame
                                  folder. Indices only, not frame numbers.
        output_name              : str — basename for the saved files, so
                                  multiple runs into the same output_dir
                                  don't overwrite each other, e.g.
                                  "predictions_3000_3100".

    Returns:
        Path to the saved {output_name}.npz
    """
    vggt_omega_shards_dir = Path(vggt_omega_shards_dir)
    if str(vggt_omega_shards_dir) not in sys.path:
        sys.path.insert(0, str(vggt_omega_shards_dir))

    from run_sharded import run_sharded

    return run_sharded(
        image_dir=str(image_dir),
        output_dir=str(output_dir),
        checkpoint_path=str(checkpoint_path),
        devices=devices,
        image_resolution=image_resolution,
        conf_thres=conf_thres,
        max_points=max_points,
        show_cam=show_cam,
        frame_range=frame_range,
        output_name=output_name,
    )
