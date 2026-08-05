import sys
import glob
from pathlib import Path

import numpy as np
import torch

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".heic", ".heif")

#BATCH unproject. currently not in use as doing it per-frame, on demand
def _unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic):
    depth = depth_map[..., 0]
    num_frames, height, width = depth.shape

    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))

    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]

    camera_points = np.stack(
        [
            (x - cx) / fx * depth,
            (y - cy) / fy * depth,
            depth,
        ],
        axis=-1,
    )

    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return np.einsum(
        "sij,shwj->shwi",
        np.transpose(rotation, (0, 2, 1)),
        camera_points - translation[:, None, None, :],
    )


def run_vggt_omega(image_dir, output_dir, checkpoint_path, vggt_omega_dir,
                    image_resolution=512, conf_thres=20.0, max_points=0, show_cam=True):
    """
    Run VGGT-Omega camera + depth inference on a directory of frames.

    Saves predictions.npz (extrinsic, intrinsic, depth, depth_conf,
    world_points_from_depth, images, pose_enc, camera_and_register_tokens)
    and a scene.glb (confidence-filtered point cloud + camera frustums) into
    output_dir.

    Args:
        image_dir        : path-like — directory of input images
        output_dir        : path-like — where predictions.npz / scene.glb are written
        checkpoint_path   : path-like — vggt_omega_1b_512.pt (or other) checkpoint
        vggt_omega_dir    : path-like — root of the vggt-omega repo
        image_resolution  : int   — input resolution (512 for the 1B-512 checkpoint)
        conf_thres        : float — percentile confidence threshold for the GLB point cloud
        max_points        : int   — cap on GLB point count; 0 disables decimation
        show_cam          : bool  — draw camera frustums in the GLB

    Returns:
        Path to the saved predictions.npz
    """
    image_dir       = Path(image_dir)
    output_dir      = Path(output_dir)
    checkpoint_path = Path(checkpoint_path)
    vggt_omega_dir  = Path(vggt_omega_dir)

    assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"
    if str(vggt_omega_dir) not in sys.path:
        sys.path.insert(0, str(vggt_omega_dir))

    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera
    from visual_util import predictions_to_glb

    image_names = sorted(glob.glob(str(image_dir / "*")))
    image_names = [p for p in image_names if p.lower().endswith(_IMG_EXTS)]
    if not image_names:
        raise FileNotFoundError(f"No images found in {image_dir}")
    print(f"{len(image_names)} frames  →  {output_dir}")

    model = VGGTOmega().to("cuda").eval()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    images = load_and_preprocess_images(image_names, image_resolution=image_resolution).to("cuda")

    with torch.inference_mode():
        predictions = model(images)

    extrinsic, intrinsic = encoding_to_camera(
        predictions["pose_enc"],
        predictions["images"].shape[-2:],
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    predictions_np = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
            if value.shape[0] == 1:
                value = value[0]
            predictions_np[key] = value

    predictions_np["world_points_from_depth"] = _unproject_depth_map_to_point_map(
        predictions_np["depth"],
        predictions_np["extrinsic"],
        predictions_np["intrinsic"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "predictions.npz"
    np.savez(npz_path, **predictions_np)
    print(f"Saved: {npz_path}")

    scene = predictions_to_glb(
        predictions_np,
        target_dir=str(output_dir),
        conf_thres=conf_thres,
        max_points=max_points,
        show_cam=show_cam,
    )
    glb_path = output_dir / "scene.glb"
    scene.export(file_obj=str(glb_path))
    print(f"Saved: {glb_path}")

    return npz_path
