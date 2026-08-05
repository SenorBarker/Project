# Adapters that assemble each non-VGGT-Omega technique's native on-disk output
# into a Reconstruction (F_post_recon_processing.py), so P_projection_mapping.py's
# BEV/compositing code -- written against VGGT-Omega's batched predictions.npz --
# works on every technique's output, not just VGGT-Omega's.
#
# Common target shape for Reconstruction.preds, matching load_VGGT_O_predictions's
# existing convention: depth/depth_conf (N,H,W), images (N,3,H,W) 0..1, intrinsic
# (N,3,3), extrinsic (N,3,4) world-to-camera.

from pathlib import Path

import cv2
import numpy as np
import torch

from A_Config import FRAME_NAME_FMT
from F_post_recon_processing import Reconstruction, build_frame_index, DEVICE


def _as_hw(arr):
    """Normalizes a per-frame depth/confidence array to plain (H,W) -- some
    model outputs carry a singleton channel dim ((H,W,1) or (1,H,W)) that breaks
    downstream code expecting exactly 2 dims (unproject()'s h, w = depth.shape,
    or the low_h, low_w = depth.shape callers below). No-op for already-2D input."""
    return np.asarray(arr).squeeze()


def _as_nhw(arr):
    """Like _as_hw but for an already-batched (N,H,W) array -- squeezes a
    singleton channel dim at either end ((N,H,W,1) or (N,1,H,W)) without
    touching the leading batch dim itself."""
    arr = np.asarray(arr)
    if arr.ndim == 4:
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.shape[1] == 1:
            arr = arr[:, 0]
    return arr


def _load_lowres_rgb_from_full(frames_dir, frame_num, low_h, low_w):
    """Downsamples the full-res source frame to (low_h, low_w) as a low-res color
    anchor for joint_bilateral_upsample -- used by techniques that don't save their
    own low-res color input (lingbot-map, plain VGGT). Returns (3,H,W) 0..1."""
    path = Path(frames_dir) / FRAME_NAME_FMT.format(frame_num)
    full = cv2.imread(str(path))
    full = cv2.cvtColor(full, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    low = cv2.resize(full, (low_w, low_h), interpolation=cv2.INTER_AREA)
    return low.transpose(2, 0, 1)


def cut3r_to_reconstruction(frames_dir, cut3r_output_dir):
    """CUT3R writes camera/depth/conf/color as separate per-frame files, all named
    by frame number ({fn:06d]) -- assumed mutually consistent (same frame set,
    same resolution) across the four subfolders, not independently re-verified here.

    camera/*.npz's "pose" is 4x4 c2w -- inverted to w2c to match Reconstruction's
    convention (same convention F_post_recon_processing.load_cut3r_trace_v2 reads
    the translation column of, just kept as translation-only there)."""
    cut3r_output_dir = Path(cut3r_output_dir)
    camera_files = sorted((cut3r_output_dir / "camera").glob("*.npz"), key=lambda f: int(f.stem))
    frame_nums = [int(f.stem) for f in camera_files]

    depth, conf, extrinsic, intrinsic, images = [], [], [], [], []
    for f, fn in zip(camera_files, frame_nums):
        cam = np.load(f)
        pose_c2w = cam["pose"]  # 4x4
        R_cw, t_cw = pose_c2w[:3, :3], pose_c2w[:3, 3]
        R_wc = R_cw.T
        t_wc = -R_wc @ t_cw
        ext = np.zeros((3, 4), dtype=np.float32)
        ext[:, :3] = R_wc
        ext[:, 3] = t_wc
        extrinsic.append(ext)
        intrinsic.append(cam["intrinsics"])

        depth.append(_as_hw(np.load(cut3r_output_dir / "depth" / f"{fn:06d}.npy")))
        conf.append(_as_hw(np.load(cut3r_output_dir / "conf" / f"{fn:06d}.npy")))

        img = cv2.imread(str(cut3r_output_dir / "color" / f"{fn:06d}.png"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        images.append(img.transpose(2, 0, 1))

    preds = {
        "depth": torch.from_numpy(np.stack(depth)).float().to(DEVICE),
        "depth_conf": torch.from_numpy(np.stack(conf)).float().to(DEVICE),
        "images": torch.from_numpy(np.stack(images)).float().to(DEVICE),
        "extrinsic": torch.from_numpy(np.stack(extrinsic)).float().to(DEVICE),
        "intrinsic": torch.from_numpy(np.stack(intrinsic)).float().to(DEVICE),
    }
    frame_to_row = {fn: i for i, fn in enumerate(frame_nums)}
    return Reconstruction(frames_dir=Path(frames_dir), preds_path=None, preds=preds, frame_to_row=frame_to_row)


def megasam_to_reconstruction(frames_dir, npz_path):
    """MegaSaM's single npz has no per-pixel confidence -- depth_conf is an
    all-ones placeholder. Every confidence-gated consumer downstream compares
    against a confidence_threshold that defaults to 5-20 for the other
    techniques, so a flat 1.0 placeholder would get filtered out entirely by any
    of those -- callers rendering a BEV from a MegaSaM-built Reconstruction MUST
    pass confidence_threshold=0 (see AA_cam_pose_sweeps.py), not whatever
    threshold is used for the other techniques. This means no filtering happens
    for MegaSaM at all: every reconstructed point, noise included, an inherent
    limitation of the data available rather than a bug.

    cam_c2w is 4x4 c2w -- inverted to w2c, same as cut3r_to_reconstruction."""
    data = np.load(npz_path)
    cam_c2w = data["cam_c2w"]  # (N,4,4)
    R_cw, t_cw = cam_c2w[:, :3, :3], cam_c2w[:, :3, 3]
    R_wc = np.transpose(R_cw, (0, 2, 1))
    t_wc = -np.einsum("nij,nj->ni", R_wc, t_cw)
    extrinsic = np.zeros((len(cam_c2w), 3, 4), dtype=np.float32)
    extrinsic[:, :, :3] = R_wc
    extrinsic[:, :, 3] = t_wc

    depth = _as_nhw(data["depths"])
    images = data["images"]
    if images.ndim == 4 and images.shape[-1] == 3:  # (N,H,W,3) -> (N,3,H,W)
        images = images.transpose(0, 3, 1, 2)
    images = images.astype(np.float32)
    if images.max() > 1.5:  # 0..255 -> 0..1, mirrors every other loader's convention
        images = images / 255.0
    depth_conf = np.ones_like(depth, dtype=np.float32)

    intrinsic = data["intrinsic"]
    if intrinsic.ndim == 2:  # single shared camera -> broadcast to per-frame
        intrinsic = np.broadcast_to(intrinsic, (len(cam_c2w), 3, 3)).copy()

    preds = {
        "depth": torch.from_numpy(depth).float().to(DEVICE),
        "depth_conf": torch.from_numpy(depth_conf).float().to(DEVICE),
        "images": torch.from_numpy(images).float().to(DEVICE),
        "extrinsic": torch.from_numpy(extrinsic).float().to(DEVICE),
        "intrinsic": torch.from_numpy(intrinsic).float().to(DEVICE),
    }
    frame_to_row = build_frame_index(frames_dir, FRAME_NAME_FMT)
    return Reconstruction(frames_dir=Path(frames_dir), preds_path=None, preds=preds, frame_to_row=frame_to_row)


def lingbot_map_to_reconstruction(frames_dir, lingbot_output_dir):
    """lingbot-map's per-frame npz has depth/depth_conf/extrinsic_w2c/intrinsic but
    no low-res color input -- images is built by downsampling the full-res source
    frame instead (see _load_lowres_rgb_from_full), used purely as
    joint_bilateral_upsample's color anchor; composite_overlay's no-mask path
    still needs *some* lowres_rgb even with masking off.

    extrinsic_w2c is already world-to-camera -- no inversion needed, unlike CUT3R/MegaSaM."""
    frames_dir = Path(frames_dir)
    files = sorted(Path(lingbot_output_dir).glob("*.npz"), key=lambda f: int(f.stem))
    frame_nums = [int(f.stem) for f in files]

    depth, conf, extrinsic, intrinsic, images = [], [], [], [], []
    for f, fn in zip(files, frame_nums):
        d = np.load(f)
        d_depth = _as_hw(d["depth"])
        d_conf = _as_hw(d["depth_conf"])
        depth.append(d_depth)
        conf.append(d_conf)
        extrinsic.append(d["extrinsic_w2c"])
        intrinsic.append(d["intrinsic"])
        low_h, low_w = d_depth.shape
        images.append(_load_lowres_rgb_from_full(frames_dir, fn, low_h, low_w))

    preds = {
        "depth": torch.from_numpy(np.stack(depth)).float().to(DEVICE),
        "depth_conf": torch.from_numpy(np.stack(conf)).float().to(DEVICE),
        "images": torch.from_numpy(np.stack(images)).float().to(DEVICE),
        "extrinsic": torch.from_numpy(np.stack(extrinsic)).float().to(DEVICE),
        "intrinsic": torch.from_numpy(np.stack(intrinsic)).float().to(DEVICE),
    }
    frame_to_row = {fn: i for i, fn in enumerate(frame_nums)}
    return Reconstruction(frames_dir=frames_dir, preds_path=None, preds=preds, frame_to_row=frame_to_row)


def vggt_plain_to_reconstruction(frames_dir, vggt_output_dir):
    """Plain VGGT's vggt_cache.npz -- key names (depth_map/extrinsic/intrinsic/
    depth_conf) confirmed against Models/VGGT/vggt/demo_colmap.py's own save call.
    Same missing-low-res-images gap and fallback as lingbot_map_to_reconstruction."""
    frames_dir = Path(frames_dir)
    data = np.load(Path(vggt_output_dir) / "vggt_cache.npz")
    # depth_map/depth_conf come out of demo_colmap.run_VGGT as (N,H,W,1) --
    # squeeze(0) there only drops the batch dim, not this trailing channel dim
    # (confirmed against Models/VGGT/vggt/demo_colmap.py's actual save call).
    depth = _as_nhw(data["depth_map"])
    depth_conf = _as_nhw(data["depth_conf"])
    extrinsic = data["extrinsic"]  # (N,3,4) w2c, same convention load_VGGT_trace reads
    intrinsic = data["intrinsic"]

    frame_to_row = build_frame_index(frames_dir, FRAME_NAME_FMT)
    low_h, low_w = depth.shape[1:]
    images = [
        _load_lowres_rgb_from_full(frames_dir, fn, low_h, low_w)
        for fn in sorted(frame_to_row, key=frame_to_row.get)
    ]

    preds = {
        "depth": torch.from_numpy(depth).float().to(DEVICE),
        "depth_conf": torch.from_numpy(depth_conf).float().to(DEVICE),
        "images": torch.from_numpy(np.stack(images)).float().to(DEVICE),
        "extrinsic": torch.from_numpy(extrinsic).float().to(DEVICE),
        "intrinsic": torch.from_numpy(intrinsic).float().to(DEVICE),
    }
    return Reconstruction(frames_dir=frames_dir, preds_path=None, preds=preds, frame_to_row=frame_to_row)
