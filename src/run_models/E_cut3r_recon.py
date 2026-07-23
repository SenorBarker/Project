import sys
import os
import time
import numpy as np
import torch
import imageio.v2 as iio
from pathlib import Path
from tqdm.auto import tqdm

_OPENGL = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]], dtype=np.float32)


def _make_raymap(c2w, h, w, intrinsics):
    """Build a (H, W, 6) raymap from an absolute camera-to-world matrix."""
    i, j   = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
    grid   = np.stack([i, j, np.ones_like(i)], axis=-1)          # H×W×3
    ro     = c2w[:3, 3]                                           # camera centre
    rd     = np.linalg.inv(intrinsics) @ grid.reshape(-1, 3).T   # unproject
    rd     = (c2w @ np.vstack([rd, np.ones_like(rd[0])])).T[:, :3].reshape(h, w, 3)
    rd     = rd / np.linalg.norm(rd, axis=-1, keepdims=True)
    
    ro     = np.broadcast_to(ro, (h, w, 3))
    return np.concatenate([ro, rd], axis=-1).astype(np.float32)  # H×W×6


def _save_ply(path, pts3d_world, colors):
    """Save a coloured point cloud as a binary PLY file.

    pts3d_world : (H, W, 3) float32 — points in world space
    colors      : (H, W, 3) float32 — RGB in [0, 1]
    """
    pts = pts3d_world.reshape(-1, 3).astype(np.float32)
    rgb = (colors.reshape(-1, 3) * 255).clip(0, 255).astype(np.uint8)
    n   = len(pts)

    header = (
        f"ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        f"property float x\nproperty float y\nproperty float z\n"
        f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"end_header\n"
    ).encode("ascii")

    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                   ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    verts = np.empty(n, dtype=dt)
    verts["x"], verts["y"], verts["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    verts["red"], verts["green"], verts["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    with open(path, "wb") as f:
        f.write(header)
        f.write(verts.tobytes())


def _export_outputs(output_dir, f_id, depth, conf, color, c2w, intrin,
                    pts3d_world, R_norm, t_norm, nvs_rgb=None, conf_threshold=1.5):
    np.save(output_dir / "depth"  / f"{f_id:06d}.npy", depth)
    np.save(output_dir / "conf"   / f"{f_id:06d}.npy", conf)
    iio.imwrite(str(output_dir / "color" / f"{f_id:06d}.png"),
                (color * 255).astype(np.uint8))
    np.savez(str(output_dir / "camera" / f"{f_id:06d}.npz"),
             pose=c2w, intrinsics=intrin)
    mask    = conf.reshape(-1) > conf_threshold
    pts_out = (pts3d_world @ R_norm.T + t_norm).reshape(-1, 3)[mask]
    col_out = color.reshape(-1, 3)[mask]
    _save_ply(output_dir / "ply" / f"{f_id:06d}.ply", pts_out, col_out)
    if nvs_rgb is not None:
        iio.imwrite(str(output_dir / "nvs_rgb" / f"{f_id:06d}.png"),
                    (nvs_rgb * 255).astype(np.uint8))


def _third_person_c2w(c2w, back_dist, up_dist):
    """
    Offset a camera backward and upward (OpenCV convention: Y points down).
    Keeps the same look direction — camera slides, does not rotate.
    """
    forward = c2w[:3, 2]   # +Z = look direction
    down    = c2w[:3, 1]   # +Y = down
    t_new   = c2w[:3, 3] - forward * back_dist - down * up_dist
    c2w_new = c2w.copy()
    c2w_new[:3, 3] = t_new
    return c2w_new


def run_cut3r(frames_dir, output_dir, ckpt_path, size=512, device="cuda",
              render_3rdperson=False, back_dist=0.5, up_dist=0.3, revisit=False,
              conf_threshold=1.5):
    """
    Streaming CUT3R inference on an image sequence.

    Args:
        frames_dir       : path-like — directory of input images (sorted by name)
        output_dir       : path-like — root for depth/ conf/ color/ camera/ subdirs
        ckpt_path        : path-like — cut3r_512_dpt_4_64.pth checkpoint
        size             : int       — rescale long edge to this (512 for DPT model)
        device           : str       — "cuda" or "cpu"
        render_3rdperson : bool      — also synthesise over-the-shoulder novel views
        back_dist        : float     — how far behind the original camera (world units)
        up_dist          : float     — how far above the original camera (world units)
    """
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    ckpt_path  = Path(ckpt_path)

    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    cut3r_dir = ckpt_path.parent.parent
    cut3r_src = str(ckpt_path.parent)

    for p in [str(cut3r_dir), cut3r_src]:
        if p not in sys.path:
            sys.path.insert(0, p)

    from demo import parse_seq_path
    from dust3r.utils.image import load_images
    from dust3r.model import ARCroco3DStereo
    from dust3r.utils.camera import pose_encoding_to_camera
    from dust3r.post_process import estimate_focal_knowing_depth
    from dust3r.inference import inference_step  # type: ignore[import]

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    print("Loading model...")
    model = ARCroco3DStereo.from_pretrained(str(ckpt_path)).to(device)
    model.eval()

    img_paths, _ = parse_seq_path(str(frames_dir))
    _IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".heic", ".heif")
    img_paths = [p for p in img_paths if p.lower().endswith(_IMG_EXTS)]
    if not img_paths:
        raise FileNotFoundError(f"No images found in {frames_dir}")
    print(f"{len(img_paths)} frames  →  {output_dir}")

    subdirs = ["depth", "conf", "color", "camera", "ply"]
    if render_3rdperson:
        subdirs.append("nvs_rgb")
    for d in subdirs:
        (output_dir / d).mkdir(parents=True, exist_ok=True)

    loaded_imgs = load_images(img_paths, size=size, verbose=False)

    state_feat = state_pos = init_state_feat = mem = init_mem = None
    R_norm = t_norm = None
    t0 = time.time()

    with torch.no_grad():
        for f_id, frame in enumerate(tqdm(loaded_imgs, desc="CUT3R")):
            img        = frame["img"].to(device)
            true_shape = torch.from_numpy(frame["true_shape"]).to(device)

            img_out, img_pos, _ = model._encode_image(img, true_shape)
            feat_i = img_out[-1]
            pos_i  = img_pos
            g      = model._get_img_level_feat(feat_i)

            if f_id == 0:
                state_feat, state_pos = model._init_state(feat_i, pos_i)
                mem             = model.pose_retriever.mem.expand(feat_i.shape[0], -1, -1).clone()
                init_state_feat = state_feat.clone()
                init_mem        = mem.clone()
                pose_feat_i     = model.pose_token.expand(feat_i.shape[0], -1, -1)
            else:
                pose_feat_i = model.pose_retriever.inquire(g, mem)

            pose_pos_i = -torch.ones(1, 1, 2, device=device, dtype=pos_i.dtype)

            new_state_feat, dec = model._recurrent_rollout(
                state_feat, state_pos, feat_i, pos_i,
                pose_feat_i, pose_pos_i, init_state_feat,
            )

            new_mem    = model.pose_retriever.update_mem(mem, g, dec[-1][:, 0:1])
            state_feat = new_state_feat
            mem        = new_mem

            head_input = [
                dec[0].float(),
                dec[model.dec_depth * 2 // 4][:, 1:].float(),
                dec[model.dec_depth * 3 // 4][:, 1:].float(),
                dec[model.dec_depth].float(),
            ]
            res = model._downstream_head(head_input, true_shape, pos=pos_i)

            pts3d_self = res["pts3d_in_self_view"].cpu()
            conf_self  = res["conf_self"].cpu()
            H, W = int(true_shape[0, 0].item()), int(true_shape[0, 1].item())

            pp    = torch.tensor([[W // 2, H // 2]], dtype=torch.float32)
            focal = estimate_focal_knowing_depth(pts3d_self, pp, focal_mode="weiszfeld")
            c2w   = pose_encoding_to_camera(res["camera_pose"].clone()).cpu()[0].numpy()

            intrin = np.eye(3, dtype=np.float32)
            intrin[0, 0] = intrin[1, 1] = focal.item()
            intrin[0, 2], intrin[1, 2]  = W // 2, H // 2

            depth = pts3d_self[0, ..., 2].numpy()
            conf  = conf_self[0].numpy()
            color = (0.5 * (img[0].permute(1, 2, 0).cpu().numpy() + 1)).clip(0, 1)

            # --- novel view synthesis (over-the-shoulder 3rd person) ---
            nvs_rgb = None
            if render_3rdperson:
                c2w_virt = _third_person_c2w(c2w, back_dist, up_dist)
                raymap   = _make_raymap(c2w_virt, H, W, intrin)

                nvs_view = {
                    "img":        torch.full((1, 3, H, W), float("nan")),
                    "ray_map":    torch.from_numpy(raymap).unsqueeze(0),  # 1×H×W×6
                    "true_shape": torch.from_numpy(np.int32([[H, W]])),
                    "idx":        f_id,
                    "instance":   str(f_id),
                    "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
                    "img_mask":   torch.tensor(False).unsqueeze(0),
                    "ray_mask":   torch.tensor(True).unsqueeze(0),
                    "update":     torch.tensor(False).unsqueeze(0),
                    "reset":      torch.tensor(False).unsqueeze(0),
                }

                state_args = (state_feat, state_pos, init_state_feat, mem, init_mem)
                nvs_out    = inference_step(nvs_view, state_args, model, device)
                nvs_rgb    = (0.5 * (nvs_out["pred"]["rgb"][0].numpy() + 1.0)).clip(0, 1)

            if not revisit:
                if f_id == 0:
                    w2c_0  = np.linalg.inv(c2w)
                    R_norm = _OPENGL @ w2c_0[:3, :3]
                    t_norm = _OPENGL @ w2c_0[:3,  3]
                pts3d_world = res["pts3d_in_other_view"].cpu()[0].numpy()
                _export_outputs(output_dir, f_id, depth, conf, color,
                                c2w, intrin, pts3d_world, R_norm, t_norm, nvs_rgb,
                                conf_threshold)

    if revisit:
        print("Pass 2 (revisiting): re-processing all frames with frozen final state...")
        frozen_state_feat = state_feat.clone()
        frozen_state_pos  = state_pos.clone()
        frozen_init_feat  = init_state_feat.clone()
        frozen_mem        = mem.clone()
        frozen_init_mem   = init_mem.clone()

        R_norm = t_norm = None
        t1 = time.time()
        with torch.no_grad():
            for f_id, img_path in enumerate(tqdm(img_paths, desc="revisit")):
                imgs = load_images([img_path], size=size, verbose=False)
                img        = imgs[0]["img"].to(device)
                true_shape = torch.from_numpy(imgs[0]["true_shape"]).to(device)

                img_out, img_pos, _ = model._encode_image(img, true_shape)
                feat_i = img_out[-1]
                pos_i  = img_pos
                g      = model._get_img_level_feat(feat_i)

                pose_feat_i = model.pose_retriever.inquire(g, frozen_mem)
                pose_pos_i  = -torch.ones(1, 1, 2, device=device, dtype=pos_i.dtype)

                _, dec = model._recurrent_rollout(
                    frozen_state_feat, frozen_state_pos, feat_i, pos_i,
                    pose_feat_i, pose_pos_i, frozen_init_feat,
                )

                head_input = [
                    dec[0].float(),
                    dec[model.dec_depth * 2 // 4][:, 1:].float(),
                    dec[model.dec_depth * 3 // 4][:, 1:].float(),
                    dec[model.dec_depth].float(),
                ]
                res = model._downstream_head(head_input, true_shape, pos=pos_i)

                pts3d_self = res["pts3d_in_self_view"].cpu()
                conf_self  = res["conf_self"].cpu()
                H, W = int(true_shape[0, 0].item()), int(true_shape[0, 1].item())

                pp    = torch.tensor([[W // 2, H // 2]], dtype=torch.float32)
                focal = estimate_focal_knowing_depth(pts3d_self, pp, focal_mode="weiszfeld")
                c2w   = pose_encoding_to_camera(res["camera_pose"].clone()).cpu()[0].numpy()

                if f_id == 0:
                    w2c_0  = np.linalg.inv(c2w)
                    R_norm = _OPENGL @ w2c_0[:3, :3]
                    t_norm = _OPENGL @ w2c_0[:3,  3]

                intrin = np.eye(3, dtype=np.float32)
                intrin[0, 0] = intrin[1, 1] = focal.item()
                intrin[0, 2], intrin[1, 2]  = W // 2, H // 2

                depth = pts3d_self[0, ..., 2].numpy()
                conf  = conf_self[0].numpy()
                color = (0.5 * (img[0].permute(1, 2, 0).cpu().numpy() + 1)).clip(0, 1)

                pts3d_world = res["pts3d_in_other_view"].cpu()[0].numpy()

                nvs_rgb = None
                if render_3rdperson:
                    c2w_virt = _third_person_c2w(c2w, back_dist, up_dist)
                    raymap   = _make_raymap(c2w_virt, H, W, intrin)
                    nvs_view = {
                        "img":         torch.full((1, 3, H, W), float("nan")),
                        "ray_map":     torch.from_numpy(raymap).unsqueeze(0),
                        "true_shape":  torch.from_numpy(np.int32([[H, W]])),
                        "idx":         f_id,
                        "instance":    str(f_id),
                        "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
                        "img_mask":    torch.tensor(False).unsqueeze(0),
                        "ray_mask":    torch.tensor(True).unsqueeze(0),
                        "update":      torch.tensor(False).unsqueeze(0),
                        "reset":       torch.tensor(False).unsqueeze(0),
                    }
                    state_args = (frozen_state_feat, frozen_state_pos, frozen_init_feat,
                                  frozen_mem, frozen_init_mem)
                    nvs_out    = inference_step(nvs_view, state_args, model, device)
                    nvs_rgb    = (0.5 * (nvs_out["pred"]["rgb"][0].numpy() + 1.0)).clip(0, 1)

                _export_outputs(output_dir, f_id, depth, conf, color,
                                c2w, intrin, pts3d_world, R_norm, t_norm, nvs_rgb,
                                conf_threshold)

        elapsed2 = time.time() - t1
        print(f"Revisit pass done. {len(img_paths)} frames in {elapsed2 / 60:.1f} min  "
              f"({elapsed2 / len(img_paths):.2f} s/frame)")

    torch.save({
        "state_feat":      state_feat.cpu(),
        "state_pos":       state_pos.cpu(),
        "init_state_feat": init_state_feat.cpu(),
        "mem":             mem.cpu(),
        "init_mem":        init_mem.cpu(),
    }, output_dir / "cut3r_state.pt")
    print(f"State saved → {output_dir / 'cut3r_state.pt'}")

    elapsed = time.time() - t0
    print(f"Done. {len(img_paths)} frames in {elapsed / 60:.1f} min  "
          f"({elapsed / len(img_paths):.2f} s/frame)")
    print(f"Output: {output_dir}")
