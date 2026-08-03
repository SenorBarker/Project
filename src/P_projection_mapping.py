"""
Project VGGT-Omega depth + pose predictions onto full-resolution source frames.
Runs on CUDA (env: MSc2_vggt).


Pipeline:
  1. Load predictions.npz (depth, depth_conf, extrinsic, intrinsic)
  2. Load matching full-res source frames
  3. Edge-aware (joint bilateral) upsample of depth to full resolution,
     guided by full-res RGB, using x/y/r/g/b distance THIS IS THE KEY THING IN HERE
  4. Unproject full-res pixels to 3D using the upsampled depth
  5. Reproject all selected frames into one reference camera's view
  6. Composite with a z-buffer (nearest point wins per output pixel)
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import math

from A_Config import case_dir, assets_dir, asset_name, report_path, to_report_path
from G_transforms_alignments import transform_RST, unproject

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os


FRAME_NAME_FMT = "{:04d}.jpg"

MASK_NAME_FMT = "{:04d}.png"


def check_frame_count(preds, frames_dir, name_fmt=FRAME_NAME_FMT):
    """predictions.npz stores no source filenames -- build_frame_index() assumes
    its rows are in the same sorted-by-frame-number order as the files in
    frames_dir, with one row per file. This at least catches a count mismatch
    (e.g. VGGT-O run with subsampling/a different frame set)."""
    n_preds = preds["extrinsic"].shape[0]
    n_files = len(list(Path(frames_dir).glob(f"*{Path(name_fmt).suffix}")))
    if n_files != n_preds:
        raise ValueError(
            f"Frame/prediction count mismatch: {n_files} image files in {frames_dir} "
            f"but {n_preds} rows in predictions.npz -- frame-number-to-row pairing "
            f"will be wrong."
        )
    
def load_full_res_frame(frames_dir, frame_idx):
    path = os.path.join(frames_dir, FRAME_NAME_FMT.format(frame_idx))
    img = cv2.imread(path) if path else None #returns blanks
    
    if img is None:
        print(f"frame {frame_idx} is missing and skipped")
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(img).to(DEVICE)  # (H, W, 3)
   

def load_mask(frame_idx, masks_dir=None):
    """None if no mask file exists for this frame or masks_dir is not set."""
    if masks_dir is None:
        return None
    path = os.path.join(masks_dir, MASK_NAME_FMT.format(frame_idx))
    if not os.path.exists(path):
        return None
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return torch.from_numpy(m > 0).to(DEVICE)  # (H, W) bool


def footprint_size(full_res_shape, lowres_shape):
    full_h, full_w = full_res_shape
    low_h, low_w = lowres_shape
    return full_w / low_w, full_h / low_h


def joint_bilateral_upsample(lowres_depth, lowres_conf, full_rgb, sigma_xy, radius=1, conf_thresh = 5):
    """Upsample lowres_depth (H', W') to full_rgb's (H, W) resolution, guided by color.
    Weight = spatial gaussian (sigma fixed = footprint size, baked in via the 0.5
    factor below) * color gaussian (sigma = local color variance, auto per-pixel)."""
    low_h, low_w = lowres_depth.shape
    full_h, full_w, _ = full_rgb.shape
    sx, sy = sigma_xy # upsample ratio . pixels per point

    valid = (lowres_conf >= conf_thresh).float()

    #convert hi res pix locations into low res ones that are fractional
    xs = (torch.arange(full_w, device=DEVICE, dtype=torch.float32) + 0.5) / sx - 0.5
    ys = (torch.arange(full_h, device=DEVICE, dtype=torch.float32) + 0.5) / sy - 0.5
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")  # 2x grids (H, W)

    gx0 = torch.floor(gx).long() #round down to an integer instead of a fraction
    gy0 = torch.floor(gy).long()

    out_depth = torch.zeros(full_h, full_w, device=DEVICE)
    weight_sum = torch.zeros(full_h, full_w, device=DEVICE)
    #sweep over the blob-sized point
    for dy in (0,1):
        for dx in (0,1):
            nx = gx0 + dx # pixel to sample from 
            ny = gy0 + dy
            in_bounds = (nx >= 0) & (nx < low_w) & (ny >= 0) & (ny < low_h)

            nx_c = nx.clamp(0, low_w - 1)#ensures we are in the bounds of the lo res image.
            ny_c = ny.clamp(0, low_h - 1)

            d_sample = lowres_depth[ny_c, nx_c]#measure depth here
            v_sample = valid[ny_c, nx_c]#is this a valid depth

            #upscale, push to centre first, then scale, coords
            cx_full = ((nx_c.float() + 0.5) * sx - 0.5).round().long().clamp(0, full_w - 1)
            cy_full = ((ny_c.float() + 0.5) * sy - 0.5).round().long().clamp(0, full_h - 1)
            sample_rgb = full_rgb[cy_full, cx_full]  # (H, W, 3)

            #bilinear
            spatial_w = (1 - (gx - nx.float()).abs()).clamp(min=0) * (1 - (gy - ny.float()).abs()).clamp(min=0)


            color_dist2 = ((full_rgb - sample_rgb) ** 2).sum(dim=-1)
            color_w = 1 - color_dist2 / 3

            w = spatial_w * color_w * in_bounds.float() * v_sample

            out_depth += w * d_sample
            weight_sum += w

    depth_out = out_depth / weight_sum.clamp_min(1e-6)

   
    return depth_out





def reproject(points_world, intrinsic_ref, extrinsic_ref):
    R = extrinsic_ref[:, :3]
    t = extrinsic_ref[:, 3]
    points_ref = points_world @ R.T + t

    fx, fy = intrinsic_ref[0, 0], intrinsic_ref[1, 1]
    cx, cy = intrinsic_ref[0, 2], intrinsic_ref[1, 2]
    z = points_ref[..., 2]
    valid = z > 1e-6
    z_safe = torch.where(valid, z, torch.ones_like(z))

    u = torch.where(valid, points_ref[..., 0] * fx / z_safe + cx, torch.full_like(z, -1.0))
    v = torch.where(valid, points_ref[..., 1] * fy / z_safe + cy, torch.full_like(z, -1.0))
    return u, v, z, valid


def scale_intrinsic(intrinsic, sx, sy):
    out = intrinsic.clone()
    out[0, 0] *= sx
    out[1, 1] *= sy
    out[0, 2] *= sx
    out[1, 2] *= sy
    return out


def bilinear_sample(img, u, v):
    """img: (H, W, C). u, v: (H', W') pixel coords (float). Returns (H', W', C)."""
    h, w = img.shape[:2]
    c = img.shape[2] if img.dim() == 3 else 1
    img_chw = img.reshape(h, w, -1).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

    grid_x = (u / max(w - 1, 1)) * 2 - 1
    grid_y = (v / max(h - 1, 1)) * 2 - 1
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H', W', 2)

    sampled = F.grid_sample(img_chw, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return sampled.squeeze(0).permute(1, 2, 0).reshape(*u.shape, c)


def upsample_confidence(lowres_conf, full_shape):
    full_h, full_w = full_shape
    conf = lowres_conf.unsqueeze(0).unsqueeze(0)  # (1, 1, H', W')
    out = F.interpolate(conf, size=(full_h, full_w), mode="bilinear", align_corners=True)
    return out.squeeze(0).squeeze(0)


def get_full_res_camera(preds, idx, full_shape):
    low_h, low_w = preds["depth"].shape[1:]
    sx, sy = footprint_size(full_shape, (low_h, low_w))
    intrinsic_full = scale_intrinsic(preds["intrinsic"][idx], sx, sy)
    return intrinsic_full, preds["extrinsic"][idx], (sx, sy)


def crop_to_mask_region(mask, extra_rgb, depth_low, conf_low, intrinsic_full, margin=8):
    """
    Crop extra_rgb/mask (full-res) and depth_low/conf_low (low-res) to the
    mask's bounding box, with shifted intrinsics to match the crop origin.
    fiddly but will massively reduce compute time.
    """
    h_full, w_full = extra_rgb.shape[:2]
    h_low, w_low = depth_low.shape[:2]
    scale_y = h_low / h_full
    scale_x = w_low / w_full

    ys, xs = mask.nonzero(as_tuple=True)
    y0 = max(0, ys.min().item() - margin)
    x0 = max(0, xs.min().item() - margin)
    y1 = min(h_full, ys.max().item() + 1 + margin)
    x1 = min(w_full, xs.max().item() + 1 + margin)

    rgb_crop = extra_rgb[y0:y1, x0:x1]
    mask_crop = mask[y0:y1, x0:x1]

    y0_lr = int(y0 * scale_y)
    x0_lr = int(x0 * scale_x)
    y1_lr = min(h_low, int(math.ceil(y1 * scale_y)))
    x1_lr = min(w_low, int(math.ceil(x1 * scale_x)))

    depth_crop = depth_low[y0_lr:y1_lr, x0_lr:x1_lr]
    conf_crop = conf_low[y0_lr:y1_lr, x0_lr:x1_lr]

    intrinsic_crop = intrinsic_full.clone()
    intrinsic_crop[0, 2] -= x0
    intrinsic_crop[1, 2] -= y0

    return rgb_crop, mask_crop, depth_crop, conf_crop, intrinsic_crop, (x0, y0)


def unproject_masked(mask, rgb_full, depth_low, conf_low, intrinsic_full, extrinsic, margin=8, conf_thresh=5):
    """Unproject the pixels covered by `mask` to VGGT-O world-space points.

    mask, rgb_full: full-resolution, matching the source frame.
    depth_low, conf_low: this camera's native-resolution row from preds["depth"]/preds["depth_conf"].
    intrinsic_full: this camera's intrinsic already scaled to rgb_full's resolution (see get_full_res_camera).
    extrinsic: this camera's extrinsic, unscaled -- straight from preds["extrinsic"].

    Returns (world_points, rgb) for just the masked pixels, or (None, None) if the mask
    is empty. World points are in the one VGGT-O world frame regardless of resolution --
    see unproject().
    """
    if mask is None or mask.sum() == 0:
        return None, None

    sigma = footprint_size(rgb_full.shape[:2], depth_low.shape)
    rgb_crop, mask_crop, depth_crop, conf_crop, intrinsic_crop, _ = crop_to_mask_region(
        mask, rgb_full, depth_low, conf_low, intrinsic_full, margin=margin
    )
    depth_crop_full = joint_bilateral_upsample(depth_crop, conf_crop, rgb_crop, sigma, conf_thresh=conf_thresh)
    world_points = unproject(depth_crop_full, intrinsic_crop, extrinsic)
    return world_points[mask_crop], rgb_crop[mask_crop]#just the 1s from the mask are returned


def _as_recon_list(recon):
    """recon may be a single  or a
    list/tuple of them (multi-shard). Normalise to a list"""
    return list(recon) if isinstance(recon, (list, tuple)) else [recon]


def find_recon_for_frame(recon, frame_idx):
    """Which shard's recon actually has this frame, and its row within that recon.
    """
    for i, r in enumerate(_as_recon_list(recon)):
        row = r.frame_to_row.get(frame_idx)
        if row is not None:
            return i, r, row
    return None, None, None


def load_frame_inputs(recon, frame_idx, masks_dir=None):
    """Everything needed to unproject one frame: full-res RGB, mask, this frame's
    native-resolution depth/confidence, full-res-scaled intrinsic + extrinsic, and
    which shard (recon_idx) it came from -- so the caller can tell whether two frames
    live in the same recon's model space or need point_cloud_xforms to bring them together
    (recon_idx is always 0 for a single-recon caller)."""

    recon_idx, recon, row = find_recon_for_frame(recon, frame_idx)

    if row is None:
        return None, None, None, None, None, None, None, None

    rgb_full = load_full_res_frame(recon.frames_dir, frame_idx)

    if rgb_full is None:
        return None, None, None, None, None, None, None, None

    mask = load_mask(frame_idx, masks_dir)
    intrinsic_full, extrinsic, sigma = get_full_res_camera(recon.preds, row, rgb_full.shape[:2])
    depth = recon.preds["depth"][row]
    depth_conf = recon.preds["depth_conf"][row]

    return rgb_full, mask, depth, depth_conf, intrinsic_full, extrinsic, sigma, recon_idx


def composite_overlay(recon,
                      main_idx,
                      extra_indices,
                      masks_dir = None,
                      point_cloud_xforms=None,
                      depth_margin=0.0,
                      splat_radius=1,
                      new_view=None,
                      conf_thresh = 5
                      ):
    """Paint cam(main)'s own photo with pixels from cam(extra) frames wherever
    cam(extra) has something CLOSER TO CAM(MAIN)

    Both depths in the z-test are expressed in cam(main)'s frame:
      - depth_main_full: cam(main)'s own predicted depth, already in its own frame.
      - z_in_main: cam(extra)'s 3D points, projected into cam(main)'s camera.

    recon: a single  or a list of VGTGT_o recons (sharded),

    point_cloud_xforms: only needed when recon is a multi-shard list AND an extra_idx can
              land in a different shard than main_idx.
    main_idx: index of an existing frame to use as cam(main). Pass None to render
              into an arbitrary synthetic camera instead -- see new_view.
    frame_to_row: links recon indices wtih frame numbers.
    new_view: The extrinsic (3,4) for a novel camera (intrinsics are calculated automatically,
              taken from recons[0]). Shard 0 is treated as the ground-truth frame -- its cam0
              is the origin all other shards get aligned to via point_cloud_xforms -- so
              new_view must be expressed in shard 0's raw model space."""
    # main_idx/extra_idx are absolute source frame numbers (for file lookups);
    # preds arrays are indexed by row position, so translate before indexing preds.
    recons = _as_recon_list(recon)

    # Resolution is constant for the whole capture, so always size the canvas off
    # recons[0]'s own row-0 frame (its anchor/reference frame -- same one
    # get_full_res_camera below keys off) rather than off extra_indices[0], which is
    # an arbitrary real frame number that may itself be missing on disk.
    row0_frame_num = next(fn for fn, row in recons[0].frame_to_row.items() if row == 0)
    full_h, full_w = load_full_res_frame(recons[0].frames_dir, row0_frame_num).shape[:2]

    if main_idx is not None:

        main_rgb, _, depth_main, depth_conf_main, intrinsic_main_full, extrinsic_main, sigma_main, main_recon_idx = load_frame_inputs(recons, main_idx, masks_dir)
        if main_rgb is None:
            raise ValueError(
                f"main_idx {main_idx} not found in any recon shard's frame_to_row, "
                f"or its image file is missing -- check it's within a shard's frame_range"
            )
        depth_main_full = joint_bilateral_upsample(
            depth_main, depth_conf_main, main_rgb, sigma_main, conf_thresh=conf_thresh
        )  # already in cam(main)'s own frame

        canvas = main_rgb.clone()
        zbuffer = depth_main_full.clone()  # what cam(main) currently believes is in front, per pixel
    else:
        intrinsic_main_full, _, _ = get_full_res_camera(recons[0].preds, 0, (full_h, full_w))
        extrinsic_main = new_view
        main_recon_idx = 0  # shard 0 is ground truth -- new_view is expressed in its raw model space
        canvas = torch.zeros(full_h, full_w, 3, device=DEVICE)
        zbuffer = torch.full((full_h, full_w), float("inf"), device=DEVICE)

    flat_canvas = canvas.view(-1, 3)
    flat_zbuffer = zbuffer.view(-1)
    subject_positions = {}
    #Paint over the canvas
    #debug
    print(extra_indices)
    for extra_idx in extra_indices:
        if extra_idx == main_idx:
            continue
        
        extra_rgb, mask, depth_extra, depth_conf_extra, intrinsic_extra_full, extrinsic_extra, sigma_extra, extra_recon_idx = load_frame_inputs(recons, extra_idx, masks_dir)
        if extra_rgb is None:
            continue

        if masks_dir is None:
            # masking is off entirely -- unproject the whole frame
            depth_extra_full = joint_bilateral_upsample(depth_extra, depth_conf_extra, extra_rgb, sigma_extra, conf_thresh=conf_thresh)
            world_points_extra = unproject(depth_extra_full, intrinsic_extra_full, extrinsic_extra).reshape(-1, 3)
            extra_rgb = extra_rgb.reshape(-1, 3)
        else:
            # masking is on -- mask is None/empty here means no subject detected
            # this frame, not "no masks" -- unproject_masked already skips that case
            world_points_extra, extra_rgb = unproject_masked(
                mask, extra_rgb, depth_extra, depth_conf_extra, intrinsic_extra_full, extrinsic_extra,
                conf_thresh=conf_thresh
            )
        if world_points_extra is None:
            continue
        #MULTI SHARDS
        # extra's points are still in extra_recon's own model space. If that's a
        # different shard than main's, chain them through the shared real-world frame
        # to bring them into main's model space, which is what intrinsic_main_full/
        # extrinsic_main (and reproject below) actually expect.
        if extra_recon_idx != main_recon_idx:
            if point_cloud_xforms is None:
                raise ValueError(
                    f"extra_idx {extra_idx} (shard {extra_recon_idx}) and main_idx {main_idx} "
                    f"(shard {main_recon_idx}) come from different shards, but point_cloud_xforms "
                    "was not provided -- pass the per-shard (R, s, t) alignment list so "
                    "extra's points can be brought into main's frame."
                )

            def _as_tensor(x):
                return torch.as_tensor(x, dtype=world_points_extra.dtype, device=world_points_extra.device)

            R_e, s_e, t_e = (_as_tensor(v) for v in point_cloud_xforms[extra_recon_idx])
            world_points_extra = transform_RST(world_points_extra, R_e, s_e, t_e)  # extra-model -> shared real-world
            if main_recon_idx is not None:
                R_m, s_m, t_m = (_as_tensor(v) for v in point_cloud_xforms[main_recon_idx])
                world_points_extra = ((world_points_extra - t_m) / s_m) @ R_m       # shared real-world -> main-model

        subject_positions[extra_idx] = world_points_extra.mean(dim=0)
        u, v, z_in_main, valid = reproject(world_points_extra, intrinsic_main_full, extrinsic_main)
        
        
        
        #splatting here
        for dv in range(-splat_radius, splat_radius + 1): 
            for du in range(-splat_radius, splat_radius + 1): # target just 1 coordinate for every splate (i.e the same corner of every splat)
                ui = (u + du).round().long() #this becomes the pixel location too be painted
                vi = (v + dv).round().long()
                in_canvas = valid & (ui >= 0) & (ui < full_w) & (vi >= 0) & (vi < full_h)#on screen check 
                
                #vectorise for speed
                flat_idx = (vi * full_w + ui).clamp(0, full_h * full_w - 1)
                flat_idx_v = flat_idx[in_canvas]
                flat_z_v = z_in_main[in_canvas]
                flat_rgb_v = extra_rgb[in_canvas]

                # closer to cam(main) than cam(main)'s own depth there -> it's in front, paint over
                in_front = flat_z_v < (flat_zbuffer[flat_idx_v] - depth_margin)

                order = torch.argsort(flat_z_v[in_front])
                idx_front = flat_idx_v[in_front][order]
                z_front = flat_z_v[in_front][order]
                rgb_front = flat_rgb_v[in_front][order]

                keep_mask = torch.ones_like(idx_front, dtype=torch.bool)
                if idx_front.numel() > 1:
                    keep_mask[1:] = idx_front[1:] != idx_front[:-1]

                winning_idx = idx_front[keep_mask]
                winning_z = z_front[keep_mask]
                winning_rgb = rgb_front[keep_mask]

                flat_zbuffer[winning_idx] = winning_z
                flat_canvas[winning_idx] = winning_rgb

    return canvas, subject_positions
import subprocess
from C_CSV_report import add_to_report
def projection_mapping_sequence(
        recon,
        main_idx= None,
        extra_indices = None,
        masks_dir = None,
        point_cloud_xforms=None,
        new_view = None,
        confidence_threshold = 5,
        framerate=15,
        ffmpeg_path="/opt/conda/envs/Msc2/bin/ffmpeg",):

    """Composite each frame in extra_indices onto main_idx's photo and write PNGs.
    recon: single Reconstruction (existing behaviour) or a list of shards -- see
    composite_overlay. main_idx/extra_indices only need to share a shard when recon
    is a single Reconstruction; with a multi-shard list they may come from different
    shards, in which case point_cloud_xforms must also be passed.
    Returns dict of {frame_idx: world_position} for each frame with a valid subject detection."""
    subject_positions = {}
    frame_paths = []
    (assets_dir() / "projection_frames").mkdir(parents=True, exist_ok=True)
    frame_tags = []
    for e_idx in extra_indices:
        # an entry is either one frame number (one extra per output frame) or a
        # list/range of them (the whole group fused into one output frame) --
        # composite_overlay always wants the flat-list form
        group = list(e_idx) if isinstance(e_idx, (list, tuple, range)) else [e_idx] # conditional listification
        tag = str(group[0]) if len(group) == 1 else f"{group[0]}-{group[-1]}_n{len(group)}"#simplify naming

        comp, subject_pos = composite_overlay(recon, main_idx, group, masks_dir=masks_dir,
                                              point_cloud_xforms=point_cloud_xforms,
                                              new_view=new_view, conf_thresh=confidence_threshold)

        subject_positions.update(subject_pos)
        out = (comp.clamp(0, 1) * 255).byte().cpu().numpy()
        frame_path = assets_dir()/ "projection_frames"/ f"{asset_name()}_proj_{tag}.png"
        cv2.imwrite(str(frame_path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        frame_paths.append(frame_path)
        frame_tags.append(tag)

    #has to load from disk. extra_indices can skip, so ffmpeg's numbered image2
    #pattern (frame_%04d.png) can't be used - it requires a contiguous +1
    #sequence. List each file explicitly via the concat demuxer instead.
    if frame_paths:
        concat_list_path = assets_dir() /"projection_frames" / f"{asset_name()}_concat_list.txt"
        frame_duration = 1 / framerate
        with open(concat_list_path, "w") as f:
            for frame_path in frame_paths:
                f.write(f"file '{frame_path.resolve()}'\n")
                f.write(f"duration {frame_duration}\n")
            #concat demuxer ignores the last entry's duration, so repeat it
            f.write(f"file '{frame_paths[-1].resolve()}'\n")

        mp4_path = assets_dir() / f"{asset_name()}_projection.mp4"
        gif_path = assets_dir() / f"{asset_name()}_projection.gif"
        subprocess.run([
            ffmpeg_path, "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list_path),
            "-vsync", "vfr", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4_path),
        ], check=True)

    
        subprocess.run([
            ffmpeg_path, "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list_path),
            "-vsync", "vfr",
            "-vf", "fps=12,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            str(gif_path),
        ], check=True)

    for_report = {
        f"projection_frame_{tag}": to_report_path(frame_path)
        for tag, frame_path in zip(frame_tags, frame_paths)
    }
    for_report["projection_mp4"] = to_report_path(mp4_path)
    for_report["projection_gif"] = to_report_path(gif_path)
    for_report["projection_mp4_thumb"] = to_report_path(frame_paths[0])
    add_to_report(for_report)

    return subject_positions
