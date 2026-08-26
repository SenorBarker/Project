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

from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import math

from A_Config import case_dir, assets_dir, asset_name, report_path, to_report_path
from Thr3D.G_transforms_alignments import transform_RST, unproject

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


def joint_bilateral_upsample(lowres_depth, lowres_conf, lowres_rgb, full_rgb, sigma_xy, radius=1, conf_thresh = 5):
    """Upsample lowres_depth (H', W') to full_rgb's (H, W) resolution, guided by color.
    Weight = spatial gaussian (sigma fixed = footprint size, baked in via the 0.5
    factor below) * color gaussian (sigma = local color variance, auto per-pixel).

    lowres_rgb: the model's own low-res color input (preds["images"][row], HWC) --
    the anchor color for each low-res depth sample is read directly from this,
    not approximated by nearest-sampling full_rgb. The low-res image was produced
    by a bicubic-weighted blend of a full-res neighbourhood (see
    vggt_omega/utils/load_fn.py's preprocessing), not a single full-res pixel, so
    sampling full_rgb at one nearest point was a materially different (noisier)
    signal than what the depth network actually saw.

    Returns (depth_out, valid) -- valid is False wherever none of the 4 sampled
    low-res corners had usable support (all failed conf_thresh), so depth_out there
    is meaningless (weight_sum ~0) and must be masked out by the caller rather than
    treated as a real depth of ~0."""
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
            sample_rgb = lowres_rgb[ny_c, nx_c]  # the model's own colour for this depth sample -- no full-res round-trip

            
            #bilinear
            spatial_w = (1 - (gx - nx.float()).abs()).clamp(min=0) * (1 - (gy - ny.float()).abs()).clamp(min=0)
            #colour linear
            color_dist2 = ((full_rgb - sample_rgb) ** 2).sum(dim=-1)
            if (color_dist2 > 3).any():
                print("COLOUR ERROR", color_dist2)
            
            color_w = 1 - color_dist2 / 3
            w = spatial_w * color_w * in_bounds.float() * v_sample
            
            '''
            # double-Gaussian joint bilateral weight, same variable names as your function
            # (needs two new params this function doesn't have: sigma_s, sigma_r)
            sigma_s = 0.1
            sigma_r = 0.1
            spatial_dist2 = (gx - nx.float()) ** 2 + (gy - ny.float()) ** 2      # ||p-q||^2
            spatial_w = torch.exp(-spatial_dist2 / (2 * sigma_s ** 2))          # domain Gaussian
            color_dist2 = ((full_rgb - sample_rgb) ** 2).sum(dim=-1)             # ||I_p-I_q||^2
            color_w = torch.exp(-color_dist2 / (2 * sigma_r ** 2))              # range Gaussian
            w = spatial_w * color_w * in_bounds.float() * v_sample               # same combine as your code
            '''
      

            out_depth += w * d_sample
            weight_sum += w

    valid = weight_sum > 1e-6
    depth_out = out_depth / weight_sum.clamp_min(1e-6)# normalise to 1

    return depth_out, valid





def reproject(points_world, intrinsic_ref, extrinsic_ref, ortho_params=None):
    """ortho_params: optional (scale_x, scale_y, cx, cy) -- when given, projects
    orthographically (fixed scale, no perspective divide) instead of through
    intrinsic_ref's pinhole model. z is still returned either way, for the z-buffer."""
    R = extrinsic_ref[:, :3]
    t = extrinsic_ref[:, 3]
    points_ref = points_world @ R.T + t
    z = points_ref[..., 2]

    if ortho_params is not None:
        scale_x, scale_y, cx, cy = ortho_params
        u = points_ref[..., 0] * scale_x + cx
        v = points_ref[..., 1] * scale_y + cy
        valid = torch.ones_like(z, dtype=torch.bool)
        return u, v, z, valid

    fx, fy = intrinsic_ref[0, 0], intrinsic_ref[1, 1]
    cx, cy = intrinsic_ref[0, 2], intrinsic_ref[1, 2]
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


def crop_to_mask_region(mask, extra_rgb, depth_low, conf_low, lowres_rgb, intrinsic_full, margin=8):
    """
    Crop extra_rgb/mask (full-res) and depth_low/conf_low/lowres_rgb (low-res) to
    the mask's bounding box, with shifted intrinsics to match the crop origin.
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
    lowres_rgb_crop = lowres_rgb[y0_lr:y1_lr, x0_lr:x1_lr]

    intrinsic_crop = intrinsic_full.clone()
    intrinsic_crop[0, 2] -= x0
    intrinsic_crop[1, 2] -= y0

    return rgb_crop, mask_crop, depth_crop, conf_crop, lowres_rgb_crop, intrinsic_crop, (x0, y0)


def unproject_masked(mask, rgb_full, depth_low, conf_low, lowres_rgb, intrinsic_full, extrinsic, margin=8, conf_thresh=5):
    """Unproject the pixels covered by `mask` to VGGT-O world-space points.
    Up-reses the depth to fit the mask using jonit bilateral upsample

    Returns (world_points, rgb) for just the masked pixels, or (None, None) if the mask
    is empty. 
    World points are in the original  VGGT-O model frame and scale regardless of resolution --
    see unproject().
    """
    if mask is None or mask.sum() == 0:
        return None, None

    sigma = footprint_size(rgb_full.shape[:2], depth_low.shape)
    rgb_crop, mask_crop, depth_crop, conf_crop, lowres_rgb_crop, intrinsic_crop, _ = crop_to_mask_region(
        mask, rgb_full, depth_low, conf_low, lowres_rgb, intrinsic_full, margin=margin
    )
    depth_crop_full, valid_crop = joint_bilateral_upsample(depth_crop, conf_crop, lowres_rgb_crop, rgb_crop, sigma, conf_thresh=conf_thresh)
    world_points = unproject(depth_crop_full, intrinsic_crop, extrinsic)
    keep = mask_crop & valid_crop
    return world_points[keep], rgb_crop[keep]#mask pixels with no usable upsample support are dropped too


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
    native-resolution depth/confidence/colour, full-res-scaled intrinsic + extrinsic,
    and which shard (recon_idx) it came from -- so the caller can tell whether two
    frames live in the same recon's model space or need point_cloud_xforms to bring
    them together (recon_idx is always 0 for a single-recon caller)."""

    recon_idx, recon, row = find_recon_for_frame(recon, frame_idx)

    if row is None:
        return None, None, None, None, None, None, None, None, None

    rgb_full = load_full_res_frame(recon.frames_dir, frame_idx)

    if rgb_full is None:
        return None, None, None, None, None, None, None, None, None

    mask = load_mask(frame_idx, masks_dir)
    intrinsic_full, extrinsic, sigma = get_full_res_camera(recon.preds, row, rgb_full.shape[:2])
    depth = recon.preds["depth"][row]
    depth_conf = recon.preds["depth_conf"][row]
    lowres_rgb = recon.preds["images"][row].permute(1, 2, 0)  # (3,H,W) -> (H,W,3), the model's own colour input

    return rgb_full, mask, depth, depth_conf, lowres_rgb, intrinsic_full, extrinsic, sigma, recon_idx


def composite_overlay(recon,
                      main_idx,
                      extra_indices,
                      masks_dir = None,
                      point_cloud_xforms=None,
                      depth_margin=0.0,
                      splat_radius=1,
                      new_view=None,
                      conf_thresh = 5,
                      ortho_params=None,
                      canvas_size=None,
                      analyse = False,
                      use_lowres_images=False
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



    if main_idx is not None:
        if use_lowres_images:
            # no disk read needed -- size the canvas off preds' own native resolution.
            full_h, full_w = recons[0].preds["depth"].shape[1:]
        else:
            # Resolution is constant for the whole capture, so always size the canvas off
            # recons[0]'s own row-0 frame (its anchor/reference frame -- same one
            # get_full_res_camera below keys off) rather than off extra_indices[0], which is
            # an arbitrary real frame number that may itself be missing on disk.
            row0_frame_num = next(fn for fn, row in recons[0].frame_to_row.items() if row == 0)
            full_h, full_w = load_full_res_frame(recons[0].frames_dir, row0_frame_num).shape[:2]

        main_rgb, _, depth_main, depth_conf_main, lowres_rgb_main, intrinsic_main_full, extrinsic_main, sigma_main, main_recon_idx = load_frame_inputs(recons, main_idx, masks_dir)
        if main_rgb is None:
            raise ValueError(
                f"main_idx {main_idx} not found in any recon shard's frame_to_row, "
                f"or its image file is missing -- check it's within a shard's frame_range"
            )
        depth_main_full, valid_main = joint_bilateral_upsample(
            depth_main, depth_conf_main, lowres_rgb_main, main_rgb, sigma_main, conf_thresh=conf_thresh
        )  # already in cam(main)'s own frame

        canvas = main_rgb.clone()
        zbuffer = depth_main_full.clone()  # what cam(main) currently believes is in front, per pixel
        # pixels with no usable upsample support have a meaningless depth_main_full
        # value -- set them to +inf ("nothing known here") so any extra frame's real
        # point can paint over them, instead of a bogus near-zero depth blocking it.
        zbuffer[~valid_main] = float("inf")
    else:
        # Synthetic/novel-view canvas defaults to HD, regardless of the source movie's
        # native resolution -- a novel view's projected pixel spread can exceed native
        # resolution, so tying canvas size to the source frame would be destructive.
        # canvas_size (from _ortho_view_from_recon) overrides this when the point
        # cloud's own density needs more than that.
        full_h, full_w = canvas_size if canvas_size is not None else (1080, 1920)
        if ortho_params is not None:
            intrinsic_main_full = None  # unused by reproject() when ortho_params is set
        else:
            intrinsic_main_full, _, _ = get_full_res_camera(recons[0].preds, 0, (full_h, full_w))
        extrinsic_main = new_view
        main_recon_idx = 0  # shard 0 is ground truth -- new_view is expressed in its raw model space
        canvas = torch.zeros(full_h, full_w, 3, device=DEVICE)
        zbuffer = torch.full((full_h, full_w), float("inf"), device=DEVICE)

    flat_canvas = canvas.view(-1, 3)
    flat_zbuffer = zbuffer.view(-1)
    subject_positions = {}

    # circular splat disk (matches U_rendering.BEV_render's offset construction) --
    # a square (all du,dv in range) would splat every point as a visible square.
    splat_offsets = [(du, dv) for du in range(-splat_radius, splat_radius + 1)
                              for dv in range(-splat_radius, splat_radius + 1)
                              if du * du + dv * dv <= splat_radius ** 2 + 0.5]

    #Paint over the canvas
    #debug
    
    for extra_idx in extra_indices:
        if extra_idx == main_idx:
            continue
        
        extra_rgb, mask, depth_extra, depth_conf_extra, lowres_rgb_extra, intrinsic_extra_full, extrinsic_extra, sigma_extra, extra_recon_idx = load_frame_inputs(recons, extra_idx, masks_dir)
        if extra_rgb is None:
            continue

        if masks_dir is None:
            # masking is off entirely -- unproject the whole frame
            depth_extra_full, valid_extra = joint_bilateral_upsample(depth_extra, depth_conf_extra, lowres_rgb_extra, extra_rgb, sigma_extra, conf_thresh=conf_thresh)
            world_points_extra = unproject(depth_extra_full, intrinsic_extra_full, extrinsic_extra).reshape(-1, 3)
            extra_rgb = extra_rgb.reshape(-1, 3)
            valid_extra = valid_extra.reshape(-1)
            world_points_extra, extra_rgb = world_points_extra[valid_extra], extra_rgb[valid_extra]
        else:
            # masking is on -- mask is None/empty here means no subject detected
            # this frame, not "no masks" -- unproject_masked already skips that case
            world_points_extra, extra_rgb = unproject_masked(
                mask, extra_rgb, depth_extra, depth_conf_extra, lowres_rgb_extra, intrinsic_extra_full, extrinsic_extra,
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
        u, v, z_in_main, valid = reproject(world_points_extra, intrinsic_main_full, extrinsic_main, ortho_params=ortho_params)
        
        
        
        #splatting here -- circular disk (splat_offsets), not every (du,dv) in the square
        for du, dv in splat_offsets:
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


def _ortho_view_from_recon(recon, conf_thresh=5,
                            camera_forward=(0.0, 1.0, 0.0), camera_up=(0.0, 0.0, 1.0),
                            margin_frac=0.05, width=1920, height=1080, analyse = False,
                            resolution="lores"):
    """Computes an orthographic top-down view (new_view extrinsic + ortho_params)
    that fits recon's whole reconstruction, with margin_frac border. camera_forward/
    camera_up default to the raw-model-space equivalent of U_rendering.BEV_render's
    (0,-1,0)/(0,0,-1) defaults -- inverted to account for VGGT_O_preds_to_ply_export's
    axis flip (points[:, 1:] *= -1), since this operates on the raw (pre-flip)
    reconstruction, not exported plys.

    width/height are a floor, not a fixed size -- see BEV_render's docstring in
    U_rendering.py for why: the canvas grows past width x height if the point
    cloud's own density needs more resolution than that to avoid collapsing
    several points onto the same output pixel.

    Bounds are derived per-frame via unproject(depth, intrinsic, extrinsic) from
    recon.preds directly, in recon's own raw model space -- the same frame
    composite_overlay's own point-painting already uses (get_full_res_camera/
    load_frame_inputs read recon.preds directly too, with no cam0 re-anchoring).
    This also makes _ortho_view_from_recon technique-agnostic: any Reconstruction
    with depth/intrinsic/extrinsic in preds works, not just VGGT-Omega's
    predictions.npz (which used to be re-read here for a precomputed,
    VGGT-Omega-only world_points_from_depth key -- numerically near-identical to
    this, since VGGT-Omega's raw extrinsic already has frame 0 ~= identity by
    construction, same invariant every other technique's loader documents)."""
    forward = torch.as_tensor(camera_forward, dtype=torch.float32, device=DEVICE)
    forward = forward / forward.norm()
    up = torch.as_tensor(camera_up, dtype=torch.float32, device=DEVICE)
    right = torch.linalg.cross(forward, up)
    right = right / right.norm()
    true_up = torch.linalg.cross(right, forward)

    depth_conf = recon.preds["depth_conf"]
    print("min depth", recon.preds["depth"].min())
    world_points = torch.stack([
        unproject(recon.preds["depth"][row], recon.preds["intrinsic"][row], recon.preds["extrinsic"][row])
        for row in range(depth_conf.shape[0])
    ])
    #this is the last time they are per frame, so analyse here
    if analyse:
        print(world_points.shape)
        import matplotlib.pyplot as plt
        from scipy.spatial import cKDTree

        min_cell_size = []
        all_nn_dists = []
        frame0_nn_dists = None
        for frame in range(len(world_points)):
            valid = (depth_conf[frame] > conf_thresh) & (recon.preds["depth"][frame] > 0)
            pts = world_points[frame][valid]
            pts = pts.detach().cpu().numpy()
            pts = pts[np.isfinite(pts).all(axis=1)]
            pts_xz = pts[:, [0, 2]]  # ignore y (height) -- top-down render plane
            # A grid cell only collides two points if they're close on BOTH x and z
            # simultaneously (a shared x or shared z alone isn't a collision -- see
            # the (0,0)/(1,0)/(0,1)/(1,1) cross example). That joint condition is
            # exactly Chebyshev (L-inf) distance, not two independent per-axis gaps.
            # So the minimum square cell size that keeps every point in its own cell
            # is the minimum Chebyshev nearest-neighbor distance across all points.
            tree = cKDTree(pts_xz)
            nn_dist, _ = tree.query(pts_xz, k=2, p=np.inf, workers=-1)
            min_cell_size.append(nn_dist[:, 1].min())
            all_nn_dists.append(nn_dist[:, 1])
            if frame == 0:
                frame0_nn_dists = nn_dist[:, 1]

        all_nn_dists = np.concatenate(all_nn_dists)

        plt.figure()
        plt.plot(range(1, len(min_cell_size) + 1), min_cell_size)
        plt.xlabel("frame")
        plt.ylabel("min canvas cell size (x,z, no collisions)")
        plt.xlim(1, 50)
        plt.ylim(bottom=0)
        outpath = assets_dir() / f"{asset_name()}_point_density.png"
        plt.savefig(outpath)

        plt.figure()
        plt.hist(all_nn_dists, bins=100, range=(0, 1.4e-6))
        plt.xlabel("nearest-neighbor cell size (x,z, no collisions)")
        plt.ylabel("count")
        hist_outpath = assets_dir() / f"{asset_name()}_point_density_hist.png"
        plt.savefig(hist_outpath)

        plt.figure()
        weights = np.full(len(all_nn_dists), 100.0 / len(all_nn_dists))
        plt.hist(all_nn_dists, bins=100, range=(0, 10 * 2e-6), cumulative=True, weights=weights)
        plt.xlabel("nearest-neighbor cell size")
        plt.ylabel("% of total")
        plt.yticks(range(0, 25, 1))
        plt.grid(axis="y", linewidth=0.5)
        cumhist_outpath = assets_dir() / f"{asset_name()}_point_density_cumhist.png"
        plt.savefig(cumhist_outpath)

        plt.figure()
        weights_frame0 = np.full(len(frame0_nn_dists), 100.0 / len(frame0_nn_dists))
        plt.hist(frame0_nn_dists, bins=100, range=(0, 10 * 2e-6), cumulative=True, weights=weights_frame0)
        plt.xlabel("nearest-neighbor cell size (frame 0 only)")
        plt.ylabel("% of frame 0 total")
        plt.yticks(range(0, 25, 1))
        plt.grid(axis="y", linewidth=0.5)
        frame0_outpath = assets_dir() / f"{asset_name()}_point_density_cumhist_frame0.png"
        plt.savefig(frame0_outpath)

        # scale = 1/cell_size: linear world-units-to-pixels conversion, same factor
        # applied to both x and z (square cell) -- not squared, since this maps a
        # linear distance to a linear pixel spacing, not an area to a point count.
        # Same bars as chart 3 (identical data/bins/shape) -- only the x tick labels
        # are remapped to scale (1/edge) instead of re-histogramming 1/d directly,
        # which distorts the shape (most mass is at large d -> small scale, so it
        # piles up at one end instead of tracing the same curve).
        # NOT saved yet -- canvas-size-vs-scale (needs x_range/y_range, computed
        # further down once we're in cam space) gets superimposed on a right-hand
        # axis before this figure is written out, see below.
        fig_scale, ax_scale = plt.subplots()
        cellsize_edges = np.linspace(1.0 / 20000, 1.0 / 1769, 101)  # scale range: 20,000 (left/max) down to 1,769 (right/min)
        ax_scale.hist(all_nn_dists, bins=cellsize_edges, cumulative=True, weights=weights, color="gray")
        tick_edges = cellsize_edges[::10]
        ax_scale.set_xticks(tick_edges, [f"{1.0 / e:,.0f}" for e in tick_edges], rotation=45)
        ax_scale.set_xlabel("scale to make 1 pixel wide and tall")
        ax_scale.set_ylabel("% of total")
        ax_scale.set_yticks(range(0, 100, 5))
        ax_scale.grid(axis="y", linewidth=0.5)

    valid = (depth_conf > conf_thresh).reshape(-1)
    points = world_points.reshape(-1, 3)[valid]
    print(points.shape)
    n_points = points.shape[0]
    if n_points == 0:
        raise ValueError(
            f"_ortho_view_from_recon: no points passed conf_thresh={conf_thresh} in any frame -- "
            f"this recon's confidence values may be on a different scale than the default threshold "
            f"assumes (e.g. MegaSaM has no real per-pixel confidence at all, see "
            f"F_transpose_to_recon_objects.megasam_to_reconstruction), or genuinely low-confidence "
            f"throughout. Try a lower confidence_threshold for this technique."
        )

    #camera now
    x_cam = (points @ right).float().cpu().numpy()
    y_cam = (points @ true_up).float().cpu().numpy()

    x_min, x_max = float(x_cam.min()), float(x_cam.max())
    y_min, y_max = float(y_cam.min()), float(y_cam.max())
    x_range, y_range = x_max - x_min, y_max - y_min
    x_min -= margin_frac * x_range; x_max += margin_frac * x_range
    y_min -= margin_frac * y_range; y_max += margin_frac * y_range
    x_range, y_range = x_max - x_min, y_max - y_min
    print("mins, maxes and ranges", x_min, x_max , x_range , y_min,  y_max,y_range )

    if analyse:
        # canvas size (pixels) needed to fit the whole padded bounding box at each
        # candidate scale -- canvas_dim = spatial_range * scale = spatial_range / cell_size.
        # Superimposed on the scale-cumulative-% figure (ax_scale) via a right-hand
        # twin axis, sharing the same x positions/tick labels, so the two are read
        # directly off one chart instead of two.
        ax_canvas = ax_scale.twinx()
        canvas_cellsizes = cellsize_edges[1:]  # skip 0 -> divide-by-zero
        canvas_width = x_range / canvas_cellsizes
        canvas_height = y_range / canvas_cellsizes
        ax_canvas.plot(canvas_cellsizes, canvas_width, color="tab:orange", label="canvas width")
        ax_canvas.plot(canvas_cellsizes, canvas_height, color="tab:green", label="canvas height")
        ax_canvas.axhline(8192, color="orange", linewidth=2, linestyle="--", label="45MP width (8192px)")
        ax_canvas.axhline(5464, color="green", linewidth=2, linestyle="--", label="45MP height (5464px)")
        ax_canvas.set_ylabel("canvas size needed (pixels)")
        ax_canvas.yaxis.set_major_formatter(lambda val, pos: f"{val:,.0f}")
        ax_canvas.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
        scalehist_outpath = assets_dir() / f"{asset_name()}_point_density_scale_cumhist.png"
        fig_scale.savefig(scalehist_outpath, bbox_inches="tight")
    # scale that fits the (padded) bounding box into width x height (the floor),
    # vs. scale implied by the point cloud's own density (avg spacing assuming a
    # roughly uniform 2D scatter) -- never shrink below the floor, but grow past
    # it if the data is denser than that. Note: n_points here is from the native
    # (pre-upsample) world_points_from_depth grid, so this underestimates true
    # density for composite_overlay's actual (upsampled, denser) render -- still
    # strictly better than a fixed canvas, just not exactly matched.
    if resolution == "hires":
        # native_scale is density-derived and can blow up on dense/degenerate point
        # clouds -- hardcode scale instead of computing it until that's made safe.
        scale = 6000
        print("scale - original" , scale)

    else:
        fit_scale = min(width / x_range, height / y_range)
        avg_spacing = (x_range * y_range / n_points) ** 0.5
        native_scale = 1.0 / avg_spacing
        scale = max(fit_scale, native_scale)
        print("scale - original" , scale)
    out_width = max(width, round(x_range * scale))
    out_height = max(height, round(y_range * scale))
    cx, cy = out_width / 2, out_height / 2
    x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2

    # new_view (3,4) extrinsic: R rows are the camera basis; points_ref = R@p + t,
    # so points_ref[0] = right.p + t[0] etc. x_center/y_center are already expressed
    # in this (right, true_up) basis (computed from x_cam/y_cam above), so t is just
    # their negation directly -- no further rotation needed. t[2] (forward/depth) is
    # left at 0 since depth isn't recentred, only used for z-buffer ordering.
    R = torch.stack([right, true_up, forward], dim=0)
    t = torch.tensor([-x_center, -y_center, 0.0], device=DEVICE)
    new_view = torch.cat([R, t.unsqueeze(1)], dim=1)

    ortho_params = (scale, -scale, cx, cy)  # y-flip matches image row-down convention
    return new_view, ortho_params, out_width, out_height


def BEV_tile_render_PM(recon, extra_indices, masks_dir=None, splat_radius=0,
                   confidence_threshold=5, margin_frac=0.05, out_path=None, analyse = False,
                   resolution="lores"):
    """Static orthographic top-down (BEV) still, fused from many frames via
    composite_overlay's masking/upsample/z-buffer machinery -- no .ply export, no
    GPS/real-world alignment, works directly on the raw reconstruction in its own
    (arbitrary/unscaled) model units -- any technique's Reconstruction works, not
    just VGGT-Omega's, see F_transpose_to_recon_objects.py. Backplate only --
    animating camera/subject position over time is a separate later compositor
    (P_trace_overlayer.py).

    out_path: if given, save there instead of assets_dir()/asset_name() and skip
    add_to_asset_list -- for callers (e.g. the density sweep) that manage their own
    output location/bookkeeping and aren't part of the single case+experiment
    report. Default None preserves the original assets_dir()-based behavior.

    Returns subject_positions (raw model space, a free by-product of
    composite_overlay) plus the new_view/ortho_params this canvas was built with,
    so that later compositor can project onto this exact canvas without
    recomputing (and risking drift from) the orthographic projection."""
    new_view, ortho_params, out_width, out_height = _ortho_view_from_recon(
        recon, conf_thresh=confidence_threshold, margin_frac=margin_frac, analyse = analyse,
        resolution=resolution,
    )

    canvas, subject_positions = composite_overlay(
        recon, main_idx=None, extra_indices=extra_indices, masks_dir=masks_dir,
        splat_radius=splat_radius, new_view=new_view, conf_thresh=confidence_threshold,
        ortho_params=ortho_params, canvas_size=(out_height, out_width),
    )

    img = (canvas.clamp(0, 1) * 255).byte().cpu().numpy()
    if out_path is None:
        outpath = assets_dir() / f"{asset_name()}_BEV_PM_{resolution}_{confidence_threshold}.png"
    else:
        outpath = Path(out_path)
        outpath.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(outpath), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    if out_path is None:
        add_to_asset_list({"BEV_tile_render_PM": to_report_path(outpath)})
    return subject_positions, new_view, ortho_params, outpath


import subprocess
from C_CSV_report import add_to_report, add_to_asset_list
def projection_mapping_sequence(
        recon,
        main_idx= None,
        extra_indices = None,
        masks_dir = None,
        point_cloud_xforms=None,
        new_view = None,
        confidence_threshold = 5,
        frame_rate=15,
        ffmpeg_path="/opt/conda/envs/Msc2/bin/ffmpeg",
        splat_radius = 1,
        use_lowres_images=False,
        name = None):

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
                                              new_view=new_view, conf_thresh=confidence_threshold, 
                                                use_lowres_images=use_lowres_images,
                                              splat_radius = splat_radius)

        subject_positions.update(subject_pos)
        out = (comp.clamp(0, 1) * 255).byte().cpu().numpy()
        if name == None:
            frame_path = assets_dir()/ "projection_frames"/ f"{asset_name()}_proj_{tag}_aug.png"
        else:
            frame_path = assets_dir()/ "projection_frames"/ name 

   
        cv2.imwrite(str(frame_path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        frame_paths.append(frame_path)
        frame_tags.append(tag)

    #has to load from disk. extra_indices can skip, so ffmpeg's numbered image2
    #pattern (frame_%04d.png) can't be used - it requires a contiguous +1
    #sequence. List each file explicitly via the concat demuxer instead.
    if frame_paths:
        concat_list_path = assets_dir() /"projection_frames" / f"{asset_name()}_concat_list.txt"
        frame_duration = 1 / frame_rate
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
    add_to_asset_list(for_report)

    return subject_positions


#---------------------autocam
def solve_auto_camera_extrinsic(points, target, R_c2w, intrinsic, image_hw,
                                 extra_distance_margin=0.0, dtype=None):
    """Translation-only solve for an auto camera: rotation R_c2w and intrinsic are
    FIXED (no zoom, no re-orient) -- this only finds how far back along its own viewing
    axis to place the camera so every point in `points` fits in frame. Same [R|t] /
    P_cam=R@P_world+t convention as reproject() and preds["extrinsic"].

    points: (N,3) world/model-space points to frame -- the subject's actual points
        (e.g. Q_subject_analysis' mask_points), 
    target: (3,) the point to centre in frame (also what R_c2w should be aimed at).
    R_c2w: (3,3) fixed camera-to-model rotation (e.g. from
        Q_subject_analysis.look_at_rotation).
    intrinsic: (3,3) fixed intrinsic (fx,fy,cx,cy) -- native low-res is fine, since
        fx/w and fy/h are resolution-invariant (see scale_intrinsic/get_full_res_camera).
    image_hw: (h, w) resolution intrinsic was measured at.
    extra_distance_margin: extra pull-back distance (same units as points), on top of
        what's needed to fit them.

    Returns a (3,4) torch tensor on DEVICE, ready for projection_mapping_sequence's
    new_view=... (must be in shard 0's raw model space, same as the inputs here).
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target, dtype=np.float64)
    R_c2w = np.asarray(R_c2w, dtype=np.float64)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    h, w = image_hw
    R_w2c = R_c2w.T

    # points re-expressed in the camera's own (rotated) axes, relative to target
    rel = (points - target) @ R_w2c.T

    # solve d PER POINT: a point's actual depth from the camera is d + rel[:,2]
    # (nearer points are closer than target, so they're magnified more than a
    # target-depth estimate accounts for). Require, for every point:
    #   |rel[:,0]| * fx / (d + rel[:,2]) <= w/2   (fits horizontally)
    #   |rel[:,1]| * fy / (d + rel[:,2]) <= h/2   (fits vertically)
    #   d + rel[:,2] > 0                          (in front of the camera)
    # each rearranges to a lower bound on d; take the max across all points/axes.
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    req_x = np.abs(rel[:, 0]) * fx / (w / 2.0) - rel[:, 2]
    req_y = np.abs(rel[:, 1]) * fy / (h / 2.0) - rel[:, 2]
    req_z = -rel[:, 2]
    d = max(req_x.max(), req_y.max(), req_z.max()) + extra_distance_margin

    forward = R_c2w[:, 2]
    C = target - d * forward   # camera centre, model space
    t = -R_w2c @ C             # P_cam = R_w2c @ P_model + t

    new_view_np = np.concatenate([R_w2c, t[:, None]], axis=1)   # (3,4)
    dtype = dtype or torch.float32
    return torch.as_tensor(new_view_np, dtype=dtype, device=DEVICE)