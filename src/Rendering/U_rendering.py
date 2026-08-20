
import numpy as np
import torch

import os
import subprocess
import webbrowser
from pathlib import Path

from A_Config import assets_dir, report_path, to_report_path, asset_name
from C_CSV_report import add_to_report, add_to_asset_list

# open3d/scenepic are picky about their conda env, so this viewer runs out-of-process
# in a separate env (see U_scenepic_o3d_worker.py) rather than importing them here --
# same reasoning as D_megasam_recon.py's `conda run -n mega_sam ...` isolation.
_ENV_BLOCKLIST = ("MPLBACKEND", "PYTHONPATH", "PYTHONHOME")


def _clean_env(extra=None):
    env = {k: v for k, v in os.environ.items() if k not in _ENV_BLOCKLIST}
    if extra:
        env.update(extra)
    return env


#interactive viewer to help pick angles - not part of the automated pipeline
def view_ply_sequence_with_scenepic(ply_dir,  point_size=0.01,
                                     pattern="*.ply", conda_env="open_3D_env"):
    output_html = Path(assets_dir()) / f"{asset_name()}_4d.html"
    output_html = Path(output_html)

    worker_path = Path(__file__).parent / "U_scenepic_o3d_worker.py"
    subprocess.run([
        "conda", "run", "--no-capture-output", "-n", conda_env,
        "python", str(worker_path),
        "--ply_dir", str(ply_dir),
        "--output_html", str(output_html),
        "--point_size", str(point_size),
        "--pattern", pattern,
    ], check=True, env=_clean_env())

    add_to_asset_list({"4D_recon_link": to_report_path(output_html)})
    webbrowser.open(str(output_html))
    return

#automated numpy renderer
from PIL import Image
from plyfile import PlyData
from C_CSV_report import add_to_report, add_to_asset_list


def _read_ply_points_colors(ply_path):
    """Read vertex positions and RGB colors (normalised to [0, 1]) from a .ply file.
    Returns (points, colors) with colors=None if the file has no red/green/blue properties."""
    vertex = PlyData.read(str(ply_path))["vertex"]
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1).astype(np.float64)
    if {"red", "green", "blue"} <= set(vertex.data.dtype.names):
        colors = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=-1).astype(np.float64) / 255.0
    else:
        colors = None
    return points, colors

# Automated, headless renderer: fixed/locked-off camera, one frame per ply,
# then encoded straight to mp4/gif. No browser, no manual navigation.
#
# Camera is level (not tilted), 1m above the origin in the ply's own y-up
# frame, looking along -Z. This is a placeholder pose until real per-frame
# camera extrinsics can be piped in from camera_poses.csv.
#
# Point size is a fixed screen-space splat radius (not world-space like the
# ScenePic sphere instancing above), so far points don't shrink to sub-pixel -
# this is what makes it look denser than the ScenePic viewer at the same data.
def basic_point_cloud_render(
    ply_dir,
    model,
    pattern="*.ply",
    width=1920,
    height=1080,
    vfov_deg=60.0,
    splat_radius=2,
    camera_eye=(0.0, 1.0, 0.0),#position
    camera_forward=(0.0, 0.0, -1.0),#angle, not point in space to look at 
    camera_up=(0.0, 1.0, 0.0),
    background_color=(30, 30, 30),
    framerate=30,
    ffmpeg_path="/opt/conda/envs/Msc2/bin/ffmpeg",
):
    ply_paths = sorted(Path(ply_dir).glob(pattern))
    if not ply_paths:
        raise ValueError(f"No ply files found in {ply_dir}")
    #store frames here
    frames_out_dir = assets_dir() / f"PC_render_frames_{model}"
    print(frames_out_dir)
    frames_out_dir.mkdir(parents=True, exist_ok=True)

    eye = np.array(camera_eye, dtype=np.float64)
    forward = np.array(camera_forward, dtype=np.float64)
    forward = forward / np.linalg.norm(forward)
    up = np.array(camera_up, dtype=np.float64)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)

    f = (height / 2) / np.tan(np.radians(vfov_deg) / 2)
    cx, cy = width / 2, height / 2

    offsets = [(dx, dy) for dx in range(-splat_radius, splat_radius + 1)
                         for dy in range(-splat_radius, splat_radius + 1)
                         if dx * dx + dy * dy <= splat_radius ** 2 + 0.5]
    offs_dx = np.array([o[0] for o in offsets])
    offs_dy = np.array([o[1] for o in offsets])
    k = len(offsets)
    background = np.array(background_color, dtype=np.uint8)

    frame_paths = []
    for i, ply_path in enumerate(ply_paths):
        ply_name = ply_path.stem
        points, colors = _read_ply_points_colors(ply_path)
        frame_path = frames_out_dir / f"{asset_name()}_{ply_name}.png"
        frame_paths.append(frame_path)
        if i == 0:
            thumb_path = frame_path

        if points.shape[0] == 0 or colors is None:
            Image.fromarray(np.tile(background, (height, width, 1))).save(frame_path)
            continue

        rel = points - eye
        x_cam = rel @ right
        y_cam = rel @ true_up
        z_cam = rel @ forward

        valid = z_cam > 0.05
        x_cam, y_cam, z_cam, c = x_cam[valid], y_cam[valid], z_cam[valid], colors[valid]

        u = f * x_cam / z_cam + cx
        v = -f * y_cam / z_cam + cy

        n = u.shape[0]
        u_rep = np.round(u).astype(np.int64).repeat(k) + np.tile(offs_dx, n)
        v_rep = np.round(v).astype(np.int64).repeat(k) + np.tile(offs_dy, n)
        z_rep = z_cam.repeat(k)
        c_rep = np.repeat(c, k, axis=0)

        in_bounds = (u_rep >= 0) & (u_rep < width) & (v_rep >= 0) & (v_rep < height)
        u_rep, v_rep, z_rep, c_rep = u_rep[in_bounds], v_rep[in_bounds], z_rep[in_bounds], c_rep[in_bounds]

        # z-buffer: draw far-to-near so nearer splats win where they overlap
        order = np.argsort(-z_rep)
        u_rep, v_rep, c_rep = u_rep[order], v_rep[order], c_rep[order]

        color_buf = np.tile(background.astype(np.float64) / 255, (height, width, 1))
        color_buf[v_rep, u_rep] = c_rep

        img = (color_buf * 255).astype(np.uint8)
        Image.fromarray(img).save(frame_path)

    mp4_path = assets_dir() / f"{asset_name()}_point_cloud_render_{model}.mp4"
    gif_path = assets_dir() / f"{asset_name()}_point_cloud_render_{model}.gif"
    mp4_name = f"{asset_name()}_point_cloud_render_{model}.mp4"
    gif_name = f"{asset_name()}_point_cloud_render_{model}.gif"

    #frame filenames aren't a contiguous %04d sequence (keyed off each ply's own
    #name), so ffmpeg's image2 pattern can't be used -- list each file explicitly
    #via the concat demuxer instead (same fix as P_projection_mapping.py).
    concat_list_path = frames_out_dir / f"{asset_name()}_concat_list.txt"
    frame_duration = 1 / framerate
    with open(concat_list_path, "w") as fh:
        for frame_path in frame_paths:
            fh.write(f"file '{frame_path.resolve()}'\n")
            fh.write(f"duration {frame_duration}\n")
        #concat demuxer ignores the last entry's duration, so repeat it
        fh.write(f"file '{frame_paths[-1].resolve()}'\n")

    subprocess.run([
        ffmpeg_path, "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_list_path),
        "-vsync", "vfr", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4_path),
    ], check=True)

    subprocess.run([
        ffmpeg_path, "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_list_path),
        "-vsync", "vfr",
        "-vf", "fps=12,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
        str(gif_path),
    ], check=True)
    
    add_to_asset_list({
    f"point_cloud_render_{model}_mp4": mp4_name,
    f"point_cloud_render_{model}_gif": gif_name,
    f"point_cloud_render_{model}_thumb": to_report_path(thumb_path),
    f"4D_recon_clicker"                      : to_report_path(thumb_path)}#for the 4d recon
    )
    return 


def _ortho_project_ply_to_cam(ply_path, right, true_up, forward, device):
    """Load one ply straight onto device and project it into (x_cam, y_cam, z_cam)
    -- called twice per ply (bounds pass, splat pass) so no ply's points ever have
    to sit in memory alongside another's."""
    points, colors = _read_ply_points_colors(ply_path)
    if points.shape[0] == 0 or colors is None:
        return None
    points_t = torch.as_tensor(points, dtype=torch.float32, device=device)
    colors_t = torch.as_tensor(colors, dtype=torch.float32, device=device)
    x_cam = points_t @ right
    y_cam = points_t @ true_up
    z_cam = points_t @ forward
    #valid = z_cam > 0.05
    
    #return x_cam[valid], y_cam[valid], z_cam[valid], colors_t[valid]
    return x_cam, y_cam, z_cam, colors_t

#make the map
from P_projection_mapping import joint_bilateral_upsample
def BEV_render(
        ply_dir,
        model,
        pattern="*.ply",
        width=1920,
        height=1080,
        margin_frac=0.05,#fraction of the fitted extent left as empty border on each side
        camera_forward=(0.0, -1.0, 0.0),#looking straight down -Y
        camera_up=(0.0, 0.0, -1.0),#defines which world direction is "up" in the frame (north)
        background_color=(30, 30, 30),
        out_path=None,
    ):
        """Single orthographic top-down still spanning all plys in ply_dir --
        bounds are read off the data itself (with margin_frac padding), not hardcoded.

        width/height are a floor, not a fixed size: the canvas fits the point cloud's
        bounding box at whichever scale is tighter -- the one that fills width x height,
        or (if the point cloud's own density needs more than that to give each point its
        own pixel) the one implied by that density, growing the canvas past width x
        height rather than collapsing multiple points onto the same output pixel.
        One point = one pixel, no splatting -- splatting a fixed-size shape over
        already-dense points just re-flattens the fine per-pixel detail upsampling
        produced in the first place.

        Two streaming passes over the plys (bounds, then paint) on GPU tensors so the
        whole capture's points never have to be merged/held in memory at once.

        out_path: if given, save there instead of assets_dir()/asset_name() and skip
        add_to_asset_list -- for callers (e.g. the density sweep) that manage their own
        output location/bookkeeping. Default None preserves the original
        assets_dir()-based behavior."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ply_paths = sorted(Path(ply_dir).glob(pattern))
        if not ply_paths:
            raise ValueError(f"No ply files found in {ply_dir}")

        forward = torch.as_tensor(camera_forward, dtype=torch.float32, device=device)#tensor the direction of the cam
        forward = forward / forward.norm()#normalise for GFX work
        #set directions on screen z up x right
        up = torch.as_tensor(camera_up, dtype=torch.float32, device=device)
        right = torch.linalg.cross(forward, up)
        right = right / right.norm()
        true_up = torch.linalg.cross(right, forward)

        # pass 1: stream each ply just to find the global XY extent and point count
        #find largest value in cam space x = x y = z
        x_min, x_max = float("inf"), float("-inf")
        y_min, y_max = float("inf"), float("-inf")
        n_points = 0
        for ply_path in ply_paths:
            proj = _ortho_project_ply_to_cam(ply_path, right, true_up, forward, device)
            if proj is None:
                continue
            x_cam, y_cam, _, _ = proj
            if x_cam.numel() == 0:
                continue
            n_points += x_cam.numel()
            x_min = min(x_min, x_cam.min().item()); x_max = max(x_max, x_cam.max().item())
            y_min = min(y_min, y_cam.min().item()); y_max = max(y_max, y_cam.max().item())
        if x_min == float("inf"):
            raise ValueError(f"No points found in any ply under {ply_dir}")
        #bounding box size in world units
        x_range, y_range = x_max - x_min, y_max - y_min
        #add margins
        x_min -= margin_frac * x_range; x_max += margin_frac * x_range
        y_min -= margin_frac * y_range; y_max += margin_frac * y_range
        x_range, y_range = x_max - x_min, y_max - y_min

        # scale that fits the (padded) bounding box into width x height, preserving
        # aspect ratio -- same as before, this is the floor.
        fit_scale = min(width / x_range, height / y_range)
        # scale implied by the point cloud's own density -- average spacing between
        # points, assuming a roughly uniform 2D scatter across the bounded area, so
        # each point gets ~1 output pixel to itself instead of several points
        # colliding onto the same pixel.
        avg_spacing = (x_range * y_range / n_points) ** 0.5
        native_scale = 1.0 / avg_spacing

        # never shrink below the requested floor, but grow past it if the data is
        # denser than that -- expand the canvas rather than losing detail.
        scale = max(fit_scale, native_scale)
        out_width = max(width, round(x_range * scale))
        out_height = max(height, round(y_range * scale))
        cx, cy = out_width / 2, out_height / 2
        x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2

        #canvas setup
        background = torch.tensor(background_color, dtype=torch.float32, device=device) / 255
        canvas = background.tile((out_height, out_width, 1)).clone()
        zbuffer = torch.full((out_height, out_width), float("inf"), device=device)
        flat_canvas = canvas.view(-1, 3)
        flat_zbuffer = zbuffer.view(-1)

        # pass 2: stream each ply again, this time painting straight into the shared
        # canvas -- one point, one pixel.
        for ply_path in ply_paths:
            proj = _ortho_project_ply_to_cam(ply_path, right, true_up, forward, device)
            if proj is None:
                continue
            x_cam, y_cam, z_cam, c = proj
            if x_cam.numel() == 0:
                continue

            u = scale * (x_cam - x_center) + cx
            v = -scale * (y_cam - y_center) + cy

            u_i = u.round().long()
            v_i = v.round().long()

            in_bounds = (u_i >= 0) & (u_i < out_width) & (v_i >= 0) & (v_i < out_height)
            u_i, v_i, z_i, c_i = u_i[in_bounds], v_i[in_bounds], z_cam[in_bounds], c[in_bounds]

            flat_idx = v_i * out_width + u_i
            # closer to the (downward-looking) camera than what's currently buffered -> paint over
            in_front = z_i < flat_zbuffer[flat_idx]
            flat_idx, z_i, c_i = flat_idx[in_front], z_i[in_front], c_i[in_front]

            # z-buffer: draw far-to-near within this ply so nearer points win where they collide
            order = torch.argsort(-z_i)
            flat_idx, c_i = flat_idx[order], c_i[order]

            flat_zbuffer[flat_idx] = z_i[order]
            flat_canvas[flat_idx] = c_i

        img = (canvas.clamp(0, 1) * 255).byte().cpu().numpy()
        if out_path is None:
            outpath = assets_dir() / f"{asset_name()}_BEV_render_{model}.png"
        else:
            outpath = Path(out_path)
            outpath.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(outpath)

        if out_path is None:
            add_to_asset_list({
                f"BEV_render_{model}": to_report_path(outpath),
            })
        return 