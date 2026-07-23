
import numpy as np

import os
import subprocess
import webbrowser
from pathlib import Path

from A_Config import assets_dir, report_path, to_report_path, asset_name
from C_CSV_report import add_to_report

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

    add_to_report({"4D_recon_link": to_report_path(output_html)})
    webbrowser.open(str(output_html))
    return




#automated numpy renderer
from PIL import Image
from plyfile import PlyData
from C_CSV_report import add_to_report


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
    
    add_to_report({
    f"point_cloud_render_{model}_mp4": mp4_name,
    f"point_cloud_render_{model}_gif": gif_name,
    f"point_cloud_render_{model}_thumb": to_report_path(thumb_path),
    f"4D_recon_clicker"                      : to_report_path(thumb_path)}#for the 4d recon
    )
    return 