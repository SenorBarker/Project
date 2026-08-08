"""
Animate the camera's (and optionally the subject's) position over BEV_tile_render_PM's
static top-down still.

BEV_tile_render_PM (P_projection_mapping.py) returns the exact orthographic camera
(new_view, ortho_params) its canvas was built with, plus subject_positions as a
free by-product. This module projects the per-frame camera centre + heading --
and, optionally, subject_positions -- through that same orthographic camera, so
the trace lands exactly on that backplate, then rasterises it as an n_frames-long
PNG sequence: a growing trail, a dot at the current position, and (camera trace
only) a heading frustum wedge showing which way the camera was pointing.

Usage (see BA_BEV_DESIGN.ipynb):
    BEV, new_view, ortho_params = BEV_tile_render_PM(recon, extra_indices, ...)
    BEV_trace_overlay(recon, extra_indices, new_view, ortho_params,
                       subject_positions=BEV, n_frames=90)
"""

import math

import numpy as np
import torch

from A_Config import assets_dir, asset_name, to_report_path
from C_CSV_report import add_to_report
from P_projection_mapping import reproject, DEVICE
from Rendering.R_map_animator import interp_pos, hex_to_rgb


def _time_smoothed(frames, positions, fps, window_sec):
    """Centered running average of positions over a window_sec-wide real-time
    window, using each frame's own frame_number/fps timestamp rather than a
    fixed sample count -- stays correct across irregular gaps (missing/
    redacted frames don't skew the window width the way a fixed N-sample
    window would). window_sec<=0 returns positions unchanged."""
    if window_sec <= 0:
        return positions
    t = np.asarray(frames, dtype=float) / fps
    positions = np.asarray(positions, dtype=float)
    half = window_sec / 2
    out = np.empty_like(positions)
    for i in range(len(t)):
        in_window = np.abs(t - t[i]) <= half
        out[i] = positions[in_window].mean(axis=0)
    return out


def BEV_trace_overlay(
    recon, extra_indices, new_view, ortho_params,
    subject_positions=None,
    n_frames=60,
    bev_png_path=None,
    max_width=1920, max_height=1080,
    camera_trail_color="#3498db", subject_trail_color="#e74c3c",
    trail_width=None, trail_opacity=0.85, dot_radius=None, dot_opacity=1.0,
    frustum_length=None, frustum_half_angle_deg=22.0,
    supersample=2,
    fps=25, smooth_window_sec=1.0,
):
    """n_frames is the only thing controlling frame count -- unlike R_map_animator,
    it is never derived from duration*fps.

    smooth_window_sec: subject_positions is smoothed with a centered running
    average over this many seconds of real video time (using each frame's own
    frame_number/fps timestamp, not a fixed sample count) before being
    reprojected -- 1.0s is walking pace, tight enough to preserve real motion
    but wide enough to average out per-frame mask/depth noise (confirmed
    against real data: a 1s/~8-sample window rides through the noise without
    smearing genuine trajectory shape -- see the confidence-floor/junk-mask
    investigation). Set to 0 to disable and use raw positions."""
    from PIL import Image, ImageDraw

    if bev_png_path is None:
        bev_png_path = assets_dir() / f"{asset_name()}_BEV_tile_render_PM.png"
    base_img = Image.open(bev_png_path).convert("RGBA")
    w0, h0 = base_img.size

    # BEV_tile_render_PM's canvas grows with point-cloud density and can be several
    # megapixels (e.g. 2106x4276) -- far more resolution than an animated overlay
    # needs. Downscaling once here (not per frame) cuts every per-frame image op
    # roughly with the pixel count: frame.save() alone benchmarked ~0.49s/frame at
    # native 9MP vs ~0.03s/frame fit to 1080p.
    render_scale = min(1.0, max_width / w0, max_height / h0)
    if render_scale < 1.0:
        base_img = base_img.resize((round(w0 * render_scale), round(h0 * render_scale)), Image.LANCZOS)
    w, h = base_img.size

    _trail_width = trail_width if trail_width is not None else min(w, h) * 0.006
    _dot_radius  = dot_radius  if dot_radius  is not None else min(w, h) * 0.010
    _frustum_len = frustum_length if frustum_length is not None else min(w, h) * 0.035

    # -- camera trace: position + heading, both projected through the BEV's own
    # orthographic camera (new_view/ortho_params), so they land on this exact canvas --
    cam_dict = recon.cam_pos_dict()
    cam_frames = sorted(fn for fn in extra_indices if fn in cam_dict)
    if len(cam_frames) < 2:
        raise ValueError("Need at least 2 camera frames in extra_indices to animate a trace.")

    cam_pos = torch.as_tensor(np.stack([cam_dict[fn] for fn in cam_frames]),
                               dtype=torch.float32, device=DEVICE)

    extrinsic = recon.preds["extrinsic"]  # (N, 3, 4), world-to-camera
    rows = [recon.frame_to_row[fn] for fn in cam_frames]
    R = extrinsic[rows, :, :3]  # (K, 3, 3)
    fwd_local = torch.tensor([0.0, 0.0, 1.0], device=DEVICE)
    # same forward_world = R^T @ [0,0,1] convention as H_Geolocation_GMAPS.py:725
    heading_world = torch.einsum("kij,j->ki", R.transpose(1, 2), fwd_local)
    cam_look = cam_pos + heading_world

    # reproject() gives pixel coords in the native BEV canvas -- scale by
    # render_scale so they land correctly on the (possibly downscaled) base_img.
    u, v, _, _ = reproject(cam_pos, None, new_view, ortho_params=ortho_params)
    cam_pts = [(x * render_scale, y * render_scale) for x, y in zip(u.cpu().tolist(), v.cpu().tolist())]
    uh, vh, _, _ = reproject(cam_look, None, new_view, ortho_params=ortho_params)
    cam_heading_pts = [(x * render_scale, y * render_scale) for x, y in zip(uh.cpu().tolist(), vh.cpu().tolist())]

    traces = [{
        "frames": cam_frames, "px_points": cam_pts, "heading_pts": cam_heading_pts,
        "trail_color": camera_trail_color, "label": "camera",
    }]

    if subject_positions is not None:
        sub_frames = sorted(fn for fn in extra_indices if fn in subject_positions)
        # drop NaN/inf positions here -- reproject()/interp_pos() have no NaN guard,
        # so a bad point silently spreads into every frame whose t_frac interpolates
        # across it (see P_trace_overlayer teleport investigation)
        sub_frames = [fn for fn in sub_frames if np.isfinite(np.asarray(subject_positions[fn])).all()]
        if len(sub_frames) >= 2:
            sub_pos_raw = np.stack([subject_positions[fn] for fn in sub_frames])
            sub_pos_smoothed = _time_smoothed(sub_frames, sub_pos_raw, fps, smooth_window_sec)
            sub_pos = torch.as_tensor(sub_pos_smoothed, dtype=torch.float32, device=DEVICE)

            us, vs, _, _ = reproject(sub_pos, None, new_view, ortho_params=ortho_params)
            sub_pts = [(x * render_scale, y * render_scale) for x, y in zip(us.cpu().tolist(), vs.cpu().tolist())]
            traces.append({
                "frames": sub_frames, "px_points": sub_pts, "heading_pts": None,
                "trail_color": subject_trail_color, "label": "subject",
            })
        else:
            print(f"Skipping subject trace -- only {len(sub_frames)} frame(s) overlap extra_indices.")

    # -- shared time axis: frame number is time-proportional for evenly-sampled video,
    # same hold-before-start/hold-after-end padding as R_map_animator.py:365-374 --
    t0_global = min(tr["frames"][0] for tr in traces)
    t1_global = max(tr["frames"][-1] for tr in traces)
    if t1_global == t0_global:
        raise ValueError("All camera/subject frame numbers are identical -- cannot animate.")

    for tr in traces:
        raw_fracs = [(fn - t0_global) / (t1_global - t0_global) for fn in tr["frames"]]
        time_fracs = raw_fracs[:]
        interp_pts = tr["px_points"][:]
        heading_pts = tr["heading_pts"][:] if tr["heading_pts"] is not None else None
        if raw_fracs[0] > 0:
            time_fracs = [0.0] + time_fracs
            interp_pts = [tr["px_points"][0]] + interp_pts
            if heading_pts is not None:
                heading_pts = [tr["heading_pts"][0]] + heading_pts
        if raw_fracs[-1] < 1:
            time_fracs = time_fracs + [1.0]
            interp_pts = interp_pts + [tr["px_points"][-1]]
            if heading_pts is not None:
                heading_pts = heading_pts + [tr["heading_pts"][-1]]
        tr["time_fracs"] = time_fracs
        tr["interp_pts"] = interp_pts
        tr["heading_pts"] = heading_pts

    # -- render n_frames frames --
    # supersample>1 draws at S x resolution then downsamples with LANCZOS, which is
    # how R_map_animator gets anti-aliased lines/dots out of PIL's non-AA ImageDraw --
    # cheap for its 640x640 map tiles, but the BEV canvas here can be several
    # megapixels, so the same trick balloons the per-frame composite+resize cost by
    # S^2 (benchmarked ~3.7s/frame at S=4 vs ~0.6s/frame at S=1 on a 2106x4276 canvas).
    # Default S=1 skips the resize round-trip entirely; raise it for a smoother final
    # render if the extra time is worth it.
    n_frames = max(2, int(n_frames))
    S = max(1, int(supersample))
    base = base_img.resize((w * S, h * S), Image.LANCZOS) if S > 1 else base_img

    frames_dir = assets_dir() / "bev_trace_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for fi in range(n_frames):
        t_frac = fi / (n_frames - 1)
        frame = base.copy()
        overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for tr in traces:
            rgb = hex_to_rgb(tr["trail_color"])
            t_alpha = int(trail_opacity * 255)
            d_alpha = int(dot_opacity * 255)
            tw = max(1, int(_trail_width * S))
            dr = max(1, int(_dot_radius * S))

            trail_pts = []
            for i, tf in enumerate(tr["time_fracs"]):
                if tf <= t_frac:
                    trail_pts.append(tr["interp_pts"][i])
                else:
                    trail_pts.append(interp_pos(t_frac, tr["time_fracs"], tr["interp_pts"]))
                    break
            trail_pts_s = [(x * S, y * S) for x, y in trail_pts]
            if len(trail_pts_s) >= 2:
                draw.line(trail_pts_s, fill=(*rgb, t_alpha), width=tw, joint="curve")

            dx0, dy0 = interp_pos(t_frac, tr["time_fracs"], tr["interp_pts"])
            dx, dy = dx0 * S, dy0 * S
            virtual_frame = t0_global + t_frac * (t1_global - t0_global)
            print(f"frame {fi:04d} t_frac={t_frac:.4f} [{tr['label']}] "
                  f"virtual_frame={virtual_frame:.1f} pos_px=({dx0:.1f}, {dy0:.1f})")
            draw.ellipse([dx - dr, dy - dr, dx + dr, dy + dr], fill=(*rgb, d_alpha))

            if tr["heading_pts"] is not None:
                lx0, ly0 = interp_pos(t_frac, tr["time_fracs"], tr["heading_pts"])
                heading_angle = math.atan2(ly0 - dy0, lx0 - dx0)
                half = math.radians(frustum_half_angle_deg)
                edge1 = (dx + _frustum_len * S * math.cos(heading_angle - half),
                         dy + _frustum_len * S * math.sin(heading_angle - half))
                edge2 = (dx + _frustum_len * S * math.cos(heading_angle + half),
                         dy + _frustum_len * S * math.sin(heading_angle + half))
                draw.polygon([(dx, dy), edge1, edge2], fill=(*rgb, d_alpha))

        frame = Image.alpha_composite(frame, overlay)
        if S > 1:
            frame = frame.resize((w, h), Image.LANCZOS)
        frame = frame.convert("RGB")
        frame.save(frames_dir / f"frame_{fi:04d}.png")

    print(f"Written: {frames_dir}  ({n_frames} frame PNGs)")
    add_to_report({"bev_trace_frames_dir": to_report_path(frames_dir)})
    return frames_dir
