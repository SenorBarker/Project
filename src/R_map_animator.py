"""
map_animator.py — fetch a Google Static Map and generate an animated SVG route.

Coords format: list of (timestamp, lat, lon)
  timestamp: ISO string "2026-06-01T10:00:00", datetime object, or unix float

Single trace:
    main(coords=my_coords, output="run.svg")

Multiple traces:
    main(traces=[
        {"coords": camera_coords, "trail_color": "#3498db"},
        {"coords": cat_coords,    "trail_color": "#e74c3c"},
    ], output="run.svg")

Output: a self-contained SVG with the map embedded as base64, a growing trail,
        and a dot that travels at real-time-proportional speed.
"""

import base64
import io
import math
import os
import urllib.request
from datetime import datetime
from pathlib import Path

import json

from C_CSV_report import add_to_report
from A_Config import report_path, asset_name, assets_dir, case_dir, to_report_path
from B_video_processing import video_fps

# ── Helpers ───────────────────────────────────────────────────────────────────

TILE = 256  # Web Mercator tile size (pixels at zoom 0)


def parse_time(t) -> float:
    """Return unix timestamp from ISO string, datetime, or float."""
    if isinstance(t, (int, float)):
        return float(t)
    if isinstance(t, datetime):
        return t.timestamp()
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return datetime.strptime(t, fmt).timestamp()
        except ValueError:
            pass
    raise ValueError(f"Cannot parse timestamp: {t!r}")


def world_xy(lat: float, lon: float):
    """Lat/lon → Web Mercator world coordinates (pixels at zoom 0)."""
    x = (lon + 180) / 360 * TILE
    s = math.sin(math.radians(lat))
    y = (0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * TILE
    return x, y


def to_image_px(lat, lon, center_lat, center_lon, zoom, img_w, img_h):
    """Project lat/lon to pixel coordinates within the map image."""
    scale = 2 ** zoom
    cx, cy = world_xy(center_lat, center_lon)
    px_x, px_y = world_xy(lat, lon)
    x = (px_x - cx) * scale + img_w / 2
    y = (px_y - cy) * scale + img_h / 2
    return x, y


def auto_zoom(coords, map_size):
    """
    Find zoom level and centre so the largest bounding-box dimension
    (N-S or E-W, in Web Mercator pixels) fills 90% of the square map.
    coords may be from multiple traces combined.
    """
    lats = [c[1] for c in coords]
    lons = [c[2] for c in coords]
    clat = (min(lats) + max(lats)) / 2
    clon = (min(lons) + max(lons)) / 2

    x_min, y_min = world_xy(min(lats), min(lons))
    x_max, y_max = world_xy(max(lats), max(lons))
    world_span = max(x_max - x_min, abs(y_max - y_min))

    if world_span == 0:
        zoom = 16
    else:
        zoom = int(math.log2(map_size * 0.90 / world_span))
        zoom = max(1, min(zoom, 20))

    return zoom, clat, clon


def fetch_map(center_lat, center_lon, zoom, map_size, scale, maptype, api_key):
    """Download map PNG from Google Static Maps API; return raw bytes."""
    url = (
        f"https://maps.googleapis.com/maps/api/staticmap"
        f"?center={center_lat},{center_lon}"
        f"&zoom={zoom}"
        f"&size={map_size}x{map_size}"
        f"&scale={scale}"
        f"&maptype={maptype}"
        f"&key={api_key}"
    )
    print(f"  Fetching: {url[:80]}...")
    with urllib.request.urlopen(url) as r:
        return r.read()


def path_d(points):
    """SVG path string from a list of (x, y) tuples."""
    pts = [f"M {points[0][0]:.3f},{points[0][1]:.3f}"]
    pts += [f"L {x:.3f},{y:.3f}" for x, y in points[1:]]
    return " ".join(pts)


def cumulative_lengths(points):
    """Cumulative Euclidean arc lengths along a polyline."""
    lengths = [0.0]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        lengths.append(lengths[-1] + math.hypot(x1 - x0, y1 - y0))
    return lengths


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def interp_pos(t_frac, time_fracs, px_points):
    """Interpolate pixel position along path at time fraction t_frac."""
    if t_frac <= time_fracs[0]:
        return px_points[0]
    if t_frac >= time_fracs[-1]:
        return px_points[-1]
    for i in range(len(time_fracs) - 1):
        if time_fracs[i] <= t_frac <= time_fracs[i + 1]:
            seg = (t_frac - time_fracs[i]) / (time_fracs[i + 1] - time_fracs[i])
            x = px_points[i][0] + seg * (px_points[i+1][0] - px_points[i][0])
            y = px_points[i][1] + seg * (px_points[i+1][1] - px_points[i][1])
            return x, y
    return px_points[-1]


def render_frames(png_bytes, trace_data, map_size, duration, fps, supersample=4):
    """Render animation frames as PIL Images using supersampling for smooth edges.

    trace_data: list of dicts, each with keys:
        px_points, time_fracs, trail_color, trail_width, trail_opacity,
        dot_radius, dot_opacity
    """
    from PIL import Image, ImageDraw

    S = supersample
    base = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    base = base.resize((map_size * S, map_size * S), Image.LANCZOS)

    n_frames = max(2, int(duration * fps))
    frames   = []

    # Pre-scale interp_pts for each trace
    for tr in trace_data:
        tr["pts_s"] = [(x * S, y * S) for x, y in tr["interp_pts"]]

    for fi in range(n_frames):
        t_frac  = fi / (n_frames - 1)
        frame   = base.copy()
        overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        draw    = ImageDraw.Draw(overlay)

        for tr in trace_data:
            rgb     = hex_to_rgb(tr["trail_color"])
            t_alpha = int(tr["trail_opacity"] * 255)
            d_alpha = int(tr["dot_opacity"]   * 255)
            tw      = max(1, int(tr["trail_width"] * S))
            dr      = max(1, int(tr["dot_radius"]  * S))

            trail_pts = []
            for i, tf in enumerate(tr["time_fracs"]):
                if tf <= t_frac:
                    trail_pts.append(tr["pts_s"][i])
                else:
                    rx, ry = interp_pos(t_frac, tr["time_fracs"], tr["interp_pts"])
                    trail_pts.append((rx * S, ry * S))
                    break

            if len(trail_pts) >= 2:
                draw.line(trail_pts, fill=(*rgb, t_alpha), width=tw, joint="curve")

            dx, dy = interp_pos(t_frac, tr["time_fracs"], tr["interp_pts"])
            dx, dy = dx * S, dy * S
            draw.ellipse([dx - dr, dy - dr, dx + dr, dy + dr],
                         fill=(*rgb, d_alpha))

        frame = Image.alpha_composite(frame, overlay)
        frame = frame.resize((map_size, map_size), Image.LANCZOS).convert("RGB")
        frames.append(frame)

    return frames


def export_gif(frames, output, fps):
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=False,
    )
    print(f"Written: {output}  ({Path(output).stat().st_size // 1024} KB GIF)")


def export_mp4(frames, output, fps):
    import cv2, numpy as np
    h, w = frames[0].size[1], frames[0].size[0]
    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for img in frames:
        writer.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"Written: {output}  ({Path(output).stat().st_size // 1024} KB MP4)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main(
    coords        = None,
    traces        = None,
    api_key       = None,
    map_size      = 640,
    map_scale     = 2,
    map_type      = "roadmap",
    zoom          = None,
    trail_color   = "#e74c3c",
    trail_width   = None,       # default: map_size * 0.006
    trail_opacity = 0.85,
    dot_radius    = None,       # default: map_size * 0.010
    dot_opacity   = 1.0,
    gif           = None,
    mp4           = None,
    fps           = None,
    manual        =False


):
    """
    Generate an animated SVG map with one or more traces.

    Single trace:
        main(coords=my_coords, api_key="...", output="run.svg")

    Multiple traces:
        main(traces=[
            {"coords": camera_coords, "trail_color": "#3498db"},
            {"coords": cat_coords,    "trail_color": "#e74c3c"},
        ], api_key="...", output="run.svg")

    Per-trace style keys: trail_color, trail_width, trail_opacity,
                          dot_radius, dot_opacity
    Omit any key to inherit the top-level default.
    """
    _map_size      = map_size
    _map_scale     = map_scale
    _map_type      = map_type
    _zoom          = zoom
    _trail_color   = trail_color
    _trail_width   = trail_width   if trail_width  is not None else map_size * 0.006
    _trail_opacity = trail_opacity
    _dot_radius    = dot_radius    if dot_radius   is not None else map_size * 0.010
    _dot_opacity   = dot_opacity
    _output        = assets_dir() / f"{asset_name()}_map.svg"

    if manual ==False:
        # Duration comes from the Producer's own MAP beat, not a caller-supplied
        # value -- it's a fixed editorial decision for the current case, read at
        # call time (not eagerly, since the Producer may not have written this
        # beat yet when the notebook's config cell runs).
        paper_edit_path = case_dir() / "012_agent_p_output" / f"{case_dir().name}_paper_edit.json"
        paper_edit = json.loads(paper_edit_path.read_text(encoding="utf-8"))
        map_beat = next(b for b in paper_edit["beats"] if b["archetype"] == "MAP")
        _duration = map_beat["duration_seconds"]

    else:
        _duration = 10

    # fps is never an independent choice -- these frames get muxed into the
    # case's actual video later (see W_video_editor.assemble_paper_edit), so
    # rendering at any other rate just forces a pointless resample there.
    # Derive it from the same single source video the rest of the case uses,
    # unless the caller explicitly overrides it (e.g. for a standalone
    # gif/mp4 export that never touches the main footage).
    _fps = fps
    if _fps is None:
        source_video = next((case_dir() / "010_source").glob("*.mp4"))
        _fps = video_fps(source_video)

    _api_key = api_key or os.environ.get("GOOGLE_MAPS_KEY", "")
    if not _api_key:
        raise ValueError("Provide api_key or set GOOGLE_MAPS_KEY env var.")

    # Normalise: traces= wins; coords= is single-trace shorthand
    if traces is None:
        if coords is None:
            raise ValueError("Provide coords= or traces=.")
        traces = [{"coords": coords}]
    elif coords is not None:
        raise ValueError("Pass either coords= or traces=, not both.")

    # Resolve each trace: parse timestamps, apply style defaults
    resolved = []
    for tr in traces:
        entries = [(parse_time(t), lat, lon) for t, lat, lon in tr["coords"]]
        entries.sort(key=lambda e: e[0])
        resolved.append({
            "entries":       entries,
            "trail_color":   tr.get("trail_color",   _trail_color),
            "trail_width":   tr.get("trail_width",   _trail_width),
            "trail_opacity": tr.get("trail_opacity", _trail_opacity),
            "dot_radius":    tr.get("dot_radius",    _dot_radius),
            "dot_opacity":   tr.get("dot_opacity",   _dot_opacity),
        })

    # Zoom / centre fitted across ALL traces
    all_entries = [e for tr in resolved for e in tr["entries"]]
    if _zoom is None:
        _zoom, clat, clon = auto_zoom(all_entries, _map_size)
        print(f"Auto-zoom: {_zoom}, centre ({clat:.5f}, {clon:.5f})")
    else:
        lats = [e[1] for e in all_entries]
        lons = [e[2] for e in all_entries]
        clat = (min(lats) + max(lats)) / 2
        clon = (min(lons) + max(lons)) / 2

    # Fetch map
    print("Fetching map...")
    png_bytes = fetch_map(clat, clon, _zoom, _map_size, _map_scale, _map_type, _api_key)
    img_b64   = base64.b64encode(png_bytes).decode()

    # Global time axis: first frame across all traces → 0, last → 1.
    # Each trace's keyTimes is padded to span exactly [0, 1] as SVG requires,
    # holding the trace hidden before it starts and fully drawn after it ends.
    t0_global = min(e[0] for tr in resolved for e in tr["entries"])
    t1_global = max(e[0] for tr in resolved for e in tr["entries"])
    if t1_global == t0_global:
        raise ValueError("All timestamps across all traces are identical — cannot animate.")

    trace_data = []
    for tr in resolved:
        entries   = tr["entries"]
        px_points = [to_image_px(lat, lon, clat, clon, _zoom, _map_size, _map_size)
                     for _, lat, lon in entries]
        cum_lens  = cumulative_lengths(px_points)
        total_len = cum_lens[-1]
        len_fracs = [l / total_len if total_len > 0 else 0.0 for l in cum_lens]

        raw_fracs  = [(e[0] - t0_global) / (t1_global - t0_global) for e in entries]
        raw_dashes = [total_len - l for l in cum_lens]

        time_fracs = raw_fracs[:]
        dash_vals  = raw_dashes[:]
        kp_vals    = len_fracs[:]
        interp_pts = px_points[:]
        if raw_fracs[0] > 0:          # hold hidden before trace starts
            time_fracs = [0.0]       + time_fracs
            dash_vals  = [total_len] + dash_vals
            kp_vals    = [0.0]       + kp_vals
            interp_pts = [px_points[0]] + interp_pts
        if raw_fracs[-1] < 1:         # hold fully drawn after trace ends
            time_fracs = time_fracs + [1.0]
            dash_vals  = dash_vals  + [0.0]
            kp_vals    = kp_vals    + [1.0]
            interp_pts = interp_pts + [px_points[-1]]

        trace_data.append({**tr,
                           "px_points":  px_points,
                           "interp_pts": interp_pts,
                           "time_fracs": time_fracs,
                           "dash_vals":  dash_vals,
                           "kp_vals":    kp_vals,
                           "cum_lens":   cum_lens,
                           "total_len":  total_len,
                           "len_fracs":  len_fracs})

    def fmt(vals, decimals=5):
        return ";".join(f"{v:.{decimals}f}" for v in vals)

    dur_str = f"{_duration}s"

    # Build SVG — one path+dot per trace
    trace_svg = ""
    for i, tr in enumerate(trace_data):
        route_id   = f"route_{i}"
        route_d    = path_d(tr["px_points"])
        dash_vals  = fmt(tr["dash_vals"], decimals=3)
        key_times  = fmt(tr["time_fracs"])
        key_points = fmt(tr["kp_vals"])
        trace_svg += f"""
  <!-- Trace {i} trail -->
  <path id="{route_id}"
        d="{route_d}"
        fill="none"
        stroke="{tr['trail_color']}"
        stroke-width="{tr['trail_width']}"
        stroke-linecap="round"
        stroke-linejoin="round"
        opacity="{tr['trail_opacity']}"
        stroke-dasharray="{tr['total_len']:.3f}"
        stroke-dashoffset="{tr['total_len']:.3f}">
    <animate attributeName="stroke-dashoffset"
             values="{dash_vals}"
             keyTimes="{key_times}"
             dur="{dur_str}"
             calcMode="linear"
             fill="freeze"/>
  </path>

  <!-- Trace {i} dot -->
  <circle r="{tr['dot_radius']}" fill="{tr['trail_color']}" opacity="{tr['dot_opacity']}">
    <animateMotion dur="{dur_str}"
                   calcMode="linear"
                   keyPoints="{key_points}"
                   keyTimes="{key_times}"
                   fill="freeze">
      <mpath xlink:href="#{route_id}"/>
    </animateMotion>
  </circle>
"""

    svg = f"""\
<?xml version="1.0" encoding="UTF-8"?>
<!-- map_animator v01 -->
<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     viewBox="0 0 {_map_size} {_map_size}"
     width="{_map_size}" height="{_map_size}">

  <!-- Map raster (scale={_map_scale}x embedded base64, square) -->
  <image x="0" y="0" width="{_map_size}" height="{_map_size}"
         preserveAspectRatio="xMidYMid slice"
         href="data:image/png;base64,{img_b64}"/>
{trace_svg}
</svg>
"""

    Path(_output).write_text(svg, encoding="utf-8")
    print(f"Written: {_output}  ({len(png_bytes)//1024} KB map, {len(svg)//1024} KB SVG)")

    # Frame-by-frame render always happens -- the image sequence is the
    # standard hand-off to the editor (frame count = duration * fps, where
    # duration is the Producer's requested duration_seconds for this beat),
    # gif/mp4 are just optional extra exports of the same frames.
    print("Rendering frames...")
    frames = render_frames(png_bytes, trace_data, _map_size, _duration, _fps)

    frames_dir = assets_dir() / "map_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for fi, img in enumerate(frames):
        img.save(frames_dir / f"frame_{fi:04d}.png")
    print(f"Written: {frames_dir}  ({len(frames)} frame PNGs)")

    if gif:
        gif_path = assets_dir() / f"{asset_name()}_map.gif"
        export_gif(frames, gif_path, _fps)
    if mp4:
        mp4_path = assets_dir() / f"{asset_name()}_map.mp4"
        export_mp4(frames, mp4_path, _fps)

    for_report = {"map_svg": to_report_path(_output), "map_frames_dir": to_report_path(frames_dir)}
    if gif:
        for_report["map_gif"] = to_report_path(gif_path)
    if mp4:
        for_report["map_mp4"] = to_report_path(mp4_path)
    add_to_report(for_report)


if __name__ == "__main__":
    main()
