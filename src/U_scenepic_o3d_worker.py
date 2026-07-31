"""Standalone worker for view_ply_sequence_with_scenepic (see U_rendering.py).
bakes  HTML that's loaded and watched
Runs inside the separate open3d/scenepic conda env via `conda run`, so it must not
import anything from the pipeline (A_Config, C_CSV_report) -- those assume the
Msc2 env and aren't guaranteed to exist here. Just writes the HTML viewer file;
the caller (in its own env/process) handles add_to_report and opening the browser.
"""
import argparse
from pathlib import Path

import numpy as np


def view_ply_sequence_with_scenepic(ply_dir, output_html, point_size=0.01, pattern="*.ply"):
    import open3d as o3d
    import scenepic as sp
    ply_paths = sorted(Path(ply_dir).glob(pattern))
    if not ply_paths:
        raise ValueError(f"No ply files found in {ply_dir}")

    scene = sp.Scene()
    canvas = scene.create_canvas_3d(width=1200, height=800)
    #centre is poosisiotn, look at is look towrds this point in tspace, up dir is where is up.
    canvas.camera = sp.Camera(center=[0, 1, 0], look_at=[0, 1, -1], up_dir=[0, 1, 0],
                               aspect_ratio=1200 / 800)

    for ply_path in ply_paths:
        pcd = o3d.io.read_point_cloud(str(ply_path))
        points = np.asarray(pcd.points, dtype=np.float32)

        if points.shape[0] == 0:
            continue

        colors = None
        if pcd.has_colors():
            colors = np.asarray(pcd.colors, dtype=np.float32)

        cloud = scene.create_mesh(f"cloud_{ply_path.stem}")
        cloud.add_sphere(color=sp.Colors.White, transform=sp.Transforms.scale(point_size))

        if colors is not None:
            cloud.enable_instancing(positions=points, colors=colors)
        else:
            cloud.enable_instancing(positions=points)

        frame = canvas.create_frame()
        frame.add_mesh(cloud)

    scene.save_as_html(str(output_html), title=Path(output_html).stem)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply_dir", required=True)
    parser.add_argument("--output_html", required=True)
    parser.add_argument("--point_size", type=float, default=0.01)
    parser.add_argument("--pattern", default="*.ply")
    args = parser.parse_args()

    view_ply_sequence_with_scenepic(args.ply_dir, args.output_html, args.point_size, args.pattern)
