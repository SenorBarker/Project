
import glob
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn.functional as F
from G_transforms_alignments import _rc_rotation_matrix, apply_cam0_frame, unproject, ortho_charts
from A_Config import FRAME_NAME_FMT, frames_for_recon_dir

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#------------------loaders-------------------
'''Loaders and transforms to turn any chosen 3d/4d recon files into the same format for further processing'''
'''all return N,3 shaped outputs as np arrays'''
def load_reality_scan_trace(csv_path):
    """Extract camera positions from a RealityScan registration CSV,
    expressed in cam0's frame of reference.

    RealityScan's own position axes (x, y, alt) don't match the up/forward
    convention implied by its yaw/pitch/roll (alt is up, y is forward), so
    positions are reordered to (x, alt, y) before applying cam0's inverse
    rotation. This makes RealityScan traces directly comparable to
    MegaSaM/CUT3R output, which already define cam0 as the origin.
    """
    df = pd.read_csv(csv_path)
    xyz = df[["x", "alt", "y"]].to_numpy()
       
    chart = ortho_charts(xyz, "Reality Scan",dataB=None, title = "Reality Scan")

    yaw0, pitch0, roll0 = df.loc[0, ["yaw", "pitch", "roll"]]
    #print(yaw0, pitch0, roll0)
    #22/06 check this - is rc 90 level?? 0 should be level
    pitch0 = pitch0 -90
    R0 = _rc_rotation_matrix(yaw0, pitch0, roll0)
    positions = apply_cam0_frame(xyz, R0, xyz[0])
    frame_nums = df["#name"].str.split(".").str[0].astype(int)
    world_pos_dict = dict(zip(frame_nums, positions))
    #positions are just the positions..while the dict is keyed by frame #
    return positions, world_pos_dict


def load_cut3r_trace(camera_dir):
    """Extract camera positions from a directory of CUT3R per-frame pose .npz files.

    Pre-fix CUT3R output only (files named by sequential loop index, not real frame
    number -- see E_cut3r_recon.py's naming fix). Kept as-is so AA_CAM_POSE_COMPARE_01.ipynb's
    existing calls against already-run, pre-fix CUT3R output keep working. New CUT3R
    runs are frame-number-named -- use load_cut3r_trace_v2 for those."""
    files = sorted(glob.glob(os.path.join(camera_dir, "*.npz")))
    positions = []
    for f in files:
        data = np.load(f)
        positions.append(data["pose"][:3, 3])

    chart = ortho_charts(np.array(positions), "CUT3R",dataB=None, title = "CUT3R")

    return np.array(positions)


def load_cut3r_trace_v2(camera_dir):
    """Extract camera positions from CUT3R's camera/*.npz, frame-number-named at
    the source (post naming-fix, see E_cut3r_recon.py). Returns (positions,
    world_pos_dict) like load_lingbot_map_trace/load_reality_scan_trace -- no
    frames_dir cross-referencing needed, the filename is the frame number."""
    files = sorted(Path(camera_dir).glob("*.npz"))
    world_pos_dict = {int(f.stem): np.load(f)["pose"][:3, 3] for f in files}
    print(world_pos_dict)
    positions = np.array([world_pos_dict[k] for k in sorted(world_pos_dict)])
    print(positions.shape)
    chart = ortho_charts(positions, "CUT3R", dataB=None, title="CUT3R")
    return positions, world_pos_dict


def load_megasam_trace(npz_path, frames_dir=None):
    """Extract camera positions from a MegaSaM {scene}_sgd_cvd_hr.npz output.

    Row-order only -- no frame numbers in the data itself -- so frames_dir is
    needed to build the frame-keyed dict (see positions_to_frame_dict). Defaults
    to A_Config.frames_for_recon_dir() (the currently active case/experiment,
    same zero-arg convention as the rest of A_Config) -- pass frames_dir
    explicitly when comparing against a different/older frame set than whatever
    is currently active via set_case()."""
    if frames_dir is None:
        frames_dir = frames_for_recon_dir()
    data = np.load(npz_path)
    positions = data["cam_c2w"][:, :3, 3]
    chart = ortho_charts(positions, "MegaSAM", dataB=None, title="MegaSAM")
    world_pos_dict = positions_to_frame_dict(positions, frames_dir)
    return positions, world_pos_dict


def load_VGGT_O_trace(npz_path, frames_dir=None):
    """Extract camera positions from a VGGT-O predictions.npz, in cam0's frame of
    reference -- same convention as Reconstruction.cam_pos_dict() (which is the
    other existing way to get frame-keyed VGGT-O positions, if you already have a
    Reconstruction loaded; this is the standalone version for when you don't).
    frames_dir defaults to A_Config.frames_for_recon_dir(), same as load_megasam_trace.

    Frame 0's extrinsic is identity by construction (verified against real output --
    see load_VGGT_trace below for the same check on plain VGGT), so no separate
    cam0-anchoring step is needed; positions are already in that frame."""
    if frames_dir is None:
        frames_dir = frames_for_recon_dir()
    d = np.load(npz_path)
    extrinsic = d["extrinsic"]  # (N, 3, 4), world-to-camera

    rotation_t = np.transpose(extrinsic[:, :3, :3], (0, 2, 1))
    positions = -np.einsum("nij,nj->ni", rotation_t, extrinsic[:, :3, 3])

    chart = ortho_charts(positions, "VGGT_O", dataB=None, title="VGGT_O")
    world_pos_dict = positions_to_frame_dict(positions, frames_dir)
    return positions, world_pos_dict


def load_VGGT_trace(sparse_reconstruction_dir, frames_dir=None):
    """Extract camera positions from plain-VGGT's extrinsic.npy (demo_colmap.py).
    frames_dir defaults to A_Config.frames_for_recon_dir(), same as load_megasam_trace.

    Frame 0 is identity by construction -- verified directly against real output
    (Data/01_walk/034_VGGT_output/.../extrinsic.npy: rotation ~= I, translation ~= 0
    to ~1e-4 float noise), same as VGGT-O and lingbot-map. No cam0-anchoring step
    needed, positions are already in that frame."""
    if frames_dir is None:
        frames_dir = frames_for_recon_dir()
    extrinsic = np.load(Path(sparse_reconstruction_dir) / "extrinsic.npy")  # (N,3,4), w2c
    rotation_t = np.transpose(extrinsic[:, :3, :3], (0, 2, 1))
    positions = -np.einsum("nij,nj->ni", rotation_t, extrinsic[:, :3, 3])
    chart = ortho_charts(positions, "VGGT", dataB=None, title="VGGT")
    world_pos_dict = positions_to_frame_dict(positions, frames_dir)
    return positions, world_pos_dict


def load_lingbot_map_trace(output_dir):
    """Extract camera positions from run_lingbot_map()'s per-frame npz output
    (B_lingbot_map.py). Already frame-number-keyed by filename -- no frames_dir
    cross-referencing needed."""
    files = sorted(Path(output_dir).glob("*.npz"))
    world_pos_dict = {}
    for f in files:
        frame_num = int(f.stem)
        d = np.load(f)
        extrinsic = d["extrinsic_w2c"]  # (3, 4), world-to-camera
        rotation_t = extrinsic[:3, :3].T
        world_pos_dict[frame_num] = -rotation_t @ extrinsic[:3, 3]
    positions = np.array([world_pos_dict[k] for k in sorted(world_pos_dict)])
    chart = ortho_charts(positions, "lingbot-map", dataB=None, title="lingbot-map")
    return positions, world_pos_dict


#load the vggt_O output file - com
def load_VGGT_O_predictions(npz_path):
    d = np.load(npz_path)
    return {
        "depth": torch.from_numpy(d["depth"][..., 0]).float().to(DEVICE),       # (N, 384, 688)
        "depth_conf": torch.from_numpy(d["depth_conf"]).float().to(DEVICE),     # (N, 384, 688)
        "images": torch.from_numpy(d["images"]).float().to(DEVICE),             # (N, 3, 384, 688), 0..1
        "extrinsic": torch.from_numpy(d["extrinsic"]).float().to(DEVICE),       # (N, 3, 4)
        "intrinsic": torch.from_numpy(d["intrinsic"]).float().to(DEVICE),       # (N, 3, 3)
        
        # (N, 384, 688, 3) -- already cam0-space, precomputed at inference time
        # (see B_VGGT_omega._unproject_depth_map_to_point_map); frame 0's
        # extrinsic is identity by construction, so no extra anchoring is needed.
    }

#----------------------------------------------------------

#turn VGGT outputs into a class and does the post recon preocessing neeced for further analysis
@dataclass
class Reconstruction:
    """Bundles everything that describes one VGGT-O reconstruction run:
    the frame image directory, the loaded predictions, and the mapping
    from source frame number to prediction row index."""
    frames_dir:   Path
    preds_path:   Path
    preds:        dict
    frame_to_row: dict

    @classmethod
    def load(cls, frames_dir, preds_path, frame_name_fmt, frame_range=None):
        frames_dir = Path(frames_dir)
        return cls(
            frames_dir   = frames_dir,
            preds_path   = Path(preds_path),
            preds        = load_VGGT_O_predictions(preds_path),
            frame_to_row = build_frame_index(frames_dir, frame_name_fmt, frame_range=frame_range),
        )

    def _cam0_anchor(self):
        """cam0's (rotation, origin) in VGGT-O's raw model space -- the shared
        anchor cam_pos_dict() and VGGT_O_preds_to_ply_export() both use, so a
        camera trace and its point clouds land in the same frame."""
        extrinsic = self.preds["extrinsic"][0].cpu().numpy()
        R0 = extrinsic[:3, :3].T
        origin = -R0 @ extrinsic[:3, 3]
        return R0, origin

    def cam_pos_dict(self):
        """Camera centres in VGGT-O model space (cam0 frame), keyed by source frame number."""
        extrinsic  = self.preds["extrinsic"].cpu().numpy()
        rotation_t = np.transpose(extrinsic[:, :3, :3], (0, 2, 1))
        cam_pos    = -np.einsum("nij,nj->ni", rotation_t, extrinsic[:, :3, 3])
        R0, origin = self._cam0_anchor()
        cam_pos    = apply_cam0_frame(cam_pos, R0, origin)
        return {frame: cam_pos[row] for frame, row in self.frame_to_row.items()}

    def VGGT_O_preds_to_ply_export(self, out_dir=None, conf_threshold=1.5,R = None, t=None, s=None, multi=False, centre_to_cam00 = None):
        """Write one coloured .ply per frame, in cam0's frame with CUT3R's axis
        convention (x right, y up, z forward) so VGGT-O and CUT3R traces can be
        viewed side by side in the same viewer (see D_cut3r_vis.py).

        R, s, t: optional real-world alignment (rotation, scale, translation)
        from umeyama_align, e.g. subject_to_metric_and_gps_space's R_cr/s_cr/t_cr
        -- the same transform already applied to the subject trace, so points
        land in real-world metres instead of VGGT-O's raw (unitless) cam0 scale.
        Applied before the axis flip below, since it was derived against
        cam_pos_dict(), which is in that same raw, un-flipped frame. Left in
        raw cam0 scale if not given.

        out_dir defaults to preds_path's own folder / "ply", next to the
        predictions.npz/.glb this Reconstruction was loaded from.
        Files are named by source frame number, not row index, since
        predictions.npz rows carry no frame numbers of their own.
        """
        out_dir = Path(out_dir) if out_dir is not None else self.preds_path.parent / "ply"
        out_dir.mkdir(parents=True, exist_ok=True)

        for frame_idx, row in self.frame_to_row.items():
            conf   = self.preds["depth_conf"][row].reshape(-1)
            image  = self.preds["images"][row]  # (3, H, W), 0..1
            # already cam0-space -- see load_VGGT_O_predictions
            points = unproject(self.preds["depth"][row], self.preds["intrinsic"][row], self.preds["extrinsic"][row])
            points = points.reshape(-1, 3).cpu().numpy()


            #rotate and scale to real world (values derived from umeyay align vs calibrated RS trace)
            if multi ==True:
                points = (points@R.T  * s +t )#cam to real world
                if centre_to_cam00 is not None:
                    points = points - centre_to_cam00
                    #then transform to make origin sensible
                else: 
                    centre_to_cam00 = t
                    points = points -t
            #scale to metres if no transform is needed
            elif s is not None:
                points = points * s 

            colors = image.permute(1, 2, 0).reshape(-1, 3).cpu().numpy()
            valid  = (conf > conf_threshold).cpu().numpy()

            points = points[valid]
            points[:, 1:] *= -1  # match CUT3R's y-up/z-forward axis convention
            _write_ply(out_dir / f"{frame_idx:06d}.ply", points, colors[valid])
        return centre_to_cam00




def _write_ply(path, points, colors):
    """Binary little-endian PLY -- same byte layout as D_cut3r_recon.py's
    writer, so files from both sources load through D_cut3r_vis.py unchanged.
    points: (N,3) float. colors: (N,3) float in [0,1].
    """
    pts = np.asarray(points, dtype=np.float32)
    rgb = (np.asarray(colors) * 255).clip(0, 255).astype(np.uint8)
    n = len(pts)

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



#-------frame indexing ----vggt only at the moment
from pathlib import Path
from B_video_processing import available_frames
def build_frame_index(frames_dir, name_fmt, frame_range=None):
    """Maps absolute source frame number -> row index into preds arrays, assuming
    predictions.npz was generated from these exact files in sorted-by-frame-number
    order (see check_frame_count).

    frame_range: optional (start, end) slice into frames_dir's sorted file listing --
    for a VGGT-O shard reconstructed from only part of a shared frame folder (see
    B_VGGT_O_shards.run_vggt_omega_shards_batched), so row 0 of that shard's
    predictions.npz lines up with frame `start`, not the first file in frames_dir.
    """
    frames = available_frames(frames_dir, name_fmt)
    if frame_range is not None:
        start, end = frame_range
        frames = frames[start:end]
    return {frame: row for row, frame in enumerate(frames)}


def positions_to_frame_dict(positions, frames_dir, frame_name_fmt=FRAME_NAME_FMT):
    """Zip a flat (N,3) trace array (row order) against the sorted frame numbers of
    the folder it was reconstructed from -- for techniques whose output doesn't carry
    frame numbers itself (MegaSaM, plain VGGT, VGGT-O). Not needed for CUT3R
    (frame-number-named at the source, see load_cut3r_trace_v2) or lingbot-map (same).
    If you already have a Reconstruction loaded for VGGT-O, its own cam_pos_dict()
    does the same job without a second frames_dir lookup."""
    frames = available_frames(frames_dir, frame_name_fmt)
    assert len(frames) == len(positions), \
        f"{len(frames)} frames in {frames_dir} vs {len(positions)} positions -- " \
        "this technique wasn't run on the frame set you're matching against"
    return {frame: positions[i] for i, frame in enumerate(frames)}



