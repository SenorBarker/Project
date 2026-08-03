
import numpy as np
import torch
import matplotlib.pyplot as plt
from itertools import combinations
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")






#-------------------Transforms  
#the standard transform
def transform_RST(positions,R, s, t):
    '''N,3 array in, N,3 array out'''
    return (positions @ R.T) * s +t

#Turn a double transform into a single transform
def combine_transforms(R_cb, s_cb, t_cb, R_ba, s_ba, t_ba):
    R_ca = R_ba @ R_cb
    s_ca = s_ba * s_cb
    t_cb_rotated = transform_RST(t_cb, R_ba, s_ba, 0)  # just rotate+scale t_ba into world units -- no translation yet
    t_ca = t_cb_rotated + t_ba                          # now it's actually the final b→world translation
    return R_ca, s_ca, t_ca


#Move from Camera-centric depth matps to xyz in the frame of reference described by camera extrinsics
#rigid transform
def unproject(depth_full, intrinsic, extrinsic):
    h, w = depth_full.shape
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    ys, xs = torch.meshgrid(
        torch.arange(h, device=DEVICE, dtype=torch.float32),
        torch.arange(w, device=DEVICE, dtype=torch.float32),
        indexing="ij",
    )
    #pinhole projection in reverse
    z = depth_full
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    points_cam = torch.stack([x, y, z], dim=-1)  # (h, w, 3)

    R = extrinsic[:, :3]  # world-to-camera rotation
    t = extrinsic[:, 3]
    # world-to-camera: P_cam = R @ P_world + t  =>  P_world = R^T @ (P_cam - t)
    points_world = (points_cam - t) @ R
    return points_world


#in case a model's system doesn't match to cam0's thi rotates and translates), no scale
def apply_cam0_frame(positions, R0, origin):
    """Re-express positions in the frame anchored at (R0, origin): shift so
    origin is at (0,0,0), then rotate by R0's inverse so R0 becomes identity.

    Generic - works for any rotation matrix regardless of which angle
    convention built it, so any source with its own scrambled angle order can
    reuse this once it builds a proper R0.
    """
    return (R0.T @ (np.asarray(positions) - origin).T).T


#rotation matrices - one axis at a time 
def _rot_x(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
def _rot_y(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
def _rot_z(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _rc_rotation_matrix(yaw, pitch, roll):
    """Build a rotation matrix from RealityScan's yaw/pitch/roll.

    RealityScan's yaw/pitch/roll don't rotate about (x, y, alt) as labelled -
    yaw is about alt (up), pitch is about x, roll is about y (forward).
    in pilot convention - LR, then up down, then spin.
    """
    return _rot_y(yaw) @ _rot_x(pitch) @ _rot_z(roll)



#----RESULT VISIALISATION-------
def ortho_charts(dataA, label_A, dataB=None, label_B = None, title = "chart", out_path = None, rmse = None, mean_dist = None):
    views = [("X", "Y", 0, 1), ("X", "Z", 0, 2), ("Z", "Y", 2, 1)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (xl, yl, xi, yi) in zip(axes, views):
        ax.plot(dataA[:, xi], dataA[:, yi], "o-", color="tab:blue",label=label_A)
        if dataB is not None:
            ax.plot(dataB[:, xi], dataB[:, yi], "o-", color="tab:red",label=label_B)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f"{xl}{yl} plane")

        ax.set_aspect("equal", adjustable="datalim")
        
        ax.grid(True)
        ax.legend()   
    fig.suptitle(f"{title}")
    if rmse is not None:
        fig.suptitle(f"Trace alignment (RMSE={rmse:.3f}, Mean_dist = {mean_dist:.3f})")
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
#match 2 models' frame indices for comparisons
#tiny frame gap as we're supposed to be matching precisely here - ideally we want
#exactly the same frames, but 5frarmes is only 160ms, so should be fine
def recon_to_recon_matcher(source_dict, target_dict, max_frame_gap=5):
    """Match each frame in source_dict to its nearest frame in target_dict,
    one-to-one -- source_dict is typically dense (one row per reconstructed
    frame) and target_dict sparse (e.g. GPS samples every ~13-27 frames), so
    naively matching every source frame independently lets a single sparse
    target sample pull in several nearby source frames, clustering the fit
    around whichever target samples happen to have neighbours and silently
    over-weighting them. Instead, keep only the single closest source frame
    per target frame.
    Returns (source_pts, target_pts) as (N,3) arrays."""
    target_frames = np.array(sorted(target_dict.keys()))
    best = {}   # target frame -> (gap, source_pos)
    for frame, pos in sorted(source_dict.items()):
        nearest = int(target_frames[np.argmin(np.abs(target_frames - frame))])
        gap = abs(nearest - frame)
        if gap <= max_frame_gap and (nearest not in best or gap < best[nearest][0]):
            best[nearest] = (gap, pos)
    src = np.array([pos for _, pos in best.values()])
    tgt = np.array([target_dict[frame] for frame in best.keys()])
    return src, tgt


def recon_to_recon_transform(recon_from,cam_poses_to_dict):
        '''
        takes a recon and obtains it's dict of frames and poses
        compares this wiht another recon's dict
        finds correspondinf frames
        aligns the matches 
        returns r s t to get from the firstone to the second
        '''
        
        matched_from, matched_to = recon_to_recon_matcher(
            recon_from.cam_pos_dict(), cam_poses_to_dict
        )
        R, s, t, _, rmse = umeyama_align(
            matched_to, matched_from, label_A="cam_real", label_B="subject_cam", with_scale=True
        )
        print(f"Subject VGGT-O → real-world: {len(matched_to)} pairs, RMSE = {rmse:.4f} m")
        return R, s, t


#--------stitch together positions dicts

def merge_pos(*sub_pos_real_dicts):
    # check every pair of shards for frame numbers they both have a subject position for,
    # and how far apart their independent estimates land in real-world metres
    for (i, a), (j, b) in combinations(enumerate(sub_pos_real_dicts), 2):
        shared_keys = sorted(set(a) & set(b))
        if not shared_keys:
            continue
        print(f"shards {i} & {j}: {len(shared_keys)} overlapping frame(s)")
        for k in shared_keys:
            pa, pb = a[k], b[k]
            dist = np.linalg.norm(pa - pb)  # distance between points - the mismatch in modelling
            print(f"  frame {k}: shard{i}={pa}  shard{j}={pb}  dist={dist:.3f} m")

    # merge - later shard wins on collision (same behaviour as before, now with visibility into what's being dropped)
    sub_pos_real_dict_all = {}
    for d in sub_pos_real_dicts:
        sub_pos_real_dict_all.update(d)
    return sub_pos_real_dict_all



 #--------------ALIGNMENTS--------------------------------------------------------------
def _umeyama_solve(A, B, ref_A, ref_B, with_scale, up_A, up_B, up_weight):
    """
    Shared Umeyama core: estimate R, s from A/B centered on ref_A/ref_B
    (the centroid for the plain alignment, the anchor point for the
    anchor-pinned alignment), then set t so ref_A <- s*R*ref_B exactly.
    """
    n = A.shape[0]
    A_c = A - ref_A
    B_c = B - ref_B
    #cross variance matrix - 3x3 for the 3 axes of the two sets of points
    H = B_c.T @ A_c / n

    if up_A is not None and up_B is not None:
        up_A = np.asarray(up_A, dtype=float) / np.linalg.norm(up_A)
        up_B = np.asarray(up_B, dtype=float) / np.linalg.norm(up_B)
        if up_weight is None:
            # Scale the vector term to match H's own magnitude (not an
            # arbitrary count), since up_A/up_B are unit vectors but H's
            # entries reflect the actual position units/scale of A and B.
            up_weight = np.linalg.norm(H)
        #this adds in the rotation needed to keep up = up, but now, up is always y axis
        H = H + up_weight * np.outer(up_B, up_A)

    U, S, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, 1])
    R = Vt.T @ D @ U.T

    if with_scale:
        var_B = (B_c ** 2).sum() / n
        s = np.sum(S) / var_B
    else:
        s = 1.0

    t = ref_A - s * R @ ref_B
    return R, s, t


def _umeyama_finish(A, B, R, s, t, label_A, label_B, out_path):
    B_aligned = (s * R @ B.T).T + t
    rmse = np.sqrt(np.mean(np.sum((A - B_aligned) ** 2, axis=1)))
    mean_dist = np.mean(np.sqrt(np.sum((A - B_aligned) ** 2, axis=1)))
    charts = ortho_charts(A,label_A = label_A , dataB=B_aligned, label_B = label_B, title = "test2", out_path = out_path, rmse = rmse, mean_dist = mean_dist )
    return B_aligned, rmse, mean_dist


def umeyama_align(A, B, label_A = "A", label_B = "B", out_path = None, with_scale=True, up_A=None, up_B=None, up_weight=None, skip_indices=None):
    """
    Align point set B onto point set A using the Umeyama algorithm
    (rotation + uniform scale + translation; no independent per-axis scaling).
    A, B: (N, 3) arrays of corresponding points, same order - MUST MATCH
    THe result is in the dimensions of set A.

    Position correspondences alone underconstrain rotation about the
    direction of travel when the path is close to a straight line (rolling
    the whole trace about its own travel axis barely changes the residual).
    Passing up_A/up_B (a single corresponding "up" direction in each frame's
    own coordinate system, e.g. from PCA least-variance axis / stable camera
    axis) breaks that degeneracy by adding a vector correspondence into the
    same cross-covariance matrix, weighted by up_weight (defaults to N, i.e.
    equal total influence to all the position correspondences combined).

    R and s minimize the total residual across all points; t is set so the
    centroids coincide. For pinning a specific point exactly (e.g. to measure
    drift from a start frame), use umeyama_align_anchor instead -- reusing
    this function's R/s with an anchor-forced t is not the least-squares
    solution for that constraint and inflates the error.

    Returns: R (3x3 rotation), s (float scale), t (3,) translation, B_aligned (N,3), rmse (float)
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    print("A,B", len (A), len(B))

    'remove outliers'
    if skip_indices is not None:
            mask = np.ones(A.shape[0], dtype=bool)
            mask[list(skip_indices)] = False
            A, B = A[mask], B[mask]

    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)
    R, s, t = _umeyama_solve(A, B, centroid_A, centroid_B, with_scale, up_A, up_B, up_weight)
    B_aligned, rmse, mean_dist = _umeyama_finish(A, B, R, s, t, label_A, label_B, out_path)
    return R, s, t, B_aligned, rmse


def umeyama_align_anchor(A, B, label_A = "A", label_B = "B", out_path = None, with_scale=True, up_A=None, up_B=None, up_weight=None, anchor_index=None, skip_indices=None):
    """
    Umeyama alignment constrained to pin A[anchor_index]/B[anchor_index]
    together exactly (e.g. frame 0), for measuring drift from that point
    rather than minimizing the overall trace error.

    This is NOT umeyama_align with t swapped for an anchor-forced value --
    that mismatches R/s (fit to minimize centroid-referenced error) against
    a different translation, which increases error rather than concentrating
    it at the anchor. Instead R and s here are solved directly for the
    anchor-referenced problem: centering both point sets on the anchor point
    (instead of the centroid) before the SVD/scale step is the correct
    least-squares solution subject to the anchor coinciding exactly.

    anchor_index indexes into A/B AFTER skip_indices has been applied.
    Defaults to 0 (the first remaining point) if None.

    Returns: R (3x3 rotation), s (float scale), t (3,) translation, B_aligned (N,3), rmse (float)
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    print("A,B", len (A), len(B))

    if skip_indices is not None:
            mask = np.ones(A.shape[0], dtype=bool)
            mask[list(skip_indices)] = False
            A, B = A[mask], B[mask]

    if anchor_index is None:
        anchor_index = 0

    ref_A = A[anchor_index]
    ref_B = B[anchor_index]
    R, s, t = _umeyama_solve(A, B, ref_A, ref_B, with_scale, up_A, up_B, up_weight)
    B_aligned, rmse, mean_dist = _umeyama_finish(A, B, R, s, t, label_A, label_B, out_path)
    return R, s, t, B_aligned, rmse