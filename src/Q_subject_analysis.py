'''SUBJECT ANALYSIS
A SERIES OF ANALYSIS TOOLS THAT OUTPUTS METRICS STRICTLY ASSOCIATED WITH THE SUBJECTS IN VIEW
THESE METRICS CAN BE FOR THE REPORT, OR TO HELP WITH DECISION MAKING ABOUT THE REPORT
'''


from pathlib import Path
import json
import re
import subprocess

import cv2
import numpy as np
import pandas as pd
from A_Config import report_path, length_units, MASK_NAME_FMT
from C_CSV_report import add_to_report
from Q_GPS_processing import android_movie_GPS
from Q_Metric_georeferencing import  metres_to_latlong
from Thr3D.G_transforms_alignments import umeyama_align, ortho_charts , transform_RST, recon_to_recon_matcher

#──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#3D analysis
from Thr3D.F_post_recon_processing import Reconstruction
from P_projection_mapping import load_frame_inputs, unproject_masked

def subject_model_positions(recon, masks_dir):
    '''DEAD CODE at the moment
    PROJECT THEN AVERAGE
    Upreses depth to mask resolution 
    Reprojects the depth into model space
    Finds a point in model space that the subject is, for each frame'''
    subject_positions = {}
    for frame_idx in recon.frame_to_row:
        rgb, mask, depth, depth_conf, lowres_rgb, intrinsic, extrinsic, sigma, _ = load_frame_inputs(recon, frame_idx, masks_dir)
        world_points, _ = unproject_masked(mask, rgb, depth, depth_conf, lowres_rgb, intrinsic, extrinsic, margin=8)
        pos = world_points.mean(dim=0) if world_points is not None else None
        subject_positions[frame_idx] = pos.cpu().numpy() if pos is not None else np.full(3, np.nan)
    return subject_positions


def meeting_calculator(A_real_dict, A_dict_entry, B_real_dict,B_dict_entry, lat_0= None, lon_0 = None,
                        B_is_camera=True, return_series=False):
    '''Returns dict of info about when movers A and B were closest together -- i.e. the incident happened.
    B is the camera by default; pass B_is_camera=False for a subject-vs-subject pair.
    Also appends that info straight to the report CSV.

    return_series=True additionally returns the full per-A-frame distance dict
    (frame -> distance) that's computed internally on the way to the closest-approach
    frame, instead of throwing it away -- e.g. for a continuous distance-from-camera
    readout rather than just the single closest moment.'''
    if B_is_camera:
        A_frames = np.array(sorted(A_real_dict.keys()))
        A_pos    = np.array([A_real_dict[f] for f in A_frames])              # (N,3)

        B_frames = np.array(sorted(B_real_dict.keys()))
        B_pos    = np.array([B_real_dict[f] for f in B_frames])              # (M,3)

        # each A frame -> its nearest available camera frame (the two frame sets don't line up 1:1)
        nearest_idx      = np.array([np.argmin(np.abs(B_frames - f)) for f in A_frames])
        matched_B_pos    = B_pos[nearest_idx]
        matched_B_frames = B_frames[nearest_idx]
    else:
        # two subjects: both keyed on the same recon frames, so only frames they're BOTH
        # present in can be compared -- nearest-matching would pair a subject against one
        # who'd already left and report a meeting that never happened
        both             = sorted(set(A_real_dict) & set(B_real_dict))
        A_frames         = np.array(both)
        A_pos            = np.array([A_real_dict[f] for f in both])
        matched_B_pos    = np.array([B_real_dict[f] for f in both])
        matched_B_frames = A_frames

    distances = np.linalg.norm(A_pos - matched_B_pos, axis=1)
    distances[np.isnan(distances)] = np.inf   # ignore frames with no valid subject mask

    meet_i       = np.argmin(distances)#index of meet
    meet_frame   = A_frames[meet_i] #frame of meet
    meet_B_frame = matched_B_frames[meet_i] #B's frame at the meet - may be different
    if lat_0:
        meet_lat_lon, _ = metres_to_latlong(matched_B_pos[meet_i:meet_i + 1], lat_0, lon_0) # position of B
        meeting_lat_lon = tuple(float(v) for v in meet_lat_lon[0])

    else:
        meeting_lat_lon = (None, None)

    #x=east/west, z=north/south -- the map plane; y (up) dropped. Origin is cam 0 either
    #way; only {{length_units}} says whether these are metres or raw model units
    meeting_xz = (float(matched_B_pos[meet_i][0]), float(matched_B_pos[meet_i][2]))

    #pair goes in the KEY -- add_to_report overwrites by key, so pair 2 would eat pair 1
    pair = f"{A_dict_entry}_vs_{B_dict_entry}"
    meeting_report = {
        f"{pair}_meeting_frame"            : float(meet_frame),    #frame A closest to B
        f"{pair}_meeting_B_frame"          : float(meet_B_frame),  #B's frame at that moment (they don't match)
        f"{pair}_meeting_lat_lon"          : meeting_lat_lon,
        f"{pair}_meeting_xz"               : meeting_xz,
        f"{{{{{pair}_meeting_distance}}}}" : float(distances[meet_i]),
        "{{length_units}}"                 : length_units(),   # "m" only when GPS data exists, else raw model units
        # needs video_path/fps (time-of-day) and multi-frame velocity (travel direction) -- not wired in yet
        "{{camera_direction}}"             : None,
    }
    add_to_report(meeting_report)
    if return_series:
        distance_series = dict(zip(A_frames.tolist(), distances.tolist()))
        return meeting_report, distance_series
    return meeting_report


def build_screen_labels(job):
    '''Merges the per-subject-per-frame fields needed for on-screen labelling
    (position, speed, heading, distance from camera, confidence) into one dict:
    job["screen_labels"][subject_slug][frame] = {"position", "speed_kmh", "heading",
    "dist_cam", "confidence"}.

    Pulls from job state that earlier cells already computed: subjects_real_pos /
    subjects_positions (position), group_metrics (speed/heading),
    subjects_distance_to_camera (distance -- only present for subjects meeting_pairs
    paired against "camera"), subjects_confidence.'''
    subjects = job.get("subjects_real_pos") or job.get("subjects_positions") or {}
    group_metrics   = job.get("group_metrics") or {}
    distance_to_cam = job.get("subjects_distance_to_camera") or {}
    confidence      = job.get("subjects_confidence") or {}

    screen_labels = {}
    for slug in subjects:
        pos_dict = subjects[slug]
        frames   = sorted(pos_dict)

        # group_metrics is keyed by the same subject slugs {"camera": ..., **subjects}
        # was built from (see speed_direction's named/multi return)
        mps, kmh, heading = group_metrics.get(slug, (None, None, None))
        speed_by_frame   = dict(zip(frames, kmh))     if kmh     is not None else {}
        heading_by_frame = dict(zip(frames, heading)) if heading is not None else {}

        dist_by_frame = distance_to_cam.get(slug, {})
        conf_by_frame = confidence.get(slug, {})

        screen_labels[slug] = {
            f: {
                "position"  : pos_dict[f],
                "speed_kmh" : speed_by_frame.get(f),
                "heading"   : heading_by_frame.get(f),
                "dist_cam"  : dist_by_frame.get(f),
                "confidence": conf_by_frame.get(f),
            }
            for f in frames
        }

    job["screen_labels"] = screen_labels
    return screen_labels


def contact_frames(recon, A_masks_dir, B_masks_dir, touch_px=2,
                   skip_depthels=1, max_depthels=6, depth_tol=0.05):
    '''Are A and B touching, per frame? Camera space, no unprojection.
    1) do the masks abut (mask res, so a thin sword survives)
    2) no -> not touching, true even if depth is broken
    3) yes -> depth profile either side of the seam decides touching vs occlusion.

    The seam itself is never measured -- those depthels are mixed pixels and a
    mislabelled mask edge sits there too. Instead each side's depth is binned by
    distance from the seam, a line is fitted to the clean bins, and both are
    extrapolated IN. Real abutment -> the two intercepts agree. A bad fit means
    the mask isn't following a surface, which is its own answer.

    skip_depthels: bins nearer the seam than this are assumed contaminated.
    Returns {frame: {..., "A_profile", "B_profile"}} -- profiles are {bin: depth} for plotting.'''
    frame_keys = sorted(recon.frame_to_row, key=recon.frame_to_row.get)
    depths     = recon.preds["depth"].cpu().numpy()
    h, w       = depths.shape[1:]

    def _profile(dist_lo, mask_lo, depth):
        '''median depth per depthel-distance bin, for one side of the seam'''
        prof = {}
        for b in range(skip_depthels, max_depthels + 1):
            sel = mask_lo & (np.round(dist_lo) == b) & (depth != 0)
            if sel.sum() >= 3:                        #need a few pixels to trust the median
                prof[b] = float(np.median(depth[sel]))
        return prof

    def _extrapolate(prof):
        '''fit depth vs distance, return (depth at the seam, fit residual)'''
        if len(prof) < 2:
            return None, None
        x, y = np.array(sorted(prof)), np.array([prof[k] for k in sorted(prof)])
        slope, intercept = np.polyfit(x, y, 1)
        return float(intercept), float(np.abs(y - (slope * x + intercept)).max())

    out = {}
    for row, f in enumerate(frame_keys):
        A = cv2.imread(str(Path(A_masks_dir) / MASK_NAME_FMT.format(f)), cv2.IMREAD_GRAYSCALE)
        B = cv2.imread(str(Path(B_masks_dir) / MASK_NAME_FMT.format(f)), cv2.IMREAD_GRAYSCALE)
        if A is None or B is None:
            continue
        A, B = A > 0, B > 0
        if not A.any() or not B.any():
            continue

        #square centred kernel: side 2r+1 grows the mask by r px all round. (n,)*2 is (n,n)
        touch_k = np.ones((touch_px * 2 + 1,) * 2, np.uint8)
        abut = cv2.dilate(A.astype(np.uint8), touch_k).astype(bool) & \
               cv2.dilate(B.astype(np.uint8), touch_k).astype(bool)
        if not abut.any():
            continue                                   #steps 1+2: not touching, done

        #distance from the OTHER mask, in maskels, for every pixel
        dist_to_B = cv2.distanceTransform((~B).astype(np.uint8), cv2.DIST_L2, 3)
        dist_to_A = cv2.distanceTransform((~A).astype(np.uint8), cv2.DIST_L2, 3)

        scale = A.shape[0] / h                         #maskels per depthel
        to_lo = lambda m: cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST_EXACT)

        depth   = depths[row]
        A_prof  = _profile(to_lo(dist_to_B) / scale, to_lo(A) > 0.5, depth)
        B_prof  = _profile(to_lo(dist_to_A) / scale, to_lo(B) > 0.5, depth)
        A_seam, A_res = _extrapolate(A_prof)
        B_seam, B_res = _extrapolate(B_prof)
        if A_seam is None or B_seam is None:           #not enough clean bins, can't judge
            continue

        gap = abs(A_seam - B_seam)
        out[int(f)] = {
            "abut_px"  : int(abut.sum()),
            "depth_gap": float(gap),
            "front"    : "A" if A_seam < B_seam else "B",
            "contact"  : bool(gap <= depth_tol * min(A_seam, B_seam)),  #fraction -> scale-free
            "fit_resid": (A_res, B_res),               #big -> mask isn't following a surface
            "A_profile": A_prof,
            "B_profile": B_prof,
        }
    return out


def subject_direction(sub_real_dict):
    '''Compass heading (degrees, 0=North, clockwise) and straight-line distance (m)
    of the subject's net travel from its first to last recorded position.
    sub_real spans the whole reconstruction, not just frames where the subject
    was actually detected, so NaN rows at the edges/gaps are normal -- use the
    first and last non-NaN rows, not sub_real[0]/sub_real[-1] directly.
    Also appends that info straight to the report CSV.'''
    frames   = sorted(sub_real_dict.keys())
    #how to turn a dict into an array
    sub_real = np.array([sub_real_dict[f] for f in frames])
    valid      = ~np.isnan(sub_real).any(axis=1)
    valid_real = sub_real[valid]
    dir  = valid_real[-1] - valid_real[0]   # (dx=East, dy=Up, dz=North)
    dist = np.linalg.norm(dir)
    heading = np.degrees(np.arctan2(dir[0], dir[2])) % 360   # East, North -- Up is irrelevant to heading
    print("subject moved",dist,"m" )
    print("subject moved in a",heading,"direction" )
    direction_report = {
        "{{subject_direction_degs}}" : heading, # 0 is north
        "{{subject_distance}}"       : dist,
    }
    add_to_report(direction_report)

    return direction_report


  

def find_mask_centroids_in_model_space(subject_recon,
            masks_dir, RA = 1, analyse = False, autocam=False, conf_thresh=0.0):
    '''Confidence-weighted centroid of every mask, one recon-worth at a time.
    No hard confidence cutoff: every in-mask pixel with nonzero depth contributes,
    weighted by its own depth_conf, so a frame where confidence never clears a fixed
    threshold still degrades to a weighted average instead of NaN-ing out entirely.
    depth_conf is floored at 1.0 (not 0) -- confirmed against real data, min=1.0,
    max~4.86 -- so weighting by raw confs gives even garbage pixels real pull;
    weighting by (confs - 1) instead zeroes that floor out properly.

    computes rolling average by convolving 

    Also returns spread_framecam_ray: the confidence-weighted depth spread (std) under
    the mask, as a vector in model space along that frame's own camera ray -- not a bare
    number, since each frame's ray points a different way once rotated into model space.
    centroid +/- spread_framecam_ray is the "subject is kinda around this blob, stretched
    along the sightline" envelope, not an exact point.
    '''
    #subject centroids found and transformed into model space
    from P_projection_mapping import MASK_NAME_FMT
    frame_keys = sorted(subject_recon.frame_to_row, key=subject_recon.frame_to_row.get)  # row order
    intrinsics = subject_recon.preds["intrinsic"].cpu().numpy()   # (N,3,3) native res, matches depth
    extrinsics = subject_recon.preds["extrinsic"].cpu().numpy()   # (N,3,4) model-to-cam
    depths     = subject_recon.preds["depth"].cpu().numpy()       # (N,h,w) native res
    confs      = subject_recon.preds["depth_conf"].cpu().numpy()  # (N,h,w) native res
    h, w       = depths.shape[1:]

    #load masks - make them model sized.
    #treat missing files as an empty mask instead of letting cv2.resize crash on it.
    mask_imgs = [cv2.imread(str(Path(masks_dir) / MASK_NAME_FMT.format(f)), cv2.IMREAD_GRAYSCALE) for f in frame_keys]
    masks = np.stack([
        cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST_EXACT) > 0 if img is not None else np.zeros((h, w), dtype=bool)
        for img in mask_imgs
    ])   # (N,h,w) bool, resized to the depth's own native resolution -- no bilateral upsampling
    ys, xs = np.indices((h, w))   # ys: row-index grid, xs: col-index grid (np.indices order)

    # 1) WEIGHT -- confidence, zeroed outside the mask, wherever depth is missing, and
    # (via conf_thresh) below a hard per-pixel confidence gate -- so a garbage pixel
    # can't sneak weight in just for being inside the mask. Pixels that pass the gate
    # still contribute proportional to (confs - 1), not equally, so a frame that's
    # barely above threshold everywhere doesn't get the same pull as one confidently so.
    # confs - 1: depth_conf is floored at 1.0, not 0 -- subtracting the floor gives
    # garbage pixels zero weight instead of the ~20% pull raw confs would give them.
    weights     = masks * (confs - 1) * (depths != 0) * (confs >= conf_thresh)
    #key PARAMETER HERE!
    weight_sums = weights.sum(axis=(1, 2))
    valid       = weight_sums > 0

    # 2) CENTROID -- confidence-weighted mean pixel (col,row) and mean depth (centroid_depth) under the mask
    #for fr in range(0,RA):
    #get totals for each dimension, every frame at once
    u_sum = np.zeros(len(frame_keys)) # 0s - will end up as NaNs once the convolve is done
    v_sum = np.zeros(len(frame_keys))
    depth_sum = np.zeros(len(frame_keys))

    
    u_sum[valid] = (xs * weights).sum(axis=(1, 2))[valid]  # valid is for each frame that has a mask
    v_sum[valid] = (ys * weights).sum(axis=(1, 2))[valid]
    depth_sum[valid] = (depths * weights).sum(axis=(1, 2))[valid]

    # for rolling average -- convolve on a dense, frame-number-indexed array (one slot
    # per integer frame in [min(frame_keys), max(frame_keys)]) instead of on the
    # compacted per-entry arrays above. frame_keys can have real gaps (dropped/missing
    # frames), and convolving the compacted arrays directly treats "next entry in the
    # array" as "next frame", silently stitching frames from across a gap together as
    # if adjacent. Scattering into a dense array first means a missing frame occupies
    # a real (zero-weight) slot in the window, so it dilutes/breaks the rolling average
    # the same way an RA-sized run of low-confidence frames would, instead of being
    # invisible to the convolution.
    frame_keys_arr = np.asarray(frame_keys)
    dense_len = frame_keys_arr.max() - frame_keys_arr.min() + 1
    dense_idx = frame_keys_arr - frame_keys_arr.min()

    def _rolling(dense_vals):
        return np.convolve(dense_vals, np.ones(RA), mode='same')[dense_idx]

    dense_weight_sums = np.zeros(dense_len)
    dense_weight_sums[dense_idx] = weight_sums
    weight_sums_R = _rolling(dense_weight_sums)  # tot pixels

    dense_u_sum = np.zeros(dense_len)
    dense_u_sum[dense_idx] = u_sum
    u = _rolling(dense_u_sum) / weight_sums_R  # average x pos (image space, so u)
    print(u[0])
    dense_v_sum = np.zeros(dense_len)
    dense_v_sum[dense_idx] = v_sum
    v = _rolling(dense_v_sum) / weight_sums_R  # average y pos (image space, so v)

    dense_depth_sum = np.zeros(dense_len)
    dense_depth_sum[dense_idx] = depth_sum
    centroid_depth = _rolling(dense_depth_sum) / weight_sums_R


    # 3) Z spread -- confidence-weighted std of depth under the mask ("how tight is this blob"),
    # still in camera-space model units along-the-ray, not yet a model-space direction
    z_var = np.zeros(len(frame_keys)) #empty
    #residual of each depth, summed to make cariance (filterd by valid mask present in the frame)
    z_var[valid] = (weights * (depths - centroid_depth[:, None, None]) ** 2).sum(axis=(1, 2))[valid]
    #rolling sum of variances (not variances now), gap-aware for the same reason as above
    dense_z_var = np.zeros(dense_len)
    dense_z_var[dense_idx] = z_var
    z_var_R = _rolling(dense_z_var)
    #then take the average (variance) and sqrt (stdev)
    z_std = np.sqrt(z_var_R / weight_sums_R)


    # 4) UNPROJECT to CAM SPACE -- (col,row,centroid_depth) -> camera-space xyz. col pairs with cx/fx (horizontal),
    # row pairs with cy/fy (vertical) -- same pinhole convention as unproject() in Thr3D, just
    # vectorized over every frame at once instead of one full depth map at a time.
    fx, fy = intrinsics[:, 0, 0], intrinsics[:, 1, 1]
    cx, cy = intrinsics[:, 0, 2], intrinsics[:, 1, 2]
    x = (u - cx) * centroid_depth / fx
    y = (v - cy) * centroid_depth / fy
    subject_centroids_cam = np.stack([x, y, centroid_depth], axis=1)   # (N,3) in each frame's own camera space

    R = extrinsics[:, :3, :3]   # model-to-cam rotation, per frame
    t = extrinsics[:, :3, 3]
    # 5) TO MODEL SPACE -- cam -> model space: P_model = R^T @ (P_cam - t)
    subject_centroids_model = np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), subject_centroids_cam - t)

    if analyse:
        import matplotlib.pyplot as plt

        # NOTE: u, v, centroid_depth, z_std, weight_sums_R, subject_centroids_model
        # are all indexed by position in frame_keys (one entry per frame this
        # function processed), NOT by subject_positions_model_dict's keys (a
        # different, filtered set) -- plotting against frame_keys directly keeps
        # the x-axis aligned with what's actually being convolved/transformed.
        fig, axes = plt.subplots(9, 1, sharex=True, figsize=(8, 18))

        for ax, vals, spread, label in zip(
            axes[:3], (u, v, centroid_depth), (None, None, z_std), ("u", "v", "centroid_depth")
        ):
            vals = np.array(vals)
            if spread is not None:
                spread = np.array(spread)
                ax.fill_between(frame_keys, vals - spread, vals + spread, color="0.7", alpha=0.6, label=f"{label} spread")
            ax.plot(frame_keys, vals, marker="o", color="black", markersize=1)
            ax.set_ylabel(label)

        # diagnostic: rolling weight-sum denominator -- a near-zero dip here right
        # at a break/gap is the signature of the RA division blowing up (tiny
        # denominator amplifying numerator noise), as opposed to a plotting artifact.
        axes[3].plot(frame_keys, weight_sums_R, marker="o", color="black", markersize=1)
        axes[3].axhline(0, color="red", linewidth=0.5)
        axes[3].set_ylabel("weight_sums_R")

        # diagnostic: camera position in model space, per frame (-R^T @ t) -- if the
        # camera pose itself jumps/discontinuities right where subject_centroids_model
        # dives, that points at bad extrinsics for those frames rather than the RA step.
        cam_pos_model = np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), -t)
        axes[4].plot(frame_keys, cam_pos_model[:, 2], marker="o", color="black", markersize=1)
        axes[4].set_ylabel("cam_pos_model z")

        axes[5].plot(frame_keys, subject_centroids_model[:, 2], marker="o", color="black", markersize=1)
        axes[5].set_ylabel("subj_model z")

        # diagnostic: frame-to-frame rotation angle, independent of t -- probes R
        # directly rather than through its (possibly deceptively smooth) effect on -t.
        R_rel = np.einsum("nij,njk->nik", np.transpose(R[:-1], (0, 2, 1)), R[1:])
        cos_angle = np.clip((np.trace(R_rel, axis1=1, axis2=2) - 1) / 2, -1, 1)
        rot_angle_deg = np.degrees(np.arccos(cos_angle))
        axes[6].plot(frame_keys[1:], rot_angle_deg, marker="o", color="black", markersize=1)
        axes[6].set_ylabel("frame-to-frame\nrotation (deg)")

        # diagnostic: camera-space x,y (post-intrinsics) -- checks whether fx/fy/cx/cy
        # are noisy/wrong for specific frames, which u/v (pre-intrinsics) wouldn't show.
        axes[7].plot(frame_keys, x, marker="o", color="black", markersize=1)
        axes[7].set_ylabel("cam-space x")
        axes[8].plot(frame_keys, y, marker="o", color="black", markersize=1)
        axes[8].set_ylabel("cam-space y")

        axes[-1].set_xlabel("frame key")

        fig.tight_layout()

    # 6) RE-OREINT Z_std (thickness + error) into model space
    #shoot ray from cam to centroid
    ray_dir_cam   = subject_centroids_cam / np.linalg.norm(subject_centroids_cam, axis=1, keepdims=True)
    #rotate frame of reference into model space
    ray_dir_model = np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), ray_dir_cam)
    #project z_std into that frame of reference
    depth_std_model = z_std[:, None] * ray_dir_model   # (N,3), NaN where z_std is NaN

    if autocam:
        from P_projection_mapping import load_frame_inputs, unproject_masked
        # box from every masked+confident pixel's own 3D point, not just each frame's
        # single centroid -- a centroid-only box only ever captures where the subject's
        # centre wandered, never its actual size.
        # Build the box from THE SAME points the renderer draws -- unproject_masked is
        # exactly what composite_overlay calls per frame (full-res mask, colour-guided
        # upsampled depth). Re-deriving our own cloud from the low-res mask + raw depth
        # instead put mask-edge pixels carrying background depth into the box, so it
        # bounded points that are nowhere in the render.
        centrality_weights = mask_centrality_weights(u, v, h, w)
        mask_points = []
        for f in np.asarray(frame_keys)[centrality_weights > 0]:
            fi = load_frame_inputs(subject_recon, int(f), masks_dir)
            rgb_full, mask, depth_low, conf_low, lowres_rgb, intrinsic_full, extrinsic = fi[:7]
            if rgb_full is None or mask is None:
                continue
            world_points, _ = unproject_masked(mask, rgb_full, depth_low, conf_low,
                                                lowres_rgb, intrinsic_full, extrinsic,
                                                conf_thresh=conf_thresh)
            if world_points is None:
                continue
            mask_points.append(world_points.cpu().numpy())
        # empty is a legitimate answer (masks all blank, or every frame unprojected to
        # nothing) -- hand back an empty (0,3) cloud so the caller can fall back to a
        # subject-free camera instead of dying on the concatenate.
        mask_points = np.concatenate(mask_points, axis=0) if mask_points else np.empty((0, 3))
        return subject_centroids_model, frame_keys, depth_std_model, weight_sums_R, centrality_weights, mask_points

    #cetroids and the Zdepth standard deviation in model space, plus the frame keys
    #and confidence weight (weight_sums_R -- already frame_keys-aligned via _rolling's [dense_idx])
    return subject_centroids_model, frame_keys, depth_std_model, weight_sums_R

def subject_centroids_to_metric_and_gps_space(
            subject_centroids_model,
            frame_keys,
            _depth_std_model,
            weight_sums_R,
            lat_0, lon_0,
            R_mw , s_mw,t_mw

           ):
    """Turns mask centroids from model space into global space and scales to metric.

    Projects mask centroids through model to world to get
    metres and lat/lon -- no separate RS-space hop needed since
    cam_poses_realworld_dict is already in real-world space.
    
    """   
      
    sub_real      = transform_RST(subject_centroids_model, R_mw, s_mw, t_mw)
    sub_real_dict = dict(zip(frame_keys, sub_real))   # frame-keyed, same pattern as cam_poses_realworld_dict
    sub_conf_dict = dict(zip(frame_keys, weight_sums_R))   # frame-keyed confidence, same alignment
    sub_latlon, _ = metres_to_latlong(sub_real, lat_0, lon_0)
    valid = ~np.isnan(sub_latlon).any(axis=1)
    return [(float(frame_keys[i]), *sub_latlon[i])
                for i in range(len(frame_keys)) if valid[i]] , sub_real , sub_real_dict, sub_conf_dict
    
#------------helpers----------------

def rolling_by_frame(frames, meas, window):
    """Centered mean of "meas" over `window` FRAMES (not samples). Gaps dilute the
    window instead of being stitched over."""
    frames = np.asarray(frames); meas = np.asarray(meas, dtype=float)
    idx = frames - frames.min()
    vals = np.zeros(idx.max() + 1); vals[idx] = meas #0s length of total time put measuements on after
    hits = np.zeros(idx.max() + 1); hits[idx] = 1 #do samples exist at each time point? lenght of total time
    k = np.ones(window)#kernel
    #then convolve
    return (np.convolve(vals, k, mode="same") / np.convolve(hits, k, mode="same"))[idx]

#──────Mapping analysis────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

        




#──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────




#-----------------cam pose analysis

#distance travelled with subject (will determine recon type) - can be gps based if I get a trace of it - no need for / 
#single continuous path? - needed for GPS fitting - maps - or I can just use the GPS trace
#is the subject in a bit that was reconstructed? - GPS fitting - maps -


###
    #only run if I haven't got this for free.. 
   # if  composite is None:
    #    subject_positions = subject_world_positions(frames_list, depth_path, mask_path, transform_matrices)

##-------AUTO CAM
    
def auto_camera_position(extrinsics, weights):
    '''Weighted average of camera CENTRES (not rotations) across frames with
    weight>0 -- "where the operators stood", weighted by mask centrality.
    extrinsics is (N,3,4) model-to-cam (P_cam = R@P_model + t), so a frame's
    camera centre in model space is C = -R^T @ t (the point where P_cam=0).
    Used with look_at_rotation() instead of auto_camera_rotation() when the
    recorded camera orientations aren't reliably pointed at the subject
    (e.g. a handheld shoot) -- position is a much more robust signal than
    orientation in that case, so aim is solved geometrically via look-at
    instead of inherited from a shaky average of raw rotations.'''
    valid = weights > 0
    if not np.any(valid):
        raise ValueError("auto_camera_position: no frame has weight > 0 -- no usable subject mask in this recon.")
    R = extrinsics[valid, :3, :3]
    t = extrinsics[valid, :3, 3]
    centres = -np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), t)
    w = weights[valid]
    return (centres * w[:, None]).sum(axis=0) / w.sum()

def look_at_rotation(eye, target, up):
    '''Camera-to-model rotation that points the camera's forward axis at `target`
    from `eye`, holding roll level against `up` (e.g. the real-world up direction
    rotated into model space) rather than inheriting whatever roll the source
    footage happened to have. Axis convention matches preds["extrinsic"]/
    P_projection_mapping.reproject(): +x=right, +y=down (image v grows downward),
    +z=forward. `up` only needs to be roughly upward and non-parallel to the
    eye->target direction -- it's just used to disambiguate roll, not followed
    exactly (the actual "up" of the result is whatever's left over after right
    and forward are fixed).'''
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.stack([right, down, forward], axis=1)   # (3,3) columns = camera axes in model space

def subject_oriented_box(points_model, margin=0.0, percentile=100.0):
    '''PCA-oriented bounding box: returns the 8 corners (8,3) in model space, ordered
    the same way as np.meshgrid(..., indexing="ij") so the usual edge list still works.
    margin: padding per side as a FRACTION of each axis' own extent (0.1 = 10% each side),
    not model units -- so one value works whatever the recon's scale.
    percentile: central % of points kept per axis (100 = raw min/max). At 100 each face
    is pinned by 1-5 stray points out of ~90k, so the box sits well outside the subject.'''
    pts = points_model[~np.isnan(points_model).any(axis=1)]
    if pts.shape[0] < 2:
        raise ValueError("subject_oriented_box: need at least 2 valid points.")
    centre = pts.mean(axis=0)
    centred = pts - centre
    _, eigvecs = np.linalg.eigh(np.cov(centred, rowvar=False))
    axes = eigvecs[:, ::-1]            # columns = principal axes, largest variance first
    local = centred @ axes             # points in the box's own frame
    if percentile >= 100:
        lo, hi = local.min(axis=0), local.max(axis=0)
    else:
        half = (100.0 - percentile) / 2.0
        lo, hi = np.percentile(local, [half, 100.0 - half], axis=0)
    pad = margin * (hi - lo)   # fraction of each axis' own extent, so it's scale-free
    lo = lo - pad
    hi = hi + pad
    corners_local = np.stack(np.meshgrid([lo[0], hi[0]], [lo[1], hi[1]], [lo[2], hi[2]],
                                          indexing="ij"), axis=-1).reshape(-1, 3)
    return corners_local @ axes.T + centre


def mask_centrality_weights(col, row, h, w):
    '''Per-frame weight for how central the subject's mask centroid (col,row) is in its
    frame, normalized to [-1,1] about the image center: w = (1-|x|)*(1-|y|). 1 = dead
    centre, 0 = at the edge. NaN (no mask that frame) or a frame with no weight -> 0.'''
    x = (col - w / 2.0) / (w / 2.0)
    y = (row - h / 2.0) / (h / 2.0)
    weights = (1 - np.abs(x)) * (1 - np.abs(y))
    weights = np.nan_to_num(weights, nan=0.0)
    return np.clip(weights, 0.0, None)



#------- auto cam