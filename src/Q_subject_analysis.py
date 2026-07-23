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
from A_Config import report_path
from C_CSV_report import add_to_report
from Q_GPS_processing import android_movie_GPS
from Q_Metric_georeferencing import  metres_to_latlong
from G_transforms_alignments import umeyama_align, ortho_charts , transform_RST, recon_to_recon_matcher

#──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#3D analysis
from F_post_recon_processing import Reconstruction
from P_projection_mapping import load_frame_inputs, unproject_masked
#helper function to turn a mask into world coordinates
def mask_centroid_model_position(mask, rgb_full, depth_low, conf_low, intrinsic_full, extrinsic, margin=8):
    """World-space centroid (mean of unprojected points) of a mask's covered pixels,
    in VGGT-O model space. Returns None if the mask is empty.
    INPUTS 
    mask - from YOLO at the moment 
    rgb_full - the image frame. needed to bilinear upsampe every mask pixel - can remove if it's too heavy
    depth_low - the depth map from 3d recon
    intrnsic full 
    extrinsics
  
    """
    world_points, _ = unproject_masked(mask, rgb_full, depth_low, conf_low, intrinsic_full, extrinsic, margin)#gets the positions of each 
    #value in the mas
    return world_points.mean(dim=0) if world_points is not None else None # does the average


def subject_model_positions(recon, masks_dir):
    '''For each frame in the recon, unprojects the YOLO mask centroid into VGGT-O models space.
    Returns dict {frame_idx: np.array([x,y,z])}, np.full(3, nan) where the mask is absent.
    masks_dir must point to the per-frame PNG masks produced by YOLO.
    '''
    subject_positions = {}
    for frame_idx in recon.frame_to_row:
        rgb, mask, depth, depth_conf, intrinsic, extrinsic, sigma, _ = load_frame_inputs(recon, frame_idx, masks_dir)
        pos = mask_centroid_model_position(mask, rgb, depth, depth_conf, intrinsic, extrinsic, margin=8)
        subject_positions[frame_idx] = pos.cpu().numpy() if pos is not None else np.full(3, np.nan)
    return subject_positions


def meeting_calculator(sub_real_dict, cam_poses_realworld_dict, lat_0, lon_0):
    '''Returns dict of info about when camera and subject were closest together -- i.e. the incident happened.
    Also appends that info straight to the report CSV.'''
    sub_frames = np.array(sorted(sub_real_dict.keys()))
    sub_pos    = np.array([sub_real_dict[f] for f in sub_frames])              # (N,3)

    cam_frames = np.array(sorted(cam_poses_realworld_dict.keys()))
    cam_pos    = np.array([cam_poses_realworld_dict[f] for f in cam_frames])   # (M,3)

    # each subject frame -> its nearest available camera frame (the two frame sets don't line up 1:1)
    nearest_idx        = np.array([np.argmin(np.abs(cam_frames - f)) for f in sub_frames])
    matched_cam_pos     = cam_pos[nearest_idx]
    matched_cam_frames  = cam_frames[nearest_idx]

    distances = np.linalg.norm(sub_pos - matched_cam_pos, axis=1)
    distances[np.isnan(distances)] = np.inf   # ignore frames with no valid subject mask

    meet_i         = np.argmin(distances)#index of meet
    meet_frame     = sub_frames[meet_i] #frame of meet
    meet_cam_frame = matched_cam_frames[meet_i] #cam frame of meet - may be different
    meet_lat_lon, _ = metres_to_latlong(matched_cam_pos[meet_i:meet_i + 1], lat_0, lon_0) # position of cam 

    meeting_report = {
        "meeting_frame"      : float(meet_frame), #frame subject closest to cam
        "meeting_cam_frame"  : float(meet_cam_frame), #frame cam closest to subject (they don't match)
        "meeting_lat_lon"    : tuple(float(v) for v in meet_lat_lon[0]),
        "{{meeting_distance_m}}" : float(distances[meet_i]),
        # needs video_path/fps (time-of-day) and multi-frame velocity (travel direction) -- not wired in yet
        "{{camera_direction}}"   : None,
    }
    add_to_report(meeting_report)
    return meeting_report


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


def subject_to_metric_and_gps_space(
            subject_recon,
            masks_dir,
            lat_0, lon_0,
            R_mw , s_mw,t_mw
           ):
    """Turn YOLO masks + VGGT-O subject recon into GPS positions.

    Projects mask centroids through model to world to get
    metres and lat/lon -- no separate RS-space hop needed since
    cam_poses_realworld_dict is already in real-world space.

    recon: an already-loaded Reconstruction (see E_post_recon_processing.Reconstruction.load) --
    shared with the other VGGT-O consumers (VGGT_O_preds_to_ply_export, projection_mapping_sequence)
    so predictions.npz is only read/moved to the GPU once.

    Returns list of (frame_number, lat, lon) for frames with a valid detection,
    the subject positions in metres (array and frame-keyed dict), and the
    R_cr/s_cr/t_cr alignment itself -- reusable to bring anything else in
    VGGT-O's raw cam0 space (e.g. VGGT_O_preds_to_ply_export's point clouds)
    into the same real-world metric frame.
     """     
    #subject centroids found and transformed into model space
    from P_projection_mapping import MASK_NAME_FMT

    frame_keys = sorted(subject_recon.frame_to_row, key=subject_recon.frame_to_row.get)  # row order
    print(frame_keys)
    intrinsics = subject_recon.preds["intrinsic"].cpu().numpy()   # (N,3,3) native res, matches depth
    extrinsics = subject_recon.preds["extrinsic"].cpu().numpy()   # (N,3,4) model-to-cam
    depths     = subject_recon.preds["depth"].cpu().numpy()       # (N,h,w) native res
    h, w       = depths.shape[1:]
    
    #load masks - make them model sized. frame 3880's mask file (and any other frame
    #missing/unreadable on disk -- no detection that frame) reads as None from cv2.imread;
    #treat that as an empty mask instead of letting cv2.resize crash on it.
    mask_imgs = [cv2.imread(str(Path(masks_dir) / MASK_NAME_FMT.format(f)), cv2.IMREAD_GRAYSCALE) for f in frame_keys]
    masks = np.stack([
        cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST) > 0 if img is not None else np.zeros((h, w), dtype=bool)
        for img in mask_imgs
    ])   # (N,h,w) bool, resized to the depth's own native resolution -- no bilateral upsampling


    ys, xs      = np.indices((h, w))
    mask_counts = masks.sum(axis=(1, 2))
    valid       = mask_counts > 0

    #pixel centroid of the mask, in depth-grid coordinates (NaN where mask is empty)
    col = np.full(len(frame_keys), np.nan)
    row = np.full(len(frame_keys), np.nan)
    col[valid] = (xs * masks).sum(axis=(1, 2))[valid] / mask_counts[valid]
    row[valid] = (ys * masks).sum(axis=(1, 2))[valid] / mask_counts[valid]

    #depth under the mask, averaged over the non-zero pixels
    depths_masked   = depths * masks
    nonzero_counts  = (depths_masked != 0).sum(axis=(1, 2))
    has_depth       = valid & (nonzero_counts > 0)
    z = np.full(len(frame_keys), np.nan)
    z[has_depth] = depths_masked.sum(axis=(1, 2))[has_depth] / nonzero_counts[has_depth]

    #seems repetive, but unproject needs a full depth map and only 1, while this does 
    #EVERY frame at once. 
    fx, fy = intrinsics[:, 0, 0], intrinsics[:, 1, 1]
    cx, cy = intrinsics[:, 0, 2], intrinsics[:, 1, 2]
    x = (col - cx) * z / fx
    y = (row - cy) * z / fy
    subject_centroids_cam = np.stack([x, y, z], axis=1)   # (N,3) in each frame's own camera space

    R = extrinsics[:, :3, :3]   # model-to-cam rotation, per frame
    t = extrinsics[:, :3, 3]
    #cam -> model space: P_model = R^T @ (P_cam - t)
    subject_centorid_model = np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), subject_centroids_cam - t)

    #moves straight from 3d-recon space into real-world frame of reference (metres, NESW)
    sub_real      = transform_RST(subject_centorid_model, R_mw, s_mw, t_mw)
    sub_real_dict = dict(zip(frame_keys, sub_real))   # frame-keyed, same pattern as cam_poses_realworld_dict
    sub_latlon, _ = metres_to_latlong(sub_real, lat_0, lon_0)
    valid = ~np.isnan(sub_latlon).any(axis=1)
    return [(float(frame_keys[i]), *sub_latlon[i])
                for i in range(len(frame_keys)) if valid[i]] , sub_real , sub_real_dict
    
def subject_reporting(sub_real_dict,cam_poses_realworld_dict,lat_0, lon_0  ):
###FOR THE REPORT##
    map_analysis    = meeting_calculator(sub_real_dict, cam_poses_realworld_dict, lat_0, lon_0)
    subject_movement = subject_direction(sub_real_dict)
    return
    

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