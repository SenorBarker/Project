"""
georeference.py — fit a local camera path to known geographic anchors.
Take objects from a scene and establish their real world location and movement based on world locations. 
"""
import math
import numpy as np
import csv as _csv
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

from Q_GPS_processing import csv_to_GPS_dict, android_movie_GPS
from Thr3D.F_post_recon_processing import build_frame_index, load_reality_scan_trace
from C_CSV_report import add_to_report, add_to_asset_list
from A_Config import assets_dir, asset_name, report_path, to_report_path, length_units
from Thr3D.G_transforms_alignments import umeyama_align, ortho_charts , transform_RST   , umeyama_align_anchor

#real-world lat long into metres, relative to the first item as origin (0,0,0).
#x ~ east-west, z ~ north-south, y ~ up (0 for every row if heights isn't given,
#i.e. treat the ground control as flat -- pass real elevations in heights to
#let the fit detect/correct genuine tilt instead of assuming it away).
def lat_long_to_metres_3dims(coords_list, heights=None):
    lat_0 = coords_list[0,0]
    lon_0 = coords_list[0,1]
    #set to origin and unpack
    dlatdlong = coords_list - coords_list[0]
    dlat = dlatdlong[:,0]
    dlon = dlatdlong[:,1]

    x = dlon *  (111000 * math.cos(math.radians(lat_0)))
    z = dlat * 111000
    if heights is None:
        y = np.zeros_like(x)
    else:
        heights = np.asarray(heights, dtype=float)
        y = heights - heights[0]
    return np.column_stack([x, y, z]), lat_0, lon_0
    

#---------get A B matches for GPS anchors
def GPS_camerapose_matcher(GPS_dict, cam_pos_dict, max_frame_gap=150):#150 is 5s 
    """Each GPS anchor picks its single nearest frame in cam_pos_dict. If two
    anchors pick the same frame, the closer one keeps it and the other is
    dropped (no fallback -- with more anchors than frames some are always
    unmatched, and any fallback frame already has its own closer claimant).
    Anchors with nothing within max_frame_gap are dropped too.
    """
    available_cams = sorted(cam_pos_dict.keys())

    nearest = {}  # frame_idx -> (gap, avail_frame)
    for frame_idx in GPS_dict:
        best_cam = min(available_cams, key=lambda a: abs(a - frame_idx))#closest matching camera pose key
        gap = abs(best_cam - frame_idx)##frames apart
        if gap <= max_frame_gap: #check it's allowed
            nearest[frame_idx] = (gap, best_cam) #nearest[gpsframe] = (gap, cam_frame)

    winners = {}  #
    for frame_idx, (gap, best_cam) in nearest.items(): #GPS index, distance, best cam index
        if best_cam not in winners or gap < winners[best_cam][0]: # add in the combo if the cam frame isnt in winners, or this gap is smaller
            winners[best_cam] = (gap, frame_idx)#keep gap for doing the above line
    matched = {frame_idx: cam for cam, (_, frame_idx) in winners.items()}#reverse back to GPS indices: camindices

    GPS_locs, object_locs = [], []
    # sorted, not insertion order: csv_to_GPS_dict builds GPS_dict straight from
    # CSV row order, so anything reading GPS_locs as a *path* (trajectory_length
    # in the cam-pose sweeps) would zigzag on an out-of-order row. umeyama_align
    # is order-invariant, so this is free for the alignment itself.
    for frame_idx, latlon in sorted(GPS_dict.items()):
        if frame_idx not in matched:
            print(f"GPS anchor at frame {frame_idx} dropped: out of range or lost to a closer anchor.")
            continue
        GPS_locs.append(latlon)
        object_locs.append(cam_pos_dict[matched[frame_idx]])

    GPS_locs = np.array(GPS_locs)
    GPS_locs, lat_0, lon_0 = lat_long_to_metres_3dims(GPS_locs, heights=None)
    object_locs = np.array(object_locs)
    assert len(GPS_locs) == len(object_locs)
    return GPS_locs, lat_0, lon_0, object_locs #both N,3.

#Data from above can be pumped into umeyama align



def _speed_direction_single(positions_dict, video_path, frame_times=None):
    #invented to run this separatly for each dict that is passed into the function below
    t = np.array(sorted(positions_dict.keys()))#time
    pos = np.array([positions_dict[k] for k in sorted(positions_dict.keys())])#position

    # a subject with no matched track has no positions -- NaNs, so it still
    # plots (as a gap) instead of taking the whole group down
    if pos.ndim != 2 or len(pos) < 3:
        nan = np.full(len(t), np.nan)
        return t, nan, nan, nan

    # t's values are real frame numbers -- true elapsed seconds between two
    # specific frames needs their real decoded timestamps, not a frame delta
    # divided by an average fps (wrong on VFR sources, where the true local
    # rate drifts from the average). frame_times, if given, is a batched
    # lookup the caller already made across every subject; otherwise resolve
    # it here for a standalone call.
    if frame_times is None:
        from Two2D.B_video_processing import frame_times_at_indices
        frame_times = frame_times_at_indices(video_path, t)
    t_seconds = np.array([frame_times[f] for f in t])

    t_0 = t_seconds[:-2]
    t_2 = t_seconds[2:]

    pos_0 = pos[:-2]
    pos_2 = pos[2:]

    delta_t = (t_2 - t_0) / 2
    delta_p = (pos_2 - pos_0) / 2
    heading = np.degrees(np.arctan2(delta_p[:, 0], delta_p[:, 2])) % 360 #East, North -- Up is irrelevant to heading
    delta_p = np.linalg.norm(delta_p, axis=1)
    mps = delta_p / delta_t  # delta_t is real elapsed seconds, so this is already metres/second

    kmh = mps /1000 * 60* 60 #km/h

    mps = np.pad(mps, (1, 1), constant_values=np.nan)
    kmh = np.pad(kmh, (1, 1), constant_values=np.nan)
    heading = np.pad(heading, (1, 1), constant_values=np.nan)

    return t, mps, kmh, heading


def speed_direction(positions_dicts, fps=None, video_path=None):
    '''positions_dicts is one of:
      - a single time->position dict -> returns one (mps, kmh, heading) tuple
      - a sequence of them, labelled camera, subject, subject_2... -> returns
        a plain list of (mps, kmh, heading) tuples, same order as the input
      - a {name: time->position dict} mapping, which keeps the real subject slugs
        -> returns a dict[name -> (mps, kmh, heading)], keyed by those same names
        instead of relying on the caller knowing the input dict's insertion order
    Each is run through the same speed/heading maths, then plotted together on
    shared axes. Saves the plots and appends to the report CSV for whichever case
    is currently active (see A_Config.set_case).

    fps is accepted but unused -- kept only so existing positional callers
    (speed_direction(positions, fps)) don't break. Speed is computed from
    real per-frame timestamps now, not fps arithmetic (inaccurate on VFR
    sources); video_path defaults to the current case's source video.'''
    named = None
    if isinstance(positions_dicts, dict):
        first = next(iter(positions_dicts.values()), None)
        if isinstance(first, dict):          # {name: positions} -- multi, keep the names
            named = list(positions_dicts)
            positions_dicts = list(positions_dicts.values())
            single = False
        else:                                # a bare positions dict
            positions_dicts = (positions_dicts,)
            single = True
    else:
        single = False

    if video_path is None:
        from A_Config import source_video_path
        video_path = source_video_path()

    labels = named or (["camera", "subject"] + [f"subject_{i}" for i in range(2, len(positions_dicts))])
    colors = plt.cm.tab10(np.linspace(0, 1, 10))[:len(positions_dicts)]

    # One batched real-timestamp lookup across every subject's frame numbers,
    # instead of a separate full-video pass (or frame/fps arithmetic) per subject.
    from Two2D.B_video_processing import frame_times_at_indices
    all_frames = sorted({f for pd_ in positions_dicts for f in pd_.keys()})
    frame_times = frame_times_at_indices(video_path, all_frames) if all_frames else {}

    results = [_speed_direction_single(pd_, video_path, frame_times=frame_times) for pd_ in positions_dicts]

    plt.figure()
    for (t, mps, kmh, heading), label, color in zip(results, labels, colors):
        plt.scatter(t, mps, color=color, label=label)
    plt.xlabel("time")
    plt.ylabel(f"velocity ({length_units()}/s)")
    plt.title("speed vs time")
    plt.legend()
    outpath = assets_dir() / f"{asset_name()}_speed_graph.png"
    plt.savefig(outpath)
    graph_path = to_report_path(outpath)

    plt.figure()
    ax = plt.subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1) #clockwise, to match compass heading
    for (t, mps, kmh, heading), label, color in zip(results, labels, colors):
        ax.scatter(np.radians(heading), t, color=color, label=label)
    ax.set_title("heading vs time")
    ax.legend()

    metrics = [(mps, kmh, heading) for (_, mps, kmh, heading) in results]

    #per-mover rows -- the ends are padded with NaN, and net heading is first-to-last
    #displacement, not the mean of the per-frame headings (which averages wrongly across
    #the 0/360 wrap). A mover with no matched track reaches here as an empty dict and an
    #empty mps, so every row below has to survive having nothing to average: nanmean/nanmax
    #warn on an all-NaN slice and nanmax is a hard error on a zero-size one, hence the
    #explicit finite-only selection rather than leaving it to NaN maths.
    speed_report = {"{{speed_units}}": f"{length_units()}/s"}
    for (t, mps, kmh, heading), label, pd_ in zip(results, labels, positions_dicts):
        #first-to-last of the FINITE rows -- a dict can start or end on NaN
        finite     = [f for f in sorted(pd_) if np.isfinite(np.asarray(pd_[f])).all()]
        net        = (np.asarray(pd_[finite[-1]]) - np.asarray(pd_[finite[0]])
                      if finite else np.full(3, np.nan))
        finite_mps = np.asarray(mps)[np.isfinite(mps)]
        speed_report[f"{{{{{label}_speed_mean}}}}"]   = float(finite_mps.mean()) if finite_mps.size else np.nan
        speed_report[f"{{{{{label}_speed_max}}}}"]    = float(finite_mps.max())  if finite_mps.size else np.nan
        speed_report[f"{{{{{label}_heading_net}}}}"]  = float(np.degrees(np.arctan2(net[0], net[2])) % 360)
        speed_report[f"{{{{{label}_distance_net}}}}"] = float(np.linalg.norm(net))
    add_to_report(speed_report)

    add_to_asset_list( {
        "speed_graph"  : graph_path
    })
        

    if single:
        return metrics[0]
    return dict(zip(named, metrics)) if named is not None else metrics



def graph_real_world(positions_dict):
    x_axis = positions_dict.keys()
    y_axis = positions_dict.values()
    return

#turn 3d world coordinates into lat lon (+ height). requires the lat lon (+
#height) of the first item in the list, matching whatever origin
#lat_long_to_metres_3dims was built with.
def metres_to_latlong (positions, lat_0, lon_0, height_0=0.0): #meas in rows
    lat = (positions[:,2] / 111000) + lat_0  #z is lat, z is north south
   
    lon = (positions[:,0] /(111000 * math.cos(math.radians(lat_0))))+lon_0  #x is east west
    
    height = positions[:,1] + height_0
 
    return np.column_stack([lat, lon]), height # keep height separate - not needed for mapping





#------CONTROL FUNCTION
def model_to_GPS_calibrated_locations (csv_path_GPS, poses, Source = "RS_path"):
    '''
    INPUTS: csv path GPS: file path to the GPS CSV. current formatting is frame, lat,lon, but will need converters for different GPS outputs
            csv_path_poses_RS: path to reality scan output will need convervion if we use another tool for this
        
    OUTPUTS: 
    All_poses_realworld :  CAMERA POSES in metres, with cam 0 roughly at 0,0,0,0,0 
    Geo-cords :            CAMERA POSES as lat lon 
    '''
    GPS_dict = csv_to_GPS_dict(csv_path_GPS)
    #loader for RS specific path
    if Source ==  "RS_path":
        _, cam_pose_dict = load_reality_scan_trace(poses)
    #or just use the dict as provided
    else:
        cam_pose_dict = poses

    #get matching poses - incoudes # frames to seek the match
    GPS_locs, lat_0, lon_0, cam_poses_filt = GPS_camerapose_matcher(GPS_dict, cam_pose_dict, max_frame_gap=150)
    # label_B = Source: reuse the string callers already pass in (previously only
    # used for the RS_path branch check above) as the chart label too, so the A/B
    # legend actually says which technique this is instead of generic "A"/"B".
    R, s, t, Cam_poses_metric, rmse = umeyama_align_anchor(GPS_locs, cam_poses_filt, label_A = "GPS", label_B = Source, out_path = None, with_scale=True, up_A=None, up_B=None, up_weight=None, skip_indices=None)

    #keep positions frame-keyed end-to-end -- never rely on cam_pose_dict's row order matching frame order
    frames_sorted = sorted(cam_pose_dict.keys())
    cam_poses_ordered = np.array([cam_pose_dict[f] for f in frames_sorted])
    cam_poses_realworld = transform_RST(cam_poses_ordered, R, s, t)
    cam_poses_realworld_dict = dict(zip(frames_sorted, cam_poses_realworld))
    #X + east - west   Z + north - south
    geo_latlon, height = metres_to_latlong(cam_poses_realworld, lat_0, lon_0)
    geo_cords = [(float(f), lat, lon) for f, (lat, lon) in zip(frames_sorted, geo_latlon)]
    return cam_poses_realworld_dict, geo_cords, R, s, t, lat_0, lon_0



