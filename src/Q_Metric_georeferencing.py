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
from F_post_recon_processing import build_frame_index, load_reality_scan_trace
from C_CSV_report import add_to_report
from A_Config import assets_dir, asset_name, report_path, to_report_path
from G_transforms_alignments import umeyama_align, ortho_charts , transform_RST   

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
def GPS_camerapose_matcher(GPS_dict, world_pos_dict, max_frame_gap=150):
    """Match each GPS anchor to the nearest frame actually present in
    world_pos_dict (image_sequencer keeps the sharpest frame per sample point,
    not every Nth frame, so the GPS anchor's own frame# is rarely a key).
    Anchors whose nearest available frame is more than max_frame_gap frames
    away are dropped - at typical walking pace that's still well inside GPS's
    own ~10m error radius, so closer matches aren't worth enforcing tighter.
    Outputs both in xyz, and the starting lat and lon we need for transforms in the future
    """
    available = np.array(sorted(world_pos_dict.keys()))
    GPS_locs, object_locs = [], []
    for frame_idx, latlon in GPS_dict.items():
        nearest = available[np.argmin(np.abs(available - frame_idx))] #smallest difference between available poses and the gps index
        gap = abs(int(nearest) - frame_idx)
        if gap > max_frame_gap:
            print(f"GPS anchor at frame {frame_idx} dropped: nearest reconstructed "
                  f"frame {nearest} is {gap} frames away (limit {max_frame_gap}).")
            continue
        GPS_locs.append(latlon)
        object_locs.append(world_pos_dict[nearest])

    GPS_locs = np.array(GPS_locs)
    GPS_locs, lat_0, lon_0 = lat_long_to_metres_3dims(GPS_locs, heights=None)
    object_locs = np.array(object_locs)
    assert len(GPS_locs) == len(object_locs)
    return GPS_locs, lat_0, lon_0, object_locs #both N,3. 

#Data from above can be pumped into umeyama align



def _speed_direction_single(positions_dict, fps):
    #invented to run this separatly for each dict that is passed into the function below
    t = np.array(sorted(positions_dict.keys()))#time
    pos = np.array([positions_dict[k] for k in sorted(positions_dict.keys())])#position

    t_0 = t[:-2]
    t_2 = t[2:]

    pos_0 = pos[:-2]
    pos_2 = pos[2:]

    delta_t = (t_2 - t_0) / 2
    delta_p = (pos_2 - pos_0) / 2
    heading = np.degrees(np.arctan2(delta_p[:, 0], delta_p[:, 2])) % 360 #East, North -- Up is irrelevant to heading
    delta_p = np.linalg.norm(delta_p, axis=1)
    vel = delta_p / delta_t  #metres per frame
    mps = vel * fps #metres/second
    kmh = mps /1000 * 60* 60 #km/h

    mps = np.pad(mps, (1, 1), constant_values=np.nan)
    kmh = np.pad(kmh, (1, 1), constant_values=np.nan)
    heading = np.pad(heading, (1, 1), constant_values=np.nan)

    return t, mps, kmh, heading


def speed_direction(positions_dicts, fps):
    '''positions_dicts is a single time->position dict, or a (camera, subject)
    pair of them -- either way each is run through the same speed/heading
    maths, then plotted together on shared axes (camera blue, subject red).
    Saves the plots and appends to the report CSV for whichever case is
    currently active (see A_Config.set_case).'''
    single = isinstance(positions_dicts, dict)
    if single:
        positions_dicts = (positions_dicts,)

    labels = ["camera", "subject"]
    colors = ["tab:blue", "tab:red"]
    results = [_speed_direction_single(pd_, fps) for pd_ in positions_dicts]

    plt.figure()
    for (t, mps, kmh, heading), label, color in zip(results, labels, colors):
        plt.scatter(t, mps, color=color, label=label)
    plt.xlabel("time")
    plt.ylabel("velocity (m/s)")
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
    add_to_report( {
        "speed_graph"  : graph_path
    })
        

    return metrics[0] if single else metrics



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
def model_to_GPS_calibrated_locations (csv_path_GPS, csv_path_poses_RS):
    '''
    INPUTS: csv path GPS: file path to the GPS CSV. current formatting is frame, lat,lon, but will need converters for different GPS outputs
            csv_path_poses_RS: path to reality scan output will need convervion if we use another tool for this
        
    OUTPUTS: 
    All_poses_realworld :  CAMERA POSES in metres, with cam 0 roughly at 0,0,0,0,0 
    Geo-cords :            CAMERA POSES as lat lon 
    '''
    GPS_dict = csv_to_GPS_dict(csv_path_GPS)
    _, cam_pose_dict = load_reality_scan_trace(csv_path_poses_RS)

    GPS_locs, lat_0, lon_0, cam_poses_filt = GPS_camerapose_matcher(GPS_dict, cam_pose_dict, max_frame_gap=150)
    R, s, t, Cam_poses_metric, rmse = umeyama_align(GPS_locs, cam_poses_filt, label_A = "A", label_B = "B", out_path = None, with_scale=True, anchor_index=None, up_A=None, up_B=None, up_weight=None, skip_indices=None)

    #keep positions frame-keyed end-to-end -- never rely on cam_pose_dict's row order matching frame order
    frames_sorted = sorted(cam_pose_dict.keys())
    cam_poses_ordered = np.array([cam_pose_dict[f] for f in frames_sorted])
    cam_poses_realworld = transform_RST(cam_poses_ordered, R, s, t)
    cam_poses_realworld_dict = dict(zip(frames_sorted, cam_poses_realworld))
    #X + east - west   Z + north - south
    geo_latlon, height = metres_to_latlong(cam_poses_realworld, lat_0, lon_0)
    geo_cords = [(float(f), lat, lon) for f, (lat, lon) in zip(frames_sorted, geo_latlon)]
    return cam_poses_realworld_dict, geo_cords, R, s, t, lat_0, lon_0



