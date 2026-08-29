'''REPORT CSV / ASSET LIST CSV
Accumulates analysis results into a key-value report CSV, and produced files
(images, videos, thumbnails, dirs...) into a separate asset list CSV -- kept
apart so the report reads as data/decisions for a viewer (movie or pptx),
not a dump of every asset the pipeline happened to render.
Any stage calls add_to_report(dict) for metrics, or add_to_asset_list(dict)
for produced files.
'''

import pandas as pd
import numpy as np

from A_Config import report_path, asset_list_path


def _add_rows(path, data: dict):
    new_rows = pd.DataFrame(data.items(), columns=["key", "value"])

    if path.exists():
        existing = pd.read_csv(path)
        merged = existing[~existing["key"].isin(new_rows["key"])]
        df = pd.concat([merged, new_rows], ignore_index=True)
    else:
        df = new_rows
    #save
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def add_to_report(data: dict):
    '''Append key-value pairs to the current case's report CSV (see
    A_Config.set_case). Creates the file if it doesn't exist; adds new rows
    for each key. Existing keys are overwritten. For metrics/decisions --
    see add_to_asset_list for produced files.
    '''
    _add_rows(report_path(), data)


def add_to_asset_list(data: dict):
    '''Append key-value pairs to the current case's asset list CSV (see
    A_Config.set_case). Same overwrite-by-key semantics as add_to_report,
    but for paths to produced files (images, videos, thumbnails, dirs...)
    rather than metrics.
    '''
    _add_rows(asset_list_path(), data)



def frames_to_time(frames, fps, timecode=pd.Timedelta(0)):
    '''INPUT  : frames   - frame index, or list/array of frame indices
                fps      - video frame rate (frames per second)
                timecode - pandas Timestamp marking the video's absolute recording start;
                           omit for elapsed time only
       OUTPUT : "HH:MM:SS" string, or list of strings, rounded to the nearest second'''
    seconds = np.asarray(frames) / fps
    times = (timecode + pd.to_timedelta(seconds, unit='s')).round('s')

    #format to remove days and make seconds the smallest unit of time
    def _fmt(t):
        if isinstance(t, pd.Timestamp):
            return t.strftime('%H:%M:%S')
        total = int(t.total_seconds())
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    if np.ndim(frames) == 0:
        return _fmt(times)
    return [_fmt(t) for t in times]#apply format to all the times

from Two2D.B_video_processing import video_start_time
def attach_times_of_day(video_path, fps):
    '''For every "key" already in the current case's report CSV (see
    A_Config.set_case) ending in "_frame", append a matching "<key>_time"
    entry converted via frames_to_time. Leaves everything else untouched --
    a report holds plenty of values (lat/lon, distances, flags...) that
    aren't frame numbers.'''
    path = report_path()
    existing = pd.read_csv(path)
    try:
        vid_start = video_start_time(video_path)
    except ValueError as e:
        print(f"skipping attach_times_of_day: {e}")
        return
    add_to_report({"{{Video_start_time}}": vid_start})

    frame_rows = existing[existing["key"].str.endswith("_frame")]
    new_rows = {
        f"{{{{{key.removesuffix('_frame')}_time}}}}": frames_to_time(float(value), fps, timecode=vid_start)
        for key, value in zip(frame_rows["key"], frame_rows["value"])
        if pd.notna(value)   #a subject that never matched a track (e.g. NO MATCH in
                              #cell 22) has no first/last frame -- an empty CSV cell,
                              #not a real 0 -- so there's no frame to convert to a time
    }
    add_to_report(new_rows)
