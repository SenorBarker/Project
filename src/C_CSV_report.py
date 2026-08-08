'''REPORT CSV
Accumulates analysis results into a single key-value CSV report.
Any stage calls add_to_report(dict) to append its metrics.
'''

import pandas as pd
import numpy as np

from A_Config import report_path


def add_to_report(data: dict):
    '''Append key-value pairs to the current case's report CSV (see
    A_Config.set_case). Creates the file if it doesn't exist; adds new rows
    for each key. Existing keys are overwritten.
    '''
    path = report_path()
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
    vid_start = video_start_time(video_path)
    add_to_report({"{{Video_start_time}}": vid_start})

    frame_rows = existing[existing["key"].str.endswith("_frame")]
    new_rows = {
        f"{{{{{key.removesuffix('_frame')}_time}}}}": frames_to_time(float(value), fps, timecode=vid_start)
        for key, value in zip(frame_rows["key"], frame_rows["value"])
    }
    add_to_report(new_rows)
