"""
A_Config.py -- current case/experiment identity, and the fixed folder layout
derived from it.

The directory structure under REPO_ROOT is fixed (Data/<case_name>/090_assets/...),
so the only thing that ever changes between runs is which case/experiment is
active. Call set_case(...) once per session (e.g. the notebook's config cell);
every other function that used to take assets_dir/asset_name/report_path as
parameters now reads them from here instead.
"""
import os
import sys
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SRC_DIR = Path(REPO_ROOT) / "src"
CLAUDE_AGENT_DIR = SRC_DIR / "claude_agent"

# Importing A_Config puts every topic folder under src/ on sys.path, so no cell has to patch it.
_ON_PATH = [
    SRC_DIR,
    CLAUDE_AGENT_DIR,
    SRC_DIR / "Two2D",
    SRC_DIR / "Thr3D",
    SRC_DIR / "Rendering",
    SRC_DIR / "run_models",
    SRC_DIR / "Experiments" / "templates",
]
for _p in _ON_PATH:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_CASE_NAME = None
_EXPERIMENT = None
# Guarded, not a plain assignment: %autoreload 2 re-execs this module in place
# whenever the file's mtime changes, so a bare `_PASS = 1` here would silently
# stomp set_pass()'s value back to 1 on the next reload after an edit -- this
# only seeds it the first time the module namespace has no _PASS yet.
if "_PASS" not in globals():
    _PASS = 1


def set_case(case_name, experiment=None):
    global _CASE_NAME, _EXPERIMENT
    _CASE_NAME = case_name
    _EXPERIMENT = experiment

def case_name(): return _CASE_NAME

def experiment(): return _EXPERIMENT

def set_pass(n):
    """Bump this in the notebook each time you hand the Producer feedback.
    Every paper-edit/draft file written after that gets '-{n}' on its name,
    so passes accumulate on disk instead of overwriting each other."""
    global _PASS
    _PASS = n

def pass_num(): return _PASS

def case_dir():
    assert _CASE_NAME is not None, "call set_case(case_name, experiment) first"
    return Path(REPO_ROOT) / "Data" / _CASE_NAME


def assets_dir():
    return _with_experiment(case_dir() / "090_assets")


def asset_name():
    return f"{_CASE_NAME}_{_EXPERIMENT}" if _EXPERIMENT else _CASE_NAME


def report_path():
    return assets_dir() / f"report_{_CASE_NAME}_{_EXPERIMENT}.csv"


def asset_list_path():
    return assets_dir() / f"asset_list_{_CASE_NAME}_{_EXPERIMENT}.csv"


def _with_experiment(base):
    '''Append the active experiment subfolder if one is set, else use base as-is --
    some cases have multiple experiments' worth of data side by side under the same
    stage folder (e.g. multiple frame sets under 020_frames_for_recon), others don't.'''
    return base / _EXPERIMENT if _EXPERIMENT else base

#make the directories either case_dir / XXX_thing, or case_dir / XXX_thing / experiment
def source_dir(): return case_dir() / "010_source"
def source_video_path(): return next(source_dir().glob("*.mp4"))
def query_dir(): return case_dir() / "012_Gemini_outputs"
def asr_dir(): return case_dir() / "013_ASR_outputs"

# The ASR stage reads the source video and nothing else, so its outputs belong to
# the case, not to an experiment -- named with case_name() rather than
# asset_name() so every experiment on a case shares the one transcript instead of
# paying for whisper+pyannote+AST again and leaving a duplicate set behind.
def asr_transcript_path(): return asr_dir() / f"{case_name()}_transcript_asr.txt"
def agent_p_output_dir(): return _with_experiment(case_dir() / "012_agent_p_output")
def yolo_masks_dir(): return _with_experiment(case_dir() / "015_YOLO")
def sam3_masks_dir(): return _with_experiment(case_dir() / "015_SAM3_masks")
def _beat(base, beat_key):
    '''Append the beat's own subfolder to a recon stage folder. Each beat that asks
    for a recon gets its own frames/predictions rather than overwriting the last one
    (see render_paper_edit.build_recon_requests). beat_key is beat["beat_id"][0],
    the same key D_2d_analysis and W_video_editor already use. None keeps the
    pre-multi-recon path, so zero-arg callers are unaffected.'''
    return base / beat_key if beat_key else base

def frames_for_cam_poses_dir(beat_key=None): return _beat(_with_experiment(case_dir() / "020_frames_for_cam_poses"), beat_key)
def frames_for_recon_dir(beat_key=None): return _beat(_with_experiment(case_dir() / "020_frames_for_recon"), beat_key)
def recon_for_MAP_dir(beat_key=None): return _beat(_with_experiment(case_dir() / "031_Recon_MAP"), beat_key)
def recon_for_EVENT_dir(beat_key=None): return _beat(_with_experiment(case_dir() / "031_Recon_EVENT"), beat_key)


def mega_SAM_output_dir(): return _with_experiment(case_dir() / "036_MEGASAM_output") 
def vggt_o_output_dir(): return _with_experiment(case_dir() / "035_VGGT_O_output")
def vggt_output_dir(): return _with_experiment(case_dir() / "034_VGGT_output")
def predictions_path(): return vggt_o_output_dir() / "predictions.npz"
def cut3r_output_dir(): return _with_experiment(case_dir() / "035_CUT3R_output")
def lingbot_map_dir(): return _with_experiment(case_dir() / "036_lingbot_map_output")
def ply_dir(): return vggt_o_output_dir() / "ply"  # already experiment-scoped via vggt_o_output_dir()
def reg_dir(): return _with_experiment(case_dir() / "030_Registrations")
def gps_data_dir(): return case_dir() / "000_GPS_Data"
def csv_path_gps(): return gps_data_dir() / "GPS_tags.csv"
def csv_path_pose(): return reg_dir() / "camera_poses.csv"


def has_gps_data():
    '''Whether this case has real-world GPS to scale against -- the video's own metadata
    isn't enough (a single tag can't align anything), so the truth is a GPS csv/json
    sitting in the GPS folder, hand-measured or otherwise.'''
    d = gps_data_dir()
    return d.exists() and (any(d.glob("*.csv")) or any(d.glob("*.json")))


def length_units(): return "m" if has_gps_data() else "model units"


FRAME_NAME_FMT = "{:04d}.jpg"
MASK_NAME_FMT  = "{:04d}.png"   # also copy-pasted in P_projection_mapping, A_YOLO_seg, A_ROBOFLOW_SAM3, W_video_editor

# Seconds tracked either side of a real beat's search window, so SAM3 has room to
# catch the subject entering/leaving rather than starting mid-presence. Nothing to
# do with cut handles -- it buys an onset to cut against, it doesn't place the cut.
MASK_SEARCH_HANDLE_S = 2.5


def to_report_path(path):
    '''Path to store in the report CSV: relative to the current case's
    assets_dir (short filename, or short subpath for files nested in a
    subfolder under it). W_populate_pptx.py resolves it back the same way,
    via assets_dir() -- also zero-argument, so nothing needs threading
    through either side.'''
    return str(Path(path).resolve().relative_to(assets_dir().resolve()))


def to_repo_path(path):
    '''Path as the Claude agent sees it: relative to REPO_ROOT, forward slashes.
    The SDK resolves every relative tool path against the project root it finds
    by walking up from cwd, so this is what goes in a prompt's OUTPUT_DIR.'''
    return Path(path).resolve().relative_to(Path(REPO_ROOT).resolve()).as_posix()
