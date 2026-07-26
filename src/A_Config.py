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
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_CASE_NAME = None
_EXPERIMENT = None


def set_case(case_name, experiment=None):
    global _CASE_NAME, _EXPERIMENT
    _CASE_NAME = case_name
    _EXPERIMENT = experiment

def case_name(): return _CASE_NAME

def case_dir():
    assert _CASE_NAME is not None, "call set_case(case_name, experiment) first"
    return Path(REPO_ROOT) / "Data" / _CASE_NAME


def assets_dir():
    return case_dir() / "090_assets"


def asset_name():
    return f"{_CASE_NAME}_{_EXPERIMENT}" if _EXPERIMENT else _CASE_NAME


def report_path():
    return assets_dir() / f"report_{_CASE_NAME}_{_EXPERIMENT}.csv"


def source_dir(): return case_dir() / "010_source"
def query_dir(): return case_dir() / "012_Gemini_outputs"
def yolo_masks_dir(): return case_dir() / "015_YOLO"
def sam3_masks_dir(): return case_dir() / "015_SAM3_masks"
def frames_for_cam_poses_dir(): return case_dir() / "020_frames_for_cam_poses"
def frames_for_recon_dir(): return case_dir() / "020_frames_for_recon"
def vggt_o_output_dir(): return case_dir() / "035_VGGT_O_output"
def predictions_path(): return vggt_o_output_dir() / "predictions.npz"
def cut3r_output_dir(): return case_dir() / "035_CUT3R_output"
def ply_dir(): return vggt_o_output_dir() / "ply"
def reg_dir(): return case_dir() / "030_Registrations"
def csv_path_gps(): return case_dir() / "000_GPS_Data" / "GPS_tags.csv"
def csv_path_pose(): return reg_dir() / "camera_poses.csv"


FRAME_NAME_FMT = "{:04d}.jpg"


def to_report_path(path):
    '''Path to store in the report CSV: relative to the current case's
    assets_dir (short filename, or short subpath for files nested in a
    subfolder under it). W_populate_pptx.py resolves it back the same way,
    via assets_dir() -- also zero-argument, so nothing needs threading
    through either side.'''
    return str(Path(path).resolve().relative_to(assets_dir().resolve()))
