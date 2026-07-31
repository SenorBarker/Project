import time, glob, threading
from pathlib import Path
import numpy as np
import viser
from plyfile import PlyData


def load_ply(path):
    plydata = PlyData.read(path)
    v = plydata['vertex'].data
    points = np.vstack([v['x'], v['y'], v['z']]).T.astype(np.float32)
    colors = np.vstack([v['red'], v['green'], v['blue']]).T.astype(np.uint8)
    return points, colors


def visualise_cut3r(output_dir):
    PLY_DIR = Path(output_dir) / "ply"

    files = sorted(PLY_DIR.glob("*.ply"))
    n_frames = len(files)
    print(f"Found {n_frames} PLY files")
    if n_frames == 0:
        raise FileNotFoundError(f"No PLY files in {PLY_DIR}")

    server = viser.ViserServer()
    server.scene.set_up_direction("+y")

    playing       = server.gui.add_checkbox("Playing",    initial_value=False)
    frame_slider  = server.gui.add_slider("Frame",        min=0, max=n_frames-1, step=1,   initial_value=0)
    fps_slider    = server.gui.add_slider("FPS",          min=1, max=30,         step=1,   initial_value=10)
    window_slider = server.gui.add_slider("Accumulate",   min=1, max=160,        step=1,   initial_value=1)
    btn_m100 = server.gui.add_button("-100")
    btn_m30  = server.gui.add_button("-30")
    btn_p30  = server.gui.add_button("+30")
    btn_p100 = server.gui.add_button("+100")
    point_size_slider = server.gui.add_slider("Point size", min=0.000001, max=0.02, step=0.0001, initial_value=0.0005)

    state = {"idx": 0}

    def clamp(v): return max(0, min(n_frames - 1, v))
    def jump(delta):
        state["idx"] = clamp(state["idx"] + delta)
        frame_slider.value = state["idx"]

    @frame_slider.on_update
    def _(_): state["idx"] = frame_slider.value
    @btn_m100.on_click
    def _(_): jump(-100)
    @btn_m30.on_click
    def _(_): jump(-30)
    @btn_p30.on_click
    def _(_): jump(+30)
    @btn_p100.on_click
    def _(_): jump(+100)

    cache   = {}   # frame_idx -> (points, colors)
    handles = {}   # frame_idx -> viser handle
    active  = set()

    def playback_loop():
        last_idx = -1
        last_win = -1
        while True:
            try:
                idx = state["idx"]
                win = window_slider.value
                if idx != last_idx or win != last_win:
                    desired = set(range(max(0, idx - win + 1), idx + 1))

                    for fi in list(active - desired):
                        h = handles.pop(fi, None)
                        active.discard(fi)
                        cache.pop(fi, None)
                        if h is not None:
                            try:
                                h.remove()
                            except Exception:
                                pass

                    for fi in sorted(desired - active):
                        if fi not in cache:
                            cache[fi] = load_ply(files[fi])
                        pts, col = cache[fi]
                        handles[fi] = server.scene.add_point_cloud(
                            f"/frames/{fi}/pc", points=pts, colors=col, point_size=point_size_slider.value, point_shape="circle"
                        )
                        active.add(fi)

                    last_idx = idx
                    last_win = win

                if playing.value:
                    state["idx"] = (idx + 1) % n_frames
                    frame_slider.value = state["idx"]
            except Exception as e:
                print(f"[error @ {state['idx']}] {e}")
            time.sleep(1.0 / fps_slider.value)

    threading.Thread(target=playback_loop, daemon=True).start()
