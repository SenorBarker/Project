"""B_lingbot_map: run the LingBot-Map interactive demo (models/lingbot-map) in the Msc2 env.

Thin wrapper around the upstream demo.py that bakes in the defaults for this machine:

- checkpoint: models/lingbot-map/checkpoints/lingbot-map-long.pt (downloaded from
  huggingface robbyant/lingbot-map)
- scene: example/courthouse with --mask_sky (the README quick-start command)
- --use_sdpa is always forced: flashinfer isn't installed in msc2-image.sif, and
  every real target machine runs this same container on a 24GB-class NVIDIA card
  (a prior attempt to also run this on MOana's older GPU never worked and left
  dead config lying around -- not a machine this wrapper needs to support).
- --offload_to_cpu: per-frame predictions go to CPU during inference to keep the
  peak GPU memory down on smaller cards

Any demo.py flag can be passed through and overrides these defaults, e.g.:

    python src/B_lingbot_map.py                                  # courthouse quick-start
    python src/B_lingbot_map.py --image_folder example/loop      # another example scene
    python src/B_lingbot_map.py --first_k 30                     # quick smoke test
    python src/B_lingbot_map.py --model_path /path/to/other.pt --image_folder /path/to/imgs

Run inside the Msc2 environment (conda activate Msc2, or
/opt/conda/envs/Msc2/bin/python src/B_lingbot_map.py). The viser viewer serves at
http://localhost:8080 once inference finishes.
"""

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

from A_Config import lingbot_map_dir, frames_for_recon_dir, frames_for_cam_poses_dir

LINGBOT_ROOT = Path(__file__).resolve().parents[2] / "Models" / "lingbot-map"
DEFAULT_CKPT = LINGBOT_ROOT / "checkpoints" / "lingbot-map-long.pt"


def _import_lingbot_demo():
    """Load Models/lingbot-map/demo.py by file path rather than `import demo` --
    Models/CUT3R/demo.py also imports under the bare name "demo", and whichever
    one is imported first wins sys.modules["demo"] for the rest of the kernel
    session (e.g. running the CUT3R cell before this one in a notebook silently
    hands this module CUT3R's demo, which has no build_parser/load_images/etc)."""
    spec = importlib.util.spec_from_file_location("lingbot_map_demo", LINGBOT_ROOT / "demo.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["lingbot_map_demo"] = module
    spec.loader.exec_module(module)
    return module


# --use_sdpa is always added: flashinfer isn't installed in msc2-image.sif, and
# every real target machine runs this same container/env (24GB-class NVIDIA card).
# The MOana attempt (different, older GPU) never worked and is dead; don't build
# per-machine detection for a machine that isn't actually a target.


def main():
    # Relative paths (example/..., skyseg.onnx cache) resolve against the lingbot-map
    # repo root, so the README commands work verbatim from anywhere.
    os.chdir(LINGBOT_ROOT)
    sys.path.insert(0, str(LINGBOT_ROOT))

    user_args = sys.argv[1:]
    defaults = []
    if "--model_path" not in user_args:
        defaults += ["--model_path", str(DEFAULT_CKPT)]
    if "--image_folder" not in user_args and "--video_path" not in user_args:
        defaults += ["--image_folder", "example/courthouse"]
        if "--mask_sky" not in user_args:
            defaults += ["--mask_sky"]
    if "--use_sdpa" not in user_args:
        defaults += ["--use_sdpa"]
    if "--offload_to_cpu" not in user_args and "--no-offload_to_cpu" not in user_args:
        defaults += ["--offload_to_cpu"]
    sys.argv = [sys.argv[0]] + defaults + user_args

    # demo.py inspects sys.argv and sets PYTORCH_CUDA_ALLOC_CONF at import time,
    # before importing torch — so sys.argv must be final before this import.
    demo = _import_lingbot_demo()
    demo.main()


if __name__ == "__main__":
    main()


# =============================================================================
# Headless pipeline stage
# =============================================================================
#
# Output layout: one {frame_num:04d}.npz per frame under lingbot_map_dir(), keyed
# by the *source* frame number (not sequence position) -- so a caller can look up
# a specific frame directly instead of needing a Reconstruction-style frame->row
# index. Each file holds that frame's depth, depth_conf, extrinsic_w2c, intrinsic,
# and world_points/world_points_conf *if the model produced them* -- GCTStream's
# streaming mode defaults enable_point=False, and DEFAULT_CKPT (lingbot-map-long.pt)
# ships no point_head weights at all, so in the current default config there is no
# world_points to save; the keys are only written when present, rather than assumed.
# `images` is intentionally omitted -- it's already on disk as the source frame
# under frames_for_recon_dir().
#
# extrinsic_w2c is predictions["extrinsic"] straight out of demo.py's postprocess(),
# despite that function's own docstring calling it c2w -- verified against real
# output (load_lingbot_map_trace's viser cross-check, see F_post_recon_processing.py):
# treating it as world-to-camera (position = -R^T @ t, same as VGGT_O/plain-VGGT)
# gives a trajectory matching the reconstruction; taking its translation column
# directly, as true c2w would allow, does not.

def _frame_npz_path(out_dir, frame_num):
    return Path(out_dir) / f"{frame_num:04d}.npz"


def save_lingbot_frame(out_dir, frame_num, **arrays):
    np.savez(_frame_npz_path(out_dir, frame_num), **arrays)


def load_lingbot_frame(out_dir, frame_num):
    with np.load(_frame_npz_path(out_dir, frame_num)) as d:
        return dict(d)


def run_lingbot_map(mode="streaming", first_k=None, stride=1,
                     launch_viewer=False, port=8080, **demo_kwargs):
    """Run lingbot-map headlessly over frames_for_recon_dir(), saving one per-frame
    npz per frame to lingbot_map_dir(). No CSV report logging -- this stage produces
    intermediate 2D->3D data, not a final reportable asset.

    Any other demo.py flag can be passed through via demo_kwargs, e.g.
    run_lingbot_map(camera_num_iterations=1) for a faster/lower-quality pass.
    Set launch_viewer=True to also open the interactive viser viewer afterwards
    (blocks until closed).
    """
    # Relative paths (skyseg.onnx cache, etc.) resolve against the lingbot-map repo
    # root, same reason main() does this.
    os.chdir(LINGBOT_ROOT)
    sys.path.insert(0, str(LINGBOT_ROOT))
    in_path = frames_for_cam_poses_dir()
    #in_path = frames_for_recon_dir()
    out_path = lingbot_map_dir()
    out_path.mkdir(parents=True, exist_ok=True)

    argv = ["--model_path", str(DEFAULT_CKPT), "--image_folder", str(in_path),
            "--offload_to_cpu", "--mode", mode]
    if first_k is not None:
        argv += ["--first_k", str(first_k)]
    if stride != 1:
        argv += ["--stride", str(stride)]
    if "use_sdpa" not in demo_kwargs:
        argv += ["--use_sdpa"]
    for k, v in demo_kwargs.items():
        flag = f"--{k}"
        if v is True:
            argv += [flag]
        elif v is not False:
            argv += [flag, str(v)]

    demo = _import_lingbot_demo()
    import torch

    args = demo.build_parser().parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images, paths, resolved_image_folder = demo.load_images(
        image_folder=args.image_folder, video_path=args.video_path,
        fps=args.fps, first_k=args.first_k, stride=args.stride,
        image_size=args.image_size, patch_size=args.patch_size,
        rotate_clockwise_90=args.rotate_clockwise_90,
    )
    model = demo.load_model(args, device)
    predictions, images_cpu = demo.run_inference(args, model, images, device)

    frame_nums = []
    for i, p in enumerate(paths):
        frame_num = int(Path(p).stem)
        frame_nums.append(frame_num)
        arrays = {
            "depth": predictions["depth"][i].numpy(),
            "depth_conf": predictions["depth_conf"][i].numpy(),
            "extrinsic_w2c": predictions["extrinsic"][i].numpy(),
            "intrinsic": predictions["intrinsic"][i].numpy(),
        }
        if "world_points" in predictions:
            arrays["world_points"] = predictions["world_points"][i].numpy()
        if "world_points_conf" in predictions:
            arrays["world_points_conf"] = predictions["world_points_conf"][i].numpy()
        save_lingbot_frame(out_path, frame_num, **arrays)

    if launch_viewer:
        from lingbot_map.vis import PointCloudViewer
        viewer = PointCloudViewer(
            pred_dict=demo.prepare_for_visualization(predictions, images_cpu),
            port=port,
            vis_threshold=args.conf_threshold,
            downsample_factor=args.downsample_factor,
            point_size=args.point_size,
            mask_sky=args.mask_sky,
            image_folder=resolved_image_folder,
            sky_mask_dir=args.sky_mask_dir,
            sky_mask_visualization_dir=args.sky_mask_visualization_dir,
        )
        print(f"3D viewer at http://localhost:{port}")
        viewer.run()

    return {"output_dir": out_path, "frame_nums": frame_nums}