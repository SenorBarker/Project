"""B_lingbot_map: run the LingBot-Map interactive demo (models/lingbot-map) in the Msc2 env.

Thin wrapper around the upstream demo.py that bakes in the defaults for this machine:

- checkpoint: models/lingbot-map/checkpoints/lingbot-map-long.pt (downloaded from
  huggingface robbyant/lingbot-map)
- scene: example/courthouse with --mask_sky (the README quick-start command)
- --use_sdpa is forced: FlashInfer requires Ampere or newer, and this GPU is a
  TITAN X (Pascal, sm_61), so the model must run on the PyTorch SDPA backend
- --offload_to_cpu: per-frame predictions go to CPU during inference to keep the
  peak under the 12 GB of the TITAN X

Any demo.py flag can be passed through and overrides these defaults, e.g.:

    python src/B_lingbot_map.py                                  # courthouse quick-start
    python src/B_lingbot_map.py --image_folder example/loop      # another example scene
    python src/B_lingbot_map.py --first_k 30                     # quick smoke test
    python src/B_lingbot_map.py --model_path /path/to/other.pt --image_folder /path/to/imgs

Run inside the Msc2 environment (conda activate Msc2, or
/opt/conda/envs/Msc2/bin/python src/B_lingbot_map.py). The viser viewer serves at
http://localhost:8080 once inference finishes.
"""

import os
import sys
from pathlib import Path

LINGBOT_ROOT = Path(__file__).resolve().parents[1] / "models" / "lingbot-map"
DEFAULT_CKPT = LINGBOT_ROOT / "checkpoints" / "lingbot-map-long.pt"


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
    import demo
    demo.main()


if __name__ == "__main__":
    main()
