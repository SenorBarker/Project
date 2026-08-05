#!/usr/bin/env python3
"""reserve_gpu.py -- hold a chunk of GPU VRAM so nobody else can grab it.

Allocates --gb of VRAM on --device and just sits there until you Ctrl+C (or
kill it), at which point the memory is freed immediately. Doesn't run any
compute, so it costs no power/cycles beyond holding the allocation -- it just
makes the card look full to anyone else's `nvidia-smi`/allocator.

Usage:
    python reserve_gpu.py                  # reserve 22GB on GPU 0
    python reserve_gpu.py --gb 20 --device 1
"""
import argparse
import signal
import sys
import time

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gb", type=float, default=22.0, help="VRAM to reserve, in GiB")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--heartbeat", type=int, default=60, help="seconds between status prints")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("no CUDA device visible -- nothing to reserve")

    device = torch.device(f"cuda:{args.device}")
    free, total = torch.cuda.mem_get_info(device)
    print(f"GPU {args.device}: {free/2**30:.1f} GiB free of {total/2**30:.1f} GiB total")

    n_bytes = int(args.gb * 2**30)
    if n_bytes >= free:
        sys.exit(f"asked for {args.gb:.1f} GiB but only {free/2**30:.1f} GiB free -- ask for less")

    # uint8 so n_elements == n_bytes exactly, no compute, just a dead allocation
    hog = torch.empty(n_bytes, dtype=torch.uint8, device=device)
    print(f"reserved {args.gb:.1f} GiB on GPU {args.device} -- holding until Ctrl+C / kill")

    stop = False

    def _release(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _release)
    signal.signal(signal.SIGTERM, _release)

    try:
        while not stop:
            time.sleep(args.heartbeat)
            free, total = torch.cuda.mem_get_info(device)
            print(f"still holding {args.gb:.1f} GiB -- {free/2**30:.1f} GiB free of {total/2**30:.1f} GiB", flush=True)
    finally:
        del hog
        torch.cuda.empty_cache()
        print("released reservation, exiting")


if __name__ == "__main__":
    main()
