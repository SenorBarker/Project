"""
Ranks a torch CUDA memory snapshot (see A_LOCAL_SAM3.dump_memory_snapshot) by
which call site is holding the most *still-live* GPU bytes at the moment it
was captured -- this replaces guessing about what's accumulating with an
actual answer.

Usage: python analyze_mem_snapshot.py <path-to-mem_snapshot_*.pickle> [top_n]
"""
import pickle
import sys
from collections import defaultdict


def main(path, top_n=25):
    with open(path, "rb") as f:
        snapshot = pickle.load(f)

    # torch's C++ allocator frames carry no filename ("" or "??"), so filtering
    # those out isn't enough -- only *.py frames are real call sites, and the
    # generic nn.Module/contextlib wrapper frames aren't useful attributions.
    SKIP_BASENAMES = {"module.py", "_contextlib.py"}

    by_site = defaultdict(lambda: [0, 0])  # site -> [live_bytes, live_blocks]
    for segment in snapshot["segments"]:
        for block in segment["blocks"]:
            if block["state"] != "active_allocated":
                continue
            frames = block.get("frames") or []
            site = next(
                (f"{fr['filename']}:{fr['line']} ({fr['name']})" for fr in frames
                 if fr["filename"].endswith(".py")
                 and fr["filename"].rsplit("/", 1)[-1] not in SKIP_BASENAMES),
                "<no .py frame>",
            )
            by_site[site][0] += block["size"]
            by_site[site][1] += 1

    ranked = sorted(by_site.items(), key=lambda kv: -kv[1][0])
    total = sum(v[0] for v in by_site.values())
    print(f"total live: {total / 1024**3:.2f} GiB across {len(by_site)} call sites\n")
    for site, (nbytes, nblocks) in ranked[:top_n]:
        print(f"{nbytes / 1024**2:9.1f} MiB  ({nblocks:5d} blocks)  {site}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 25)
