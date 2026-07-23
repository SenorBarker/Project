import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def visualise_megasam(npz_path, n_samples=6):
    """
    Two-panel visualisation of a MegaSaM {scene}_sgd_cvd_hr.npz output.

    1. RGB / depth pairs for n_samples evenly-spaced frames.
    2. Camera trajectory (top-down X vs Z).

    Args:
        npz_path  : path-like — {scene}_sgd_cvd_hr.npz
                    (keys: images, depths, intrinsic, cam_c2w)
        n_samples : int — how many frames to show in the grid
    """
    d = np.load(str(Path(npz_path)))
    images, depths, cam_c2w = d["images"], d["depths"], d["cam_c2w"]
    n_total = len(images)

    indices = np.linspace(0, n_total - 1, min(n_samples, n_total), dtype=int)

    fig, axes = plt.subplots(2, len(indices), figsize=(3 * len(indices), 6))
    for col, idx in enumerate(indices):
        axes[0, col].imshow(images[idx])
        axes[0, col].axis("off")
        axes[0, col].set_title(f"{idx}", fontsize=8)

        im = axes[1, col].imshow(depths[idx], cmap="turbo_r")
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("rgb", fontsize=8)
    axes[1, 0].set_ylabel("depth", fontsize=8)
    fig.suptitle(f"MegaSaM  —  {n_total} frames  (showing {len(indices)})", fontsize=10)
    plt.colorbar(im, ax=axes[1, :].tolist(), fraction=0.02)
    plt.tight_layout()
    plt.show()

    positions = cam_c2w[:, :3, 3]
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(positions[:, 0], positions[:, 2], c=np.arange(n_total),
                     cmap="viridis", s=10, zorder=3)
    ax.plot(positions[:, 0], positions[:, 2], color="grey", linewidth=0.5,
            alpha=0.5, zorder=2)
    ax.scatter(*positions[0, [0, 2]],  color="green", s=50, zorder=4, label="start")
    ax.scatter(*positions[-1, [0, 2]], color="red",   s=50, zorder=4, label="end")
    plt.colorbar(sc, ax=ax, label="frame index")
    ax.set_title(f"Camera trajectory (X vs Z)  ({n_total} frames)", fontsize=10)
    ax.set_xlabel("X"); ax.set_ylabel("Z")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()
