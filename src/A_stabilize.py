"""
Video Painter — canvas-based video stabilization with similarity transforms.

The camera paints onto a large world canvas as it moves and rotates.
The output viewport follows the smoothed camera path (translation + rotation).
No black borders: border pixels show whatever was last painted by an earlier frame.

Usage:
    python stabilize.py input.mp4 output.mp4 [--smooth 30]
"""

import argparse
import sys
import cv2
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------

def estimate_similarity(gray1, gray2):
    """
    Estimate similarity transform (translation + rotation + uniform scale)
    mapping gray1 feature points into gray2. Returns 3x3 matrix or None.
    """
    corners = cv2.goodFeaturesToTrack(gray1, maxCorners=300,
                                      qualityLevel=0.01, minDistance=20)
    if corners is None or len(corners) < 10:
        return None
    corners2, status, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, corners, None)
    good = status.ravel() == 1
    if good.sum() < 5:
        return None
    pts1 = corners[good]
    pts2 = corners2[good]
    M, inliers = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC,
                                              ransacReprojThreshold=3.0)
    if M is None:
        return None
    A = np.eye(3, dtype=np.float64)
    A[:2] = M
    return A


def decompose(M):
    """(tx, ty, angle_rad, scale) from a 3x3 similarity matrix."""
    tx    = M[0, 2]
    ty    = M[1, 2]
    scale = np.sqrt(M[0, 0]**2 + M[1, 0]**2)
    angle = np.arctan2(M[1, 0], M[0, 0])
    return tx, ty, angle, scale


def build(tx, ty, angle, scale):
    """Reconstruct a 3x3 similarity matrix from components."""
    c = scale * np.cos(angle)
    s = scale * np.sin(angle)
    return np.array([[c, -s, tx],
                     [s,  c, ty],
                     [0,  0,  1]], dtype=np.float64)


def smooth_transforms(transforms, radius):
    """
    Smooth a sequence of 3x3 similarity matrices with a rolling average.
    Each component (tx, ty, angle, scale) is smoothed independently.
    Angle is unwrapped before averaging to avoid wraparound artefacts.
    """
    n = len(transforms)
    params = np.array([decompose(M) for M in transforms])  # (n, 4)
    params[:, 2] = np.unwrap(params[:, 2])                  # unwrap angle

    smoothed = np.empty_like(params)
    for i in range(n):
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        smoothed[i] = params[lo:hi].mean(axis=0)

    return [build(*row) for row in smoothed]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process(input_path, output_path, smooth_radius):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        sys.exit(f"Cannot open: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Input: {fw}x{fh} @ {fps:.2f} fps")

    # ------------------------------------------------------------------
    # Pass 1 — read frames, estimate pairwise transforms, build path
    # ------------------------------------------------------------------
    print("\nPass 1: reading frames and estimating motion...")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    n = len(frames)
    print(f"  {n} frames read.")

    print("  Estimating pairwise transforms...")
    pairwise = []   # pairwise[i] maps frame i coords -> frame i+1 coords
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for i in tqdm(range(1, n)):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        M = estimate_similarity(prev_gray, curr_gray)
        if M is None:
            M = np.eye(3, dtype=np.float64)
        pairwise.append(M)
        prev_gray = curr_gray

    # Cumulative transforms: cum[i] maps frame i coords -> world space.
    # cum[0] = I (world is defined by frame 0).
    # cum[i] = cum[i-1] @ inv(pairwise[i-1])
    # (pairwise[i-1] maps frame i-1 -> frame i, so inv maps frame i -> frame i-1
    #  and composing with cum[i-1] carries us all the way to world.)
    cum = [np.eye(3, dtype=np.float64)]
    for M in pairwise:
        cum.append(cum[-1] @ np.linalg.inv(M))

    # Smooth the cumulative path
    print("  Smoothing path...")
    smooth = smooth_transforms(cum, smooth_radius)

    # Canvas bounds: bounding box of all frame corners warped by cum[i],
    # plus all viewport corners warped by smooth[i].
    corners_frame = np.array([[0, 0], [fw, 0], [fw, fh], [0, fh]],
                              dtype=np.float64)

    all_pts = []
    for i in range(n):
        for M in (cum[i], smooth[i]):
            pts_h = np.column_stack([corners_frame,
                                     np.ones(4)]) @ M[:2, :].T  # 4x2 in world
            all_pts.append(pts_h)

    all_pts = np.vstack(all_pts)  # (8n, 2)
    pad = 4
    ox = int(np.floor(all_pts[:, 0].min())) - pad
    oy = int(np.floor(all_pts[:, 1].min())) - pad
    cw = int(np.ceil(all_pts[:, 0].max())) - ox + pad
    ch = int(np.ceil(all_pts[:, 1].max())) - oy + pad

    # Shift all transforms so world origin maps to canvas pixel (0,0)
    T_shift = np.array([[1, 0, -ox],
                        [0, 1, -oy],
                        [0, 0,   1]], dtype=np.float64)
    cum    = [T_shift @ M for M in cum]
    smooth = [T_shift @ M for M in smooth]

    print(f"  Canvas: {cw} x {ch}  (origin world offset {ox}, {oy})")

    # ------------------------------------------------------------------
    # Pass 2 — paint sequentially, sample viewport, write output
    # ------------------------------------------------------------------
    canvas = np.zeros((ch, cw, 3), dtype=np.uint8)

    print("\nPass 2: painting canvas and sampling output...")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (fw, fh))

    for i in tqdm(range(n)):
        # Paint frame i onto canvas using its cumulative world transform.
        # BORDER_TRANSPARENT preserves existing canvas pixels outside frame i.
        cv2.warpPerspective(frames[i], cum[i], (cw, ch),
                            dst=canvas,
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_TRANSPARENT)

        # Sample output viewport via the smoothed transform.
        # inv(smooth[i]) maps output pixel coords -> canvas pixel coords.
        viewport = cv2.warpPerspective(canvas,
                                       np.linalg.inv(smooth[i]),
                                       (fw, fh),
                                       flags=cv2.INTER_LINEAR)
        writer.write(viewport)

    writer.release()
    print(f"Done: {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Video Painter — canvas stabilization")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--smooth", type=int, default=30,
                   help="Smoothing window radius in frames (default: 30)")
    args = p.parse_args()
    process(args.input, args.output, smooth_radius=args.smooth)
