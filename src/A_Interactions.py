import cv2
import numpy as np
import matplotlib
matplotlib.use("module://matplotlib_inline.backend_inline")  # force inline rendering so plt.show() blocks before input()
import matplotlib.pyplot as plt
import torch

from ultralytics import YOLO
from C_CSV_report import add_to_report
from Two2D.B_video_processing import seek_exact

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "Models/yolo_checkpoints/yolo26l-seg.pt"


def subject_selector(
    video_path: str,
    detections: list[dict],
    model_path: str = MODEL_PATH,
    conf: float = 0.05,
) -> dict | None:
    """
    Grabs a mid-point frame for each detection, runs a quick YOLO predict,
    overlays the subject mask, and displays them with buttons for the user to pick.

    Returns the chosen detection dict, or None if cancelled.
    """
    raw_subject = detections[0]["subject"]
    subject_type = raw_subject.lower().removeprefix("the ").removeprefix("a ").removeprefix("an ").strip()

    add_to_report({"{{subject}}": subject_type})

    model = YOLO(model_path)
    model.to(DEVICE)
    #convert the name into an id first it's a lookup dict, then the selection
    #match by word-subset so adjectives (e.g. "stray cat") don't defeat an exact-name lookup
    name_to_id = {name: idx for idx, name in model.names.items()}

    cap = cv2.VideoCapture(video_path)
    native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # round up to the nearest multiple of YOLO's max stride (32) so predict()
    # doesn't silently re-pad (and warn about it) on every single call
    predict_imgsz = (-(-native_h // 32) * 32, -(-native_w // 32) * 32)

    previews = []
    for i, det in enumerate(detections):
        det_subject = det["subject"].lower().removeprefix("the ").removeprefix("a ").removeprefix("an ").strip()
        subject_words = set(det_subject.split())
        class_matches = [name for name in name_to_id if set(name.split()) <= subject_words]
        if not class_matches:
            print(f"'{det_subject}' not on list — no matching YOLO class, skipping detection {i + 1}.")
            continue
        subject_ids = [name_to_id[max(class_matches, key=len)]]

        mid_frame = int((det["start_s"] + det["end_s"]) / 2 * fps)
        start_f   = int(det["start_s"] * fps)
        end_f     = int(det["end_s"]   * fps)

        # one seek + sequential decode over the whole range, instead of a seek per
        # candidate — cap.set() replays from the last keyframe every time, so
        # seeking to scattered candidate frames re-decodes the same span repeatedly
        seek_exact(cap, start_f)
        frames_by_idx = {}
        for candidate in range(start_f, end_f + 1):
            ret, frame = cap.read()
            if not ret:
                break
            frames_by_idx[candidate] = frame

        found_frame, found_r = None, None
        for offset in range(0, (end_f - start_f) // 2 + 1):
            for candidate in ([mid_frame + offset] if offset == 0 else [mid_frame - offset, mid_frame + offset]):
                frame = frames_by_idx.get(candidate)
                if frame is None:
                    continue
                print(f"checking frame {candidate}")
                #try to apply a mask
                r = model.predict(frame, classes=subject_ids, conf=conf,
                                  imgsz=predict_imgsz, retina_masks=True, verbose=False)[0]
                if r.masks is not None and len(r.masks.data): #found a frame, so stop looing
                    found_frame, found_r = frame, r
                    break
            if found_frame is not None:
                break
                
        if found_frame is None:
            print(f"  Warning: no mask found for detection {i + 1} — skipping preview.")
            continue

        display = cv2.cvtColor(found_frame, cv2.COLOR_BGR2RGB)
        mask = (found_r.masks.data.cpu().numpy() > 0.5).any(axis=0)
        display[mask] = (display[mask] * 0.6 + np.array([0, 200, 0]) * 0.4).astype(np.uint8)
        previews.append(display)

    cap.release()

    if not previews:
        print("No preview frames could be extracted.")
        return None

    n = len(previews)
    _, axes = plt.subplots(1, n, figsize=(7 * n, 5))
    if n == 1:
        axes = [axes]
    for i, (ax, img) in enumerate(zip(axes, previews)):
        ax.imshow(img)
        ax.set_title(f"{subject_type} {i + 1}", fontsize=14)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    while True:
        try:
            choice = int(input(f"Which is the correct {subject_type}? Enter 1-{n}: "))
            if 1 <= choice <= n:
                return [detections[choice - 1]]
        except ValueError:
            pass
        print(f"Please enter a number between 1 and {n}.")
