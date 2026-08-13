"""
Control video for testing Gemini's actual video frame-sampling behavior.
5 min, 1920x1080, NTSC drop-frame rate 29.97fps (30000/1001), black
background, real frame index (0-based) as white text centered in frame.
"""
import cv2
import numpy as np

OUT_PATH = "/workspace/project/Data/control_frame_index.mp4"
WIDTH, HEIGHT = 1920, 1080
FPS = 30000 / 1001

DURATION_S = 5 * 60
N_FRAMES = round(DURATION_S * FPS)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 6
FONT_THICKNESS = 10

writer = cv2.VideoWriter(OUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))

for i in range(N_FRAMES):
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    text = str(i)
    (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
    x = (WIDTH - tw) // 2
    y = (HEIGHT + th) // 2
    cv2.putText(frame, text, (x, y), FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA)
    writer.write(frame)

writer.release()
print(f"wrote {N_FRAMES} frames ({DURATION_S}s @ {FPS:.5f}fps) to {OUT_PATH}")
