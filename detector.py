"""Transition / scene detection.

Uses a content-aware frame-difference algorithm (no external model required):
1. Decode the video, downscale each frame to a small grayscale image.
2. Compute the mean absolute pixel difference between consecutive frames.
3. A new scene starts where the difference spikes above an adaptive threshold
   (robust to encoding noise, catches hard cuts including resolution changes
   like an ad -> main-video boundary).
"""
import cv2
import numpy as np


def detect_scenes(path, threshold=18.0, min_scene_len=8, progress_cb=None,
                   sample_w=320, sample_h=180):
    """Detect scene boundaries.

    Returns a list of dicts: {"start": float, "end": float, "duration": float}
    covering the whole video contiguously.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("无法打开视频文件: " + path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    prev = None
    frame_idx = 0
    diffs = []  # diffs[i] = diff between frame i and frame i+1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (sample_w, sample_h))
        if prev is not None:
            d = float(cv2.absdiff(prev, g).mean())
            diffs.append(d)
        prev = g
        frame_idx += 1
        if progress_cb and n_frames and frame_idx % 30 == 0:
            progress_cb(frame_idx, n_frames)

    cap.release()

    if not diffs:
        # Could not read any frame pair (single-frame file, etc.)
        return [{"start": 0.0, "end": 0.0, "duration": 0.0}]

    arr = np.array(diffs, dtype=np.float64)
    med = float(np.median(arr))
    # Adaptive threshold: at least `threshold`, and well above the noise floor.
    thr = max(float(threshold), med * 2.5 + 5.0)

    cuts = [0]  # frame index where a new scene begins
    last_cut = 0
    for i, d in enumerate(arr):
        fi = i + 1  # transition is between frame i and frame i+1
        if d > thr and (fi - last_cut) >= min_scene_len:
            cuts.append(fi)
            last_cut = fi
    cuts.append(frame_idx)  # append end sentinel

    scenes = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        s = a / fps
        e = b / fps
        scenes.append({"start": s, "end": e, "duration": e - s})
    return scenes


if __name__ == "__main__":
    import sys
    segs = detect_scenes(sys.argv[1] if len(sys.argv) > 1 else "")
    for i, s in enumerate(segs):
        print(f"#{i:02d}  {s['start']:.2f}s - {s['end']:.2f}s  ({s['duration']:.2f}s)")
    print(f"共 {len(segs)} 段")
