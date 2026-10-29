# -*- coding: utf-8 -*-
"""高密度信号下的三路线偏移曲线（就近配对，容忍漏检）。"""
import os
import subprocess
import sys

APP = r"C:\Users\excel\WorkBuddy\Claw\video-ad-cutter"
sys.path.insert(0, APP)
sys.path.insert(0, os.path.join(APP, "bench"))
import merger  # noqa: E402
from test_avsync import _probe_lavfi, beep_starts  # noqa: E402
from utils import get_ffmpeg  # noqa: E402

T = r"C:\Users\excel\AppData\Local\Temp\avsync"
_, FP = get_ffmpeg()


def white_frames2(path, ymin=220.0):
    out = []
    for line in _probe_lavfi(
            path, "movie={f},signalstats",
            "frame=pts_time:frame_tags=lavfi.signalstats.YAVG"):
        parts = line.split(",")
        if len(parts) >= 2:
            try:
                if float(parts[1]) > ymin:
                    out.append(float(parts[0]))
            except ValueError:
                pass
    return out


def cluster(ts, gap=0.5):
    groups = []
    for t in sorted(ts):
        if not groups or t - groups[-1][-1] > gap:
            groups.append([t])
        else:
            groups[-1].append(t)
    return [g[0] for g in groups]


def offsets(path, ymin=220.0):
    wf = cluster(white_frames2(path, ymin))
    bs = cluster(beep_starts(path))
    pairs = []
    for w in wf:
        c = [b for b in bs if abs(w - b) < 0.45]
        if c:
            pairs.append((w, w - min(c, key=lambda b: abs(w - b))))
    return pairs, len(wf), len(bs)


def seg(a, b):
    return {"start": float(a), "end": float(b), "duration": float(b) - a}


src = os.path.join(T, "hevc_hd.mp4")
print("=== 源基准 ===")
pairs, nw, nb = offsets(src)
print(f"白帧组 {nw} / beep组 {nb}，配对 {len(pairs)}")
devs = [d for _w, d in pairs]
print(f"源偏移: max|dev|={max(abs(d) for d in devs):.3f} "
      f"（超0.1的组 {sum(1 for d in devs if abs(d) > 0.1)}）")

CASES = [
    ("lossless5", [seg(6, 16), seg(22, 33), seg(41, 52), seg(58, 67),
                   seg(73, 84)], "lossless"),
    ("smartcut5", [seg(7, 16), seg(22, 33), seg(41, 52), seg(58, 67),
                   seg(73, 84)], "precise"),
]
for tag, kept, mode in CASES:
    out = os.path.join(T, f"hd_{tag}.mp4")
    if os.path.exists(out):
        os.remove(out)
    ok, msg = merger.merge_streamcopy(src, kept, out, mode=mode)
    print(f"\n=== {tag} 导出:{'成功' if ok else '失败'} ===")
    if not ok:
        print(msg[:200])
        continue
    pairs, nw, nb = offsets(out)
    print(f"白帧组 {nw} / beep组 {nb}，配对 {len(pairs)}")
    if not pairs:
        continue
    devs = [d for _w, d in pairs]
    bad = [(w, d) for w, d in pairs if abs(d) > 0.1]
    print(f"偏移: min={min(devs):+.3f} max={max(devs):+.3f} "
          f"超0.1s组数={len(bad)}/{len(devs)}")
    if bad:
        print("  超差样例:", [f"{w:.1f}s:{d:+.3f}" for w, d in bad[:10]])
