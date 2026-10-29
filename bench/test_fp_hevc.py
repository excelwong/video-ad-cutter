# -*- coding: utf-8 -*-
"""HEVC 三路线指纹对齐：独立输出文件，量每个输出的视频窗/音频窗。"""
import os
import sys

APP = r"C:\Users\excel\WorkBuddy\Claw\video-ad-cutter"
sys.path.insert(0, APP)
sys.path.insert(0, os.path.join(APP, "bench"))
import merger  # noqa: E402
from test_avsync import white_frames, beep_starts, cluster  # noqa: E402
from utils import get_ffmpeg  # noqa: E402
import subprocess  # noqa: E402

T = r"C:\Users\excel\AppData\Local\Temp\avsync"
SRC = os.path.join(T, "hevc.mp4")
FPS = 25.0
_, FP = get_ffmpeg()


def yavg_series(path, max_frames=None):
    old = os.getcwd()
    os.chdir(os.path.dirname(path))
    try:
        cmd = [FP, "-v", "error", "-f", "lavfi",
               f"movie={os.path.basename(path)},signalstats",
               "-show_entries", "frame_tags=lavfi.signalstats.YAVG",
               "-of", "csv=p=0"]
        res = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=1200, encoding="utf-8",
                             errors="replace")
        vals = []
        for line in res.stdout.splitlines():
            try:
                vals.append(float(line.split(",")[0]))
            except (ValueError, IndexError):
                vals.append(0.0)
            if max_frames and len(vals) >= max_frames:
                break
        return vals
    finally:
        os.chdir(old)


def window(src_series, out_series, tag="head"):
    """输出前(或后) 150 帧在源中的位置。"""
    n = min(150, len(out_series))
    if tag == "head":
        q = out_series[:n]
    else:
        q = out_series[-n:]
    best, best_i = None, -1
    rng = range(0, len(src_series) - n + 1)
    for i in rng:
        d = sum(abs(src_series[i + j] - q[j]) for j in range(0, n, 2))
        if best is None or d < best:
            best, best_i = d, i
    return best_i, best


def seg(a, b):
    return {"start": float(a), "end": float(b), "duration": float(b) - a}


CASES = [
    ("HEVC lossless 5段", [seg(6, 16), seg(22, 33), seg(41, 52),
                           seg(58, 67), seg(73, 84)], "lossless"),
    ("HEVC smartcut 5段", [seg(7, 16), seg(22, 33), seg(41, 52),
                           seg(58, 67), seg(73, 84)], "precise"),
]

print("提取源指纹…")
src_series = yavg_series(SRC)
print(f"源 {len(src_series)} 帧")

planned_total = 0.0
for name, kept, mode in CASES:
    out = os.path.join(T, f"fp_{mode}.mp4")
    if os.path.exists(out):
        os.remove(out)
    ok, msg = merger.merge_streamcopy(SRC, kept, out, mode=mode)
    print(f"\n===== {name} 导出:{'成功' if ok else '失败'} =====")
    if not ok:
        print(msg[:200])
        continue
    out_series = yavg_series(out)
    h_i, h_d = window(src_series, out_series, "head")
    t_i, t_d = window(src_series, out_series, "tail")
    v_start = h_i / FPS
    v_end = (t_i + min(150, len(out_series)) - 1) / FPS
    # 视频窗总长（帧级）
    v_len = len(out_series) / FPS
    print(f"  视频窗: 源 {v_start:.3f}s 起（匹配差 {h_d:.1f}），"
          f"末帧 {v_end:.3f}s（匹配差 {t_d:.1f}），长 {v_len:.3f}s")
    bs = cluster(beep_starts(out))
    wf = cluster(white_frames(out))
    print(f"  白帧组({len(wf)}): {['%.3f' % t for t in wf]}")
    print(f"  beep组({len(bs)}): {['%.3f' % t for t in bs]}")
    if wf and bs:
        pairs = []
        for w in wf:
            c = [b for b in bs if -0.3 <= w - b < 1.5]
            pairs.append(f"{w - min(c):+.3f}" if c else "N/A")
        print(f"  白帧-beep偏移: {pairs}")
