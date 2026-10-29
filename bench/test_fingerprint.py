# -*- coding: utf-8 -*-
"""帧指纹对齐：把输出文件的视频/音频内容反推到源时间轴，量化内容窗错位。

视频指纹 = 全帧 YAVG 序列（testsrc2 逐帧变化独特）。
输出序列在源序列里滑窗匹配（最小绝对差），得视频内容窗起点（帧精度）。
音频指纹 = beep 包络。源里 beep 每 10s 一个（5.0,15.0,...），输出里检测到
的 beep 组间距可反推音频窗。
"""
import os
import subprocess
import sys

APP = r"C:\Users\excel\WorkBuddy\Claw\video-ad-cutter"
sys.path.insert(0, APP)
sys.path.insert(0, os.path.join(APP, "bench"))
from test_avsync import white_frames, beep_starts, cluster  # noqa: E402

T = r"C:\Users\excel\AppData\Local\Temp\avsync"
FF, FP = merger_ff, merger_fp = None, None
from utils import get_ffmpeg  # noqa: E402
FF, FP = get_ffmpeg()


def yavg_series(path, step=1):
    """全帧 YAVG 序列（每帧）。"""
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
        return vals
    finally:
        os.chdir(old)


def match_offset(src, out):
    """输出序列在源序列中的起始帧号（最小绝对差滑窗）。"""
    n = len(out)
    best, best_i = None, -1
    for i in range(0, len(src) - n + 1):
        d = sum(abs(src[i + j] - out[j]) for j in range(0, n, 3))
        if best is None or d < best:
            best, best_i = d, i
    return best_i, best


def main():
    src = os.path.join(T, "hevc.mp4")
    out = os.path.join(T, "hevc_out.mp4")
    print("提取源视频指纹…")
    s = yavg_series(src)
    print(f"源 {len(s)} 帧")
    print("提取输出视频指纹…")
    o = yavg_series(out)
    print(f"输出 {len(o)} 帧")
    i, d = match_offset(s, o)
    fps = 25.0
    print(f"视频内容窗起点 = 源第 {i} 帧 = 源 {i / fps:.3f}s (匹配差 {d:.1f})")
    print(f"视频内容窗终点 = 源 {(i + len(o) - 1) / fps:.3f}s")

    # 音频：源 beep 在 5+10k；输出 beep 组反推
    bs = cluster(beep_starts(out))
    wf = cluster(white_frames(out))
    print(f"输出白帧组: {['%.3f' % t for t in wf]}")
    print(f"输出beep组: {['%.3f' % t for t in bs]}")
    # 已知视频窗起点 v0（源秒），输出里白闪是源 15.0/25.0/... 的帧：
    # 输出位置 = 源t - v0（若视频窗从 v0 起连续）
    for w in wf:
        cands = [x for x in (15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0)
                 if 0 <= x - (w + 0) < 60]
        pass
    # 更直接：用视频窗起点推每个白帧对应的源时间，看它落在哪个信号点
    print("\n假设视频窗连续，白帧输出位置 w 对应源时间 w + v0：")
    for w in wf:
        print(f"  白帧 {w:.3f}s -> 源 {w + i / fps:.3f}s")


if __name__ == "__main__":
    main()
