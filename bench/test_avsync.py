# -*- coding: utf-8 -*-
"""音画同步端到端测试：白闪+beep 同轴参考信号。

素材 signal.mp4：每 10s 周期在 t=5.0 起有 3 帧白闪 + 0.2s beep（同相位）。
源内音画偏移 ≈ 0（白帧 5.000 vs beep 4.992，21ms 检测粒度）。

每个场景导出后，测量输出中【同一周期】白帧 pts 与 beep 起始 pts 之差，
即用户可感知的音画偏移。
"""
import os
import subprocess
import sys

APP = r"C:\Users\excel\WorkBuddy\Claw\video-ad-cutter"
sys.path.insert(0, APP)
import merger  # noqa: E402
from utils import get_ffmpeg  # noqa: E402

T = r"C:\Users\excel\AppData\Local\Temp\avsync"
SRC = os.path.join(T, "signal.mp4")
FF, FP = get_ffmpeg()


def _probe_lavfi(path, graph, entries, timeout=600):
    """lavfi 图里不能用 Windows 绝对路径（':' 是滤镜参数分隔符）。
    切到文件所在目录后用纯文件名。"""
    old = os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(path)))
    base = os.path.basename(path)
    try:
        cmd = [FP, "-v", "error", "-f", "lavfi", graph.format(f=base),
               "-show_entries", entries, "-of", "csv=p=0"]
        res = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout, encoding="utf-8",
                             errors="replace")
        return res.stdout.splitlines()
    finally:
        os.chdir(old)


def white_frames(path):
    """返回所有纯白帧的 pts_time 列表。"""
    out = []
    for line in _probe_lavfi(
            path, "movie={f},signalstats",
            "frame=pts_time:frame_tags=lavfi.signalstats.YAVG"):
        parts = line.split(",")
        if len(parts) >= 2 and parts[1].strip() in ("235", "235.0"):
            try:
                out.append(float(parts[0]))
            except ValueError:
                pass
    return out


def beep_starts(path, thresh=-25.0):
    """返回每个 beep 段的起始 pts（第一个 RMS 超阈帧）。"""
    out, prev_on = [], False
    for line in _probe_lavfi(
            path, "amovie={f},astats=metadata=1:reset=1",
            "frame=pts_time:frame_tags=lavfi.astats.Overall.RMS_level"):
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            t = float(parts[0])
            rms = float(parts[1])
        except ValueError:
            continue
        on = rms > thresh
        if on and not prev_on:
            out.append(t)
        prev_on = on
    return out


def cluster(times, gap=1.0):
    """把相近时间点聚成组，返回每组首个时间。"""
    out = []
    for t in sorted(times):
        if not out or t - out[-1][-1] > gap:
            out.append([t])
        else:
            out[-1].append(t)
    return [g[0] for g in out]


def av_offset(path):
    """输出里各周期白帧组首帧 与 beep 起点之差（秒）。0 = 同步。"""
    wf = [t for t in white_frames(path) if t > 0.5]   # 跳过边界残缺
    bs = [t for t in beep_starts(path) if t > 0.5]
    if not wf or not bs:
        return None, wf, bs
    wf_g = cluster(wf)
    bs_g = cluster(bs)
    pairs = []
    for w in wf_g:
        cands = [b for b in bs_g if -0.3 <= w - b < 1.5]
        pairs.append(w - min(cands) if cands else None)
    return pairs, wf_g, bs_g


def report(name, path):
    pairs, wf, bs = av_offset(path)
    print(f"[{name}] 白帧组: {['%.3f' % t for t in wf]}")
    print(f"[{name}] beep 起始: {['%.3f' % t for t in bs]}")
    if pairs is None:
        print(f"[{name}] ❌ 信号检测失败（无白帧或无 beep）")
        return None
    ok = all(p is not None and abs(p) <= 0.10 for p in pairs)
    print(f"[{name}] 音画偏移(白帧-beep): "
          f"{['%.3f' % p if p is not None else 'N/A' for p in pairs]} "
          f"-> {'✅ 同步' if ok else '❌ 错位'}")
    return ok


def seg(a, b):
    return {"start": a, "end": b, "duration": b - a}


def main():
    cases = [
        ("lossless 两段[6-16,18-28] 切点均在KF",
         [{"start": 6.0, "end": 16.0, "duration": 10.0},
          {"start": 18.0, "end": 28.0, "duration": 10.0}],
         dict(mode="lossless"), SRC),
        ("lossless 全保留[0-60]",
         [{"start": 0.0, "end": 60.0, "duration": 60.0}],
         dict(mode="lossless"), SRC),
        ("smartcut 头部非KF[7-16]",
         [{"start": 7.0, "end": 16.0, "duration": 9.0}],
         dict(mode="precise"), SRC),
        ("smartcut 两段[7-16,23-31] 内部切点均非KF",
         [{"start": 7.0, "end": 16.0, "duration": 9.0},
          {"start": 23.0, "end": 31.0, "duration": 8.0}],
         dict(mode="precise"), SRC),
        ("big_offset 源(音频晚0.479) lossless[6-16]",
         [{"start": 6.0, "end": 16.0, "duration": 10.0}],
         dict(mode="lossless"), "big_offset.mp4"),
        ("VFR 源 lossless 两段[6-16,18-28]",
         [{"start": 6.0, "end": 16.0, "duration": 10.0},
          {"start": 18.0, "end": 28.0, "duration": 10.0}],
         dict(mode="lossless"), "vfr.mp4"),
        ("VFR 源 smartcut 两段[7-16,23-31]",
         [{"start": 7.0, "end": 16.0, "duration": 9.0},
          {"start": 23.0, "end": 31.0, "duration": 8.0}],
         dict(mode="precise"), "vfr.mp4"),
    ]
    ffmpeg, ffprobe = FF, FP
    results = []
    for i, (name, kept, kw, src) in enumerate(cases):
        src_path = src if os.path.isabs(src) else os.path.join(T, src)
        out = os.path.join(T, f"exp_{i}.mp4")
        ok, msg = merger.merge_streamcopy(src_path, kept, out,
                                          mode=kw["mode"])
        print(f"\n===== 场景{i + 1}: {name} =====")
        print(f"  导出: {'成功' if ok else '失败'}")
        if not ok:
            print("  " + msg[:300])
            results.append((name, None))
            continue
        results.append((name, report(f"场景{i + 1}", out)))

    # 兜底路线 _try_copy_cut：现在同样吸附关键帧（修复后行为）
    out = os.path.join(T, "exp_fb.mp4")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        ok, msg = merger._try_copy_cut(ffmpeg, ffprobe, SRC,
                                       [{"start": 7.0, "end": 16.0,
                                         "duration": 9.0}],
                                       out, td, None, 0)
    print("\n===== 场景5: copy兜底路线 [7-16]（切点不处理） =====")
    print(f"  导出: {'成功' if ok else '失败'}")
    if ok:
        results.append(("copy兜底", report("场景5", out)))
    else:
        print("  " + msg[:200])
        results.append(("copy兜底", None))

    print("\n========== 总结论 ==========")
    for name, r in results:
        mark = "?" if r is None else ("✅" if r else "❌ 错位")
        print(f"  {mark} {name}")


if __name__ == "__main__":
    main()
