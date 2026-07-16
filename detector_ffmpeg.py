"""转场检测 —— ffmpeg 加速版（GPU 可用时自动走 NVDEC）。

为什么更快
----------
原 detector.py 用 cv2.VideoCapture 在 Python 里逐帧软解码 + 灰度 +
缩放 + absdiff + mean，每帧都有 Python 解释器开销，大视频会非常慢。

本模块把“解码 + 计算相邻帧差异 + 阈值判定”整个交给 ffmpeg 的 C
实现（内置的 scdet 场景检测滤镜，原理就是相邻帧差异，与原来的帧差一致）：
  - 软件解码时，ffmpeg 的 C 解码器本身就远快于 OpenCV 的 Python 循环；
  - 若机器上 ffmpeg 的 NVDEC 可用，加 `-hwaccel cuda`（现代 nvdec
    路径）可让解码进一步跑在 GPU 上（3~5 倍）。注意：老的
    `-c:v h264_cuvid` 独立解码器已废弃且在本机 nightly ffmpeg 上失效，
    不要再用。

注意（2026-07-15 更正）：本机原驱动 591.74 时，nvidia-smi 虽已显示
支持 CUDA 13.1，但自带的 nightly ffmpeg（2026-07-13 编）所需的
NVENC/NVDEC 编解码 SDK 版本超前于驱动，导致 NVENC 硬件编码
（Function not implemented）与老的 h264_cuvid 解码器（cuvid decode callback
error）失效。升级到驱动 610.74 后：NVENC 硬件编码恢复正常；NVDEC 走
现代「-hwaccel cuda」（nvdec）路径也恢复正常（0 解码错误）。本模块检测
阶段优先用 -hwaccel cuda GPU 硬解，失败回退软件 ffmpeg，再失败回退
原 OpenCV 实现，保证 GUI 永远有结果。

API 与原 detect_scenes 完全一致，可作为 drop-in 替换；任意一步失败都会
自动回退到原 OpenCV 实现，保证 GUI 永远有结果。

进度条实现要点（2026-07-16 修正）
------------------------------------
之前用 `select='gt(scene,X)'+metadata=print:file=tmp` 有双重问题，导致
“检测转场”进度条直接跳到完成、没有渐进过程：
  1) select 会【丢弃】非切点帧 → 输出帧数 = 切点数量（稀疏），
     ffmpeg 进度 `frame=` 统计的是【输出】帧，于是对整段扫描几乎不变；
  2) metadata=print 的临时文件是【结尾才一次性写出】，运行途中读不到，
     也不能当作进度源。
改为用 **scdet** 滤镜：它只是“标记/上报”场景切换、【不丢帧】，
故 `frame=` 平滑地从 0 增长到总帧数，正好当作“扫描进度”；每个切点
以 `lavfi.scd.time: N`(秒) 实时打到 stderr，直接收集即可（无需临时文件）。
另外 ffmpeg 进度行用 `\r` 原地覆盖（非 `\n` 换行），必须用
“按块读 + 按 `\r` 切分”来解析，否则 readline 会阻塞到 EOF。
"""
import os
import re
import uuid
import tempfile
import subprocess
import threading

from utils import _find, probe_video
# 兜底实现
from detector import detect_scenes as _cv2_detect

# scdet 场景检测：ffmpeg 不过滤帧（输出=全部输入帧），故 stderr 的
# `frame=` 会从 0 平滑增长到总帧数 —— 正好当作“扫描进度”。
# 每个检测到的场景切换以 `lavfi.scd.time: N`(秒) 实时打到 stderr。
# 注意：_pump 读取的是 bytes（proc.stderr），正则必须用 bytes 模式！
_RE_FRAME = re.compile(rb"frame=\s*(\d+)")
_RE_SCD = re.compile(rb"lavfi\.scd\.time:\s*(\d+(?:\.\d+)?)")


def _try_ffmpeg(path, threshold, min_scene_len, progress_cb, hwaccel=False):
    """用 ffmpeg 的 scdet 滤镜检测转场，返回 scenes 列表；失败抛异常。

    为什么用 scdet 而不是旧的 select='gt(scene,X)'+metadata=print：
    - select 会【丢弃】非切点帧 → 输出帧数=切点数量（稀疏），ffmpeg 进度
      `frame=` 统计的是输出帧，于是对整段扫描几乎不变，进度条“直接跳到完成”；
    - metadata=print 的临时文件是【结尾才一次性写出】，运行途中读不到，
      也不能当作进度源；
    - scdet 只是“标记/上报”场景切换、【不丢帧】，故 `frame=` 平滑地
      从 0 增长到总帧数，正好当作“扫描进度”；每个切点以
      `lavfi.scd.time: N`(秒) 实时打到 stderr，直接收集即可。
    hwaccel=True 时加 `-hwaccel cuda`（现代 nvdec GPU 硬解）；
    软解（hwaccel=False）与硬解结果一致，仅速度不同。

    threshold: scdet 阈值(0~100)，越小越灵敏。默认 10 对硬切较敏感、
    平滑视频误报为 0；软切/淡入淡出可调到 20~40。
    """
    ff = _find("ffmpeg")
    info = probe_video(path)
    fps = info["video"]["fps"] if info.get("video") else 0.0
    fps = fps or 30.0
    duration = info.get("duration") or 0.0
    n_frames = int(round(duration * fps)) if duration else 0

    # 阈值：接口 threshold 直接按 scdet 0~100 尺度解释（默认 10）。
    # 旧实现用 scene 滤镜 0~1 尺度、默认 0.3；两者不是线性 1:100 关系
    # （实测硬切 scene≈1.0 ↔ scdet.score≈15.6），故不再做 *100 换算，
    # 直接以 scdet 尺度使用，默认 10 既能捕获硬切、又不会在平滑视频误报。
    try:
        scdet_thr = int(round(threshold))
    except (TypeError, ValueError):
        scdet_thr = 10
    if scdet_thr < 1:
        scdet_thr = 10

    cmd = [ff, "-hide_banner"]
    if hwaccel:
        cmd += ["-hwaccel", "cuda"]
    cmd += [
        "-i", path,
        "-vf", f"scdet=threshold={scdet_thr}",
        "-f", "null", "-",
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )

    cuts = []  # 秒，来自 scdet 的 lavfi.scd.time

    def _pump():
        # 同一路 stderr：解析 frame=(进度) 与 lavfi.scd.time(切点)。
        # ffmpeg 进度行用 \r 原地覆盖（非 \n 换行），故按块读取 + 按 \r 切分。
        buf = b""
        while True:
            chunk = proc.stderr.read(4096)
            if not chunk:
                break
            buf += chunk
            parts = buf.split(b"\r")     # 用 \r 切分进度行
            buf = parts.pop()               # 末段可能不完整，留到下次
            for raw in parts:
                m = _RE_FRAME.search(raw)
                if m and progress_cb and n_frames:
                    progress_cb(int(m.group(1).decode()), n_frames)
                s = _RE_SCD.search(raw)
                if s:
                    try:
                        cuts.append(float(s.group(1).decode()))
                    except ValueError:
                        pass
        # 处理残留（及可能的 \n 结尾）
        for raw in buf.split(b"\n"):
            s = _RE_SCD.search(raw)
            if s:
                try:
                    cuts.append(float(s.group(1).decode()))
                except ValueError:
                    pass

    t = threading.Thread(target=_pump, daemon=True)
    t.start()
    rc = proc.wait(timeout=3600)
    t.join(timeout=5)
    if rc != 0:
        raise RuntimeError(f"ffmpeg 退出码 {rc}")

    # 收集到的切点 → scenes（连续覆盖整段）
    cuts = sorted(set(cuts))
    # 过滤掉贴近开头/结尾的无效切点
    cuts = [c for c in cuts if 0.0 < c < (duration or c + 1)]
    # 合并过近的切点（< min_scene_len 帧）
    min_gap = (min_scene_len / fps) if fps else 0.0
    merged = []
    for c in cuts:
        if merged and (c - merged[-1]) < min_gap:
            continue
        merged.append(c)

    if not merged:
        # 无任何转场：整段作为一个场景
        if duration:
            return [{"start": 0.0, "end": duration, "duration": duration}]
        return [{"start": 0.0, "end": 0.0, "duration": 0.0}]

    bounds = [0.0] + merged + [duration or merged[-1]]
    scenes = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        scenes.append({"start": a, "end": b, "duration": b - a})
    return scenes


def detect_scenes(path, threshold=10, min_scene_len=8, progress_cb=None,
                  sample_w=320, sample_h=180):
    """检测场景边界。

    优先用 ffmpeg（软件或 NVDEC 硬解码）+ scdet 场景检测；任意失败回退到
    原 OpenCV 逐帧实现。返回 [{start, end, duration}, ...] 连续覆盖整段。

    threshold: scdet 阈值(0~100)，越小越灵敏。默认 10 对硬切较敏感，
               软切/淡入淡出可调到 20~40；想更灵敏可降到 5。
    """
    # 1) GPU 硬解（现代 nvdec via -hwaccel cuda）
    try:
        return _try_ffmpeg(path, threshold, min_scene_len, progress_cb, hwaccel=True)
    except Exception:
        pass
    # 2) 软件解码
    try:
        return _try_ffmpeg(path, threshold, min_scene_len, progress_cb, hwaccel=False)
    except Exception:
        pass
    # 3) 兜底：原 OpenCV 逐帧实现
    return _cv2_detect(path, progress_cb=progress_cb,
                       sample_w=sample_w, sample_h=sample_h)


if __name__ == "__main__":
    import sys
    segs = detect_scenes(sys.argv[1] if len(sys.argv) > 1 else "")
    for i, s in enumerate(segs):
        print(f"#{i:02d}  {s['start']:.2f}s - {s['end']:.2f}s  ({s['duration']:.2f}s)")
    print(f"共 {len(segs)} 段")
