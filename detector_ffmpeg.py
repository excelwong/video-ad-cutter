"""转场检测 —— ffmpeg 加速版（默认 CPU 软解，GPU 硬解作一级回退）。

判定逻辑沿用 scdet 滤镜自带的阈值（相邻帧差异，与原 OpenCV 帧差思路一致）：
scdet 只在**真正判定为转场**的帧上，把 ``lavfi.scd.score`` 与
``lavfi.scd.time`` 打到 stderr，直接收集即可。

2026-08-29 加速改动
--------------------
两处改动，实测 5 分钟 1080p 片（306MB）：

  原实现  -hwaccel cuda + scdet(全分辨率)      16.84s  (17.8x)
  + -an（跳过音频解码）                        16.88s  (17.8x)   几乎无变化
  + GPU 端降采样后再回传  -> **本模块采用**    12.26s  (24.5x)   **1.37x**

核心点：``-hwaccel cuda`` 解出的帧在显存里，直接接 scdet（CPU 滤镜）时
ffmpeg 会自动插入 hwdownload **把整幅 1080p 帧搬回内存**，这一步很贵。
先用 ``scale_cuda=320:180`` 在 GPU 上缩小、再 ``hwdownload``，回传量降到
1/36，故提速 37%。

降采样**不影响检测质量**（实测，realistic_cuts.mp4，切点真值 15/30/45s）：
    全分辨率      score=26.876 / 32.565 / 32.325   time=15.021 / 30.021 / 45.021
    320x180       score=26.882 / 32.370 / 32.199   time=15.021 / 30.021 / 45.021
    160x90        score=26.795 / 32.349 / 31.865   time=15.021 / 30.021 / 45.021
打分与切点时间几乎完全一致，故 320x180 可放心使用（与 detector.py 兜底
实现的 320x180 取样尺度也一致）。

另外实测：**并行分块扫描没有收益**（2/4/6 路并发均为 ~12.1~12.6s，反而
因每路重复 seek 略慢）——单路已把解码链路跑满，故不采用。

一点排错经验（写给以后的自己）
------------------------------
* scdet 的元数据**只在判定为转场时才出现**。调试时若整段没有任何
  ``lavfi.scd.time`` 输出，先别急着改正则——很可能是内容本身没有
  高于阈值的帧差（曾误判为"正则失效"，实则是测试片用了高频噪声底图
  + 色相偏移，scdet 打分仅 0.7，低于阈值 10，属于**正确**的"无转场"）。
* ``metadata=print`` 拿到的是每帧的 ``lavfi.scd.mafd`` / ``lavfi.scd.score``，
  与 stderr 上那条 ``lavfi.scd.time`` 是两套机制，不要混用。
* ``-hwaccel_output_format cuda`` 之后**必须**显式写 ``hwdownload``
  （再补 ``format=nv12``），否则 scdet 吃不了 CUDA 帧，ffmpeg 会以
  退出码 -40 报 "Conversion failed"。

历史备注（驱动 / NVDEC）
------------------------
本机原驱动 591.74 时，nvidia-smi 虽显示支持 CUDA 13.1，但自带 nightly
ffmpeg（2026-07-13 编）所需的 NVENC/NVDEC SDK 版本超前于驱动，导致 NVENC
硬件编码（Function not implemented）与老的 h264_cuvid 解码器（cuvid decode
callback error）失效。升级到驱动 610.74 后恢复正常。

2026-10-21 路径重排：CPU 软解优先
----------------------------------
实测 realistic_cuts.mp4（150s 1080p30）：``-hwaccel cuda`` + scale_cuda
6.41s vs 纯 CPU 软解 + scale 3.65s，**检测结果完全一致**（10 段同切点）。
原因是 NVDEC native 通路每帧有 CUDA 同步开销（≈703fps），而纯软解
≈1590fps；≤1080p 源根本喂不饱 GPU。故降级链改为
**CPU → GPU → OpenCV**（4K+ 源可改回 GPU 优先，届时解码压力才真正
落在 GPU 一侧）。原「GPU 优先」的实测数据仍记录在此，供换机器后复测。

API 与原 detect_scenes 完全一致，可作为 drop-in 替换；任意一步失败都会
自动回退到原 OpenCV 实现。

进度条实现要点（2026-07-16 修正）
------------------------------------
之前用 ``select='gt(scene,X)'+metadata=print:file=tmp`` 有双重问题，导致
"检测转场"进度条直接跳到完成、没有渐进过程：
  1) select 会【丢弃】非切点帧 -> 输出帧数 = 切点数量（稀疏），
     ffmpeg 进度 ``frame=`` 统计的是【输出】帧，于是对整段扫描几乎不变；
  2) metadata=print 的临时文件是【结尾才一次性写出】，运行途中读不到，
     也不能当作进度源。
改为用 **scdet** 滤镜：它只是"标记/上报"场景切换、【不丢帧】，
故 ``frame=`` 平滑地从 0 增长到总帧数，正好当作"扫描进度"；每个切点
以 ``lavfi.scd.time: N``(秒) 实时打到 stderr，直接收集即可（无需临时文件）。
另外 ffmpeg 进度行用 ``\\r`` 原地覆盖（非 ``\\n`` 换行），必须用
"按块读 + 按 ``\\r`` 切分"来解析，否则 readline 会阻塞到 EOF。

2026-08-29 修「检测转场卡死/长时间无反应」（bench/test.mp4，45.9 分钟）
------------------------------------------------------------------------
现象：45.9 分钟竖屏片（720x1284, 69112 帧）点"检测转场"后进度条长时间不动，
末段尤其明显，用户以为程序吊死。实测发现三处问题，均已修复：

1) **进度回调饿死（元凶）** —— 读数用的是 ``proc.stderr.read(4096)``。
   ``BufferedReader.read(n)`` 语义是"阻塞到凑满 n 字节或 EOF"，而 ffmpeg
   每 0.5s 才写一行约 146 字节的进度，攒满 4096 字节要约 20 秒。实测
   回调时刻：12.63s(22.8%) → 32.23s(67.8%) → 45.37s(100%)，即
   **45 秒里进度条只跳了 3 次，中间两次各静止 19.6s / 13.1s**，
   开头 12.6s 更是完全为 0。
   修复：改用 ``read1(8192)``。``read1`` 只做**一次**底层读，管道上读到
   多少立刻返回多少（约 146 字节即返回），故进度变成 0.5s 粒度平滑推进。

2) **竖屏被压扁** —— 原来硬写 ``scale_cuda=320:180``，对 720x1284 的竖屏
   会输出 ``320x180 [SAR 203:642]``（像素被横向压缩 3.2 倍）。
   修复：按源宽高比、按**像素总量恒定**（≈320x180=57600）算目标尺寸，
   720x1284 → 180x320，1920x1080 → 320x180；回传量不随画幅变化。

3) **无看门狗，ffmpeg 真卡死时 GUI 会干等到 1 小时** —— 原来只有
   ``proc.wait(timeout=3600)``；且若 progress_cb 抛异常，_pump 线程会静默
   死掉，stderr 管道(64KB)被写满后 ffmpeg 阻塞在写，形成真正死锁。
   修复：① progress_cb 包 try/except；② 主线程看门狗循环：
   超过 ``stall_timeout``(默认 30s) **完全没有新输出** 即判定卡死并 kill，
   抛异常自动降级到下一档；③ 总时限兜底（GPU duration/10，CPU duration/3）。
"""
import math
import os
import re
import subprocess
import threading
import time

from utils import _find, probe_video
# 兜底实现
from detector import detect_scenes as _cv2_detect

# scdet 场景检测：ffmpeg 不过滤帧（输出=全部输入帧），故 stderr 的
# `frame=` 会从 0 平滑增长到总帧数 —— 正好当作"扫描进度"。
# 每个检测到的场景切换以 `lavfi.scd.time: N`(秒) 实时打到 stderr。
# 注意：_pump 读取的是 bytes（proc.stderr），正则必须用 bytes 模式！
_RE_FRAME = re.compile(rb"frame=\s*(\d+)")
_RE_SCD = re.compile(rb"lavfi\.scd\.time:\s*(\d+(?:\.\d+)?)")
# ffmpeg 进度行用 \r 原地覆盖，scdet 元数据行用 \n 结尾 —— 两种都要切
_RE_SPLIT = re.compile(rb"[\r\n]+")

_SAMPLE_W = 320      # 降采样基准宽（与 detector.py 兜底实现一致）
_SAMPLE_H = 180      # 降采样基准高

# 看门狗：ffmpeg 连续这么久"一个字节都没吐"就判定卡死并 kill（秒）
_STALL_TIMEOUT = 30.0

# 最后一次检测实际走的路径（"gpu" / "cpu" / "opencv"），供 GUI 状态栏显示
_last_method = "unknown"
# 各档失败原因（降级链路的排障信息），供 GUI 提示
_last_detail = ""


def get_last_method():
    """返回最后一次检测实际使用的路径：gpu / cpu / opencv。"""
    return _last_method


def get_last_detail():
    """返回降级链路的失败原因（成功时为空串），供 GUI 排障提示。"""
    return _last_detail


def _sample_size(info):
    """按源宽高比计算降采样尺寸，保持像素总量 ≈ _SAMPLE_W*_SAMPLE_H。

    为什么不能硬写 320:180：竖屏（如 720x1284）会被压成 320x180，
    SAR 变成 203:642（横向压缩 3.2 倍）。按面积等比缩放后
    720x1284 → 180x320、1920x1080 → 320x180，画幅正确且回传量恒定。
    宽高取偶数（nv12 / scale_cuda 更稳）。
    """
    v = (info or {}).get("video") or {}
    w = int(v.get("width") or 0)
    h = int(v.get("height") or 0)
    if w < 2 or h < 2:
        return _SAMPLE_W, _SAMPLE_H
    scale = (_SAMPLE_W * _SAMPLE_H) / float(w * h)
    if scale >= 1.0:                       # 源比取样还小，不放大
        return w - (w % 2), h - (h % 2)
    s = math.sqrt(scale)
    tw = max(2, int(round(w * s / 2.0)) * 2)
    th = max(2, int(round(h * s / 2.0)) * 2)
    return tw, th


def _try_ffmpeg(path, threshold, min_scene_len, progress_cb, hwaccel=False,
                stall_timeout=_STALL_TIMEOUT, hard_timeout=None):
    """用 ffmpeg 的 scdet 滤镜检测转场，返回 scenes 列表；失败抛异常。

    为什么用 scdet 而不是旧的 select='gt(scene,X)'+metadata=print：
    - select 会【丢弃】非切点帧 -> 输出帧数=切点数量（稀疏），ffmpeg 进度
      `frame=` 统计的是输出帧，于是对整段扫描几乎不变，进度条"直接跳到完成"；
    - metadata=print 的临时文件是【结尾才一次性写出】，运行途中读不到，
      也不能当作进度源；
    - scdet 只是"标记/上报"场景切换、【不丢帧】，故 `frame=` 平滑地
      从 0 增长到总帧数，正好当作"扫描进度"；每个切点以
      `lavfi.scd.time: N`(秒) 实时打到 stderr，直接收集即可。

    hwaccel=True 时走 GPU 硬解 + GPU 降采样；hwaccel=False 为纯 CPU
    软解 + 降采样（**当前默认首选**，≤1080p 实测更快，见模块头 2026-10-21
    路径重排）。两者检测结果一致（实测打分误差 <1%），仅速度不同。

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

    # 滤镜链：先在 GPU 上缩小再回传给 CPU 侧的 scdet。瓶颈是 hwdownload
    # 的回传量（1080p 全幅 -> 320x180 即降到 1/36，故提速 37%）。
    # 注意 -hwaccel_output_format cuda 之后必须显式 hwdownload + format=nv12，
    # 否则 scdet 无法消费 CUDA 帧，ffmpeg 会以退出码 -40 失败。
    # 尺寸按源宽高比、按像素总量恒定算（见 _sample_size），竖屏不会被压扁，
    # 且回传量不随画幅变化（720x1284 -> 180x320，同为 57600 像素）。
    tw, th = _sample_size(info)
    if hwaccel:
        vf = (f"scale_cuda={tw}:{th},hwdownload,format=nv12,"
              f"scdet=threshold={scdet_thr}")
    else:
        vf = f"scale={tw}:{th},scdet=threshold={scdet_thr}"

    cmd = [ff, "-y", "-hide_banner", "-an"]
    if hwaccel:
        cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    cmd += ["-i", path, "-vf", vf, "-f", "null", "-"]

    proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )

    cuts = []  # 秒，来自 scdet 的 lavfi.scd.time
    state = {"last": time.monotonic()}   # 最近一次收到字节的时刻（看门狗用）

    def _emit(frame):
        # progress_cb 由 GUI 提供，绝不能让它把 _pump 线程搞死 ——
        # 否则 stderr 管道(64KB)写满后 ffmpeg 阻塞在写，形成真正死锁。
        if progress_cb and n_frames:
            try:
                progress_cb(frame, n_frames)
            except Exception:
                pass

    def _consume(raw):
        m = _RE_FRAME.search(raw)
        if m:
            try:
                _emit(int(m.group(1).decode()))
            except ValueError:
                pass
        for s in _RE_SCD.finditer(raw):
            try:
                cuts.append(float(s.group(1).decode()))
            except ValueError:
                pass

    def _pump():
        # 同一路 stderr：解析 frame=(进度) 与 lavfi.scd.time(切点)。
        # ⚠️ 必须用 read1 而不是 read：BufferedReader.read(n) 会阻塞到凑满
        # n 字节，而 ffmpeg 每 0.5s 只写约 146 字节，攒满 4096 要约 20 秒，
        # 表现为"进度条 20 秒才跳一次"。read1 只做一次底层读、读到多少
        # 返回多少，故进度是 0.5s 粒度平滑推进的。
        buf = b""
        try:
            while True:
                try:
                    chunk = proc.stderr.read1(8192)
                except (AttributeError, ValueError):
                    chunk = proc.stderr.read(1)      # 极端兜底
                if not chunk:
                    break
                state["last"] = time.monotonic()
                buf += chunk
                parts = _RE_SPLIT.split(buf)   # 进度行按 \r、元数据行按 \n
                buf = parts.pop()              # 末段可能不完整，留到下次
                for raw in parts:
                    _consume(raw)
        except Exception:
            pass
        finally:
            for raw in _RE_SPLIT.split(buf):   # 收尾残留
                _consume(raw)

    if hard_timeout is None:
        # CPU 是现在的首选路径（软解 ≈10x 实时，取 /3 已很宽松）；
        # GPU 通路 ≈60x 实时，取 /10。两者都至少保留下限，防短视频误杀。
        hard_timeout = (max(300.0, duration / 3.0) if not hwaccel
                        else max(180.0, duration / 10.0))

    t = threading.Thread(target=_pump, daemon=True)
    t.start()
    t_start = time.monotonic()
    try:
        # 看门狗：不再用 proc.wait(timeout=3600) 干等。ffmpeg 若真卡死，
        # 这里会在 stall_timeout 后 kill 它并抛异常 → 自动降级到下一档。
        while proc.poll() is None:
            now = time.monotonic()
            if now - state["last"] > stall_timeout:
                try:
                    proc.kill()
                except Exception:
                    pass
                raise RuntimeError(
                    f"ffmpeg 连续 {stall_timeout:.0f}s 无任何输出（疑似卡死）")
            if now - t_start > hard_timeout:
                try:
                    proc.kill()
                except Exception:
                    pass
                raise RuntimeError(
                    f"ffmpeg 超过总时限 {hard_timeout:.0f}s")
            time.sleep(0.25)
        rc = proc.wait(timeout=30)
    finally:
        t.join(timeout=10)
        try:
            proc.stderr.close()
        except Exception:
            pass
    if rc != 0:
        raise RuntimeError(f"ffmpeg 退出码 {rc}")

    # 收集到的切点 -> scenes（连续覆盖整段）
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
                  sample_w=_SAMPLE_W, sample_h=_SAMPLE_H, status_cb=None):
    """检测场景边界。

    优先用 ffmpeg（CPU 软解 + GPU 降采样回退链）+ scdet 场景检测；
    任意失败回退到原 OpenCV 逐帧实现。返回
    [{start, end, duration}, ...] 连续覆盖整段。

    路径顺序（2026-10-21 实测定序，realistic_cuts.mp4 1080p30）：
      1. CPU 软解 + scale 降采样 —— **3.65s**（默认首选）
      2. GPU 硬解 + scale_cuda 降采样 —— 6.41s（一级回退）
      3. OpenCV 逐帧兜底
    ≤1080p 源 CPU 软解反而更快：纯软解 ≈1590fps，而 -hwaccel cuda 的
    NVDEC native 通路每帧有 CUDA 同步开销、仅 ≈703fps；且两者检测
    结果完全一致（同一切点、打分误差 <1%）。4K+ 源可改用 GPU 档
    （届时回传量/解码压力才真正落在 GPU 一侧）。

    threshold: scdet 阈值(0~100)，越小越灵敏。默认 10 对硬切较敏感，
               软切/淡入淡出可调到 20~40；想更灵敏可降到 5。
    status_cb: 可选，(str) -> None。降级时被调用，让 GUI 能实时显示
               "CPU 路径失败(原因)，正在回退 GPU 硬解…"，避免用户在
               换挡期间以为程序又卡住了。
    """
    global _last_method, _last_detail
    _last_detail = ""
    reasons = []

    def _say(msg):
        if status_cb:
            try:
                status_cb(msg)
            except Exception:
                pass

    # 1) CPU 软解 + 降采样（实测 ≤1080p 最快，且不占显卡）
    try:
        scenes = _try_ffmpeg(path, threshold, min_scene_len, progress_cb,
                             hwaccel=False)
        _last_method = "cpu"
        return scenes
    except Exception as e:
        _last_method = "cpu(失败)"
        reasons.append(f"CPU: {e}")
        _last_detail = "；".join(reasons)
        _say(f"CPU 软解未成功（{e}），正在回退 GPU 硬解…")
    # 2) GPU 硬解（现代 nvdec via -hwaccel cuda）+ GPU 降采样
    try:
        scenes = _try_ffmpeg(path, threshold, min_scene_len, progress_cb,
                             hwaccel=True)
        _last_method = "gpu"
        return scenes
    except Exception as e:
        _last_method = "gpu(失败)"
        reasons.append(f"GPU: {e}")
        _last_detail = "；".join(reasons)
        _say(f"GPU 硬解也未成功（{e}），正在回退 OpenCV 逐帧（较慢）…")
    # 3) 兜底：原 OpenCV 逐帧实现
    _last_method = "opencv"
    return _cv2_detect(path, progress_cb=progress_cb,
                       sample_w=sample_w, sample_h=sample_h)


if __name__ == "__main__":
    import sys
    prog = (lambda f, n: print(f"\r扫描进度 {f}/{n} ({100.0*f/max(1,n):.1f}%)",
                               end="", flush=True)) if "-p" in sys.argv else None
    args = [a for a in sys.argv[1:] if a != "-p"]
    segs = detect_scenes(args[0] if args else "", progress_cb=prog,
                         status_cb=lambda m: print(f"\n[降级] {m}", flush=True))
    print()
    for i, s in enumerate(segs):
        print(f"#{i:02d}  {s['start']:.2f}s - {s['end']:.2f}s  ({s['duration']:.2f}s)")
    print(f"共 {len(segs)} 段   （检测路径: {get_last_method()}）"
          + (f"   降级原因: {get_last_detail()}" if get_last_detail() else ""))
