"""Utility helpers: locate ffmpeg/ffprobe, probe video metadata, formatting."""
import json
import os
import re
import subprocess
import threading
import time

_BASE = os.path.dirname(os.path.abspath(__file__))


def _find(name):
    """Locate an ffmpeg/ffprobe executable: prefer the bundled build, else PATH."""
    for cand in (
        os.path.join(_BASE, "ffmpeg", "bin", name + ".exe"),
        os.path.join(_BASE, "ffmpeg", "bin", name),
    ):
        if os.path.isfile(cand):
            return cand
    return name


def get_ffmpeg():
    return _find("ffmpeg"), _find("ffprobe")


_nvenc_cache = None  # None=unknown, True/False=probed

def has_nvenc():
    """Return True only if the GPU NVENC encoder is *actually usable*.

    Compiling in ``h264_nvenc`` is not enough: the installed NVIDIA
    driver's NVENC/NVDEC codec SDK version must match what ffmpeg was
    built against. On the old driver 591.74 our nightly ffmpeg (built
    2026-07-13) failed NVENC with "Function not implemented"; upgrading
    to driver 610.74 fixed it. So this probe validates the encoder at
    runtime. IMPORTANT: the test frame must be >= NVENC's minimum
    dimension (64x64 is BELOW it and yields a false negative
    "Frame Dimension less than the minimum supported value"), so we use
    320x240. We probe once, cache the result, and report False so
    callers fall back to libx264 without per-segment failures.
    """
    global _nvenc_cache
    if _nvenc_cache is not None:
        return _nvenc_cache
    ffmpeg = _find("ffmpeg")
    try:
        res = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi",
             "-i", "color=c=black:s=320x240:d=0.04",
             "-c:v", "h264_nvenc", "-frames:v", "1", "-f", "null", "-"],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=60,
        )
        ok = res.returncode == 0 and "nvenc" not in (res.stderr + res.stdout).lower()
        _nvenc_cache = bool(ok)
    except Exception:
        _nvenc_cache = False
    return _nvenc_cache


def _to_float(s, default=0.0):
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def probe_video(path):
    """Return a dict with duration and the first video/audio stream properties."""
    ffprobe = _find("ffprobe")
    cmd = [ffprobe, "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", path]
    res = subprocess.run(cmd, capture_output=True, text=True,
                        encoding="utf-8", errors="replace", timeout=120)
    if res.returncode != 0:
        raise RuntimeError("ffprobe failed: " + res.stderr)
    data = json.loads(res.stdout)
    fmt = data.get("format", {})
    info = {
        "duration": _to_float(fmt.get("duration", 0)),
        "video": None,
        "audio": None,
    }
    for s in data.get("streams", []):
        ctype = s.get("codec_type")
        if ctype == "video" and info["video"] is None:
            fr = s.get("avg_frame_rate", "0/1")
            num, _, den = fr.partition("/")
            fps = _to_float(num) / _to_float(den, 1) if _to_float(den, 1) else 0
            info["video"] = {
                "width": int(_to_float(s.get("width", 0))),
                "height": int(_to_float(s.get("height", 0))),
                "fps": fps,
                "codec": s.get("codec_name"),
                "pix_fmt": s.get("pix_fmt"),
            }
        elif ctype == "audio" and info["audio"] is None:
            info["audio"] = {
                "sample_rate": int(_to_float(s.get("sample_rate", 0))),
                "channels": int(_to_float(s.get("channels", 0))),
                "codec": s.get("codec_name"),
            }
    return info


def probe_segment(path, start, dur=2.0):
    """Fast copy-probe of a short slice to learn its native resolution/fps/audio."""
    ffmpeg, ffprobe = get_ffmpeg()
    import tempfile
    fd, tmp = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        subprocess.run(
            [ffmpeg, "-y", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
             "-i", path, "-c", "copy", tmp],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=120,
        )
        return probe_video(tmp)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def human_time(sec):
    """Format seconds as MM:SS or HH:MM:SS."""
    sec = max(0.0, float(sec))
    s = int(round(sec))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# ffmpeg 的进度是**原地刷新**的：一行里用 \r 覆盖上一次的输出，整条
# 转码/拷贝过程中一个 \n 都不发。所以必须按 \r 和 \n 一起切分。
_SPLIT_RE = re.compile(rb"[\r\n]+")
# 只保留最近这么多 stderr 用于报错（进度输出量很大，不能全存）
_ERR_KEEP = 262144          # 256KB
_PENDING_MAX = 1 << 20      # 单行超过 1MB 强制断开，防止无界增长


def run_capture(cmd, progress_cb=None, timeout=None, inactivity_timeout=None):
    """Run a command, streaming stderr to progress_cb (raw line). Returns (rc, stderr).

    三个坑（2026-08-29 修，test3.mp4 上表现为"导出卡死、毫无动静"）：

    1. **不能用 ``readline``**：ffmpeg 进度用 ``\\r`` 刷新、不带 ``\\n``，
       ``iter(proc.stderr.readline, b"")`` 会一直阻塞到换行才返回 —— 于是
       整个拷贝/转码期间进度回调**一次都不触发**，界面看着像死了。
       改用 ``read1`` + 按 ``[\\r\\n]`` 切分（与 detector_ffmpeg 同一套）。
    2. **stderr 不能无上限累积**：所有 ``\\r`` 进度会被拼进同一个字符串，
       长任务下能涨到几百 MB。这里只保留最近 256KB，够报错用。
    3. **ffmpeg 挂死无兜底**：``read1`` 是阻塞读，若 ffmpeg 在某个损坏文件上
       **静默卡住**（不出进度、不退出、不 EOF），``read1`` 永远阻塞，墙钟
       ``timeout``（``proc.wait(timeout=...)``）根本到不了 —— 上层 UI 就会
       **永久转圈、零反馈**。因此新增 ``inactivity_timeout``：另起一个读取线程
       持续收输出，主线程以 0.25s 粒度看门狗监控「最近一次收到输出的时刻」，
       一旦静默超过 ``inactivity_timeout`` 秒即判定挂死、kill 并回传
       ``(1, stderr + "[TIMEOUT: ffmpeg 已 N 秒无任何输出...]")``。
       注意：ffmpeg 健康运行时每隔零点几秒就吐一次进度，所以一个较宽松的
       阈值（如 180s）不会误杀正常导出；只有真挂死才触发。

    另外 stdout 直接丢弃：本函数从不返回 stdout，留着 PIPE 又不读的话，
    一旦 ffmpeg 往 stdout 写东西就会把管道写满、阻塞住整个编码。

    Reads in binary and decodes each line as UTF-8 with replacement so a
    stray non-GBK byte (e.g. a Chinese filename in ffmpeg's output on a
    Chinese-Windows machine) can never crash the reader thread.
    """
    proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    chunks = []
    total = 0
    pending = b""
    last = [time.monotonic()]          # 最近一次收到输出的时刻（闭包可变）
    done = threading.Event()

    def emit(line):
        nonlocal total
        chunks.append(line)
        total += len(line)
        if progress_cb:
            try:
                progress_cb(line)
            except Exception:
                # 回调抛异常绝不能让读取线程死掉：否则 stderr 管道没人读、
                # 会被写满，ffmpeg 阻塞在 write 上 —— 真正的死锁。
                pass
        if total > _ERR_KEEP:
            keep = "".join(chunks)[-_ERR_KEEP:]
            chunks.clear()
            chunks.append(keep)
            total = len(keep)

    def reader():
        nonlocal pending
        try:
            while True:
                try:
                    data = proc.stderr.read1(8192)
                except (OSError, ValueError):
                    break
                if not data:
                    break
                last[0] = time.monotonic()
                buf = pending + data
                if len(buf) > _PENDING_MAX:      # 一直没有分隔符也别无限攒
                    emit(buf.decode("utf-8", "replace"))
                    pending = b""
                    continue
                parts = _SPLIT_RE.split(buf)
                pending = parts.pop()            # 末段可能还没写完
                for raw in parts:
                    if raw:
                        emit(raw.decode("utf-8", "replace"))
            if pending:
                emit(pending.decode("utf-8", "replace"))
        finally:
            done.set()

    th = threading.Thread(target=reader, daemon=True)
    th.start()

    started = time.monotonic()
    while not done.is_set():
        if proc.poll() is not None:
            break
        now = time.monotonic()
        if timeout is not None and (now - started) > timeout:
            proc.kill()
            th.join(timeout=5)
            return 1, "".join(chunks) + "\n[TIMEOUT]"
        if inactivity_timeout is not None and (now - last[0]) > inactivity_timeout:
            proc.kill()
            th.join(timeout=5)
            return 1, "".join(chunks) + (
                "\n[TIMEOUT: ffmpeg 已 %d 秒无任何输出，疑似挂死，已终止]"
                % int(inactivity_timeout))
        time.sleep(0.25)
    th.join(timeout=10)
    # reader 结束后进程多半已退出，但 returncode 未必被设置；显式 reap 一下，
    # 否则可能返回 None（上层误判为"未知结果"）。
    if proc.poll() is None:
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
    return proc.returncode, "".join(chunks)
