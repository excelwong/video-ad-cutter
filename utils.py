"""Utility helpers: locate ffmpeg/ffprobe, probe video metadata, formatting."""
import json
import os
import subprocess

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


def run_capture(cmd, progress_cb=None, timeout=None):
    """Run a command, streaming stderr to progress_cb (raw line). Returns (rc, stderr).

    Reads in binary and decodes each line as UTF-8 with replacement so a
    stray non-GBK byte (e.g. a Chinese filename in ffmpeg's output on a
    Chinese-Windows machine) can never crash the reader thread.
    """
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    stderr_chunks = []
    try:
        for raw in iter(proc.stderr.readline, b""):
            line = raw.decode("utf-8", "replace")
            stderr_chunks.append(line)
            if progress_cb:
                progress_cb(line.rstrip("\n"))
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        return 1, "".join(stderr_chunks) + "\n[TIMEOUT]"
    return proc.returncode, "".join(stderr_chunks)
