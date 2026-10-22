"""Cut and re-combine the kept video segments.

三条导出路线
------------
1. ``merge_streamcopy(mode="lossless")`` —— **GUI 默认**。关键帧对齐的
   **纯流拷贝**：切点吸附到关键帧，全程 ``-c copy``，**零重编码**、画质
   100% 无损、速度最快。代价是每个内部切点最多丢掉一个 GOP（test2.mp4
   约 4s）的正片。详见 ``_plan_lossless``。

2. ``merge_streamcopy(mode="precise")`` —— smart cut：先合并首尾相连的
   保留段，再对每块做「头部重编码 + 主体原编码流拷贝 + 尾部重编码」。
   切点精确，重编码通常只占 4% 左右。作为 lossless 不可用时的自动兜底，
   也可由调用方显式指定。详见 ``_try_smartcut``。

3. ``merge()`` —— 逐段重新编码到统一规格（分辨率/fps/音频取自参考段）。
   用于保留段规格不一致时的兜底；会把整段重编码。

为什么默认改成关键帧对齐
------------------------
用户明确选择「对齐关键帧更重要」（2026-08-29）：宁可每个切点多删一点点
正片，也不要有任何一帧被重新编码。两条路线的**对齐方向**都遵守同一条
铁律 —— **绝不把被丢弃的内容拷回来**（这是当初「去了广告还在」的根因）。
lossless 路线把它推到极致：头边界只能向【前】吸附、尾边界只能向【后】
吸附，因此只会少保留，永远不会多拷。

GPU: tries h264_nvenc first (RTX 2060 / NVENC), and transparently falls
back to libx264 (CPU) if the GPU encoder is unavailable or errors.
"""
import bisect
import datetime
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from utils import (get_ffmpeg, has_nvenc, probe_video, probe_segment,
                   run_capture)

# 应用所在目录（错误日志写到这里）
_APP_DIR = os.path.dirname(os.path.abspath(__file__))

# 导出时 ffmpeg 静默超过此秒数即判定「挂死」并被强制终止（见 utils.run_capture
# 的 inactivity_timeout）。健康 ffmpeg 每零点几秒就吐一次进度行，180s 的阈值
# 不会误杀正常长任务（如 1 小时 4K 重编码）；只有真卡死（不出进度、不退出）
# 才触发，避免 GUI 无限转圈、零反馈。
EXPORT_INACTIVITY_TIMEOUT = 180

_TIME_RE = re.compile(r"time=(\d+:\d+:\d+\.\d+)")
_TIME2_RE = re.compile(r"time=(\d+\.\d+)")


def _parse_time(line):
    m = _TIME_RE.search(line)
    if m:
        h, mm, ss = m.group(1).split(":")
        return int(h) * 3600 + int(mm) * 60 + float(ss)
    m = _TIME2_RE.search(line)
    if m:
        return float(m.group(1))
    return None


def pick_encoder():
    return "h264_nvenc" if has_nvenc() else "libx264"


def determine_target(video_path, kept_scenes, full_probe):
    """Target spec = properties of the last kept segment (the main video tail).

    Resolution uses the reference segment's probed size, falling back to the
    preview-frame size (frame_w/frame_h) captured during thumbnail extraction,
    then to the source's first stream. FPS falls back to the source FPS.
    """
    ref = kept_scenes[-1]
    w, h, fps = ref.get("frame_w"), ref.get("frame_h"), None
    try:
        info = probe_segment(video_path, ref["start"], dur=2.0)
        v = (info or {}).get("video") if info else None
        if v and v["width"] and v["height"]:
            w, h = v["width"], v["height"]
        if v and v.get("fps", 0) > 0:
            fps = v["fps"]
    except Exception:
        pass
    if not w or not h:
        vv = full_probe["video"]
        w, h = vv["width"], vv["height"]
    if not fps or fps <= 0:
        fps = full_probe["video"].get("fps", 0) or 30.0
    a = full_probe.get("audio")
    sr = a["sample_rate"] if a else 48000
    ch = a["channels"] if a else 2
    return {
        "width": int(w), "height": int(h), "fps": float(fps),
        "sample_rate": int(sr), "channels": int(ch),
        "has_audio": a is not None,
    }


def _cut_cmd(ffmpeg, video_path, seg, target, out_path, encoder, preset, cq):
    w, h = target["width"], target["height"]
    fps = target["fps"]
    vf = (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
          f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps:.4f}")
    cmd = [ffmpeg, "-y"]
    cmd += ["-ss", f"{seg['start']:.3f}", "-t", f"{seg['duration']:.3f}",
            "-i", video_path]
    cmd += ["-vf", vf, "-pix_fmt", "yuv420p"]
    if target["has_audio"]:
        cmd += ["-af", f"aresample={target['sample_rate']}"]
    cmd += ["-c:v", encoder]
    if encoder == "h264_nvenc":
        cmd += ["-preset", preset, "-cq", str(cq)]
    else:
        cmd += ["-crf", str(cq), "-preset", preset]
    if target["has_audio"]:
        cmd += ["-ar", str(target["sample_rate"]),
                "-ac", str(target["channels"]), "-c:a", "aac"]
    else:
        cmd += ["-an"]
    cmd += ["-movflags", "+faststart", out_path]
    return cmd


def _run_merge(ffmpeg, video_path, kept_scenes, out_path, target,
               encoder, preset, cq, tmp_dir, progress_cb):
    """Cut + concat with a fixed encoder. Returns (ok, message)."""
    total_dur = sum(s["duration"] for s in kept_scenes) or 1.0
    done_dur = 0.0
    seg_files = []
    for i, seg in enumerate(kept_scenes):
        seg_out = os.path.join(tmp_dir, f"seg_{i:03d}.mp4")
        cmd = _cut_cmd(ffmpeg, video_path, seg, target, seg_out,
                       encoder, preset, cq)

        def _cb(line, _i=i, _d=seg["duration"]):
            t = _parse_time(line)
            if t is not None and progress_cb:
                # ffmpeg reports the *output* timestamp for this single-segment
                # encode, which starts at 0 and grows to _d.
                local = max(0.0, min(t, _d))
                frac = (done_dur + local) / total_dur
                progress_cb(min(0.99, max(0.0, frac)),
                           f"处理片段 {_i + 1}/{len(kept_scenes)}")

        rc, err = run_capture(cmd, progress_cb=_cb,
                            inactivity_timeout=EXPORT_INACTIVITY_TIMEOUT)
        if rc != 0 or not os.path.isfile(seg_out) or os.path.getsize(seg_out) == 0:
            return False, f"片段 {i + 1} 处理失败:\n{err[-2000:]}"
        seg_files.append(seg_out)
        done_dur += seg["duration"]

    if progress_cb:
        progress_cb(0.99, "合并片段中...")
    list_path = os.path.join(tmp_dir, "list.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for sf in seg_files:
            f.write(f"file '{os.path.abspath(sf)}'\n")
    concat_cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0",
                  "-i", list_path, "-c", "copy", out_path]
    rc, err = run_capture(concat_cmd,
                        inactivity_timeout=EXPORT_INACTIVITY_TIMEOUT)
    if rc != 0 or not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
        return False, f"合并失败:\n{err[-2000:]}"
    return True, (f"已导出: {out_path}\n分辨率 {target['width']}x{target['height']}, "
                  f"帧率 {target['fps']:.3f}fps  (编码器: {encoder})")


def merge(video_path, kept_scenes, out_path, encoder=None, preset="p4",
          cq=21, progress_cb=None, tmp_dir=None, target_override=None):
    """Cut + concat kept segments into out_path. Returns (ok, message).

    Transparently falls back from h264_nvenc to libx264 when the GPU encoder
    errors (e.g. driver/NVENC-API mismatch). target_override optionally forces
    the output spec.
    """
    ffmpeg, _ = get_ffmpeg()
    if encoder is None:
        encoder = pick_encoder()
    full_probe = probe_video(video_path)
    if target_override:
        target = dict(target_override)
    else:
        target = determine_target(video_path, kept_scenes, full_probe)

    if not kept_scenes:
        return False, "没有保留任何片段，无法导出。"

    own_tmp = tmp_dir is None
    if own_tmp:
        tmp_dir = tempfile.mkdtemp(prefix="vacut_")
    os.makedirs(tmp_dir, exist_ok=True)

    attempts = [encoder]
    if encoder == "h264_nvenc":
        attempts.append("libx264")
    last_err = "未知错误"
    try:
        for enc in attempts:
            epreset = "medium" if enc == "libx264" else preset
            ok, msg = _run_merge(ffmpeg, video_path, kept_scenes, out_path,
                                 target, enc, epreset, cq, tmp_dir, progress_cb)
            if ok:
                return True, msg
            last_err = msg
            low = msg.lower()
            if "nvenc" not in low and "driver" not in low and "cuda" not in low:
                break
        return False, last_err
    finally:
        if own_tmp:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# --------------------------------------------------------------------------
# 原编码流拷贝路径（需求：切割 + 拼接都用主视频原编码，不重新编码）
# --------------------------------------------------------------------------
_INDEX_KEYS = ("moov atom", "moov", "no moov", "atom not found",
               "could not find codec", "invalid data found", "missing moov",
               "index", "non-monoton", "dts/pts", "out of range")


def _looks_like_index_issue(msg):
    """粗略判断 ffmpeg 报错是否像「视频索引/moov 缺失」类问题。"""
    low = (msg or "").lower()
    return any(k in low for k in _INDEX_KEYS)


def _rebuild_index(ffmpeg, video_path, tmp_dir):
    """重建视频索引（把 moov atom 移到文件头并重建索引）后返回新文件路径；失败返回 None。"""
    rb = os.path.join(tmp_dir, "_rebuilt_index.mp4")
    # 优先：copy + faststart（把 moov 移到头部）
    rc, _ = run_capture([ffmpeg, "-y", "-i", video_path,
                          "-c", "copy", "-movflags", "+faststart", rb],
                        inactivity_timeout=EXPORT_INACTIVITY_TIMEOUT)
    if rc == 0 and os.path.isfile(rb) and os.path.getsize(rb) > 0:
        return rb
    # 回退：仅 copy（某些封装不需要 faststart）
    rc, _ = run_capture([ffmpeg, "-y", "-i", video_path, "-c", "copy", rb],
                        inactivity_timeout=EXPORT_INACTIVITY_TIMEOUT)
    if rc == 0 and os.path.isfile(rb) and os.path.getsize(rb) > 0:
        return rb
    return None


def _write_error_log(app_dir, video_path, out_path, msg, kept_scenes):
    """把详细错误（ffmpeg 命令/返回码/stderr/片段清单）写入应用目录的日志文件。"""
    os.makedirs(app_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(app_dir, f"vacut_error_{ts}.log")
    lines = [
        "=== video-ad-cutter 流拷贝失败（详细错误）===\n",
        f"时间: {ts}\n",
        f"源视频: {video_path}\n",
        f"目标输出: {out_path}\n",
        f"保留片段数: {len(kept_scenes)}\n",
    ]
    for i, s in enumerate(kept_scenes):
        lines.append(
            f"  段{i + 1}: {s.get('start')} -> {s.get('end')} "
            f"时长 {s.get('duration')}\n")
    lines.append("\n--- ffmpeg 错误信息 ---\n")
    lines.append(msg or "(无)\n")
    lines.append(
        "\n说明: 已尝试 smart cut（切点处少量重编码 + 主体 '-c copy' 流拷贝）；"
        "若提示索引/moov 缺失，已重建索引重试；此处为仍失败的其他问题。\n")
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
    except Exception:
        pass
    return log_path


def _try_copy_cut(ffmpeg, video_path, kept_scenes, out_path,
                  tmp_dir, progress_cb, attempt):
    """用 -c copy 切割每个保留段 + concat -c copy 拼接（段间并行）。

    返回 (ok, error_msg)。成功时 error_msg 是成功提示；失败时包含失败命令与 ffmpeg stderr。
    """
    total = len(kept_scenes)
    jobs = []
    for i, seg in enumerate(kept_scenes):
        seg_out = os.path.join(tmp_dir, f"seg_{i:03d}.mp4")
        cmd = [ffmpeg, "-y", "-ss", f"{seg['start']:.3f}", "-i", video_path,
               "-t", f"{seg['duration']:.3f}", "-c", "copy", seg_out]
        desc = f"片段 {i + 1}/{total} 流拷贝 (attempt={attempt})"
        jobs.append((i, cmd, seg_out, float(seg["duration"]), desc))

    # copy 段纯 IO，默认 3 路并发；段数 ≤2 时顺序跑
    workers = _default_workers(None, total)
    seg_files, fail = _run_jobs_parallel(jobs, workers, progress_cb,
                                         "流拷贝片段", EXPORT_INACTIVITY_TIMEOUT)
    if fail:
        return False, fail

    if progress_cb:
        progress_cb(0.97, "合并片段中(原编码)...")
    list_path = os.path.join(tmp_dir, "list.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for sf in seg_files:
            f.write(f"file '{os.path.abspath(sf)}'\n")
    concat_cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0",
                  "-i", list_path, "-c", "copy", out_path]
    rc, err = run_capture(concat_cmd,
                        inactivity_timeout=EXPORT_INACTIVITY_TIMEOUT)
    if rc != 0 or not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
        return False, (f"合并(concat -c copy)失败 (attempt={attempt}):\n"
                        f"命令: {' '.join(concat_cmd)}\n返回码: {rc}\n"
                        f"{err[-3000:]}")
    return True, (f"已导出（兜底路径：切点已对齐关键帧）：{out_path}\n"
                  f"共 {total} 段，可能有零点几秒的出入。")


def _ffconcat_quote(path):
    """给 ffconcat 列表用的路径转义（单引号需转义，反斜杠换成正斜杠）。

    ffconcat 的 file 行是单引号包裹的字符串，路径里的 ' 必须写成 '\\''；
    另外统一用正斜杠，避免 Windows 反斜杠被当成转义符。
    """
    return path.replace("\\", "/").replace("'", "'\\''")


# ⚠️ 已废弃：_try_concat_inpoint（用 ffconcat 的 inpoint/outpoint 单进程直出）
#
# 曾作为"最快路径"（0.24s vs 逐段的 1.8s），但实测在真实视频上【不可靠】，
# 已于 2026-08-29 从 merge_streamcopy 中彻底移除、不再调用：
#   1. -c copy 只能从关键帧切割。inpoint 不在关键帧上时，ffmpeg 会向前对齐
#      到上一个关键帧，把本该丢弃的内容拷回来（test2.mp4：期望 141.077s
#      实际 147.400s，被丢弃的广告原样保留，体积 +19.6%）。
#   2. 即便切点【看起来】在关键帧上也不一定安全：test2.mp4 的关键帧表里
#      存在重复时间戳（31.169, 31.169, 31.174，疑似 VFR / 编辑列表所致），
#      此时 inpoint=31.169 的 seek 会漂移 2.445s，输出比期望长 2.4s。
#   3. 时间精度只有 1ms（"{:.3f}"），对 VFR 视频不够。
# 结论：inpoint/outpoint 只在"切点严格等于关键帧且关键帧表干净"时成立，
# 而这在真实文件上无法事先保证。宁可多起一个 concat 进程，也要用
# _try_smartcut（切点处重编码、主体流拷贝）——它在退化情形下同样是无损流拷贝。


# --------------------------------------------------------------------------
# Smart cut（2026-08-29 修复「去掉的片段还在 / 输出反而变大」）
#
# ❗ 原实现用 concat 解复用器的 inpoint/outpoint 直出，看似优雅，但有一个
# **致命前提被忽略**：`-c copy` 只能从【关键帧】开始切割。inpoint 落在非
# 关键帧时，ffmpeg 会向前对齐到上一个关键帧，于是把被丢弃的内容也拷了回来。
#
# 实测（bench/test2.mp4，关键帧间隔约 4s，切点大多不在关键帧上）：
#   期望 141.077s  -> 实际 147.400s（= 源全长），多出的 6.323s 正好等于
#   被丢弃三段之和；输出 6520 帧 vs 源 5507 帧，体积 +19.6%；对新文件重新
#   检测仍是原来的 13 段。
#   之前 e2e 没暴露，是因为测试片由 ffmpeg 生成、【每个切点本身就是关键帧】。
#
# 修法：合并首尾相连的保留段（用户场景里 10 段其实是一整块），再对每块做
#   头部重编码(到下一个关键帧) + 主体流拷贝 + 尾部重编码，即 smart cut。
#   重编码只占 4.6%（6.46s/141s），主体仍是无损流拷贝。
# --------------------------------------------------------------------------
_SPLIT_TOL = 0.04   # 小于此值(秒)视为"切点正好落在关键帧上"，无需重编码


_KF_CACHE = {}
_KF_CACHE_MAX = 8


def _keyframe_scan_packet(ffprobe, path):
    """packet 级关键帧扫描（快）：只读包头的 flags，不解码帧。

    实测 bench/test3.mp4（955s）：0.63s vs frame 级 2.5s，快约 4 倍，
    且结果与 frame 级完全一致。返回 (ok, times)；老版本 ffprobe 不支持
    ``packet=pts_time`` 时返回 (False, [])。
    """
    cmd = [ffprobe, "-v", "error", "-select_streams", "v:0",
           "-show_packets", "-show_entries", "packet=pts_time,flags",
           "-of", "csv=p=0", path]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=600)
    except Exception:
        return False, []
    if res.returncode != 0:
        return False, []
    out = []
    for line in res.stdout.splitlines():
        parts = line.split(",")
        if len(parts) < 2 or "K" not in parts[-1]:
            continue
        try:
            out.append(float(parts[0]))
        except ValueError:
            continue
    return bool(out), sorted(out)


def _keyframe_scan_frame(ffprobe, path):
    """frame 级关键帧扫描（慢但兼容性最好）：解码器跳过非关键帧。"""
    cmd = [ffprobe, "-v", "error", "-select_streams", "v:0",
           "-skip_frame", "nokey", "-show_entries", "frame=pts_time",
           "-of", "csv=p=0", path]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=600)
    except Exception:
        return []
    if res.returncode != 0:
        return []
    out = []
    for tok in res.stdout.split():
        try:
            out.append(float(tok.split(",")[0]))
        except ValueError:
            continue
    return sorted(out)


def _keyframe_times(ffprobe, path):
    """返回视频流关键帧时间戳列表（秒，升序）；失败/无则返回 []。

    按 (路径, mtime, 大小) 缓存：GUI 每次勾选片段都要预估对齐损失，
    而 ffprobe 扫一遍关键帧在长视频上要几百毫秒到数秒，不能每次都跑。

    优先用 packet 级扫描（不解码，快 3~4 倍）；若返回空（老 ffprobe 或
    异常格式）再回退到 frame 级扫描，避免"快但扫不到"导致功能静默退化。
    """
    try:
        st = os.stat(path)
        key = (path, st.st_mtime, st.st_size)
    except Exception:
        key = None
    if key and key in _KF_CACHE:
        return _KF_CACHE[key]

    ok, out = _keyframe_scan_packet(ffprobe, path)
    if not ok:
        out = _keyframe_scan_frame(ffprobe, path)

    if key:
        if len(_KF_CACHE) >= _KF_CACHE_MAX:
            _KF_CACHE.pop(next(iter(_KF_CACHE)), None)
        _KF_CACHE[key] = out
    return out


def _probe_encode_spec(ffprobe, path):
    """探测源流的编码参数，供重编码片段对齐——必须与流拷贝部分一致，
    否则最后的 `concat -c copy` 会因参数不匹配而失败。"""
    cmd = [ffprobe, "-v", "error", "-show_entries",
           "stream=codec_type,codec_name,profile,width,height,pix_fmt,"
           "sample_aspect_ratio,bit_rate,sample_rate,channels",
           "-show_entries", "format=bit_rate", "-of", "json", path]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=120)
        if res.returncode != 0:
            return None
        data = json.loads(res.stdout)
    except Exception:
        return None
    spec = {"has_audio": False, "vbit": 0, "abit": 128000,
            "width": 0, "height": 0, "pix_fmt": "yuv420p",
            "sar": "1:1", "profile": None, "sr": 48000, "ch": 2}
    try:
        spec["vbit"] = int(data.get("format", {}).get("bit_rate") or 0)
    except (TypeError, ValueError):
        pass
    for s in data.get("streams", []):
        try:
            if s.get("codec_type") == "video" and not spec["width"]:
                spec["width"] = int(s.get("width") or 0)
                spec["height"] = int(s.get("height") or 0)
                spec["pix_fmt"] = s.get("pix_fmt") or "yuv420p"
                spec["sar"] = s.get("sample_aspect_ratio") or "1:1"
                spec["profile"] = s.get("profile")
                b = s.get("bit_rate")
                if b and str(b).isdigit():
                    spec["vbit"] = int(b)
            elif s.get("codec_type") == "audio" and not spec["has_audio"]:
                spec["has_audio"] = True
                spec["sr"] = int(s.get("sample_rate") or 48000)
                spec["ch"] = int(s.get("channels") or 2)
                b = s.get("bit_rate")
                if b and str(b).isdigit():
                    spec["abit"] = int(b)
        except (TypeError, ValueError):
            continue
    return spec if spec["width"] and spec["height"] else None


def _merge_adjacent(scenes, tol=1e-3):
    """合并首尾相连的保留段。

    广告通常在开头/结尾，中间大段是连续的；不合并的话会在【每一个】内部
    边界上都切一刀，而每刀都可能因关键帧对齐引入误差。用户场景里保留的
    10 段其实是连续的一整块，合并后只需在两头各切一次。
    """
    out = []
    for s in scenes:
        s0 = float(s["start"])
        s1 = float(s.get("end") if s.get("end") is not None
                   else s0 + float(s["duration"]))
        if out and abs(s0 - out[-1]["end"]) <= tol:
            out[-1]["end"] = s1
            out[-1]["duration"] = out[-1]["end"] - out[-1]["start"]
        else:
            out.append({"start": s0, "end": s1, "duration": s1 - s0})
    return out


def _kf_at_or_before(kfs, t):
    """返回 <= t 的最后一个关键帧；没有则 0.0（文件开头）。

    kfs 升序。bisect 版等价格代原来的 ``max([k for k in kfs if k <= t], default=0.0)``。
    """
    i = bisect.bisect_right(kfs, t)
    return kfs[i - 1] if i > 0 else 0.0


def _plan_block(kfs, start, end, tol=_SPLIT_TOL,
                head_strict=True, tail_strict=True):
    """为一个连续块规划切割方案。

    返回 [(mode, a, b, dec_from), ...]，mode ∈ {"copy", "encode"}：
      - "copy"   纯流拷贝（无损、极快）
      - "encode" 重编码 [a, b)；因 [a,b) 内没有可用关键帧对齐，必须从
                 ≤ a 的关键帧 dec_from 开始解码，再用 trim 精确截出 [a, b)

    head_strict / tail_strict 表示**该边界之外还有被丢弃的内容**：
      - head_strict=False（start 就是文件开头）→ 前面没东西可"误拷"，
        直接从 start 流拷贝即可，不必为了精确而重编码；
      - tail_strict=False（end 就是文件末尾）→ 同理，一路拷到 end。
    这样"全部保留"这种常见场景仍是 100% 无损流拷贝（实测 0.24s vs 3.3s）。
    """
    # 块内严格位于 (start+tol, end-tol) 的关键帧。kfs 升序，用 bisect 取区间
    # （原来是一次列表推导线性扫描，保留段上千时这里是热点）。
    lo = bisect.bisect_right(kfs, start + tol)
    hi = bisect.bisect_left(kfs, end - tol)
    inner = kfs[lo:hi]

    if head_strict and not inner:
        # 块内一个关键帧都没有（块比一个 GOP 还短）且起点必须精确 -> 整块重编码
        k0 = _kf_at_or_before(kfs, start + tol)
        return [("encode", start, end, max(0.0, k0))]

    parts = []
    cur = start
    if head_strict:
        k1 = inner[0]
        if k1 - cur > tol:                   # start 不在关键帧上 -> 重编码头部
            k0 = _kf_at_or_before(kfs, start + tol)
            parts.append(("encode", start, k1, max(0.0, k0)))
            cur = k1

    if tail_strict:
        klast = inner[-1] if inner else None
        if klast is not None and klast > cur + tol:
            parts.append(("copy", cur, klast, cur))    # 主体：关键帧到关键帧
            cur = klast
        if end - cur > tol:                            # 尾部 -> 重编码
            parts.append(("encode", cur, end, cur))
    else:
        if end - cur > tol:
            parts.append(("copy", cur, end, cur))      # 一路拷到末尾

    if not parts:                                      # 退化：整块流拷贝
        parts = [("copy", start, end, start)]
    return parts


def _part_cmd(ffmpeg, video_path, part, spec, encoder, preset, out_path):
    """生成单个 part 的 ffmpeg 命令（copy 或 encode）。"""
    mode, a, b, dec_from = part
    if mode == "copy":
        return [ffmpeg, "-y", "-hide_banner", "-ss", f"{dec_from:.6f}",
                "-i", video_path, "-t", f"{b - a:.6f}", "-c", "copy", out_path]

    # encode：从 ≤ a 的关键帧开始解码，用 trim/atrim 精确截出 [a, b)
    pre = max(0.0, a - dec_from)
    span = max(pre + 0.01, b - dec_from)
    cmd = [ffmpeg, "-y", "-hide_banner", "-ss", f"{dec_from:.6f}",
           "-i", video_path, "-t", f"{span:.6f}",
           "-vf", f"trim=start={pre:.6f}:end={span:.6f},setpts=PTS-STARTPTS"]
    if spec["has_audio"]:
        cmd += ["-af", f"atrim=start={pre:.6f}:end={span:.6f},"
                       f"asetpts=PTS-STARTPTS,aresample={spec['sr']}"]
    cmd += ["-pix_fmt", spec["pix_fmt"], "-c:v", encoder]
    if encoder == "h264_nvenc":
        cmd += ["-preset", preset, "-rc", "vbr"]
        if spec["vbit"] > 0:
            vb = spec["vbit"]
            cmd += ["-b:v", str(vb), "-maxrate", str(int(vb * 1.5)),
                    "-bufsize", str(int(vb * 3)), "-cq", "26"]
        else:
            cmd += ["-cq", "26"]
    else:
        if spec["vbit"] > 0:
            cmd += ["-b:v", str(spec["vbit"])]
        cmd += ["-crf", "20", "-preset", preset]
    if spec.get("profile"):
        cmd += ["-profile:v", str(spec["profile"]).lower()]
    if spec["has_audio"]:
        cmd += ["-ar", str(spec["sr"]), "-ac", str(spec["ch"]),
                "-b:a", str(spec["abit"]), "-c:a", "aac"]
    else:
        cmd += ["-an"]
    cmd += [out_path]
    return cmd


def _default_workers(encoder, n_parts):
    """并行切割的并发数。

    - 纯 copy 段（无损路线，encoder=None）：3 路。copy 段只做解复用+IO，
      并发收益直接；每段是一次独立 ffmpeg，进程启动(~0.3s)在多段时被摊薄。
    - NVENC 重编码段：2 路。消费级显卡并发编码会话数通常 ≤3，留余量免得到位
      失败——失败会触发整条路线回退，代价比少一路并发大得多。
    - libx264：3 路，多核 CPU 吃得下。
    - 段数 ≤2 不起池（顺序跑更省）。
    """
    if n_parts <= 2:
        return 1
    return 2 if encoder == "h264_nvenc" else 3


def _run_jobs_parallel(jobs, max_workers, progress_cb, tag,
                       inactivity_timeout, sub_progress=False):
    """并行跑一批互不依赖的 ffmpeg 切割任务。

    jobs: ``[(i, cmd, out_path, dur, desc), ...]``，i 是段序号（结果按它归位）。
    sub_progress: True 时按 ffmpeg 的 time= 行做段内细粒度进度（重编码段用）。

    返回 ``(files, fail_msg)``：成功时 files[i] 是第 i 段产出；任一失败返回
    ``(None, 详细错误)``，且不再向池里补新任务（已在跑的会跑完，无法中断
    ffmpeg 子进程，但也就是几秒钟的事）。

    线程安全：每段独立子进程/独立输出文件；共享的进度计数用锁保护；
    progress_cb 由 GUI 提供（内部只往 queue 里 put），本身线程安全。
    """
    n = len(jobs)
    total = sum(j[3] for j in jobs) or 1.0
    results = [None] * n
    # dur: 已完成段时长累计；live: 在跑段各自的当前输出时间；count: 完成
    # 段数；last: 已发出的最大 frac（单调钳制）
    state = {"dur": 0.0, "count": 0, "live": {}, "last": 0.0}
    lock = threading.Lock()

    def _emit(frac, label):
        """发进度：只许前进、不许回退。

        并发段的完成顺序 ≠ 提交顺序，各路 time= 行交替到达，锁内算出的
        frac 完全可能「后算出高值、先到达 GUI」（竞态窗口 = 锁内计算到
        回调之间）。GUI 的进度条是 FIFO 忠实执行，不回钳制就会肉眼可见
        地回跳。这里用 last 兜底。
        """
        with lock:
            if frac > state["last"]:
                state["last"] = frac
            else:
                frac = state["last"]
        if progress_cb:
            progress_cb(frac, label)

    def _run(job):
        i, cmd, out_path, dur, desc = job

        def _line(line):
            # 段内细粒度进度：全局 frac = (已完成 + Σ在跑段当前进度) / 总时长。
            # ⚠️ 不能用「本段 local」单兵作战——3 路并发时三路的 time= 行
            # 交替到达，单段视角会锯齿振荡（用户实测 5%→7%→5% 来回跳），
            # 必须汇总所有在跑段的实时进度才是真实总进度。
            if not progress_cb:
                return
            t = _parse_time(line)
            if t is None:
                return
            local = max(0.0, min(t, dur))
            with lock:
                state["live"][i] = local
                frac = (state["dur"] + sum(state["live"].values())) / total
                cnt = state["count"]
            _emit(min(0.97, max(0.0, frac * 0.95)), f"{tag} {cnt}/{n}")

        rc, err = run_capture(cmd, progress_cb=(_line if sub_progress else None),
                              inactivity_timeout=inactivity_timeout)
        ok = (rc == 0 and os.path.isfile(out_path)
              and os.path.getsize(out_path) > 0)
        with lock:
            state["dur"] += dur
            state["count"] += 1
            # 本段已全额计入 dur，撤出 live，防止刚完成的段被双算
            state["live"].pop(i, None)
            # 同样用汇总公式：完成时若还有其他段在跑，真实进度高于
            # 「仅已完成时长」（否则这个回调会被钳制成停滞）
            frac = min(0.97, (state["dur"] + sum(state["live"].values()))
                       / total * 0.95)
            cnt = state["count"]
        _emit(frac, f"{tag} {cnt}/{n}")
        if not ok:
            return (i, False,
                    f"{desc}失败：\n命令: {' '.join(cmd)}\n返回码: {rc}\n"
                    f"{err[-2500:]}")
        results[i] = out_path
        return (i, True, None)

    if max_workers <= 1 or n <= 1:
        for job in jobs:
            _i, ok, msg = _run(job)
            if not ok:
                return None, msg
        return results, None

    fail_msg = None
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        pending = {}
        nxt = 0
        # 预填满池
        while nxt < n and len(pending) < max_workers:
            pending[ex.submit(_run, jobs[nxt])] = nxt
            nxt += 1
        while pending and fail_msg is None:
            done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
            for f in done:
                pending.pop(f, None)
                _i, ok, msg = f.result()
                if not ok:
                    fail_msg = msg
                # 补位（保持池里有 max_workers 个在跑）
                while nxt < n and len(pending) < max_workers and fail_msg is None:
                    pending[ex.submit(_run, jobs[nxt])] = nxt
                    nxt += 1
        if fail_msg:
            for f in pending:
                f.cancel()      # 还没开始的直接取消；已在跑的跑完即止
    return (None, fail_msg) if fail_msg else (results, None)


def _render_parts(ffmpeg, video_path, parts, spec, encoder, preset,
                  tmp_dir, out_path, progress_cb, tag="切割", prefix="p",
                  attempt=0):
    """执行 parts 计划：并行跑各段，再 concat -c copy 合成。

    parts: ``[(mode, a, b, dec_from), ...]``，mode ∈ {"copy", "encode"}。
    两条路线（lossless / smart cut）共用这一段，保证「跑段 + 合并」的行为
    完全一致，只有前面的规划器不同。

    返回 ``(ok, payload)``：成功时 payload 是统计 dict；失败时是错误字符串。
    """
    total_dur = sum(b - a for _, a, b, _ in parts) or 1.0
    enc_dur = sum(b - a for m, a, b, _ in parts if m == "encode")

    # 快路径：只有一段时不必再起一次 concat 进程（"全部保留"、以及去掉
    # 头尾后中间只剩一整块，都是这个最常见的情形）。直接输出到目标文件，
    # 补上 concat 本来会加的两个参数即可。
    if len(parts) == 1:
        mode = parts[0][0]
        cmd = _part_cmd(ffmpeg, video_path, parts[0], spec, encoder, preset,
                        out_path)
        if progress_cb:
            progress_cb(0.5, f"{tag} 1/1")
        # ⚠️ -avoid_negative_ts make_zero 只能加在【重编码】段上，不能加在
        # 【纯流拷贝】段上。它在 bench/test2.mp4 上实测会让输出凭空多出
        # 61 帧 / 2.3s：规划 18.769s -> 实际 21.214s（首帧还被推后
        # 0.137s）。重编码段确实需要它（AAC 编码延迟会让首帧时间戳为负，
        # 播放器会跳过开头），流拷贝段则本来就从 0 开始。
        extra = ["-avoid_negative_ts", "make_zero"] if mode == "encode" else []
        cmd = cmd[:-1] + extra + ["-movflags", "+faststart"] + [cmd[-1]]
        rc, err = run_capture(cmd,
                            inactivity_timeout=EXPORT_INACTIVITY_TIMEOUT)
        if rc != 0 or not os.path.isfile(out_path) \
                or os.path.getsize(out_path) == 0:
            return False, (f"{tag} 失败 (attempt={attempt}):\n"
                           f"命令: {' '.join(cmd)}\n返回码: {rc}\n{err[-2500:]}")
        if progress_cb:
            progress_cb(1.0, "完成")
        return True, {"total_dur": total_dur, "enc_dur": enc_dur, "nparts": 1}

    # 多段：并行切割。段与段之间无依赖（各自 seek、各自输出文件），
    # 段数多时（几十~上千）省下的是「每段一次的进程启动 + seek 开销」。
    jobs = []
    for i, part in enumerate(parts):
        mode, a, b, _df = part
        seg_out = os.path.join(tmp_dir, f"{prefix}_{i:03d}.mp4")
        cmd = _part_cmd(ffmpeg, video_path, part, spec, encoder, preset,
                        seg_out)
        desc = f"{tag} 第 {i + 1}/{len(parts)} 段({mode} {a:.3f}-{b:.3f})"
        jobs.append((i, cmd, seg_out, b - a, desc))

    workers = _default_workers(encoder, len(parts))
    files, fail = _run_jobs_parallel(jobs, workers, progress_cb, tag,
                                     EXPORT_INACTIVITY_TIMEOUT,
                                     sub_progress=True)
    if fail:
        return False, f"{fail} (attempt={attempt})"

    if progress_cb:
        progress_cb(0.97, "合并片段中…")
    list_path = os.path.join(tmp_dir, f"list_{prefix}.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for sf in files:
            f.write(f"file '{_ffconcat_quote(os.path.abspath(sf))}'\n")
    # -avoid_negative_ts make_zero：拼接后首帧时间戳常为负（AAC 编码延迟，
    # concat 输入实测 start=-0.023），会让播放器跳过开头一小段。
    cmd = [ffmpeg, "-y", "-hide_banner", "-f", "concat", "-safe", "0",
           "-i", list_path, "-c", "copy", "-avoid_negative_ts", "make_zero",
           "-movflags", "+faststart", out_path]
    rc, err = run_capture(cmd)
    if rc != 0 or not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
        return False, (f"{tag} 合并失败 (attempt={attempt}):\n"
                       f"命令: {' '.join(cmd)}\n返回码: {rc}\n{err[-2500:]}")
    return True, {"total_dur": total_dur, "enc_dur": enc_dur,
                  "nparts": len(parts)}


# --------------------------------------------------------------------------
# 路线 1：关键帧对齐的纯流拷贝（GUI 默认，零重编码）
# --------------------------------------------------------------------------
_MIN_PART = 0.05   # 对齐后短于此(秒)的块视为"被关键帧吃掉了"


def _snap_head(kfs, t, tol=_SPLIT_TOL):
    """返回 >= t-tol 的第一个关键帧（只能向【前】吸附）。

    向前 = 少保留一点正片，但绝不把被丢弃的内容拷回来。没有则返回 None。
    kfs 已升序，用 bisect 直接定位（保留段上千时比线性扫描快一个量级）。
    """
    i = bisect.bisect_left(kfs, t - tol)
    return kfs[i] if i < len(kfs) else None


def _snap_tail(kfs, t, tol=_SPLIT_TOL):
    """返回 <= t+tol 的最后一个关键帧（只能向【后】吸附）。没有则 None。"""
    i = bisect.bisect_right(kfs, t + tol)
    return kfs[i - 1] if i > 0 else None


def _plan_lossless(kfs, start, end, head_strict=True, tail_strict=True,
                   tol=_SPLIT_TOL):
    """把块边界吸附到关键帧，产出**只用 -c copy** 的方案（零重编码）。

    head_strict / tail_strict 含义同 ``_plan_block``：该边界之外还有被丢弃
    的内容。区别是这里**不重编码**，而是直接移动切点：

      * 头边界严格 -> 吸附到第一个 >= start - tol 的关键帧（最多少保留一个 GOP）
      * 尾边界严格 -> 吸附到最后一个 <= end + tol 的关键帧（最多少保留一个 GOP）
      * 边界贴着文件开头/结尾 -> **不需要吸附**，直接从 0 拷 / 一路拷到尾，
        既无损也不丢内容（"全部保留"因此是 100% 无损流拷贝）。

    tol（默认 40ms ≈ 一帧）的作用：切点离关键帧只差几十毫秒时，吸附方向
    允许"跨过去"一点点（最多一帧）。这不是妥协——``-ss 12.02 -c copy`` 时
    ffmpeg 本来就会回退到 12.0 那个关键帧、从完整 GOP 开始拷；直接对齐到
    关键帧反而保证了输出首个视频帧一定是关键帧，不会出现开头花屏。
    若要求严格"一帧都不多拷"，tol 可设为 0，但那样每个切点几乎都会白丢
    一整个 GOP（检测时间点很难精确落在关键帧上）。

    返回 ``(parts, info)``。parts 为空表示该块被关键帧完全吃掉
    （info["dropped"]=True）——比一个 GOP 还短的保留段会遇到，此时调用方
    应回退到 smart cut，而不是把这段正片弄丢。
    """
    a = _snap_head(kfs, start, tol) if head_strict else start
    b = _snap_tail(kfs, end, tol) if tail_strict else end
    if a is None or b is None or (b - a) <= _MIN_PART:
        return [], {"dropped": True, "a": a, "b": b,
                    "head_shift": 0.0, "tail_shift": 0.0}
    return ([("copy", a, b, a)],
            {"dropped": False, "a": a, "b": b,
             "head_shift": a - start, "tail_shift": end - b})


def _plan_all_lossless(kfs, blocks, total_src):
    """对全部块做关键帧对齐规划，返回 ``(parts, shifts, dropped)``。

    shifts: ``[(块序号, 头部前进秒, 尾部回退秒), ...]``，供导出后向用户汇报。
    dropped: 被关键帧完全吃掉的块 —— 无损路线无法保留，需回退 smart cut。
    """
    parts, shifts, dropped = [], [], []
    for idx, blk in enumerate(blocks):
        head_strict = blk["start"] > _SPLIT_TOL
        tail_strict = ((total_src <= 0.0)
                       or (blk["end"] < total_src - _SPLIT_TOL))
        p, info = _plan_lossless(kfs, blk["start"], blk["end"],
                                 head_strict, tail_strict)
        if info["dropped"]:
            dropped.append((idx, blk["start"], blk["end"]))
            continue
        parts += p
        shifts.append((idx, max(0.0, info["head_shift"]),
                       max(0.0, info["tail_shift"])))
    return parts, shifts, dropped


def preview_lossless(ffprobe, video_path, kept_scenes):
    """无损路线的**事前**预估：对齐关键帧后会少保留多少。

    GUI 在实际导出前用它提示用户；测试用它算期望时长。返回 dict
    ``{"kept", "raw", "lost", "dropped", "usable"}``，或 None（拿不到关键帧）。
    ``dropped > 0`` / ``usable=False`` 表示无损路线不可行，会回退 smart cut。
    """
    kfs = _keyframe_times(ffprobe, video_path)
    if not kfs:
        return None
    try:
        total_src = float(probe_video(video_path).get("duration") or 0.0)
    except Exception:
        total_src = 0.0
    blocks = _merge_adjacent(kept_scenes)
    parts, _shifts, dropped = _plan_all_lossless(kfs, blocks, total_src)
    kept = sum(b - a for _m, a, b, _d in parts)
    raw = sum(float(s.get("duration")
                    if s.get("duration") is not None
                    else s["end"] - s["start"]) for s in kept_scenes)
    return {"kept": kept, "raw": raw, "lost": max(0.0, raw - kept),
            "dropped": len(dropped), "usable": bool(parts) and not dropped}


def _try_lossless(ffmpeg, ffprobe, video_path, kept_scenes, out_path,
                  tmp_dir, progress_cb, attempt=0):
    """关键帧对齐的纯流拷贝：全程 -c copy，零重编码。

    返回 (ok, msg)。失败时 msg 说明原因，交给上层回退到 smart cut。
    """
    blocks = _merge_adjacent(kept_scenes)
    kfs = _keyframe_times(ffprobe, video_path)
    if not kfs:
        return False, "无法读取关键帧表（无损裁剪需要）"
    try:
        total_src = float(probe_video(video_path).get("duration") or 0.0)
    except Exception:
        total_src = 0.0

    parts, shifts, dropped = _plan_all_lossless(kfs, blocks, total_src)

    if dropped:
        detail = "\n".join(f"    块{i + 1}: {a:.3f}-{b:.3f} "
                           f"(长 {b - a:.3f}s，短于一个 GOP)"
                           for i, a, b in dropped)
        return False, ("以下保留段比一个关键帧间隔(GOP)还短，关键帧对齐后"
                       "会被完全吃掉，无损裁剪无法保留它们，需改用 smart cut：\n"
                       + detail)
    if not parts:
        return False, "关键帧对齐后无可保留内容"

    # copy 分支不读 spec，传 None 即可（保持 _part_cmd 签名统一）
    ok, payload = _render_parts(ffmpeg, video_path, parts, None, None, None,
                                tmp_dir, out_path, progress_cb,
                                tag="无损切割", prefix="los", attempt=attempt)
    if not ok:
        return False, payload

    # 事后校验：规划再准，也架不住「-ss 到关键帧」在某些文件上落错位置。
    # bench/test2.mp4 带编辑列表(video start_time=0.130) 且关键帧时间戳
    # 重复(31.169/31.169/31.174)，`-ss 31.169 -c copy` 实测会偏离 2.4s。
    # 这类问题事前无法预知，只能在导出后量一次；差太多就回退 smart cut
    # （它头部是重编码 + trim，自带修正能力）。
    planned = sum(b - a for _m, a, b, _d in parts)
    try:
        got = float(probe_video(out_path).get("duration") or 0.0)
    except Exception:
        got = 0.0
    if got > 0 and abs(got - planned) > max(0.5, 0.02 * planned):
        return False, (f"无损裁剪结果校验失败：规划 {planned:.3f}s，"
                       f"实际 {got:.3f}s（偏差 {got - planned:+.3f}s）。"
                       f"该文件上 `-c copy` 的定位不可靠，改用 smart cut。")

    lost = sum(h + t for _i, h, t in shifts)
    kept = payload["total_dur"]
    msg = [f"已导出（无损流拷贝，全程未重编码）：{out_path}",
           f"共 {len(kept_scenes)} 段（合并为 {len(blocks)} 个连续块）；"
           f"切点已对齐关键帧。"]
    if lost > 0.005:
        pct = lost / (kept + lost) * 100 if (kept + lost) else 0
        msg.append(f"因对齐关键帧共少保留 {lost:.2f}s（占原保留时长 {pct:.1f}%）："
                   f"头部前进合计 {sum(h for _, h, _ in shifts):.2f}s，"
                   f"尾部回退合计 {sum(t for _, _, t in shifts):.2f}s。")
        if len(shifts) <= 6:
            for i, h, t in shifts:
                if h > 0.005 or t > 0.005:
                    msg.append(f"    块{i + 1}: 头 +{h:.2f}s / 尾 -{t:.2f}s")
    else:
        msg.append("所有切点恰好落在关键帧上，未丢失任何内容。")
    return True, "\n".join(msg)


# --------------------------------------------------------------------------
# 路线 2：smart cut（切点精确，切点处少量重编码）
# --------------------------------------------------------------------------
def _try_smartcut(ffmpeg, ffprobe, video_path, kept_scenes, out_path,
                  tmp_dir, progress_cb, attempt=0):
    """合并连续段后，头部重编码 + 主体流拷贝 + 尾部重编码，再 concat。"""
    blocks = _merge_adjacent(kept_scenes)
    kfs = _keyframe_times(ffprobe, video_path)
    spec = _probe_encode_spec(ffprobe, video_path)
    if spec is None:
        return False, "无法探测源编码参数（smart cut 需要它来对齐重编码片段）"

    # 关键帧拿不到（probe 失败/纯音频）时无法做关键帧规划，交给上层兜底
    if not kfs:
        return False, "无法读取关键帧表（smart cut 需要）"

    # 规划：位于文件开头/末尾的块，那一边不必为了精确而重编码
    try:
        total_src = float(probe_video(video_path).get("duration") or 0.0)
    except Exception:
        total_src = 0.0
    parts = []
    for blk in blocks:
        head_strict = blk["start"] > _SPLIT_TOL
        tail_strict = (total_src <= 0.0) or (blk["end"] < total_src - _SPLIT_TOL)
        parts += _plan_block(kfs, blk["start"], blk["end"], _SPLIT_TOL,
                             head_strict=head_strict, tail_strict=tail_strict)
    if not parts:
        return False, "切割方案为空"

    encoder = pick_encoder()
    epreset = "medium" if encoder == "libx264" else "p4"
    ok, payload = _render_parts(ffmpeg, video_path, parts, spec, encoder,
                                epreset, tmp_dir, out_path, progress_cb,
                                tag="精确切割", prefix="sc", attempt=attempt)
    if not ok:
        return False, payload

    pct = payload["enc_dur"] / payload["total_dur"] * 100 \
        if payload["total_dur"] else 0
    return True, (f"已导出：{out_path}\n"
                  f"共 {len(kept_scenes)} 段（合并为 {len(blocks)} 个连续块）；"
                  f"仅切点处 {payload['enc_dur']:.1f}s 重新编码（占 {pct:.1f}%），"
                  f"其余原编码流拷贝。")


def merge_streamcopy(video_path, kept_scenes, out_path, progress_cb=None,
                     tmp_dir=None, app_dir=None, mode="lossless"):
    """裁剪并拼接保留片段，尽量用主视频原编码（流拷贝）。

    mode:
      ``"lossless"``（**默认**）—— 关键帧对齐的纯流拷贝，全程 ``-c copy``、
      零重编码；切点吸附到关键帧，每个内部切点最多少保留一个 GOP。
      ``"precise"`` —— smart cut，切点精确，切点处少量重编码。

    流程（2026-08-29 重写）：
      0. 先合并首尾相连的保留段（减少切割点，也减少关键帧对齐的损失）；
      1. 按 mode 决定的顺序尝试两条路线（lossless -> precise 或反过来）；
         每条路线失败且疑似索引问题 -> 重建索引后重试一次；
      2. 都不行 -> 回退「逐段 -c copy + concat」（切点对齐关键帧，略有出入）；
      3. 还失败 -> 写详细错误日志。

    两条路线都遵守铁律：**绝不把被丢弃的内容拷回来**（这是当初「去了广告
    还在 + 文件反而变大」的根因）。lossless 路线靠"头只往前、尾只往后"的
    吸附方向保证；smart cut 靠"切点处重编码"保证。

    ⚠️ 这里刻意【不使用】ffconcat 的 inpoint/outpoint 直出（旧实现）：
    它在真实视频上会把本该丢弃的内容拷回来。详见 _ffconcat_quote 上方的
    废弃说明。
    """
    ffmpeg, ffprobe = get_ffmpeg()
    app_dir = app_dir or _APP_DIR
    if not kept_scenes:
        return False, "没有保留任何片段，无法导出。"

    order = (["lossless", "precise"] if mode != "precise"
             else ["precise", "lossless"])

    def _run(route, src, attempt):
        if route == "lossless":
            return _try_lossless(ffmpeg, ffprobe, src, kept_scenes, out_path,
                                 tmp_dir, progress_cb, attempt)
        return _try_smartcut(ffmpeg, ffprobe, src, kept_scenes, out_path,
                             tmp_dir, progress_cb, attempt)

    own_tmp = tmp_dir is None
    if own_tmp:
        tmp_dir = tempfile.mkdtemp(prefix="vacut_sc_")

    try:
        logs = []
        for route in order:
            ok, msg = _run(route, video_path, 0)
            if ok:
                return True, msg
            logs.append(f"[{route} @ 原文件]\n{msg}")

            if _looks_like_index_issue(msg):
                rb = _rebuild_index(ffmpeg, video_path, tmp_dir)
                if rb:
                    ok, msg = _run(route, rb, 1)
                    if ok:
                        return True, msg
                    logs.append(f"[{route} @ 重建索引]\n{msg}")

        # 兜底：逐段 -c copy + concat（切点对齐关键帧，可能有出入）
        ok, fallback_msg = _try_copy_cut(ffmpeg, video_path, kept_scenes,
                                         out_path, tmp_dir, progress_cb,
                                         attempt=0)
        if ok:
            return True, (fallback_msg +
                          "\n（注意：切点已对齐关键帧，可能有零点几秒出入）")

        if _looks_like_index_issue(fallback_msg):
            rb = _rebuild_index(ffmpeg, video_path, tmp_dir)
            if rb:
                ok, msg3 = _try_copy_cut(ffmpeg, rb, kept_scenes, out_path,
                                         tmp_dir, progress_cb, attempt=1)
                if ok:
                    return True, msg3
                fallback_msg = msg3

        # 写详细错误日志到应用目录
        log_path = _write_error_log(
            app_dir, video_path, out_path,
            "\n\n".join(logs) + f"\n\n[逐段切割兜底路线]\n{fallback_msg}",
            kept_scenes)
        return False, (f"导出失败。详细错误已写入：\n{log_path}")
    finally:
        if own_tmp:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    import sys
    vp = sys.argv[1]
    op = sys.argv[2]
    ranges = sys.argv[3] if len(sys.argv) > 3 else ""
    segs = []
    for part in ranges.split(","):
        if "-" in part:
            a, b = part.split("-")
            segs.append({"start": float(a), "end": float(b),
                         "duration": float(b) - float(a)})
    ok, msg = merge(vp, segs, op, encoder="libx264", preset="medium", cq=21)
    print(msg)
