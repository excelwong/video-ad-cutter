"""Cut and re-combine video segments with a unified target resolution/fps/audio.

Pipeline per kept segment:
  ffmpeg -ss S -t D -i in -vf "<scale/pad/fps>" -af "<aresample>" \
         -c:v h264_nvenc -c:a aac out_i.mp4
then the concat demuxer joins all out_i.mp4 (identical params) into the final.

The target resolution / fps / audio are taken from the *reference* segment
(the last kept one, since ads are appended at the front -> the tail is the main
video). This guarantees requirement #5: output follows the main video's spec,
even if the ad segment had a different resolution/fps.

GPU: tries h264_nvenc first (RTX 2060 / NVENC), and transparently falls
back to libx264 (CPU) if the GPU encoder is unavailable or errors.
"""
import datetime
import os
import re
import shutil
import tempfile

from utils import (get_ffmpeg, has_nvenc, probe_video, probe_segment,
                   run_capture)

# 应用所在目录（错误日志写到这里）
_APP_DIR = os.path.dirname(os.path.abspath(__file__))

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

        rc, err = run_capture(cmd, progress_cb=_cb)
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
    rc, err = run_capture(concat_cmd)
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
                          "-c", "copy", "-movflags", "+faststart", rb])
    if rc == 0 and os.path.isfile(rb) and os.path.getsize(rb) > 0:
        return rb
    # 回退：仅 copy（某些封装不需要 faststart）
    rc, _ = run_capture([ffmpeg, "-y", "-i", video_path, "-c", "copy", rb])
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
        "\n说明: 已尝试用主视频原编码流拷贝('-c copy')；若提示索引/moov 缺失，"
        "已重建索引重试；此处为仍失败的其他问题，未做任何重新编码。\n")
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
    except Exception:
        pass
    return log_path


def _try_copy_cut(ffmpeg, video_path, kept_scenes, out_path,
                  tmp_dir, progress_cb, attempt):
    """用 -c copy 切割每个保留段 + concat -c copy 拼接。

    返回 (ok, error_msg)。成功时 error_msg 是成功提示；失败时包含失败命令与 ffmpeg stderr。
    """
    total = len(kept_scenes)
    seg_files = []
    for i, seg in enumerate(kept_scenes):
        seg_out = os.path.join(tmp_dir, f"seg_{i:03d}.mp4")
        cmd = [ffmpeg, "-y", "-ss", f"{seg['start']:.3f}", "-i", video_path,
                "-t", f"{seg['duration']:.3f}", "-c", "copy", seg_out]
        rc, err = run_capture(cmd)
        if rc != 0 or not os.path.isfile(seg_out) or os.path.getsize(seg_out) == 0:
            return False, (f"片段 {i + 1}/{total} 流拷贝失败 (attempt={attempt}):\n"
                            f"命令: {' '.join(cmd)}\n返回码: {rc}\n"
                            f"{err[-3000:]}")
        seg_files.append(seg_out)
        if progress_cb:
            progress_cb(min(0.95, (i + 1) / total * 0.95),
                        f"流拷贝片段 {i + 1}/{total}")
    if progress_cb:
        progress_cb(0.97, "合并片段中(原编码)...")
    list_path = os.path.join(tmp_dir, "list.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for sf in seg_files:
            f.write(f"file '{os.path.abspath(sf)}'\n")
    concat_cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0",
                  "-i", list_path, "-c", "copy", out_path]
    rc, err = run_capture(concat_cmd)
    if rc != 0 or not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
        return False, (f"合并(concat -c copy)失败 (attempt={attempt}):\n"
                        f"命令: {' '.join(concat_cmd)}\n返回码: {rc}\n"
                        f"{err[-3000:]}")
    return True, (f"已导出(主视频原编码流拷贝，未重新编码): {out_path}\n"
                  f"共 {total} 段；体积与原片保持一致（仅裁剪掉未保留部分）。")


def merge_streamcopy(video_path, kept_scenes, out_path, progress_cb=None,
                     tmp_dir=None, app_dir=None):
    """用主视频**原编码**裁剪并拼接保留片段（不重新编码）。

    流程：
      1. 逐段 `-c copy` 切割 + `concat -c copy` 拼接；
      2. 若失败且疑似「视频索引/moov 缺失」，重建索引后重试一次；
      3. 仍为其他问题 -> 把详细错误写入应用目录的 vacut_error_<时间戳>.log。
    """
    ffmpeg, _ = get_ffmpeg()
    app_dir = app_dir or _APP_DIR
    if not kept_scenes:
        return False, "没有保留任何片段，无法导出。"

    own_tmp = tmp_dir is None
    if own_tmp:
        tmp_dir = tempfile.mkdtemp(prefix="vacut_sc_")

    # 第一次尝试（原视频直接流拷贝）
    ok, msg = _try_copy_cut(ffmpeg, video_path, kept_scenes, out_path,
                             tmp_dir, progress_cb, attempt=0)
    if ok:
        if own_tmp:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return True, msg

    # 疑似索引问题 -> 重建索引后重试
    if _looks_like_index_issue(msg):
        rb = _rebuild_index(ffmpeg, video_path, tmp_dir)
        if rb:
            ok2, msg2 = _try_copy_cut(ffmpeg, rb, kept_scenes, out_path,
                                       tmp_dir, progress_cb, attempt=1)
            if ok2:
                if own_tmp:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                return True, msg2
            msg = msg2

    # 其他问题 -> 写详细错误日志到应用目录
    log_path = _write_error_log(app_dir, video_path, out_path, msg, kept_scenes)
    if own_tmp:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return False, (f"流拷贝失败（非索引问题，或重建索引后仍失败）。\n"
                    f"未重新编码。详细错误已写入：\n{log_path}")


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
