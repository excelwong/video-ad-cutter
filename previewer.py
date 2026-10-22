"""Extract the first frame of each detected segment as a preview thumbnail.

2026-10-21 重写：批量抽帧（按块 seek + select 滤镜，单遍解码扫全片）
------------------------------------------------------------------
原实现对**每个场景单独 spawn 一次 ffmpeg**（``-ss`` + ``-frames:v 1``）：
2913 个场景实测耗时 53.4 分钟（~1.07s/次：进程启动 + 重复 seek + 重复
解码），是「检测 → 缩略图 → 渲染」三步里最慢的一步，也是 868 张卡之后
界面空白的体感元凶之一（等太久）。

改为**按块 seek + select 批量抽帧**：每个块 seek 到块首、``-t`` 收敛到
块尾，所有目标帧经 select 一次抽出，全片累计只解码一遍：

    ffmpeg -an -sn -dn -ss {块首秒} -i video -t {块跨度}
           -vf "scale=240:135,select='eq(n,F1-F0)+eq(n,F2-F0)+…'"
           -fps_mode passthrough -q:v 3 -f image2 临时目录/b_%05d.jpg

实测结论（2026-10-21，bench/realistic_cuts.mp4，1080p30，RTX 2060）——
这些数字决定了下面的路径选择，换机器/换 ffmpeg 后可复跑验证：

1. **select 的 n 与 t 在输入 -ss 之后都从 0 重新计数**（实测
   ``-ss 20`` 后 ``eq(n,0)`` 精确命中跳转点、输出 time 为相对值），
   故块内用 ``eq(n, F-F0)`` 表达。
2. **⚠️ ``-t`` 必须放在 ``-i`` 之前（输入侧）**：输出侧 ``-t`` 统计的
   是 select 放出帧的 pts 累计——最后一项之后 select 不再出帧、输出
   时间冻结，``-t span`` 永远满足不了，ffmpeg 会**一路解码到文件尾**。
   这个坑极隐蔽：单块内容可快速解码（实测 0.96s），但块块都解到 EOF，
   37 块累计 276.57s（≈ Σ(文件尾−块首)/解码速度，与实测吻合）。
   移到输入侧后 3000 场景 37 块 ≈ 20s 量级。
3. **CPU 软解优先于 GPU**：本机纯软解 1080p ≈ 1590fps，而
   ``-hwaccel cuda`` 的 NVDEC native 通路仅 ≈ 703fps（每帧 CUDA
   同步开销）；检测器同款对比 3.65s(CPU) vs 6.41s(GPU)、检测结果
   完全一致。≤1080p 源软解更快；GPU 批量路径保留为二级回退
   （4K+ 源或软解异常时更可能受益，也可用 ``prefer="gpu"`` 强制）。
3. **``-c:v h264_cuvid``（真 NVDEC，≈8000fps）在本 nightly 构建里
   不可用**：与 select / image2 组合必报
   ``Impossible to convert … auto_scale_0``（ffmpeg 给 cuvid 解码器
   自动插入滤镜的 bug，auto_scale 无法输出 cuda 格式）。已试遍
   hwaccel_output_format / hwdownload 前置 / 去掉 -fps_mode 等组合，
   均失败；换 ffmpeg 版本后可重新评估，届时是真·数量级的提升。
4. **ffmpeg select 表达式的项数硬限制 ~100**：实测超过 100 个
   ``eq(n,X)`` 相加就直接 ``Error initializing filters`` +
   ``Cannot allocate memory``（rc=-12），与字符数无关。故按 **80 项/块**
   切块（2913 场景 = 37 块），每块 seek 到块首、``-t`` 收敛到块尾，
   全片累计只解码一遍。
5. 缩略图直接按 **240x135** 抽取（GUI 显示尺寸，gui._THUMB_W），
   与原实现「抽全分辨率再 PIL 缩放到 240x135」视觉效果一致，但省掉
   全分辨率解码后的缩放与磁盘 IO。竖屏会被轻微压扁——原实现同样如此
   （GUI 的 _load_thumb_image 恒 resize 到 240x135），非回归。
6. frame_w/frame_h 改从 probe 直接取**源分辨率**（原实现用 PIL 读
   抽出的全分辨率图，值相同但每张多一次解码开销）。

健壮性：任何一块失败都会按 CPU → GPU → 逐段 -ss 三级回退，保证任何
环境都有结果；块内缺失的帧号由逐段兜底补抽，绝不让某一段没有预览。
"""
import math
import os
import re
import shutil

from utils import get_ffmpeg, probe_video, run_capture

# 缩略图尺寸：与 GUI 显示尺寸一致（gui._THUMB_W / _load_thumb_image）
_THUMB_W, _THUMB_H = 240, 135

# select 表达式的**项数**硬限制（2026-10-21 实测，本 nightly ffmpeg 构建）：
# ``eq(n,X)+eq(n,Y)+…`` 超过 100 项即报
# ``[AVFilterGraph] Error initializing filters`` / ``Cannot allocate memory``
# （rc=-12 ENOMEM）——与表达式**字符数无关**（101 个短项仅 899 字符也挂，
# 100 个长项 1036 字符反而通过），是解析器对项数/函数调用的限制。
# 取 80 留 20% 余量。2913 个场景 → 37 块，每块 seek 到块首，全片累计
# 仍只解码一遍（块间串行，单路已跑满解码）。
_MAX_TERMS = 80

# 字符数二级保护：Windows CreateProcess 命令行上限 32,767 字符，
# 给 ffmpeg 路径、滤镜链其它部分留足余量（正常不会触发）
_MAX_EXPR = 16000

# 批量抽帧的临时子目录（建在 thumb_dir 下，硬链接/改名同卷，结束后删除）
_TMP_NAME = "_batch_tmp"

# ffmpeg 进度行里的输出帧数（= select 已写出的帧数）
_RE_FRAME = re.compile(r"frame=\s*(\d+)")


def extract_first_frame(video_path, start_sec, out_path, fast_seek=True):
    """Extract one JPEG frame near `start_sec` into out_path.

    fast_seek=True uses input-side -ss (fast, lands on nearest keyframe; fine
    for a thumbnail). Returns True on success.

    现仅作**逐段兜底**路径（批量抽帧缺失的少数帧）；正常流程不会走到这。
    """
    ffmpeg, _ = get_ffmpeg()
    cmd = [ffmpeg, "-y"]
    if fast_seek:
        cmd += ["-ss", f"{start_sec:.3f}"]
    cmd += ["-i", video_path]
    if not fast_seek:
        cmd += ["-ss", f"{start_sec:.3f}"]
    cmd += ["-frames:v", "1", "-q:v", "2", out_path]
    rc, _ = run_capture(cmd)
    return rc == 0 and os.path.isfile(out_path) and os.path.getsize(out_path) > 0


def _probe_meta(video_path):
    """一次 probe，拿全片共用的帧率 / 宽高 / start_time / 总帧数。

    返回 (frame_dur, fps, w, h, start_time, total_frames)：
    frame_dur = 每帧秒数（有理数精确值，29.97 片源不会漂移）；
    fps = 浮点帧率（仅用于进度/兜底计算）。
    """
    info = probe_video(video_path)
    v = (info or {}).get("video") or {}
    fps = float(v.get("fps") or 0.0) or 30.0
    num = int(v.get("fps_num") or 0)
    den = int(v.get("fps_den") or 0)
    frame_dur = (den / num) if (num > 0 and den > 0) else (1.0 / fps)
    w = int(v.get("width") or 0)
    h = int(v.get("height") or 0)
    start_time = float(v.get("start_time") or 0.0)
    duration = float(info.get("duration") or 0.0)
    total_frames = int(round(duration / frame_dur)) if duration > 0 else 0
    return frame_dur, fps, w, h, start_time, total_frames


# 帧号取整的浮点保护：(T-start)/frame_dur 恰好是整数时，二进制浮点
# 可能算出 449.9999999 或 450.0000001，减掉 1e-6 再 ceil 两种都不误判
_EPS = 1e-6


def _frame_numbers(scenes, frame_dur, start_time, total_frames):
    """场景 start(秒, pts) -> 0-based 帧号（与 select 的 n 对齐）。

    语义与老实现的逐段 ``-ss T`` 一致：取**第一个 pts ≥ T 的帧**
    （accurate seek 的行为），即向上取整——不能用 round：切点处
     round 会落到上一场景最后一帧，缩略图就成了一张完全不同的画面
    （实测 realistic_cuts.mp4 切点差异 >100/255）。
    减去 start_time 是因为首帧 pts 不一定为 0（如 0.021354）：
    帧 i 的 pts = start_time + i × frame_dur。
    """
    out = []
    for sc in scenes:
        t = float(sc.get("start") or 0.0) - start_time
        f = math.ceil(t / frame_dur - _EPS)
        if total_frames > 0:
            f = max(0, min(f, total_frames - 1))
        else:
            f = max(0, f)
        out.append(int(f))
    return out


def _plan_chunks(frames):
    """把去重后的帧号表切成若干块。

    双重约束：① 每块 ≤ _MAX_TERMS 项（ffmpeg select 解析器的硬限制，
    超过 ~100 项直接 ENOMEM，见上方注释）；② 表达式字符数 ≤ _MAX_EXPR
    （Windows 命令行上限，超长帧号时兜底）。帧号已按场景顺序（即时间
    顺序）排列且去重，直接顺序切块。
    """
    chunks, cur, cur_len = [], [], 0
    for f in frames:
        if cur and (len(cur) >= _MAX_TERMS
                    or cur_len + len(str(f)) + 8 > _MAX_EXPR):
            chunks.append(cur)
            cur, cur_len = [], 0
        cur.append(f)
        cur_len += len(str(f)) + 8      # "eq(n,<f>)+" 的约长
    if cur:
        chunks.append(cur)
    return chunks


def _chunk_cmd(ffmpeg, video_path, chunk, out_pattern, frame_dur, gpu):
    """组装一个块的 ffmpeg 命令（CPU 软解或 GPU NVDEC native 通路）。"""
    f0, f_last = chunk[0], chunk[-1]
    # -ss 精度给到微秒：块首 pts 必须精确命中，否则首帧对不上、整块偏移
    t0 = f0 * frame_dur
    # 覆盖到最后一个被选帧 + 2 帧余量；+0.5s 兜底帧率/容器时间基舍入
    span = (f_last - f0) * frame_dur + 2.0 * frame_dur + 0.5
    rel = "+".join(f"eq(n,{f - f0})" for f in chunk)
    if gpu:
        # NVDEC native 通路（-hwaccel cuda）：GPU 解码 + GPU 缩放后回传。
        # 注意必须显式 hwdownload + format=nv12，否则 select 吃不了 CUDA 帧。
        vf = (f"scale_cuda={_THUMB_W}:{_THUMB_H},hwdownload,format=nv12,"
              f"select='{rel}'")
        pre = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    else:
        vf = f"scale={_THUMB_W}:{_THUMB_H},select='{rel}'"
        pre = []
    # ⚠️ -t 必须在 -i 之前（输入侧）：输出侧 -t 统计的是 select 放出帧的
    # pts 累计，最后一项之后 select 不再出帧、输出时间冻结，-t 永远满足
    # 不了，ffmpeg 会一路解码到文件尾（3000 场景 37 块实测 276s → 20s）。
    return ([ffmpeg, "-y", "-hide_banner", "-an", "-sn", "-dn"] + pre +
            ["-ss", f"{t0:.6f}", "-t", f"{span:.3f}", "-i", video_path,
             "-vf", vf, "-fps_mode", "passthrough",
             "-q:v", "3", "-f", "image2", out_pattern])


def _run_chunk(cmd, progress_cb, base_done, total,
               stall_timeout=120.0, hard_timeout=3600.0):
    """跑一个块，返回 (是否成功, 实际抽出的帧数)。

    按**非空**输出文件计数比 rc 更可靠：rc==0 但 -t 截断导致少帧的情况
    确实存在；个别 0 字节文件也会被这里识别出来（走逐段兜底补抽）。
    """
    tmp_dir = os.path.dirname(cmd[-1])
    os.makedirs(tmp_dir, exist_ok=True)

    state = {"last_done": -1}

    def _line(line):
        # ffmpeg 的 frame= 是**输出**帧数 = select 已写出的帧数，
        # 正好映射成"已完成场景数"，把批量抽帧的进度做实（原来是硬编码 100%）。
        m = _RE_FRAME.search(line)
        if not m or not progress_cb:
            return
        try:
            done = base_done + int(m.group(1))
        except ValueError:
            return
        if done != state["last_done"]:
            state["last_done"] = done
            try:
                progress_cb(min(done, total), total)
            except Exception:
                pass    # 回调异常绝不能弄死读取线程（会死锁 stderr 管道）

    rc, _err = run_capture(cmd, progress_cb=_line,
                           inactivity_timeout=stall_timeout,
                           timeout=hard_timeout)
    got = 0
    try:
        got = sum(1 for n in os.listdir(tmp_dir)
                  if n.startswith("b_") and n.endswith(".jpg")
                  and os.path.getsize(os.path.join(tmp_dir, n)) > 0)
    except OSError:
        pass
    return rc == 0, got


def build_previews(video_path, scenes, thumb_dir, progress_cb=None,
                   prefer="cpu"):
    """Extract a thumbnail for each scene; also record each frame's native size.

    Mutates `scenes` in place adding "thumb" (path) and "frame_w"/"frame_h".

    prefer: "cpu"（默认，实测 ≤1080p 更快）或 "gpu"（强制 GPU 优先）。
    两条批量路径都失败时，逐段 -ss 兜底，保证每段都有预览。
    """
    os.makedirs(thumb_dir, exist_ok=True)
    total = len(scenes)
    if total == 0:
        return scenes

    frame_dur, fps, vw, vh, start_time, total_frames = _probe_meta(video_path)
    # frame_w/frame_h = 源分辨率（与原实现 PIL 读全分辨率抽图等价，零成本）
    for sc in scenes:
        sc["frame_w"], sc["frame_h"] = (vw or None), (vh or None)
        sc["thumb"] = None

    ffmpeg, _ = get_ffmpeg()
    frames = _frame_numbers(scenes, frame_dur, start_time, total_frames)

    # 帧号 -> 场景下标（多个场景落到同一帧时只抽一次，再分发给各段）
    frame_to_scenes = {}
    for idx, f in enumerate(frames):
        frame_to_scenes.setdefault(f, []).append(idx)
    distinct = sorted(frame_to_scenes)          # 已天然按时间升序
    chunks = _plan_chunks(distinct)

    tmp_root = os.path.join(thumb_dir, _TMP_NAME)
    if os.path.isdir(tmp_root):
        shutil.rmtree(tmp_root, ignore_errors=True)
    os.makedirs(tmp_root, exist_ok=True)

    order = ["gpu", "cpu"] if prefer == "gpu" else ["cpu", "gpu"]

    def _place_first(src, dst):
        """首个场景：直接重命名（一次系统调用，无需删除临时文件）。"""
        try:
            os.replace(src, dst)
            return True
        except OSError:
            return False

    def _place_extra(src, dst):
        """同帧的其它场景：硬链接/复制（src 留给你后面的清理）。"""
        try:
            os.link(src, dst)
            return True
        except FileExistsError:
            return True                     # 上次运行的残留，内容以本次为准
        except OSError:
            try:
                shutil.copyfile(src, dst)
                return True
            except OSError:
                return False

    base_done = 0
    try:
        for ci, chunk in enumerate(chunks):
            n_expected = len(chunk)
            span_sec = (chunk[-1] - chunk[0]) * frame_dur + 1.0
            # 每块独立子目录：重试/失败时整目录扔掉，绝不与下一块的文件名碰撞
            chunk_dir = os.path.join(tmp_root, f"c{ci:03d}")
            os.makedirs(chunk_dir, exist_ok=True)
            pattern = os.path.join(chunk_dir, "b_%05d.jpg")
            got = 0
            for mode in order:
                gpu = (mode == "gpu")
                cmd = _chunk_cmd(ffmpeg, video_path, chunk, pattern,
                                 frame_dur, gpu)
                ok, got = _run_chunk(
                    cmd, progress_cb, base_done, total,
                    hard_timeout=max(900.0, span_sec / 2.0))
                if ok and got > 0:
                    break
                # 本路径失败：整目录扔掉重试（rename 是一次系统调用，
                # 比逐个 os.remove 便宜得多——后者在某些环境里 0.2s/文件）
                try:
                    os.replace(chunk_dir, chunk_dir + "_fail")
                    shutil.rmtree(chunk_dir + "_fail", ignore_errors=True)
                    os.makedirs(chunk_dir, exist_ok=True)
                except OSError:
                    shutil.rmtree(chunk_dir, ignore_errors=True)
                    os.makedirs(chunk_dir, exist_ok=True)
            if got > 0:
                # 第 j 个输出文件（j 从 1 起）对应块内第 j 个帧号；
                # select 按流顺序输出、块内帧号升序，故位置映射成立
                # （-t 截断只会缺尾部帧，不影响已有文件与场景的对应）。
                for j, f in enumerate(chunk, start=1):
                    src = os.path.join(chunk_dir, f"b_{j:05d}.jpg")
                    if not os.path.isfile(src):
                        continue
                    owners = frame_to_scenes[f]
                    for k, idx in enumerate(owners):
                        dst = os.path.join(thumb_dir, f"seg_{idx:03d}.jpg")
                        if k == 0:
                            placed = _place_first(src, dst)
                        else:
                            placed = _place_extra(src, dst)
                        if placed:
                            scenes[idx]["thumb"] = dst
            # 无论成败都推进进度基数（缺失的帧由后面逐段兜底补）
            base_done += n_expected
            if progress_cb:
                try:
                    progress_cb(min(base_done, total), total)
                except Exception:
                    pass
            shutil.rmtree(chunk_dir, ignore_errors=True)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    # 兜底：批量没抽到的少数帧（seek 语义异常 / -t 截断 / 路径全挂），
    # 用原来的逐段 -ss 补抽，绝不让某一段没有预览。
    missing = [i for i, sc in enumerate(scenes) if not sc.get("thumb")]
    for k, i in enumerate(missing):
        out = os.path.join(thumb_dir, f"seg_{i:03d}.jpg")
        if extract_first_frame(video_path, scenes[i]["start"], out):
            scenes[i]["thumb"] = out
        if progress_cb:
            try:
                progress_cb(min(base_done + k + 1, total), total)
            except Exception:
                pass
    return scenes
