# -*- coding: utf-8 -*-
"""回归测试：concat 时基不一致 -> 视频时间戳拉伸（音画渐进错位）。

根因（2026-10-29 用户真实片源实锤）：
  流拷贝段继承片源视频时基（下载片源常见 1/30000，每帧 1000 tick），
  重编码段 mp4 默认时基 1/15360（每帧 512 tick）。concat demuxer 对
  时基不一致的输入【不做 tick 换算】：列表以重编码段开头时（smart cut
  的头部重编码正是如此），copy 段 tick 被原样透传，视频时间戳整体拉伸
  30000/15360 ~= 1.953 倍 —— 音画渐进漂移、有效帧率减半。

修复：merger._part_cmd / _cut_cmd / _try_copy_cut 对所有段显式钉死
  -video_track_timescale 15360，让 concat 输入时基永远同质。

本测试用 1/30000 时基的合成片源（模拟真实下载片源），触发 smart cut
产生「encode 开头 + copy 主体」的混合列表，断言导出结果音画时长一致。
"""
import os
import subprocess
import sys

APP = r"C:\Users\excel\WorkBuddy\Claw\video-ad-cutter"
sys.path.insert(0, APP)
import merger  # noqa: E402
from utils import get_ffmpeg  # noqa: E402

T = r"C:\Users\excel\AppData\Local\Temp\timescale"
SRC = os.path.join(T, "src30000.mp4")
OUT_MIX = os.path.join(T, "out_smartcut.mp4")
OUT_COPY = os.path.join(T, "out_lossless.mp4")
FF, FP = get_ffmpeg()


def probe(path, entries, sel="v:0"):
    r = subprocess.run(
        [FP, "-v", "error", "-select_streams", sel, "-show_entries",
         f"stream={entries}", "-of", "default=nw=1", path],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=120)
    out = {}
    for line in r.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k] = v.strip()
    return out


def make_source():
    """10s 合成片源，时基钉成 1/30000（模拟真实下载片源）。"""
    os.makedirs(T, exist_ok=True)
    cmd = [FF, "-y", "-loglevel", "error",
           "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=10",
           "-f", "lavfi", "-i", "sine=frequency=440:duration=10",
           "-c:v", "libx264", "-g", "30", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-video_track_timescale", "30000", SRC]
    subprocess.run(cmd, check=True, capture_output=True, timeout=300)


def main():
    make_source()
    v = probe(SRC, "time_base,duration")
    assert v["time_base"] == "1/30000", f"素材时基应是 1/30000: {v}"
    print(f"[fixture] 源时基 {v['time_base']}，时长 {v['duration']}s")

    # 场景1：smart cut -> parts = [encode 3-4, copy 4-7, encode 7-8]
    # 列表以 encode 段（1/15360）开头，copy 段（修复前 1/30000）触发拉伸。
    scenes = [{"start": 3.0, "end": 8.0, "duration": 5.0}]
    ok, msg = merger.merge_streamcopy(SRC, scenes, OUT_MIX, mode="precise")
    assert ok, f"smart cut 失败:\n{msg}"
    v = probe(OUT_MIX, "time_base,duration,nb_frames")
    a = probe(OUT_MIX, "duration", sel="a:0")
    vd, ad = float(v["duration"]), float(a["duration"])
    print(f"[smartcut] 视频 {vd:.3f}s / 音频 {ad:.3f}s / "
          f"帧数 {v['nb_frames']} / 时基 {v['time_base']}")
    assert v["time_base"] == "1/15360", \
        f"输出时基应为 1/15360: {v['time_base']}"
    assert abs(vd - ad) < 0.25, \
        f"音画时长差 {abs(vd - ad):.3f}s —— 时基拉伸 bug 复发！"
    assert 4.7 < vd < 5.6, \
        f"视频时长 {vd:.3f}s 偏离规划 5s —— 时间戳仍被拉伸！"

    # 场景2：lossless（全 copy，单一 part 快路径）不受影响
    scenes2 = [{"start": 1.0, "end": 9.0, "duration": 8.0}]
    ok, msg = merger.merge_streamcopy(SRC, scenes2, OUT_COPY, mode="lossless")
    assert ok, f"lossless 失败:\n{msg}"
    v = probe(OUT_COPY, "duration")
    vd = float(v["duration"])
    print(f"[lossless] 视频 {vd:.3f}s（规划 8s）")
    assert 7.7 < vd < 8.4, f"lossless 时长异常: {vd}"

    print("PASS：混合时基拼接已归一化，无音画拉伸。")


if __name__ == "__main__":
    main()
