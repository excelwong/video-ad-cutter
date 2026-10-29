# -*- coding: utf-8 -*-
"""音画同步诊断工具（用户反馈用）。

用法:
    python bench/avsync_check.py <视频文件> [导出结果文件]

对片源/导出结果做流级检查并打印诊断：
- 各流编码参数、start_time、首包 pts、流时长
- 音视频起点差 / 时长差（>0.5s 会标记 ⚠️）
- 关键帧表概况（前 5 个 + 平均 GOP）
- 是否存在多余的视频流（封面图）/ 多条音轨

若「导出结果」的起点差或时长差显著大于「片源」的，说明管线在该片源上
产生了音画偏移 —— 请把本工具的完整输出发给开发者。
"""
import json
import subprocess
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_ffmpeg  # noqa: E402


def probe(path):
    _, fp = get_ffmpeg()
    r = subprocess.run(
        [fp, "-v", "error", "-show_entries",
         "stream=index,codec_type,codec_name,width,height,sample_rate,"
         "channels,start_time,duration:format=duration",
         "-of", "json", path],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=120)
    return json.loads(r.stdout)


def first_pts(path, sel):
    _, fp = get_ffmpeg()
    r = subprocess.run(
        [fp, "-v", "error", "-select_streams", sel, "-read_intervals",
         "%+#4", "-show_entries", "packet=pts_time", "-of", "csv=p=0",
         path],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=60)
    for ln in r.stdout.splitlines():
        ln = ln.strip().rstrip(",")
        if not ln or ln.lower() == "n/a":
            continue
        try:
            return float(ln)
        except ValueError:
            continue
    return None


def keyframes(path):
    _, fp = get_ffmpeg()
    r = subprocess.run(
        [fp, "-v", "error", "-select_streams", "v:0", "-show_packets",
         "-show_entries", "packet=pts_time,flags", "-of", "csv=p=0", path],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=600)
    kfs = []
    for line in r.stdout.splitlines():
        parts = line.split(",")
        if len(parts) >= 2 and "K" in parts[-1]:
            try:
                kfs.append(float(parts[0]))
            except ValueError:
                pass
    return sorted(kfs)


def report(path, tag):
    print(f"\n===== {tag}: {path} =====")
    try:
        data = probe(path)
    except Exception as e:
        print(f"  ❌ probe 失败: {e}")
        return None
    streams = data.get("streams", [])
    vids = [s for s in streams if s.get("codec_type") == "video"]
    auds = [s for s in streams if s.get("codec_type") == "audio"]
    others = [s for s in streams
              if s.get("codec_type") not in ("video", "audio")]
    print(f"  视频流 {len(vids)} 条, 音频流 {len(auds)} 条, 其他 {len(others)} 条")
    if len(vids) > 1:
        print("  ⚠️ 存在多条视频流（封面图?）:")
        for s in vids:
            print(f"    #{s['index']} {s.get('codec_name')} "
                  f"{s.get('width')}x{s.get('height')} "
                  f"start={s.get('start_time')}")
    if len(auds) > 1:
        print("  ⚠️ 存在多条音轨:")
        for s in auds:
            print(f"    #{s['index']} {s.get('codec_name')} "
                  f"{s.get('sample_rate')}Hz {s.get('channels')}ch "
                  f"start={s.get('start_time')}")

    def one(sel):
        if not (vids if sel == "v:0" else auds):
            return None
        s = (vids if sel == "v:0" else auds)[0]
        st = s.get("start_time")
        du = s.get("duration")
        try:
            st = float(st) if st not in (None, "N/A") else None
        except ValueError:
            st = None
        try:
            du = float(du) if du not in (None, "N/A") else None
        except ValueError:
            du = None
        return st, du, s

    v = one("v:0")
    a = one("a:0")
    if not v:
        print("  （无视频流）")
        return None
    v_st, v_du, v_s = v
    v_fp = first_pts(path, "v:0")
    print(f"  视频: {v_s.get('codec_name')} start={v_st} 首包pts={v_fp} "
          f"时长={v_du}")
    if a:
        a_st, a_du, a_s = a
        a_fp = first_pts(path, "a:0")
        print(f"  音频: {a_s.get('codec_name')} start={a_st} 首包pts={a_fp} "
              f"时长={a_du}")
        flags = []
        if v_fp is not None and a_fp is not None:
            d = v_fp - a_fp
            print(f"  → 起点差(视频-音频): {d:+.3f}s"
                  + ("  ⚠️ 超过 0.5s" if abs(d) > 0.5 else ""))
            if abs(d) > 0.5:
                flags.append("start-gap")
        if v_du and a_du:
            d = v_du - a_du
            print(f"  → 时长差(视频-音频): {d:+.3f}s"
                  + ("  ⚠️ 超过 0.5s" if abs(d) > 0.5 else ""))
            if abs(d) > 0.5:
                flags.append("dur-gap")
        return {"v_start": v_st, "a_start": a_st, "v_first": v_fp,
                "a_first": a_fp, "v_dur": v_du, "a_dur": a_du,
                "flags": flags}
    print("  （无音轨）")
    return None


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    src = report(sys.argv[1], "片源")
    if len(sys.argv) >= 3:
        out = report(sys.argv[2], "导出结果")
        if src and out:
            print("\n===== 结论 =====")
            diffs = []
            for k in ("v_start", "v_first", "a_start", "a_first",
                      "v_dur", "a_dur"):
                if src.get(k) is not None and out.get(k) is not None:
                    d = out[k] - src[k]
                    if abs(d) > 0.05:
                        diffs.append(f"  {k}: 片源 {src[k]:.3f} → "
                                     f"导出 {out[k]:.3f} (Δ{d:+.3f}s)")
            if out.get("flags") and not src.get("flags"):
                diffs.append("  导出结果出现片源没有的起点/时长差 ⚠️")
            if diffs:
                print("  导出相对片源的变化：")
                print("\n".join(diffs))
                print("  ⚠️ 若 Δ 显著（>0.3s），请把以上完整输出发给开发者。")
            else:
                print("  流级指标与片源一致（Δ≤0.05s），未见封装层错位。")
    kfs = keyframes(sys.argv[1])
    if kfs:
        gaps = [b - a for a, b in zip(kfs, kfs[1:])]
        avg = sum(gaps) / len(gaps) if gaps else 0
        print(f"\n片源关键帧: {len(kfs)} 个, 平均 GOP {avg:.2f}s, "
              f"前 5 个: {['%.3f' % t for t in kfs[:5]]}")


if __name__ == "__main__":
    main()
