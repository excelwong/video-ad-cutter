# -*- coding: utf-8 -*-
"""真实感素材（44.1k/25fps/5s GOP/x264 slow）上的三路线音画矩阵。"""
import os
import sys
import tempfile

APP = r"C:\Users\excel\WorkBuddy\Claw\video-ad-cutter"
sys.path.insert(0, APP)
sys.path.insert(0, os.path.join(APP, "bench"))
import merger  # noqa: E402
from test_avsync import report  # noqa: E402

T = r"C:\Users\excel\AppData\Local\Temp\avsync"
SRC = os.path.join(T, "real.mp4")


def seg(a, b):
    return {"start": float(a), "end": float(b), "duration": float(b) - a}


CASES = [
    ("lossless 5段 [6-16,22-33,41-52,58-67,73-84]",
     [seg(6, 16), seg(22, 33), seg(41, 52), seg(58, 67), seg(73, 84)],
     "lossless"),
    ("smartcut 5段 同切点(均非KF)",
     [seg(7, 16), seg(22, 33), seg(41, 52), seg(58, 67), seg(73, 84)],
     "precise"),
    ("lossless 全保留[0-120]",
     [seg(0, 120)], "lossless"),
    ("lossless 单段[13-29] 两切点均在段中间",
     [seg(13, 29)], "lossless"),
]

results = []
for i, (name, kept, mode) in enumerate(CASES):
    out = os.path.join(T, f"real_{i}.mp4")
    ok, msg = merger.merge_streamcopy(SRC, kept, out, mode=mode)
    print(f"\n===== {name} =====")
    print(f"  导出: {'成功' if ok else '失败'}")
    if not ok:
        print("  " + msg[:400])
        results.append((name, None))
        continue
    results.append((name, report(name.split()[0], out)))

# copy 兜底路线多段
out = os.path.join(T, "real_fb.mp4")
with tempfile.TemporaryDirectory() as td:
    ok, msg = merger._try_copy_cut(
        merger.get_ffmpeg()[0], merger.get_ffmpeg()[1], SRC,
        [seg(7, 16), seg(22, 33), seg(41, 52)], out, td, None, 0)
print("\n===== copy兜底 3段 =====")
print(f"  导出: {'成功' if ok else '失败'}")
if ok:
    results.append(("copy兜底3段", report("copyfb", out)))
else:
    print("  " + msg[:200])

print("\n========== 总结论 ==========")
for name, r in results:
    mark = "?" if r is None else ("✅" if r else "❌ 错位")
    print(f"  {mark} {name}")
