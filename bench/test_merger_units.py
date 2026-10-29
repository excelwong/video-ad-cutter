import os, sys, time, random
sys.path.insert(0, os.getcwd())
import merger

random.seed(42)
ok = True

# ---- 1) bisect 版与线性版语义一致 ---------------------------------------
for trial in range(300):
    n = random.randint(0, 40)
    kfs = sorted(round(random.uniform(0, 300), 3) for _ in range(n))
    # 有意注入重复时间戳（真实文件存在，见 test2.mp4 备注）
    if kfs and random.random() < 0.5:
        kfs.insert(random.randint(0, len(kfs)), random.choice(kfs))
        kfs.sort()
    t = round(random.uniform(-5, 305), 3)
    tol = random.choice([0.0, 0.04, 1.0])

    # _snap_head: 第一个 >= t-tol
    old = next((k for k in kfs if k >= t - tol), None)
    new = merger._snap_head(kfs, t, tol)
    if old != new:
        print(f"FAIL snap_head kfs={kfs} t={t} tol={tol}: {old} vs {new}"); ok = False; break
    # _snap_tail: 最后一个 <= t+tol
    old = None
    for k in kfs:
        if k <= t + tol: old = k
        else: break
    new = merger._snap_tail(kfs, t, tol)
    if old != new:
        print(f"FAIL snap_tail kfs={kfs} t={t} tol={tol}: {old} vs {new}"); ok = False; break
    # _kf_at_or_before
    old = max([k for k in kfs if k <= t], default=0.0)
    new = merger._kf_at_or_before(kfs, t)
    if old != new:
        print(f"FAIL kf_at_or_before kfs={kfs} t={t}: {old} vs {new}"); ok = False; break
print(f"[1] bisect 等价性（900 组随机用例）: {'OK' if ok else 'FAIL'}")

# ---- 2) _plan_block bisect 版 vs 参考线性版 ------------------------------
def ref_plan_block(kfs, start, end, tol=0.04, head_strict=True, tail_strict=True):
    inner = [k for k in kfs if start + tol < k < end - tol]
    if head_strict and not inner:
        k0 = max([k for k in kfs if k <= start + tol], default=0.0)
        return [("encode", start, end, max(0.0, k0))]
    parts = []
    cur = start
    if head_strict:
        k1 = inner[0]
        if k1 - cur > tol:
            k0 = max([k for k in kfs if k <= start + tol], default=0.0)
            parts.append(("encode", start, k1, max(0.0, k0)))
            cur = k1
    if tail_strict:
        klast = inner[-1] if inner else None
        if klast is not None and klast > cur + tol:
            parts.append(("copy", cur, klast, cur)); cur = klast
        if end - cur > tol:
            parts.append(("encode", cur, end, cur))
    else:
        if end - cur > tol:
            parts.append(("copy", cur, end, cur))
    if not parts:
        parts = [("copy", start, end, start)]
    return parts

ok2 = True
for trial in range(300):
    n = random.randint(0, 30)
    kfs = sorted(round(random.uniform(0, 300), 3) for _ in range(n))
    a = round(random.uniform(0, 290), 3)
    b = a + round(random.uniform(0.1, 15), 3)
    hs = random.random() < 0.8
    ts = random.random() < 0.8
    r1 = merger._plan_block(kfs, a, b, 0.04, hs, ts)
    r2 = ref_plan_block(kfs, a, b, 0.04, hs, ts)
    if r1 != r2:
        print(f"FAIL plan_block kfs={kfs} {a}-{b} hs={hs} ts={ts}:\n  {r1}\n  {r2}"); ok2 = False; break
print(f"[2] _plan_block bisect 等价（300 组随机用例）: {'OK' if ok2 else 'FAIL'}")

# ---- 3) _plan_all_lossless / preview_lossless 不炸 -----------------------
kfs = [i * 2.0 for i in range(50)]
scenes = [{"start": i * 3.0, "end": i * 3.0 + 1.5, "duration": 1.5} for i in range(10)]
parts, shifts, dropped = merger._plan_all_lossless(kfs, merger._merge_adjacent(scenes), 100.0)
print(f"[3] _plan_all_lossless: parts={len(parts)} shifts={len(shifts)} dropped={len(dropped)}")

# ---- 4) 并行引擎：结果归位、失败即停、进度单调 ----------------------------
import tempfile, threading
tmp = tempfile.mkdtemp()
prog = []
def make_job(i, fail=False, dur=0.05):
    out = os.path.join(tmp, f"j_{i:03d}.bin")
    cmd = ["python", "-c",
           f"import time, sys; time.sleep({dur}); "
           f"open(r'{out}','w').write('x'*10)" + ("; sys.exit(1)" if fail else "")]
    return (i, cmd, out, 1.0, f"任务{i}")

jobs = [make_job(i) for i in range(12)]
files, fail = merger._run_jobs_parallel(jobs, 3, lambda f, s: prog.append((f, s)), "任务", 180)
assert fail is None, fail
assert files == [os.path.join(tmp, f"j_{i:03d}.bin") for i in range(12)], "结果未按序号归位"
fracs = [f for f, _ in prog]
assert fracs == sorted(fracs) and fracs[-1] <= 0.97, f"进度异常: {fracs}"
print(f"[4] 并行引擎 12 任务: 归位 OK, 进度 {len(prog)} 次单调 OK")

# 失败即停
jobs = [make_job(i, fail=(i == 3)) for i in range(12)]
t0 = time.monotonic()
files, fail = merger._run_jobs_parallel(jobs, 3, None, "任务", 180)
dt = time.monotonic() - t0
assert files is None and fail and "任务3" in fail, fail
print(f"[5] 失败即停: {dt:.2f}s 内返回, 错误信息含失败任务 OK  (fail={fail.splitlines()[0][:40]})")

# 单任务/双任务不起池
jobs = [make_job(0)]
files, fail = merger._run_jobs_parallel(jobs, merger._default_workers(None, 1), None, "任务", 180)
assert files and fail is None
print(f"[6] 单/双任务顺序执行路径 OK")

# _default_workers 策略
assert merger._default_workers(None, 10) == 3
assert merger._default_workers("h264_nvenc", 10) == 2
assert merger._default_workers("libx264", 10) == 3
assert merger._default_workers(None, 2) == 1
print(f"[7] 并发策略: copy/x264=3, nvenc=2, <=2段=1 OK")

import shutil; shutil.rmtree(tmp, ignore_errors=True)
print("\nmerger 单元验证:", "全部通过 ✅" if (ok and ok2) else "存在失败 ❌")
sys.exit(0 if (ok and ok2) else 1)
