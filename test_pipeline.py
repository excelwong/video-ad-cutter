"""End-to-end test for the video ad cutter.

Builds expectations:
  * full_clean.mp4: 1920x1080 @ 24fps, 15s (5s ad + 10s main)
  * full.mp4: copy-concat of 1280x720@30 (ad) + 1920x1080@24 (main) -> resolution change present
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import utils
import detector
import previewer
import merger

BASE = os.path.dirname(os.path.abspath(__file__))
TEST = os.path.join(BASE, "test")
OUT = os.path.join(BASE, "output")
os.makedirs(OUT, exist_ok=True)


def log(*a):
    print("[TEST]", *a)


def enc_for_test():
    # Prefer NVENC if it actually works; otherwise CPU fallback so the
    # pipeline logic is still validated.
    if merger.has_nvenc():
        try:
            rc = os.system(
                '"%s" -y -hide_banner -f lavfi -i testsrc=size=128x96:rate=30:duration=1 '
                "-c:v h264_nvenc -preset p4 -cq 23 %s >nul 2>&1"
                % (utils.get_ffmpeg()[0], os.path.join(TEST, "_nvprobe.mp4"))
            )
            if rc == 0:
                return "h264_nvenc"
        except Exception:
            pass
    return "libx264"


def main():
    enc = enc_for_test()
    log("using encoder:", enc)
    assert os.path.isfile(utils.get_ffmpeg()[0]), "ffmpeg missing"

    # ---- 1) CLEAN content test (1080p24) ----
    clean = os.path.join(TEST, "full_clean.mp4")
    scenes = detector.detect_scenes(clean)
    log("clean scenes:", [(round(s["start"], 2), round(s["end"], 2))
                          for s in scenes])
    assert len(scenes) >= 2, "expected >=2 scenes"
    kept = [scenes[-1]]  # keep the main part (drop the ad)
    ok, msg = merger.merge(clean, kept, os.path.join(OUT, "clean_out.mp4"),
                           encoder=enc, preset="p4" if enc == "h264_nvenc" else "medium",
                           cq=23)
    log("clean merge:", ok, msg)
    assert ok, msg
    info = utils.probe_video(os.path.join(OUT, "clean_out.mp4"))
    v = info["video"]
    log("clean_out video:", v)
    assert v["width"] == 1920 and v["height"] == 1080, "resolution must be 1920x1080"
    assert abs(v["fps"] - 24) < 1.0, "fps must be ~24"
    assert abs(info["duration"] - 10) < 1.0, "duration must be ~10s (ad dropped)"
    log("==> CLEAN CONTENT TEST PASS")

    # ---- 2) RESOLUTION NORMALIZATION (target override 1280x720@30) ----
    ok2, msg2 = merger.merge(
        clean, [scenes[-1]], os.path.join(OUT, "res_out.mp4"),
        encoder=enc, preset="p4" if enc == "h264_nvenc" else "medium", cq=23,
        target_override={"width": 1280, "height": 720, "fps": 30.0,
                         "sample_rate": 48000, "channels": 2, "has_audio": False})
    log("res merge:", ok2, msg2)
    assert ok2, msg2
    info2 = utils.probe_video(os.path.join(OUT, "res_out.mp4"))
    v2 = info2["video"]
    log("res_out video:", v2)
    assert v2["width"] == 1280 and v2["height"] == 720, "override res must be 1280x720"
    assert abs(v2["fps"] - 30) < 1.0, "override fps must be ~30"
    log("==> RESOLUTION NORMALIZATION TEST PASS")

    # ---- 3) probe_segment reads per-segment resolution on a mixed-res file ----
    mixed = os.path.join(TEST, "full.mp4")
    if os.path.isfile(mixed):
        vmain = utils.probe_segment(mixed, 7.0, dur=2.0)["video"]
        log("mixed @t=7 video:", vmain)
        assert vmain["width"] == 1920 and vmain["height"] == 1080, \
            "mixed file main part must read 1920x1080"
        log("==> MIXED-RES PROBE TEST PASS")

    log("ALL TESTS PASSED (encoder=%s)" % enc)


if __name__ == "__main__":
    main()
