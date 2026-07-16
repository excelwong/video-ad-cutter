"""Extract the first frame of each detected segment as a preview thumbnail."""
import os

from utils import get_ffmpeg, run_capture


def extract_first_frame(video_path, start_sec, out_path, fast_seek=True):
    """Extract one JPEG frame near `start_sec` into out_path.

    fast_seek=True uses input-side -ss (fast, lands on nearest keyframe; fine for a
    thumbnail). Returns True on success.
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


def build_previews(video_path, scenes, thumb_dir, progress_cb=None):
    """Extract a thumbnail for each scene; also record each frame's native size.

    Mutates `scenes` in place adding "thumb" (path) and "frame_w"/"frame_h".
    """
    os.makedirs(thumb_dir, exist_ok=True)
    for i, sc in enumerate(scenes):
        out = os.path.join(thumb_dir, f"seg_{i:03d}.jpg")
        ok = extract_first_frame(video_path, sc["start"], out)
        sc["thumb"] = out if ok else None
        if ok:
            try:
                from PIL import Image
                with Image.open(out) as im:
                    sc["frame_w"], sc["frame_h"] = im.size
            except Exception:
                sc["frame_w"], sc["frame_h"] = None, None
        else:
            sc["frame_w"], sc["frame_h"] = None, None
        if progress_cb:
            progress_cb(i + 1, len(scenes))
    return scenes
