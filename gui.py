"""GUI for the video ad-removal tool (tkinter)."""
import os
import queue
import shutil
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import utils
# 转场检测：优先用 ffmpeg 加速版（去掉 OpenCV 逐帧 Python 循环，
# 大视频显著更快；任意失败自动回退原 OpenCV 实现，API 一致）。
import detector_ffmpeg as detector
import previewer
import merger

_BASE = os.path.dirname(os.path.abspath(__file__))
_OUTPUT_DIR = os.path.join(_BASE, "output")
_ENC = None  # chosen encoder (set after probe)

# 支持的视频扩展名（拖入文件时用于识别第一个有效视频）
_VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".flv", ".wmv", ".webm", ".m4v",
               ".mpg", ".mpeg", ".ts", ".m2ts"}

# 拖放支持：用 ctypes 子类化 Tk 窗口过程（WNDPROC），在回调里**只**用
# 系统 shell32/user32 读取拖入路径、写入一个普通 Python 列表，绝不调用 Tk；
# 真正的 GUI 操作放到主线程的 after 轮询器里执行——否则在 Python 3.13 / Tk
# 下会与 Tcl 的 GIL 管理冲突，于 Tk_MainLoop 内触发致命崩溃：
#   PyEval_RestoreThread: GIL is released (thread state is NULL)
# （windnd / 旧版 ctypes 实现都因在回调里碰了 Tk 而同样崩溃。）
# 本机「应用程序控制策略」(WDAC) 会拦截未受信任的第三方 DLL，故不能用
# tkinterdnd2；本方案仅依赖系统 DLL，无此问题。非 Windows 自动降级为
# 「仅文件对话框选择」。


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("视频广告去除工具  ·  Video Ad Cutter")
        self.root.geometry("1100x740")
        try:
            self.root.state("zoomed")
        except Exception:
            pass

        self.video_path = None
        self.full_probe = None
        self.scenes = []
        self.vars = []
        self.thumb_imgs = []
        self.q = queue.Queue()
        self.busy = False
        self.last_out = None
        self.encoder = merger.pick_encoder()
        self.output_dir = _OUTPUT_DIR

        self._build()

    # ---------- UI construction ----------
    def _build(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")

        ttk.Label(top, text="视频文件:").grid(row=0, column=0, sticky="w")
        self.path_var = tk.StringVar()
        self.path_entry = ttk.Entry(top, textvariable=self.path_var, width=70)
        self.path_entry.grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(top, text="选择视频", command=self.on_select).grid(row=0, column=2)
        self.detect_btn = ttk.Button(top, text="检测转场", command=self.on_detect)
        self.detect_btn.grid(row=0, column=3, padx=6)
        self.detect_btn.config(state="disabled")

        # 拖放基于系统 shell32/user32（ctypes），Windows 下默认可用
        dnd_hint = "（也可将视频文件直接拖入本窗口）"
        self.info_var = tk.StringVar(value="未选择视频  " + dnd_hint)
        ttk.Label(top, textvariable=self.info_var, foreground="#555").grid(
            row=1, column=0, columnspan=4, sticky="w", pady=(4, 0))
        top.columnconfigure(1, weight=1)

        # 注册拖放：把视频文件拖入输入框或整个窗口即可加载
        self._setup_dnd()

        # Segment area (scrollable)
        mid = ttk.Frame(self.root)
        mid.pack(fill="both", expand=True, padx=8, pady=4)
        self.canvas = tk.Canvas(mid, bg="#fafafa")
        self.scroll = ttk.Scrollbar(mid, orient="vertical",
                                    command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scroll.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scroll.pack(side="right", fill="y")
        self.inner = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        # 鼠标滚轮滚动（画布区域内的滚轮才生效，避免干扰其它控件）
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)  # Linux 上滚
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)  # Linux 下滚

        # Bottom controls
        bot = ttk.Frame(self.root, padding=8)
        bot.pack(fill="x")
        ttk.Button(bot, text="全选保留", command=self._sel_all).grid(row=0, column=0, padx=3)
        ttk.Button(bot, text="全不选(删除全部)", command=self._sel_none).grid(
            row=0, column=1, padx=3)
        self.target_var = tk.StringVar(value="目标规格: 勾选“保留”的片段将按顺序拼接；"
                                              "最终视频按主体视频(保留片段的最后一段)的分辨率/帧率导出。")
        ttk.Label(bot, textvariable=self.target_var, foreground="#444",
                  wraplength=560, justify="left").grid(
            row=0, column=2, padx=6, sticky="w")

        self.export_btn = ttk.Button(bot, text="导出视频", command=self.on_export)
        self.export_btn.grid(row=0, column=3, padx=6)
        self.export_btn.config(state="disabled")
        self.open_btn = ttk.Button(bot, text="打开输出目录", command=self.on_open)
        self.open_btn.grid(row=0, column=4)
        self.open_btn.config(state="disabled")

        self.progress = ttk.Progressbar(self.root, orient="horizontal",
                                        mode="determinate", maximum=100)
        self.progress.pack(fill="x", padx=8, pady=(0, 2))
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(self.root, textvariable=self.status_var, foreground="#333").pack(
            fill="x", padx=8, pady=(0, 6))

        enc_name = "GPU (NVENC)" if self.encoder == "h264_nvenc" else "CPU (libx264)"
        self.info_var.set(f"编码器: {enc_name}    CUDA 13.1 / RTX 2060 兼容")

    # ---------- drag & drop ----------
    def _setup_dnd(self):
        """注册文件拖放（ctypes 子类化 WNDPROC + 主线程轮询器）。

        ⚠️ 关键：WNDPROC 回调里**绝不调用 Tk**（哪怕 widget.after(0, ...)
        也会在 Python 3.13 / Tk 下与 Tcl 的 GIL 管理冲突，于
        Tk_MainLoop 内触发致命崩溃：
            PyEval_RestoreThread: GIL is released (thread state is NULL)
        （windnd / 旧版 ctypes 实现都因在回调里碰了 Tk 而同样崩溃。）
        故 WNDPROC 只把拖入路径存入 self._pending_drops，真正的
        GUI 操作(_on_drop_files)放到主线程 after 轮询器里执行（此时安全）。
        """
        try:
            import ctypes
            from ctypes import wintypes
        except Exception:
            return
        if not hasattr(ctypes.windll, "shell32"):
            return

        shell32 = ctypes.windll.shell32
        user32 = ctypes.windll.user32
        WM_DROPFILES = 0x0233
        GWL_WNDPROC = -4

        try:
            user32.GetWindowLongPtrW.restype = ctypes.c_void_p
            user32.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
            user32.SetWindowLongPtrW.restype = ctypes.c_void_p
            user32.SetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_void_p]
            user32.CallWindowProcW.restype = ctypes.c_int64
            user32.CallWindowProcW.argtypes = [
                ctypes.c_void_p, wintypes.HWND, wintypes.UINT,
                wintypes.WPARAM, wintypes.LPARAM,
            ]
            shell32.DragQueryFileW.restype = ctypes.c_int
            shell32.DragQueryFileW.argtypes = [
                wintypes.HANDLE, ctypes.c_uint, ctypes.c_wchar_p, ctypes.c_uint,
            ]
            shell32.DragFinish.argtypes = [wintypes.HANDLE]
            shell32.DragAcceptFiles.argtypes = [wintypes.HWND, ctypes.c_int]
        except Exception:
            return

        WndProc = ctypes.WINFUNCTYPE(
            ctypes.c_int64, wintypes.HWND, wintypes.UINT,
            wintypes.WPARAM, wintypes.LPARAM,
        )

        def _hook(widget):
            hwnd = widget.winfo_id()
            old_proc = user32.GetWindowLongPtrW(hwnd, GWL_WNDPROC)

            def wndproc(h, msg, wp, lp):
                if msg == WM_DROPFILES:
                    try:
                        count = shell32.DragQueryFileW(wp, 0xFFFFFFFF, None, 0)
                        paths = []
                        for i in range(count):
                            length = shell32.DragQueryFileW(wp, i, None, 0) + 1
                            buf = ctypes.create_unicode_buffer(length)
                            shell32.DragQueryFileW(wp, i, buf, length)
                            paths.append(buf.value)
                        shell32.DragFinish(wp)
                        if paths:
                            if not isinstance(getattr(self, "_pending_drops", None), list):
                                self._pending_drops = []
                            self._pending_drops.extend(paths)
                    except Exception:
                        pass
                    return 0
                return user32.CallWindowProcW(old_proc, h, msg, wp, lp)

            new_proc = WndProc(wndproc)
            user32.SetWindowLongPtrW(hwnd, GWL_WNDPROC, ctypes.cast(new_proc, ctypes.c_void_p))
            shell32.DragAcceptFiles(hwnd, True)
            widget._drop_wndproc = new_proc  # 保活

        try:
            # 拖到输入框或主窗口任意位置都能接住
            _hook(self.path_entry)
            _hook(self.root)
        except Exception:
            pass

        # 主线程轮询器：在 mainloop 正常上下文里处理拖入（此时调用 Tk 安全）
        def _poll():
            try:
                pending = getattr(self, "_pending_drops", None)
                if isinstance(pending, list) and pending:
                    self._pending_drops = []
                    self._on_drop_files(pending)
            except Exception:
                try:
                    self._pending_drops = []
                except Exception:
                    pass
            self.root.after(150, _poll)

        self._pending_drops = []
        self.root.after(150, _poll)

    def _on_drop_files(self, files):
        """主线程轮询器回调（在 mainloop 正常上下文里触发，可安全操作 GUI）：
        取第一个视频文件加载。files 为 str 路径列表（WNDPROC 已用
        DragQueryFileW 读出），兼容直接传入；如仍是 bytes 则按 gbk / utf-8 /
        mbcs 依次解码以兼容中文路径。
        """
        if self.busy:
            messagebox.showinfo("提示", "正在处理中，请稍候再拖入新文件。")
            return
        paths = []
        for f in files or []:
            if isinstance(f, bytes):
                for enc in ("gbk", "utf-8", "mbcs"):
                    try:
                        f = f.decode(enc)
                        break
                    except Exception:
                        continue
                else:
                    continue
            paths.append(f)
        # 取第一个扩展名为视频的文件；都不是则取第一个并提示
        video = next((p for p in paths
                      if os.path.splitext(p)[1].lower() in _VIDEO_EXTS), None)
        if not video:
            if paths:
                messagebox.showwarning(
                    "提示", "拖入的文件不是受支持的视频格式：\n" + paths[0])
            return
        self._load_path(video)

    # ---------- handlers ----------
    def on_select(self):
        path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.mkv *.mov *.avi *.flv *.wmv *.webm *.m4v"),
                       ("所有文件", "*.*")])
        if not path:
            return
        self._load_path(path)

    def _load_path(self, path):
        """加载一个视频路径（文件对话框选择 / 拖入 共用）。"""
        if not path or not os.path.isfile(path):
            messagebox.showerror("提示", f"文件不存在：\n{path}")
            return
        self.video_path = path
        self.path_var.set(path)
        self.detect_btn.config(state="normal")
        self.export_btn.config(state="disabled")
        self.open_btn.config(state="disabled")
        self._reset()
        try:
            self.full_probe = utils.probe_video(path)
            v = self.full_probe["video"]
            a = self.full_probe["audio"]
            info = (f"时长 {utils.human_time(self.full_probe['duration'])}  |  "
                    f"分辨率 {v['width']}x{v['height']}  |  帧率 {v['fps']:.3f}fps  |  "
                    f"音频 {'有' if a else '无'}")
        except Exception as e:
            info = f"已选择，但读取信息失败: {e}"
        self.info_var.set(info)
        self.status_var.set("点击“检测转场”开始切分视频。")

    def on_detect(self):
        if not self.video_path or self.busy:
            return
        self._set_busy(True)
        self.progress["value"] = 0
        self.status_var.set("检测转场中（首次会扫描整段视频，请稍候）...")
        self.detect_btn.config(state="disabled")
        threading.Thread(target=self._detect_worker, daemon=True).start()
        self._poll()

    def _detect_worker(self):
        try:
            if self.full_probe is None:
                self.full_probe = utils.probe_video(self.video_path)
            scenes = detector.detect_scenes(
                self.video_path, progress_cb=self._detect_progress)
            thumb_dir = os.path.join(self.output_dir, "thumbs")
            shutil.rmtree(thumb_dir, ignore_errors=True)
            previewer.build_previews(self.video_path, scenes, thumb_dir,
                                     progress_cb=None)
            self.q.put(("detected", scenes))
        except Exception as e:
            self.q.put(("error", str(e)))

    def _detect_progress(self, frame, n_frames):
        """检测进度回调（由检测器在后台线程调用）。
        进度以「已扫描时长 / 总时长」衡量：frac≈frame/n_frames≈elapsed/duration。
        仅把进度推到队列，由主线程 _poll 更新进度条，避免跨线程操作 tkinter。
        """
        try:
            frac = (frame / n_frames) if n_frames else 0.0
            frac = max(0.0, min(1.0, frac))
            dur = 0.0
            fps = 0.0
            if self.full_probe:
                dur = self.full_probe.get("duration") or 0.0
                v = self.full_probe.get("video") or {}
                fps = v.get("fps") or 0.0
            elapsed = (frame / fps) if fps else (frac * dur)
            pct = int(round(frac * 100))
            status = (f"检测转场中… 已扫描 {elapsed:.1f}s / 共 {dur:.1f}s  ({pct}%)")
            self.q.put(("progress", frac, status))
        except Exception:
            pass

    def on_export(self):
        if not self.scenes or self.busy:
            return
        kept = [s for i, s in enumerate(self.scenes) if self.vars[i].get() == 1]
        if not kept:
            messagebox.showerror("提示", "请至少保留一段视频。")
            return
        out = filedialog.asksaveasfilename(
            title="保存导出视频", defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4"), ("MKV", "*.mkv")],
            initialfile="output_no_ad.mp4")
        if not out:
            return
        if os.path.abspath(out) == os.path.abspath(self.video_path):
            messagebox.showerror("提示", "输出文件不能与源文件相同。")
            return
        self._set_busy(True)
        self.open_btn.config(state="disabled")
        threading.Thread(target=self._export_worker, args=(kept, out),
                        daemon=True).start()
        self._poll()

    def _export_worker(self, kept, out):
        def cb(frac, status):
            self.q.put(("progress", frac, status))
        try:
            # 用主视频原编码流拷贝裁剪+拼接（不重新编码）；
            # 疑似索引缺失会自动重建索引重试，其他问题写详细日志到应用目录。
            ok, msg = merger.merge_streamcopy(
                self.video_path, kept, out,
                progress_cb=cb, app_dir=_BASE)
        except Exception as e:
            ok, msg = False, f"异常: {e}"
        self.q.put(("done", ok, msg, out))

    def _cleanup_thumbs(self):
        """导出完成后删除 output/thumbs 里的缩略图。

        界面上的缩略图早已以 PNG 数据载入内存（self.thumb_imgs），删除磁盘
        上的 jpg 不会影响已显示的卡片，也不动 output 下的其它文件。
        """
        thumb_dir = os.path.join(self.output_dir, "thumbs")
        try:
            if os.path.isdir(thumb_dir):
                shutil.rmtree(thumb_dir, ignore_errors=True)
        except Exception:
            pass

    def on_open(self):
        if self.last_out and os.path.isfile(self.last_out):
            d = os.path.dirname(os.path.abspath(self.last_out))
        else:
            d = self.output_dir
        if os.path.isdir(d):
            os.startfile(d)

    def _sel_all(self):
        for v in self.vars:
            v.set(1)
        self._refresh_target()

    def _sel_none(self):
        for v in self.vars:
            v.set(0)
        self._refresh_target()

    # ---------- rendering ----------
    def _on_mousewheel(self, event):
        """画布区域内的鼠标滚轮 → 纵向滚动。"""
        cx, cy = self.canvas.winfo_rootx(), self.canvas.winfo_rooty()
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        # 仅当指针落在画布滚动区时才接管滚动，否则放行给其它控件
        if not (cx <= event.x_root <= cx + cw and cy <= event.y_root <= cy + ch):
            return
        if getattr(event, "delta", 0):        # Windows / 多数鼠标
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        elif getattr(event, "num", 0) == 4:   # Linux 上滚
            self.canvas.yview_scroll(-1, "units")
        elif getattr(event, "num", 0) == 5:   # Linux 下滚
            self.canvas.yview_scroll(1, "units")
        return "break"

    def _reset(self):
        for w in list(self.inner.children.values()):
            w.destroy()
        self.scenes = []
        self.vars = []
        self.thumb_imgs = []
        self.export_btn.config(state="disabled")
        self.target_var.set("目标规格: 勾选“保留”的片段将按顺序拼接；"
                            "最终视频按主体视频(保留片段的最后一段)的分辨率/帧率导出。")

    def _render_segments(self):
        for w in list(self.inner.children.values()):
            w.destroy()
        self.thumb_imgs = []
        cols = 3
        for idx, sc in enumerate(self.scenes):
            self._add_card(idx, sc, cols)
        self._refresh_target()

    def _load_thumb_image(self, thumb):
        """把缩略图解码/缩放后转成 Tk 原生 PhotoImage。

        改用「PIL 解码+缩放 -> PNG bytes -> tk.PhotoImage(data=...)」的方式，
        彻底绕开 Pillow 的 ImageTk._imagingtk DLL——该 DLL 在本机会被
        「应用程序控制策略」拦截导致加载失败（表现为“预览生成失败”）。
        Tk 8.6 自带 PNG 解码器，因此这条路径不依赖任何被封的 DLL。
        """
        import io
        import base64
        from PIL import Image
        with Image.open(thumb) as im:
            im = im.convert("RGB").resize((240, 135))
            buf = io.BytesIO()
            im.save(buf, format="PNG")
        data = base64.b64encode(buf.getvalue())
        return tk.PhotoImage(data=data)

    def _add_card(self, idx, sc, cols):
        f = ttk.LabelFrame(self.inner, text=f"段 {idx + 1}")
        r, c = divmod(idx, cols)
        f.grid(row=r, column=c, padx=6, pady=6, sticky="nsew")
        self.inner.columnconfigure(c, weight=1)

        thumb = sc.get("thumb")
        if thumb and os.path.isfile(thumb):
            try:
                img = self._load_thumb_image(thumb)
                lbl = tk.Label(f, image=img, bg="#222")
                lbl.image = img
                self.thumb_imgs.append(img)
            except Exception:
                lbl = tk.Label(f, text="(预览生成失败)", width=30, height=8)
        else:
            lbl = tk.Label(f, text="(无预览)", width=30, height=8)
        lbl.grid(row=0, column=0, pady=2)

        info = (f"{utils.human_time(sc['start'])} → {utils.human_time(sc['end'])}\n"
                f"时长 {sc['duration']:.1f}s")
        if sc.get("frame_w"):
            info += f"\n{sc['frame_w']}×{sc['frame_h']}"
        ttk.Label(f, text=info, justify="left").grid(row=1, column=0, sticky="w", padx=4)

        var = tk.IntVar(value=1)
        self.vars.append(var)
        # 点击截图即可切换“保留”复选框状态（带选中高亮反馈）
        self._bind_thumb_toggle(lbl, var)   # 必须在 var 定义之后调用
        cb = ttk.Checkbutton(f, text="保留（取消=删除）", variable=var,
                              command=self._refresh_target)
        cb.grid(row=2, column=0, sticky="w", padx=4, pady=2)


    def _bind_thumb_toggle(self, lbl, var):
        """点击缩略图即可切换对应片段的“保留/删除”复选状态。"""
        def on_click(_event):
            var.set(0 if var.get() else 1)
            self._refresh_target()
        lbl.bind("<Button-1>", on_click)
        lbl.config(cursor="hand2")  # 提示可点击
        def reflect(*_args):
            try:
                on = var.get() == 1
                # 选中(保留)的片段截图：鲜红色粗框围起来，方便识别
                lbl.config(relief="solid" if on else "flat",
                           highlightthickness=4 if on else 0,
                           highlightbackground="#ff0000",
                           highlightcolor="#ff0000")
            except Exception:
                pass
        var.trace_add("write", reflect)
        reflect()  # 初始化边框状态

    def _refresh_target(self):
        if not self.scenes:
            return
        kept = [s for i, s in enumerate(self.scenes) if self.vars[i].get() == 1]
        if not kept:
            self.target_var.set("目标规格: （未选择保留片段）")
            return
        ref = kept[-1]
        w = ref.get("frame_w") or (self.full_probe["video"]["width"]
                                   if self.full_probe else "?")
        h = ref.get("frame_h") or (self.full_probe["video"]["height"]
                                   if self.full_probe else "?")
        fps = self.full_probe["video"]["fps"] if self.full_probe else 0
        sr = (self.full_probe["audio"]["sample_rate"]
              if self.full_probe and self.full_probe.get("audio") else 0)
        self.target_var.set(
            f"输出规格(主视频原编码·流拷贝，不重新编码): {w}×{h}  |  {fps:.3f}fps"
            + (f"  |  音频 {sr}Hz" if sr else "  |  无音频"))

    # ---------- poll ----------
    def _poll(self):
        try:
            while True:
                item = self.q.get_nowait()
                kind = item[0]
                if kind == "detected":
                    self.scenes = item[1]
                    self._render_segments()
                    self.export_btn.config(state="normal")
                    self.status_var.set(
                        f"检测到 {len(self.scenes)} 段。取消勾选要删除的广告片段，"
                        f"然后点击“导出视频”。")
                    self._set_busy(False)
                elif kind == "error":
                    messagebox.showerror("错误", item[1])
                    self.status_var.set("检测失败")
                    self._set_busy(False)
                elif kind == "progress":
                    self.progress["value"] = max(0, min(100, item[1] * 100))
                    self.status_var.set(item[2])
                elif kind == "done":
                    ok, msg, out = item[1], item[2], item[3]
                    self.progress["value"] = 100 if ok else self.progress["value"]
                    self._set_busy(False)
                    if ok:
                        self.last_out = out
                        self.open_btn.config(state="normal")
                        self._cleanup_thumbs()
                        self.status_var.set(msg)
                        messagebox.showinfo("完成", msg)
                    else:
                        self.status_var.set("导出失败")
                        messagebox.showerror("失败", msg)
        except queue.Empty:
            pass
        if self.busy:
            self.root.after(120, self._poll)

    def _set_busy(self, b):
        self.busy = b
        state = "disabled" if b else "normal"
        self.detect_btn.config(state=state if self.video_path else "disabled")
        if not b:
            self.export_btn.config(state="normal" if self.scenes else "disabled")


def _crash(tb_text):
    """Best-effort: persist traceback to crash.log and show a popup."""
    try:
        with open(os.path.join(_BASE, "crash.log"), "w", encoding="utf-8") as f:
            f.write("=== 程序异常 / startup crash ===\n")
            f.write(tb_text)
    except Exception:
        pass
    try:
        r = tk.Tk()
        r.withdraw()
        from tkinter import messagebox as _mb
        _mb.showerror(
            "启动失败 / Startup failed",
            "程序发生异常并退出。\n错误信息已写入 crash.log：\n\n" + tb_text[:3000])
        r.destroy()
    except Exception:
        pass


def main():
    try:
        os.makedirs(_OUTPUT_DIR, exist_ok=True)
        root = tk.Tk()
        App(root)
        root.mainloop()
    except Exception:
        import traceback
        _crash(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
