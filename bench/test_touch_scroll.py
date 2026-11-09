# -*- coding: utf-8 -*-
"""无头回归测试：触摸屏双指上下拖动滚屏 + 滚轮像素级滚动。

被测逻辑（gui.App）：
  - _feed_pan()            WNDPROC 里累积手指纵向位移（不碰 Tk）
  - _touch_scroll_tick()   主线程消费累积位移
  - _scroll_canvas_pixels() 像素级滚动 + 首尾夹取（滚轮与触摸共用）
  - _gesture_over_canvas() 用缓存矩形判断手势落点
  - _on_mousewheel()       兼容传统滚轮(±120)与高精度触控板小 delta

不弹窗、不阻塞：窗口 withdraw + 手动 pump 事件循环（改动 GUI 后的标准
验证手法）。真机触摸屏手势无法在 CI 里模拟，这里覆盖的是手势消息到达
之后的全部逻辑链。
"""
import os
import sys

APP = r"C:\Users\excel\WorkBuddy\Claw\video-ad-cutter"
sys.path.insert(0, APP)

import tkinter as tk  # noqa: E402

import gui  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + name + ("  " + detail if detail else ""))
    if not cond:
        FAILS.append(name)


class Ev:
    def __init__(self, delta=0, num=0, x=0, y=0):
        self.delta, self.num = delta, num
        self.x_root, self.y_root = x, y


def main():
    gui.messagebox.showinfo = lambda *a, **k: None
    gui.messagebox.showerror = lambda *a, **k: None
    gui.messagebox.showwarning = lambda *a, **k: None

    root = tk.Tk()
    root.withdraw()
    app = gui.App(root)
    for _ in range(3):
        root.update()

    # 视口高度取 Tk 的真实几何（withdraw 的窗口依然会算出几何，实测 848），
    # 内容高 5000 —— 夹取边界由 Tk 按真实视口算，所以断言必须用它。
    app.canvas.configure(scrollregion=(0, 0, 400, 5000))
    for _ in range(2):
        root.update()
    VIEW = float(app.canvas.winfo_height())
    TOTAL = 5000.0
    print("   （视口 %.0fpx / 内容 %.0fpx）" % (VIEW, TOTAL))
    assert VIEW > 1 and VIEW < TOTAL, "画布几何不可用: view=%s" % VIEW

    def top_px():
        return app.canvas.yview()[0] * TOTAL

    # ---------- 1. 手势接线是否装上 ----------
    check("拖放 WNDPROC 仍在（未被手势钩子顶掉）",
          getattr(root, "_drop_wndproc", None) is not None)
    if os.name == "nt":
        check("画布手势钩子已安装", bool(app._canvas_hwnd))
        check("手势 WNDPROC 已挂在画布上",
              getattr(app.canvas, "_gesture_wndproc", None) is not None)
        check("手势 WNDPROC 已挂在主窗口上",
              getattr(root, "_gesture_wndproc", None) is not None)

        # 直接用一条非法 WM_GESTURE 消息调用回调：GetGestureInfo 必然失败，
        # 应安全走到「转发原窗口过程」分支而不崩（验证回调签名/类型没写错）
        try:
            fn = root._gesture_wndproc
            fn(root.winfo_id(), 0x0119, 0, 0)
            ok = True
        except Exception as e:
            ok = False
            print("   异常:", e)
        check("WM_GESTURE 回调可安全转发未处理消息", ok)
        root.update()

    # ---------- 2. 像素级滚动 ----------
    app.canvas.yview_moveto(0)
    app._scroll_canvas_pixels(100)
    d = top_px()
    check("向下滚 100px", 95 < d < 105, "实际 %.1fpx" % d)

    app.canvas.yview_moveto(0)
    app._scroll_canvas_pixels(-80)
    check("顶部再向上滚被夹住", abs(top_px()) < 0.5, "top=%.1f" % top_px())

    app.canvas.yview_moveto(1)
    app._scroll_canvas_pixels(300)
    check("底部再向下滚被夹住", abs(top_px() - (TOTAL - VIEW)) < 2.0,
          "top=%.1f 期望≈%.1f" % (top_px(), TOTAL - VIEW))

    # 内容不足一屏时不应动
    app.canvas.configure(scrollregion=(0, 0, 400, 200))
    app.canvas.yview_moveto(0)
    before = app.canvas.yview()[0]
    app._scroll_canvas_pixels(500)
    check("内容未超出视口时不滚动", app.canvas.yview()[0] == before)
    app.canvas.configure(scrollregion=(0, 0, 400, 5000))

    # ---------- 3. 双指位移累积（模拟 GID_PAN 序列） ----------
    app.canvas.yview_moveto(0)
    app._touch_dy = 0.0
    app._feed_pan(1, 500)          # 手势第一帧：只定基准，不产生位移
    check("首个 GID_PAN 不产生位移", app._touch_dy == 0.0,
          "dy=%s" % app._touch_dy)
    app._feed_pan(1, 440)          # 手指上移 60px → 内容向下 60px
    check("手指上滑 60px → 待滚 +60px", abs(app._touch_dy - 60.0) < 0.01,
          "dy=%.1f" % app._touch_dy)
    app._feed_pan(1, 470)          # 手指下移 30px → 反向
    check("手指下滑 30px → 待滚 +30px", abs(app._touch_dy - 30.0) < 0.01,
          "dy=%.1f" % app._touch_dy)

    app._touch_scroll_tick()
    check("轮询器消费后清零", app._touch_dy == 0.0)
    d = top_px()
    check("轮询器把位移应用到画布", 25 < d < 35, "实际 %.1fpx" % d)

    # 新手势序列要重置基准（否则跨手势会瞬移）
    app._touch_dy = 0.0
    app._feed_pan(2, 100)
    check("换手势序列不产生跳变", app._touch_dy == 0.0)

    # ---------- 4. 落点判断 ----------
    app._canvas_hwnd = 0          # 排除 hwnd 直通分支，只测矩形
    app._canvas_rect = (100, 100, 400, 300)
    check("落点在画布内 → 接管", app._gesture_over_canvas(0, 200, 200) is True)
    check("落点在画布外(左) → 放行", app._gesture_over_canvas(0, 50, 200) is False)
    check("落点在画布外(上) → 放行", app._gesture_over_canvas(0, 200, 50) is False)
    app._canvas_hwnd = 12345
    check("手势目标就是画布句柄 → 接管",
          app._gesture_over_canvas(12345, 0, 0) is True)

    # ---------- 5. 滚轮：传统 vs 高精度触控板 ----------
    # Tk 在 Windows 上 delta=+120 表示滚轮「上滚」→ 视口应向上（top 减小）
    app._canvas_rect = (8, 118, 2015, 848)
    mid = lambda: app.canvas.yview_moveto(0.5) or top_px()

    app.canvas.yview_moveto(0.5)
    t0 = top_px()
    app._on_mousewheel(Ev(delta=120, x=200, y=500))     # 传统滚轮上滚一格
    check("滚轮上滚一格(120) 上移约 48px", abs(t0 - top_px() - 48) < 6,
          "实际 %.1fpx" % (t0 - top_px()))

    app.canvas.yview_moveto(0.5)
    t0 = top_px()
    app._on_mousewheel(Ev(delta=-120, x=200, y=500))    # 传统滚轮下滚一格
    check("滚轮下滚一格(-120) 下移约 48px", abs(top_px() - t0 - 48) < 6,
          "实际 %.1fpx" % (top_px() - t0))

    app.canvas.yview_moveto(0.5)
    t0 = top_px()
    app._on_mousewheel(Ev(delta=40, x=200, y=500))      # 高精度双指滑动
    check("高精度小 delta(40) 不再被截断成 0",
          abs(t0 - top_px() - 40) < 6, "实际 %.1fpx" % (t0 - top_px()))

    app.canvas.yview_moveto(0)
    app._on_mousewheel(Ev(delta=120, x=200, y=500))     # 顶部继续上滚
    check("顶部继续上滚被夹住", abs(top_px()) < 0.5, "top=%.1f" % top_px())

    app.canvas.yview_moveto(0.5)
    before = top_px()
    app._on_mousewheel(Ev(delta=120, x=9999, y=9999))   # 指针在画布外
    check("指针不在画布内时滚轮被忽略", abs(top_px() - before) < 0.01)

    root.destroy()
    print("\nRESULT: " + ("ALL PASS" if not FAILS else "FAILED -> " + ", ".join(FAILS)))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
