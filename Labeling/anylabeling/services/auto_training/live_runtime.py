# -*- coding: utf-8 -*-
"""
live_runtime.py — 自我演化检测系统 · 第 3c 步
实时主循环:摄像头/视频 → 当前线上模型推理(后台自动热替换)→ 检测框连续过渡 → 画框+角标 → 显示/写出。

把 3a(ModelServer 热替换)+ 3b(BoxSmoother 连续过渡)串成完整效果:
一边后台训练持续产出更好的模型自动热替换,一边画面里框越收越准、识别不中断。

用法:
    # 摄像头实时显示(与训练共用 GPU 时 serving 建议 cpu)
    python -m anylabeling.services.auto_training.live_runtime --source 0 --device cpu
    # 跑视频文件并写出标注视频
    python -m anylabeling.services.auto_training.live_runtime --source in.mp4 --save out.mp4
另开一个终端持续训练,画面里会自动热替换:
    python -m anylabeling.services.auto_training.train_with_registry train --epochs 50 --device 0 --workers 0

角标显示:模型版本 / mAP50-95 / 热替换次数 / FPS / 当前帧检测数 / 跟踪数,换模型瞬间顶部高亮提示。
中文类别名通过 PIL 渲染(draw_text),自动查找系统 CJK 字体;无 PIL/字体时降级用 cv2 画 ASCII。

数据流核心(step / hud_lines / FpsMeter 等)不依赖 cv2,便于测试;cv2 读帧/画框/显示在本机跑。
"""
from __future__ import annotations

import argparse
import os
import os.path as osp
import sys
import time

try:
    from anylabeling.services.auto_training.model_server import ModelServer, resolve_dataset
    from anylabeling.services.auto_training.box_smoother import BoxSmoother
except ImportError:
    from model_server import ModelServer, resolve_dataset
    from box_smoother import BoxSmoother

# 类别配色(BGR),按 cls 取
PALETTE = [
    (66, 135, 245), (80, 200, 120), (60, 76, 231), (235, 180, 52),
    (180, 120, 255), (255, 200, 80), (52, 235, 210), (200, 80, 180),
    (120, 200, 60), (90, 90, 235),
]


def color_for_class(cls):
    return PALETTE[int(cls) % len(PALETTE)]


def clamp_box(x1, y1, x2, y2, w, h):
    """裁到画面内并转成 int。"""
    xi1 = max(0, min(int(round(x1)), w - 1))
    yi1 = max(0, min(int(round(y1)), h - 1))
    xi2 = max(0, min(int(round(x2)), w - 1))
    yi2 = max(0, min(int(round(y2)), h - 1))
    if xi2 < xi1:
        xi1, xi2 = xi2, xi1
    if yi2 < yi1:
        yi1, yi2 = yi2, yi1
    return xi1, yi1, xi2, yi2


class FpsMeter:
    """EMA 平滑的 FPS 计。tick() 内部用 perf_counter;测试可传 now。"""

    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.last = None
        self.fps = 0.0

    def tick(self, now=None):
        now = now if now is not None else time.perf_counter()
        if self.last is not None:
            dt = now - self.last
            if dt > 0:
                inst = 1.0 / dt
                self.fps = inst if self.fps == 0.0 else self.alpha * inst + (1 - self.alpha) * self.fps
        self.last = now
        return self.fps


def hud_lines(info, fps, ndet, ntracks):
    """角标文字(ASCII)。info 来自 server.info()。"""
    if info:
        mid = info.get("id", "?")
        mp = (info.get("metrics", {}) or {}).get("map5095")
        swaps = info.get("swaps", 0)
    else:
        mid, mp, swaps = "?", None, 0
    mp_s = f"{mp:.3f}" if isinstance(mp, (int, float)) else "-"
    return [
        f"Model {mid}   mAP50-95 {mp_s}   swaps {swaps}",
        f"FPS {fps:.1f}   dets {ndet}   tracks {ntracks}",
    ]


def step(server, smoother, frame, fpsm, state, now=None):
    """每帧的核心处理(无 cv2):推理 → 平滑 → 角标。返回 (dets, sboxes, lines, banner)。"""
    dets = server.infer(frame)
    sboxes = smoother.update(dets)
    t = now if now is not None else time.time()
    banner = None
    b = state.get("banner")
    if b and t < b[1]:
        banner = b[0]
    lines = hud_lines(server.info(), fpsm.tick(now), len(dets), len(sboxes))
    return dets, sboxes, lines, banner


# --------------------------------------------------------------------------------------
# 中文文字渲染(cv2.putText 只画 ASCII,中文类别名会变 ??? ;改用 PIL 画)
# 版本标记:若 Live Detection 启动日志里没有出现下面这行 banner,说明跑的还是旧代码/旧 .pyc。
# --------------------------------------------------------------------------------------
LIVE_RUNTIME_TEXT_RENDERER = "pil-cjk-v3"   # 版本号,方便确认确实加载了新代码

_FONT_CACHE = {}          # px -> (font, is_cjk)
_FONT_PATH = None         # 解析一次后缓存路径;None 表示尚未解析,False 表示找不到
_PIL_OK = None            # 是否能 import PIL
_DRAWTEXT_DIAG_DONE = False   # draw_text 首次实际渲染中文时,打一行诊断


def _has_cjk(s):
    return any("\u4e00" <= ch <= "\u9fff" for ch in str(s))


def _find_cjk_font():
    """跨平台找一个能显示中文的字体文件。优先环境变量 ANYLABELING_CJK_FONT。找不到返回 None。"""
    import os
    env = os.environ.get("ANYLABELING_CJK_FONT", "").strip()
    if env and os.path.exists(env):
        return env
    candidates = [
        # Windows
        r"C:\Windows\Fonts\msyh.ttc",      # 微软雅黑
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\msyh.ttf",
        r"C:\Windows\Fonts\simhei.ttf",    # 黑体
        r"C:\Windows\Fonts\simsun.ttc",    # 宋体
        r"C:\Windows\Fonts\simkai.ttf",    # 楷体
        r"C:\Windows\Fonts\Deng.ttf",      # 等线
        r"C:\Windows\Fonts\STSONG.TTF",
        # Linux
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        # macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    # 兜底 1:扫常见字体目录里任何带 cjk/hei/song/yahei/noto 的字体
    for d in (r"C:\Windows\Fonts", "/usr/share/fonts", "/Library/Fonts",
              os.path.expanduser("~/.fonts"),
              os.path.expanduser("~/.local/share/fonts")):
        if not os.path.isdir(d):
            continue
        try:
            for root, _dirs, files in os.walk(d):
                for fn in files:
                    low = fn.lower()
                    if low.endswith((".ttf", ".ttc", ".otf")) and any(
                        k in low for k in ("cjk", "hei", "song", "sun", "yahei",
                                           "msyh", "noto", "pingfang", "wqy",
                                           "kai", "fangsong", "yuan")
                    ):
                        return os.path.join(root, fn)
        except Exception:
            pass
    # 兜底 2:借助 matplotlib 的字体管理器(X-AnyLabeling 环境常带 matplotlib)
    try:
        from matplotlib import font_manager
        for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                     "WenQuanYi Zen Hei", "PingFang SC", "Source Han Sans SC"):
            try:
                p = font_manager.findfont(name, fallback_to_default=False)
                if p and os.path.exists(p):
                    return p
            except Exception:
                continue
    except Exception:
        pass
    # 兜底 3:用 Ultralytics 自己的字体缓存目录里能显示中文的字体。
    # 验证图(val_batch*.jpg)能正常画中文,说明这台机器上 Ultralytics 已经有可用中文字体,
    # 直接复用最稳妥。常见位置:用户目录下 .config/Ultralytics 或 AppData\Roaming\Ultralytics。
    try:
        from pathlib import Path
        ud_candidates = []
        try:
            from ultralytics.utils import USER_CONFIG_DIR  # 新版 ultralytics
            ud_candidates.append(Path(USER_CONFIG_DIR))
        except Exception:
            pass
        ud_candidates += [
            Path.home() / "AppData" / "Roaming" / "Ultralytics",
            Path.home() / ".config" / "Ultralytics",
            Path.home() / "Library" / "Application Support" / "Ultralytics",
        ]
        for ud in ud_candidates:
            if not ud or not ud.is_dir():
                continue
            for fn in ud.iterdir():
                low = fn.name.lower()
                if low.endswith((".ttf", ".ttc", ".otf")) and any(
                    k in low for k in ("unicode", "cjk", "hei", "noto", "song",
                                       "yahei", "msyh", "wqy", "pingfang")
                ):
                    return str(fn)
    except Exception:
        pass
    return None


_FONT_LOGGED = False
_RENDER_DIAG = ""          # 对外可读的渲染诊断(HUD 会显示),便于无终端时定位问号


def get_render_diag():
    """返回一行渲染诊断,供 HUD/界面显示(GUI 无终端时也能看到)。"""
    return _RENDER_DIAG


def _ultralytics_font(px):
    """复用 Ultralytics 自己的字体加载(它在本机已能正常画中文,最可靠)。失败返回 None。"""
    try:
        from ultralytics.utils.plotting import check_pil_font
        # check_pil_font 会在需要时自动下载并缓存可用字体(含非拉丁)
        return check_pil_font(size=px)
    except Exception:
        return None


def _get_font(px):
    """取(并缓存)指定像素大小的字体,返回 (font, is_cjk)。
    顺序:系统/环境变量中文字体 → Ultralytics 字体 → PIL 默认(豆腐块)。
    is_cjk=True 表示拿到的是能画中文的真字体。"""
    global _FONT_PATH, _RENDER_DIAG
    px = int(px)
    if px in _FONT_CACHE:
        return _FONT_CACHE[px]
    from PIL import ImageFont
    if _FONT_PATH is None:
        _FONT_PATH = _find_cjk_font() or False
    font, is_cjk, src = None, False, "none"
    # 1) 我们查到的系统/环境变量字体
    if _FONT_PATH:
        try:
            font = ImageFont.truetype(_FONT_PATH, px)
            is_cjk, src = True, _FONT_PATH
        except Exception:
            font = None
    # 2) Ultralytics 的字体(已在本机证明能画中文)
    if font is None:
        uf = _ultralytics_font(px)
        if uf is not None:
            font, is_cjk, src = uf, True, "ultralytics"
    # 3) 实在没有:PIL 默认(画中文是豆腐块,但好过崩溃)
    if font is None:
        try:
            font = ImageFont.load_default()
            src = "pil-default"
        except Exception:
            font = None
            src = "none"
    _RENDER_DIAG = f"renderer={LIVE_RUNTIME_TEXT_RENDERER} font_src={src}"
    _FONT_CACHE[px] = (font, is_cjk)
    return _FONT_CACHE[px]


def _pil_available():
    global _PIL_OK
    if _PIL_OK is None:
        try:
            import PIL  # noqa: F401
            _PIL_OK = True
        except Exception:
            _PIL_OK = False
    return _PIL_OK


def _text_size(text, px):
    """量文字像素尺寸 (w, h)。"""
    font = None
    if _pil_available():
        font, _ = _get_font(px)
    if font is None:
        return (len(str(text)) * px // 2, px)
    try:
        l, t, r, b = font.getbbox(str(text))
        return (r - l, b - t)
    except Exception:
        try:
            return font.getsize(str(text))
        except Exception:
            return (len(str(text)) * px // 2, px)


def _draw_text_cv2(img, text, org, px, color_bgr):
    import cv2
    fs = max(0.4, px / 22.0)
    x, y = int(org[0]), int(org[1])
    cv2.putText(img, text, (x, y + px - 2),
                cv2.FONT_HERSHEY_SIMPLEX, fs, color_bgr, 1, cv2.LINE_AA)


def _draw_text_pil(img, text, org, px, color_bgr, font):
    import numpy as np
    from PIL import Image, ImageDraw
    pil = Image.fromarray(img[:, :, ::-1])         # BGR -> RGB
    d = ImageDraw.Draw(pil)
    rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    d.text((int(org[0]), int(org[1])), text, font=font, fill=rgb)
    img[:, :, :] = np.asarray(pil)[:, :, ::-1]     # RGB -> BGR,写回原数组


def draw_text(img, text, org, px, color_bgr):
    """在 img 的 org=(x, y_top_left) 处画文字,支持中文。color 为 BGR。

    关键修正:只要 PIL 可用,**含中文一律走 PIL**(绝不退回 cv2,cv2 画中文必是 ?)。
    纯 ASCII 才用 cv2(更快)。PIL 渲染若抛异常,把错误写进可读诊断,而不是悄悄变 ?。
    """
    global _RENDER_DIAG
    text = str(text)
    has_cjk = _has_cjk(text)
    # 纯 ASCII:cv2 足够快
    if not has_cjk:
        _draw_text_cv2(img, text, org, px, color_bgr)
        return
    # 含中文但没有 PIL:无解,只能 cv2(会是 ?),并在诊断里写明
    if not _pil_available():
        _RENDER_DIAG = "renderer=NO-PIL 中文将显示为问号,请 pip install pillow"
        _draw_text_cv2(img, text, org, px, color_bgr)
        return
    # 含中文 + 有 PIL:无条件用 PIL 渲染(哪怕只是豆腐块也不退回 ?)
    font, _is_cjk = _get_font(px)
    if font is None:
        _RENDER_DIAG = "renderer=PIL-no-font 字体加载失败"
        _draw_text_cv2(img, text, org, px, color_bgr)
        return
    try:
        _draw_text_pil(img, text, org, px, color_bgr, font)
    except Exception as e:        # 不让异常变成静默的 ?
        _RENDER_DIAG = f"renderer=PIL-ERROR {type(e).__name__}: {e}"
        _draw_text_cv2(img, text, org, px, color_bgr)


# --------------------------------------------------------------------------------------
# cv2 绘制(本机运行)
# --------------------------------------------------------------------------------------
def _draw_one(img, x1, y1, x2, y2, color, label):
    import cv2

    px = 18
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    tw, th = _text_size(label, px)
    by2 = y1
    by1 = max(0, y1 - th - 6)
    cv2.rectangle(img, (x1, by1), (x1 + tw + 6, by2), color, -1)
    draw_text(img, label, (x1 + 3, by1 + 1), px, (0, 0, 0))


def draw_boxes(img, sboxes):
    import cv2

    h, w = img.shape[:2]
    for b in sboxes:
        color = color_for_class(b.cls)
        x1, y1, x2, y2 = clamp_box(b.x1, b.y1, b.x2, b.y2, w, h)
        label = f"{b.id} {b.label} {b.conf:.2f}"
        if b.alpha >= 0.99:
            _draw_one(img, x1, y1, x2, y2, color, label)
        else:                       # 渐隐渐显:按 alpha 混合这一个框
            ov = img.copy()
            _draw_one(ov, x1, y1, x2, y2, color, label)
            cv2.addWeighted(ov, float(b.alpha), img, 1 - float(b.alpha), 0, img)


def draw_hud(img, lines, banner=None):
    import cv2

    pad, px, lh = 8, 18, 26
    texts = ([banner] if banner else []) + list(lines)
    tw = max(_text_size(t, px)[0] for t in texts)
    pw, ph = tw + 2 * pad, lh * len(texts) + pad
    ov = img.copy()
    cv2.rectangle(ov, (8, 8), (8 + pw, 8 + ph), (20, 20, 20), -1)
    cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)
    y = 8 + pad
    for i, t in enumerate(texts):
        col = (0, 215, 255) if (banner and i == 0) else (235, 235, 235)
        draw_text(img, t, (8 + pad, y), px, col)
        y += lh


# --------------------------------------------------------------------------------------
# 主循环
# --------------------------------------------------------------------------------------
def run(args):
    try:
        import cv2
    except ImportError:
        print("需要 opencv-python:pip install opencv-python", file=sys.stderr)
        return 1

    ds = resolve_dataset(args.dataset)
    if not ds:
        print("[live] 没指定 --dataset,也没从 QSettings 读到发布输出目录", file=sys.stderr)
        return 1

    server = ModelServer(ds, device=args.device, conf=args.conf, iou=args.iou, imgsz=args.imgsz)
    state = {"banner": None}

    def on_swap(old_id, entry):
        m = (entry.get("metrics", {}) or {}).get("map5095")
        m_s = f"{m:.3f}" if isinstance(m, (int, float)) else "-"
        state["banner"] = (f"swapped {old_id} -> {entry['id']}  mAP {m_s}", time.time() + 2.5)
        print(f"🔁 {state['banner'][0]}")

    server.on_swap = on_swap
    try:
        sid = server.load_current()
    except Exception as ex:  # noqa: BLE001
        print("[live] 加载当前模型失败:", ex, file=sys.stderr)
        return 1
    print(f"[live] 当前线上: {sid}  device={args.device or 'auto'}  source={args.source}")
    server.start_watch(interval=args.interval)

    smoother = BoxSmoother(smooth=args.smooth, iou_match=args.iou_match,
                           fade_in=args.fade_in, fade_out=args.fade_out, max_age=args.max_age)
    fpsm = FpsMeter()

    src = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("[live] 无法打开视频源:", args.source, file=sys.stderr)
        server.stop()
        return 1

    win = "self-evolving detection"
    if not args.no_window:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    writer = None
    frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            _, sboxes, lines, banner = step(server, smoother, frame, fpsm, state)
            draw_boxes(frame, sboxes)
            draw_hud(frame, lines, banner)

            if args.save:
                if writer is None:
                    h, w = frame.shape[:2]
                    fps_out = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    if fps_out <= 1:
                        fps_out = 25.0
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps_out, (w, h))
                writer.write(frame)
            if not args.no_window:
                cv2.imshow(win, frame)
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                    break
            frames += 1
            if args.max_frames and frames >= args.max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_window:
            cv2.destroyAllWindows()
        server.stop()
    sw = server.info()["swaps"] if server.info() else 0
    print(f"[live] 结束,共 {frames} 帧,热替换 {sw} 次" + (f",已写出 {args.save}" if args.save else ""))
    return 0


def build_parser():
    p = argparse.ArgumentParser(description="实时主循环:推理(热替换)+ 检测框连续过渡")
    p.add_argument("--source", default="0", help="摄像头编号(0)或视频文件路径")
    p.add_argument("--dataset", default=None, help="数据集目录(默认读 QSettings)")
    p.add_argument("--device", default=None, help="0 / cpu;与训练共用 GPU 时建议 cpu")
    p.add_argument("--interval", type=float, default=2.0, help="热替换轮询间隔秒")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--smooth", type=float, default=0.4, help="EMA 系数:大跟手小平滑")
    p.add_argument("--iou-match", type=float, default=0.3, help="跟踪关联 IoU 阈值")
    p.add_argument("--fade-in", type=int, default=3)
    p.add_argument("--fade-out", type=int, default=6)
    p.add_argument("--max-age", type=int, default=8, help="目标丢失多少帧后移除")
    p.add_argument("--save", default=None, help="写出标注视频路径(.mp4)")
    p.add_argument("--no-window", action="store_true", help="不显示窗口(配合 --save)")
    p.add_argument("--max-frames", type=int, default=0, help="跑这么多帧后停(0=不限)")
    return p


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
