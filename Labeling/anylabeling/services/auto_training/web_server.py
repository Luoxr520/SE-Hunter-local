# -*- coding: utf-8 -*-
"""SE-Hunter Web 实时检测服务(集成版)。

复用现有真实检测管线(ModelServer + ByteTrack + draw_tracks/draw_hud),
对浏览器提供三个端点:
    GET /              炫酷 HUD 前端页面
    GET /stream.mjpg   MJPEG 视频流(画好框的真实检测画面)
    GET /status.json   实时状态(真实目标列表 / FPS / 模型信息 / 告警)

仅用 Python 标准库(http.server),不引入额外依赖。默认监听 127.0.0.1。
数据集/模型与 Live Detection 同源(由调用方从 QSettings 取好传入)。

典型用法(在 PyQt 里点按钮启动):
    from anylabeling.services.auto_training.web_server import WebDetectServer
    srv = WebDetectServer(dataset=ds, source="0", mode="camera", device="0",
                          conf=0.25, model_weight=None)
    srv.start()           # 后台线程起服务 + 推理
    url = srv.url()        # http://127.0.0.1:8420
    ...
    srv.stop()
"""
from __future__ import annotations

import json
import os
import os.path as osp
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# 复用现有管线
try:
    from anylabeling.services.auto_training.model_server import ModelServer
    from anylabeling.services.auto_training.live_runtime import (
        FpsMeter, hud_lines, draw_hud,
    )
    from anylabeling.views.labeling.widgets.live_dialog import (
        draw_tracks, fit_display_size, _expand_video_source, _expand_image_source,
    )
except Exception:  # 允许在 evo 目录直接调试
    from model_server import ModelServer
    from live_runtime import FpsMeter, hud_lines, draw_hud
    # 直接跑时这些来自 live_dialog,可能不可用;调用方需保证在包内运行
    draw_tracks = fit_display_size = _expand_video_source = _expand_image_source = None


# 前端页面文件:与本模块同目录的 se_hunter_web.html
_HTML_PATH = osp.join(osp.dirname(osp.abspath(__file__)), "se_hunter_web.html")


class _Engine:
    """后台推理引擎:循环读帧→真实跟踪→画框→缓存最新 JPEG 与目标数据。"""

    def __init__(self, dataset, source, mode, device, conf, imgsz,
                 model_weight, max_disp_w, dwell_ms):
        self.dataset = dataset or "."
        self.source = source
        self.mode = mode          # camera / video / images
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.model_weight = model_weight
        self.max_disp_w = max_disp_w
        self.dwell_ms = dwell_ms

        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._jpeg = None          # 最新一帧 JPEG bytes
        self._status = {           # 最新状态(供 /status.json)
            "fps": 0.0, "dets": 0, "tracks": 0, "alerts_total": 0,
            "frame": 0, "model": {}, "targets": [], "banner": None,
            "running": False, "error": "",
        }
        self._server = None        # ModelServer

    # --------------------------------------------------- 生命周期
    def start(self):
        self._thread = threading.Thread(target=self._run, name="web-detect", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        try:
            if self._server:
                self._server.stop()
        except Exception:
            pass
        if self._thread:
            self._thread.join(timeout=5)

    # --------------------------------------------------- 取数据(给 HTTP 处理器)
    def latest_jpeg(self):
        with self._lock:
            return self._jpeg

    def status(self):
        with self._lock:
            return dict(self._status)

    # --------------------------------------------------- 推理主循环
    def _run(self):
        import cv2

        try:
            self._server = ModelServer(self.dataset, device=self.device,
                                       conf=self.conf, iou=0.45, imgsz=self.imgsz)
            state = {"banner": None}

            def on_swap(old_id, entry):
                m = (entry.get("metrics", {}) or {}).get("map5095")
                ms = f"{m:.3f}" if isinstance(m, (int, float)) else "-"
                state["banner"] = (f"swapped {old_id} -> {entry['id']}  mAP {ms}",
                                   time.time() + 2.5)

            mw = self.model_weight
            if mw:
                weight = mw
                try:
                    from anylabeling.services.auto_training.train_with_registry import (
                        _looks_like_official_weight, _ensure_official_weight,
                    )
                    if _looks_like_official_weight(mw):
                        weight = _ensure_official_weight(mw)
                except Exception:
                    weight = mw
                self._server.load_ckpt(weight)
            else:
                self._server.on_swap = on_swap
                self._server.load_current()
                self._server.start_watch(interval=2.0)
        except Exception as ex:  # noqa: BLE001
            with self._lock:
                self._status["error"] = f"加载模型失败: {ex}"
                self._status["running"] = False
            return

        fpsm = FpsMeter()
        alerts_total = 0
        frame_count = 0
        with self._lock:
            self._status["running"] = True

        def process(frame):
            nonlocal alerts_total, frame_count
            h0, w0 = frame.shape[:2]
            nw, nh, _ = fit_display_size(w0, h0, self.max_disp_w)
            small = cv2.resize(frame, (nw, nh)) if nw != w0 else frame
            tracks = self._server.track(small)            # ← 真实 YOLO + ByteTrack
            banner = None
            b = state.get("banner")
            if b and time.time() < b[1]:
                banner = b[0]
            fps = fpsm.tick()
            lines = hud_lines(self._server.info(), fps, len(tracks), len(tracks))
            draw_tracks(small, tracks)                    # ← 复用 OpenCV 画框(含中文 PIL 渲染)
            draw_hud(small, lines, banner)
            ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_count += 1

            # 组装目标数据(供右栏真实显示)。按类名粗分敌我/植被。
            targets = []
            cur_alerts = 0
            for t in tracks:
                name = str(getattr(t, "label", ""))
                cls_kind = _classify(name)
                if cls_kind == "alert":
                    cur_alerts += 1
                targets.append({
                    "tid": getattr(t, "tid", None),
                    "name": name,
                    "kind": cls_kind,
                    "conf": round(float(getattr(t, "conf", 0.0)), 3),
                    "x1": int(getattr(t, "x1", 0)), "y1": int(getattr(t, "y1", 0)),
                    "x2": int(getattr(t, "x2", 0)), "y2": int(getattr(t, "y2", 0)),
                })
            if cur_alerts:
                alerts_total += 0   # 累计告警按"出现告警的帧"统计太频繁,这里改为不累加瞬时

            info = self._server.info() or {}
            metrics = info.get("metrics", {}) or {}
            with self._lock:
                if ok:
                    self._jpeg = bytes(buf)
                self._status.update({
                    "fps": round(float(fps), 1),
                    "dets": len(tracks),
                    "tracks": len(tracks),
                    "frame": frame_count,
                    "alerts_now": cur_alerts,
                    "model": {
                        "id": info.get("id", "-"),
                        "map5095": metrics.get("map5095"),
                        "map50": metrics.get("map50"),
                        "swaps": info.get("swaps", 0),
                    },
                    "targets": targets,
                    "banner": banner,
                })

        try:
            if self.mode == "images":
                paths = _expand_image_source(self.source)
                if not paths:
                    self._set_err("没有找到图片: %s" % self.source)
                    return
                i = 0
                while not self._stop.is_set():
                    frame = cv2.imread(paths[i])
                    if frame is not None:
                        process(frame)
                    i = (i + 1) % len(paths)
                    self._sleep_ms(self.dwell_ms)
            else:
                vids = _expand_video_source(self.source)
                if not vids:
                    self._set_err("没有视频源")
                    return
                vi = 0
                while not self._stop.is_set():
                    one = vids[vi]
                    src = int(one) if str(one).isdigit() else one
                    cap = cv2.VideoCapture(src)
                    if not cap.isOpened():
                        self._set_err("无法打开视频源: %s" % one)
                        cap.release()
                        break
                    while not self._stop.is_set():
                        ok_, frame = cap.read()
                        if not ok_:
                            break          # 该视频结束
                        process(frame)
                    cap.release()
                    vi = (vi + 1) % len(vids)   # 多视频循环播放
        finally:
            try:
                self._server.stop()
            except Exception:
                pass
            with self._lock:
                self._status["running"] = False

    def _set_err(self, msg):
        with self._lock:
            self._status["error"] = msg
            self._status["running"] = False

    def _sleep_ms(self, ms):
        end = time.time() + ms / 1000.0
        while time.time() < end and not self._stop.is_set():
            time.sleep(0.01)


# 类名 → 敌我/植被 粗分类(可按你的真实类名调整)
_ALERT_WORDS = ("可疑", "敌", "hostile", "unknown", "未知", "车辆", "热源")
_OK_WORDS = ("友方", "friendly", "person", "我方", "平民")


def _classify(name):
    s = str(name).lower()
    for w in _ALERT_WORDS:
        if w.lower() in s:
            return "alert"
    for w in _OK_WORDS:
        if w.lower() in s:
            return "ok"
    return "neutral"   # 树木/植被等


# ======================================================================
# HTTP 服务
# ======================================================================
class WebDetectServer:
    def __init__(self, dataset, source="0", mode="camera", device=None,
                 conf=0.25, imgsz=640, model_weight=None, max_disp_w=1280,
                 dwell_ms=1200, host="127.0.0.1", port=8420):
        self.host = host
        self.port = port
        self.engine = _Engine(dataset, source, mode, device, conf, imgsz,
                               model_weight, max_disp_w, dwell_ms)
        self._httpd = None
        self._http_thread = None

    def url(self):
        return f"http://{self.host}:{self.port}"

    def start(self):
        self.engine.start()
        engine = self.engine

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # 静默,不刷屏
                pass

            def _no_cache(self):
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")

            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path in ("/", "/index.html"):
                    self._serve_page()
                elif path == "/stream.mjpg":
                    self._serve_mjpeg()
                elif path == "/status.json":
                    self._serve_status()
                else:
                    self.send_error(404)

            def _serve_page(self):
                try:
                    with open(_HTML_PATH, "rb") as f:
                        data = f.read()
                except Exception:
                    data = b"<h1>se_hunter_web.html not found</h1>"
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _serve_status(self):
                data = json.dumps(engine.status(), ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self._no_cache()
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _serve_mjpeg(self):
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame")
                self._no_cache()
                self.end_headers()
                try:
                    while True:
                        jpeg = engine.latest_jpeg()
                        if jpeg is None:
                            time.sleep(0.05)
                            continue
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(
                            ("Content-Length: %d\r\n\r\n" % len(jpeg)).encode())
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                        time.sleep(0.03)    # ~30fps 上限
                except (BrokenPipeError, ConnectionResetError):
                    pass    # 浏览器断开,正常

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self._http_thread = threading.Thread(
            target=self._httpd.serve_forever, name="web-http", daemon=True)
        self._http_thread.start()

    def stop(self):
        try:
            if self._httpd:
                self._httpd.shutdown()
                self._httpd.server_close()
        except Exception:
            pass
        self.engine.stop()


# 命令行直接跑(调试用)
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--source", default="0")
    p.add_argument("--mode", default="camera", choices=["camera", "video", "images"])
    p.add_argument("--device", default=None)
    p.add_argument("--port", type=int, default=8420)
    args = p.parse_args()
    srv = WebDetectServer(args.dataset, source=args.source, mode=args.mode,
                          device=args.device, port=args.port)
    srv.start()
    print("Web 检测已启动:", srv.url())
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        srv.stop()
