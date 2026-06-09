# -*- coding: utf-8 -*-
"""训练登记后的附带产物导出:ONNX + 两份 X-AnyLabeling 可加载的 yaml 配置。

在 ModelRegistry.add() 存好 <id>.pt 之后调用 export_for_entry(),
会在 registry/models/ 下额外生成(与 <id>.pt 同目录):
    <id>.onnx        由 .pt 导出的 ONNX(X-AnyLabeling 的 YOLO 加载器原生吃 ONNX)
    <id>_onnx.yaml   指向 <id>.onnx 的配置(推荐:在「Load Custom Model」里最稳)
    <id>_pt.yaml     指向 <id>.pt  的配置(若你的版本支持直接加载 .pt 则可用)

设计原则:任何一步失败都只跳过、绝不抛出,避免影响训练/登记主流程。
classes 用数据集里的原始名(含中文),yaml 用 allow_unicode 写出,不转义。
"""
from __future__ import annotations

import os
import os.path as osp


# X-AnyLabeling 配置里的 type 字段:按起始权重名粗略推断;认不出就用通用 "yolo11"
def _infer_type(base, ckpt):
    s = (str(base or "") + " " + str(ckpt or "")).lower()
    # 越具体的放前面
    for key in ("yolo11", "yolo26", "yolov10", "yolov9", "yolov8", "yolov6",
                "yolov5", "yolo12"):
        if key in s:
            # 配置里历史命名是 yolov10/yolov8…,而 11/12/26 用不带 v 的写法
            return key
    return "yolo11"


def _write_yaml(path, *, mtype, name, display_name, model_path, classes,
                iou=0.45, conf=0.25):
    """手写 yaml(不依赖 pyyaml 也能跑;中文直接写,UTF-8 落盘,不转义)。"""
    lines = [
        f"type: {mtype}",
        f"name: {name}",
        "provider: Ultralytics",
        f"display_name: {display_name}",
        # 用正斜杠,跨平台都安全(Windows 也认)
        f"model_path: {str(model_path).replace(os.sep, '/')}",
        f"iou_threshold: {iou}",
        f"conf_threshold: {conf}",
        "classes:",
    ]
    for c in (classes or []):
        # 类别名里若含特殊字符,用双引号包裹更稳;一般中文名直接写即可
        c = str(c)
        if any(ch in c for ch in (":", "#", "'", '"')) or c.strip() != c:
            c = '"' + c.replace('"', '\\"') + '"'
        lines.append(f"  - {c}")
    text = "\n".join(lines) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def _export_onnx(pt_path):
    """用 ultralytics 把 .pt 导成 .onnx,返回 onnx 路径;失败返回 None。"""
    try:
        from ultralytics import YOLO
    except Exception:
        return None
    try:
        model = YOLO(pt_path)
        # opset 固定一个较通用的值;imgsz 用 640 与训练默认一致
        out = model.export(format="onnx", opset=12, imgsz=640)
        # 新版返回字符串/路径,老版可能返回别的;统一推断目标路径
        onnx_path = str(out) if out else osp.splitext(pt_path)[0] + ".onnx"
        if not osp.exists(onnx_path):
            # export 有时把文件落在 .pt 同名 .onnx 处
            cand = osp.splitext(pt_path)[0] + ".onnx"
            onnx_path = cand if osp.exists(cand) else None
        return onnx_path
    except Exception:
        return None


def export_for_entry(entry, classes, *, want_onnx=True, want_pt_yaml=True,
                     log=None):
    """主入口:为一个已登记 entry 生成 onnx + 两份 yaml。

    entry: ModelRegistry.add 返回的 dict(含 id / ckpt / base)
    classes: 类别名列表(含中文)
    返回 dict:{"onnx":..., "onnx_yaml":..., "pt_yaml":...},失败项为 None。
    """
    def _say(msg):
        if log:
            try:
                log(msg)
            except Exception:
                pass

    result = {"onnx": None, "onnx_yaml": None, "pt_yaml": None}
    try:
        pt_path = entry.get("ckpt")
        eid = entry.get("id", "model")
        if not pt_path or not osp.exists(pt_path):
            _say(f"[export] 跳过:找不到 .pt({pt_path})")
            return result
        models_dir = osp.dirname(osp.abspath(pt_path))
        mtype = _infer_type(entry.get("base"), pt_path)

        # 1) 指向 .pt 的 yaml(总能生成,不依赖导出是否成功)
        if want_pt_yaml:
            try:
                pt_yaml = osp.join(models_dir, f"{eid}_pt.yaml")
                _write_yaml(
                    pt_yaml, mtype=mtype, name=f"{eid}-pt",
                    display_name=f"{eid} (PT)",
                    model_path=osp.abspath(pt_path), classes=classes,
                )
                result["pt_yaml"] = pt_yaml
                _say(f"[export] 已生成 {osp.basename(pt_yaml)}")
            except Exception as e:
                _say(f"[export] 写 pt yaml 失败:{e}")

        # 2) 导出 onnx + 指向 onnx 的 yaml
        if want_onnx:
            onnx_path = _export_onnx(pt_path)
            if onnx_path and osp.exists(onnx_path):
                # 规范化到 <id>.onnx,放在 models_dir
                target = osp.join(models_dir, f"{eid}.onnx")
                try:
                    if osp.abspath(onnx_path) != osp.abspath(target):
                        import shutil
                        shutil.move(onnx_path, target)
                    onnx_path = target
                except Exception:
                    pass  # 移动失败就用原路径
                result["onnx"] = onnx_path
                _say(f"[export] 已导出 {osp.basename(onnx_path)}")
                try:
                    onnx_yaml = osp.join(models_dir, f"{eid}_onnx.yaml")
                    _write_yaml(
                        onnx_yaml, mtype=mtype, name=f"{eid}-onnx",
                        display_name=f"{eid} (ONNX)",
                        model_path=osp.abspath(onnx_path), classes=classes,
                    )
                    result["onnx_yaml"] = onnx_yaml
                    _say(f"[export] 已生成 {osp.basename(onnx_yaml)}")
                except Exception as e:
                    _say(f"[export] 写 onnx yaml 失败:{e}")
            else:
                _say("[export] ONNX 导出失败/跳过(可能缺 onnx 依赖);"
                     "仍可用 _pt.yaml")
    except Exception as e:
        _say(f"[export] 异常,已跳过:{e}")
    return result
