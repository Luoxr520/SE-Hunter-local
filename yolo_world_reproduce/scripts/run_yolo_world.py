"""
Legacy notebook-exported script kept for reference.
The runnable CLI implementation starts after this docstring.

from pathlib import Path
import json
from datetime import datetime

from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT_DIR / "data" / "images"
VIS_DIR = ROOT_DIR / "outputs" / "visualized"
JSON_DIR = ROOT_DIR / "outputs" / "json"

VIS_DIR.mkdir(parents=True, exist_ok=True)
JSON_DIR.mkdir(parents=True, exist_ok=True)


# 轻量模型，先看效果
MODEL_NAME = "yolov8s-worldv2.pt"

# 你可以先用 COCO 常见类别测试
PROMPT_CLASSES = [
    "person",
    "car",
    "truck",
    "bus",
    "bicycle",
    "motorcycle",
    "umbrella",
    "boat",
]

# 后面可以换成更接近项目的提示词
# PROMPT_CLASSES = [
#     "person",
#     "vehicle",
#     "truck",
#     "tent",
#     "camouflage net",
#     "temporary shelter",
#     "construction equipment",
# ]


def collect_images(image_dir: Path):
    image_paths = []
    for suffix in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_paths.extend(image_dir.glob(suffix))
    return sorted(image_paths)


def main():
    image_paths = collect_images(IMAGE_DIR)

    if not image_paths:
        raise RuntimeError(f"没有找到测试图片，请把图片放到：{IMAGE_DIR}")

    print(f"图片数量：{len(image_paths)}")

    print(f"加载模型：{MODEL_NAME}")
    model = YOLO(MODEL_NAME)

    print("设置提示词类别：")
    for i, name in enumerate(PROMPT_CLASSES):
        print(i, name)

    model.set_classes(PROMPT_CLASSES)

    all_records = []

    for index, image_path in enumerate(image_paths, start=1):
        print(f"[{index}/{len(image_paths)}] {image_path.name}")

        results = model.predict(
            source=str(image_path),
            imgsz=640,
            conf=0.20,
            iou=0.45,
            device=0,
            verbose=False,
        )

        result = results[0]

        save_vis_path = VIS_DIR / image_path.name
        result.save(filename=str(save_vis_path))

        detections = []

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().tolist()

            class_name = (
                PROMPT_CLASSES[class_id]
                if 0 <= class_id < len(PROMPT_CLASSES)
                else str(class_id)
            )

            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 6),
                "bbox_xyxy": [round(float(v), 3) for v in xyxy],
            })

        record = {
            "image_name": image_path.name,
            "image_path": str(image_path),
            "visualized_path": str(save_vis_path),
            "prompt_classes": PROMPT_CLASSES,
            "num_detections": len(detections),
            "detections": detections,
        }

        all_records.append(record)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_name": MODEL_NAME,
        "prompt_classes": PROMPT_CLASSES,
        "num_images": len(all_records),
        "num_total_detections": sum(item["num_detections"] for item in all_records),
        "records": all_records,
    }

    json_path = JSON_DIR / "yolo_world_results.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("YOLO-World 测试完成")
    print("可视化结果目录：", VIS_DIR)
    print("JSON 结果：", json_path)
    print("=" * 80)


if __name__ == "__main__":
    main()
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIR = ROOT_DIR / "data" / "images"
DEFAULT_VIS_DIR = ROOT_DIR / "outputs" / "visualized"
DEFAULT_JSON_DIR = ROOT_DIR / "outputs" / "json"
DEFAULT_MODEL = "yolov8s-worldv2.pt"
DEFAULT_PROMPT_CLASSES = [
    "person",
    "car",
    "truck",
    "bus",
    "bicycle",
    "motorcycle",
    "umbrella",
    "boat",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO-World inference and export visualized images plus JSON results."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model path or Ultralytics model name.")
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR), help="Directory containing input images.")
    parser.add_argument("--vis-dir", default=str(DEFAULT_VIS_DIR), help="Directory for visualized images.")
    parser.add_argument("--json-dir", default=str(DEFAULT_JSON_DIR), help="Directory for JSON output.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for inference: auto, cpu, 0, 1, etc.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_PROMPT_CLASSES,
        help="Prompt classes for YOLO-World.",
    )
    return parser.parse_args()


def resolve_device(device: str) -> str | int:
    if device != "auto":
        return int(device) if device.isdigit() else device

    try:
        import torch

        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def resolve_model(model_name: str) -> str:
    model_path = Path(model_name)
    if model_path.exists():
        return str(model_path)

    local_model = ROOT_DIR / model_name
    if local_model.exists():
        return str(local_model)

    return model_name


def collect_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    return sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def result_to_record(
    result: Any,
    image_path: Path,
    visualized_path: Path,
    prompt_classes: list[str],
) -> dict[str, Any]:
    detections: list[dict[str, Any]] = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().tolist()
            class_name = (
                prompt_classes[class_id]
                if 0 <= class_id < len(prompt_classes)
                else str(class_id)
            )
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 6),
                    "bbox_xyxy": [round(float(value), 3) for value in xyxy],
                }
            )

    return {
        "image_name": image_path.name,
        "image_path": str(image_path),
        "visualized_path": str(visualized_path),
        "prompt_classes": prompt_classes,
        "num_detections": len(detections),
        "detections": detections,
    }


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    vis_dir = Path(args.vis_dir)
    json_dir = Path(args.json_dir)
    prompt_classes = list(args.classes)
    device = resolve_device(args.device)
    model_name = resolve_model(args.model)

    vis_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(image_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in: {image_dir}")

    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Images: {len(image_paths)}")
    print("Prompt classes:")
    for index, class_name in enumerate(prompt_classes):
        print(index, class_name)

    model = YOLO(model_name)
    model.set_classes(prompt_classes)

    records: list[dict[str, Any]] = []
    for index, image_path in enumerate(image_paths, start=1):
        print(f"[{index}/{len(image_paths)}] {image_path.name}")
        results = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=device,
            verbose=False,
        )

        result = results[0]
        visualized_path = vis_dir / image_path.name
        result.save(filename=str(visualized_path))
        records.append(
            result_to_record(result, image_path, visualized_path, prompt_classes)
        )

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_name": model_name,
        "device": device,
        "prompt_classes": prompt_classes,
        "num_images": len(records),
        "num_total_detections": sum(record["num_detections"] for record in records),
        "records": records,
    }

    json_path = json_dir / "yolo_world_results.json"
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("Done")
    print(f"Visualized images: {vis_dir}")
    print(f"JSON result: {json_path}")


if __name__ == "__main__":
    main()
