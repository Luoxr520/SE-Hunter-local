from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = ROOT_DIR / "datasets" / "coco128" / "images" / "train2017"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "stage1_coco128_inference"
DEFAULT_MODEL = ROOT_DIR / "models" / "yolo26n.pt"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO inference on COCO128 images and export visualized images plus JSON results."
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL if DEFAULT_MODEL.exists() else "yolo26n.pt"),
        help="Model path or Ultralytics model name. Default: models/yolo26n.pt if present.",
    )
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE_DIR),
        help="Image file or image directory.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for inference: auto, cpu, 0, 1, etc.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of images for a quick smoke test. 0 means all images.",
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


def resolve_model(model: str) -> str:
    model_path = Path(model)
    if model_path.exists():
        return str(model_path)

    local_model = ROOT_DIR / "models" / model
    if local_model.exists():
        return str(local_model)

    return model


def collect_images(source: Path) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image suffix: {source.suffix}")
        return [source]

    if not source.exists():
        raise FileNotFoundError(f"Image source does not exist: {source}")

    image_paths = [
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    return sorted(image_paths)


def get_class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def result_to_record(
    model: YOLO, result: Any, image_path: Path, visualized_path: Path
) -> dict[str, Any]:
    detections: list[dict[str, Any]] = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().tolist()
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": get_class_name(model.names, class_id),
                    "confidence": round(confidence, 6),
                    "bbox_xyxy": [round(float(value), 3) for value in xyxy],
                }
            )

    return {
        "image_name": image_path.name,
        "image_path": str(image_path),
        "visualized_path": str(visualized_path),
        "num_detections": len(detections),
        "detections": detections,
    }


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    output_dir = Path(args.output)
    annotated_dir = output_dir / "annotated_images"
    json_dir = output_dir / "json_results"

    annotated_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(source)
    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    if not image_paths:
        raise RuntimeError(f"No images found in: {source}")

    model_name = resolve_model(args.model)
    device = resolve_device(args.device)
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Images: {len(image_paths)}")

    model = YOLO(model_name)
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
        visualized_path = annotated_dir / image_path.name
        result.save(filename=str(visualized_path))

        record = result_to_record(model, result, image_path, visualized_path)
        records.append(record)

        single_json_path = json_dir / f"{image_path.stem}.json"
        with open(single_json_path, "w", encoding="utf-8") as file:
            json.dump(record, file, ensure_ascii=False, indent=2)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_name": model_name,
        "device": device,
        "source": str(source),
        "num_images": len(records),
        "num_total_detections": sum(record["num_detections"] for record in records),
        "records": records,
    }

    summary_path = output_dir / "all_detections.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("Done")
    print(f"Annotated images: {annotated_dir}")
    print(f"JSON results: {json_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
