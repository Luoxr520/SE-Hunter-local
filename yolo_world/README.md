# YOLO-World 复现实验说明

这个目录用于复现 YOLO-World 在一批图片上的开放词表检测效果。已有模型、图片和输出都按相对路径组织，请保持目录结构。

## 目录结构

| 路径 | 说明 |
| --- | --- |
| `yolov8s-worldv2.pt` | 默认 YOLO-World 模型 |
| `data/images` | 输入图片 |
| `scripts/run_yolo_world.py` | 推理脚本 |
| `outputs/visualized` | 可视化检测结果 |
| `outputs/json/yolo_world_results.json` | 汇总 JSON 结果 |

## 安装环境

```powershell
cd yolo_world_reproduce
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

## 运行

CPU：

```powershell
python scripts\run_yolo_world.py --device cpu
```

GPU：

```powershell
python scripts\run_yolo_world.py --device 0
```

自动选择设备：

```powershell
python scripts\run_yolo_world.py
```

## 常用参数

```powershell
python scripts\run_yolo_world.py --model yolov8s-worldv2.pt --image-dir data\images --conf 0.20 --iou 0.45 --classes person car truck bus
```

默认类别为：

```text
person, car, truck, bus, bicycle, motorcycle, umbrella, boat
```

输出位置：

- 可视化图片：`outputs/visualized`
- 汇总 JSON：`outputs/json/yolo_world_results.json`
