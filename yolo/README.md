# YOLO 实验说明

这个目录包含 YOLO 推理和自动标注实验。已有数据、模型和输出都按相对路径组织，请不要随意移动 `datasets`、`models`、`outputs`。

## 目录结构

| 路径 | 说明 |
| --- | --- |
| `datasets/coco128` | COCO128 示例数据集 |
| `datasets/raw_images/unlabeled` | 待自动标注或复核的原始图片 |
| `datasets/labeled_yolo_v1` | 复核后的标注数据和类别文件 |
| `models` | 本地模型文件 |
| `outputs` | 推理和检查结果 |
| `scripts/infer_coco128.py` | 可直接运行的 COCO128 推理脚本 |
| `scripts/stage1_yolo_coco128_inference.ipynb` | 第一阶段 YOLO 推理 notebook |
| `scripts/stage2_auto_labeling_system.ipynb` | 第二阶段自动标注和 X-AnyLabeling 调用 notebook |

## 安装环境

```powershell
cd yolo
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

如果要运行第二阶段的 X-AnyLabeling 标注流程，还需要安装旁边的主项目：

```powershell
pip install -e "..\X-AnyLabeling-dev[cpu]"
xanylabeling checks
```

有 CUDA 环境时，可把上面的 `[cpu]` 换成 `[gpu]` 或 `[gpu-cu11]`。

## 直接运行推理脚本

CPU 跑通：

```powershell
python scripts\infer_coco128.py --device cpu
```

GPU 跑通：

```powershell
python scripts\infer_coco128.py --device 0
```

常用参数：

```powershell
python scripts\infer_coco128.py --model models\yolo26n.pt --source datasets\coco128\images\train2017 --output outputs\stage1_coco128_inference --conf 0.25 --limit 20
```

结果位置：

- 可视化图片：`outputs/stage1_coco128_inference/annotated_images`
- 单图 JSON：`outputs/stage1_coco128_inference/json_results`
- 汇总 JSON：`outputs/stage1_coco128_inference/all_detections.json`

## 使用 notebook

启动 Jupyter：

```powershell
jupyter notebook scripts
```

建议按顺序运行：

1. `stage1_yolo_coco128_inference.ipynb`
2. `stage2_auto_labeling_system.ipynb`

notebook 里有部分兜底路径会指向 `d:\code\yolo`。为了让别人复制项目后也能正常运行，请从 `yolo` 根目录或 `yolo/scripts` 目录打开 notebook。

## 调用 X-AnyLabeling 进行复核

安装好 `X-AnyLabeling-dev` 后，可以手动启动：

```powershell
xanylabeling datasets\raw_images\unlabeled --output datasets\labeled_yolo_v1 --labels datasets\labeled_yolo_v1\classes.txt
```

打开界面后检查并保存标注，输出会落在 `datasets/labeled_yolo_v1`。
