# 本地安装与启动说明

本目录是 X-AnyLabeling 主项目源码。项目依赖写在 `pyproject.toml` 中，Python 版本要求为 3.11 及以上，推荐使用 Python 3.12。

## Windows CPU 环境

```powershell
cd X-AnyLabeling-dev
py -3.12 -m venv .venv-cpu
.\.venv-cpu\Scripts\Activate.ps1
python -m pip install -U pip uv
uv pip install -e ".[cpu]"
```

验证安装：

```powershell
xanylabeling checks
xanylabeling version
```

启动图形界面：

```powershell
xanylabeling
```

## Windows GPU 环境

CUDA 12.x：

```powershell
cd X-AnyLabeling-dev
py -3.12 -m venv .venv-cu12
.\.venv-cu12\Scripts\Activate.ps1
python -m pip install -U pip uv
uv pip install -e ".[gpu]"
xanylabeling
```

CUDA 11.x：

```powershell
cd X-AnyLabeling-dev
py -3.12 -m venv .venv-cu11
.\.venv-cu11\Scripts\Activate.ps1
python -m pip install -U pip uv
uv pip install -e ".[gpu-cu11]"
xanylabeling
```

注意：不要在同一个环境里同时安装 `onnxruntime` 和 `onnxruntime-gpu`。CPU、CUDA 11、CUDA 12 建议分别建独立虚拟环境。

## 开发环境

需要运行测试、打包或参与开发时安装 `dev` 依赖：

```powershell
uv pip install -e ".[cpu,dev]"
```

常用命令：

```powershell
xanylabeling help
xanylabeling checks
xanylabeling config
pytest
```

## 常见启动问题

- 找不到 `xanylabeling`：确认已经激活虚拟环境，并在 `X-AnyLabeling-dev` 目录下执行过 `uv pip install -e ".[cpu]"`。
- Qt 或界面启动失败：优先尝试全新虚拟环境，避免和其他项目的 PyQt/PySide 版本冲突。
- GPU 推理不可用：确认 CUDA、cuDNN、显卡驱动和 `onnxruntime-gpu` 版本匹配。无法确认时先用 CPU 环境跑通。
- 换了目录后代码不生效：editable install 会绑定安装时的源码目录，移动目录后需要重新执行安装命令。

## 和旁边实验目录的关系

`../yolo` 的自动标注 notebook 会调用本项目的 `xanylabeling` 命令。建议先在本目录安装并确认 `xanylabeling` 可用，再进入 `../yolo` 运行实验。
