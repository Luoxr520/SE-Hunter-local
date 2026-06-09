# 项目使用总览

这个目录下面放了多个项目和实验。为了方便别人接手使用，建议从本文件开始看，再进入对应子目录阅读更具体的启动说明。

我没有删除、移动或重命名现有文件和文件夹。`datasets`、`models`、`outputs`、`assets`、`.git`、`.vscode` 等目录都可能被代码或工具引用，分享给别人时请尽量保持原有相对路径。

## 目录说明

| 目录 | 作用 | 推荐入口 |
| --- | --- | --- |
| `X-AnyLabeling-dev` | 主项目，X-AnyLabeling 标注工具源码 | `X-AnyLabeling-dev/LOCAL_SETUP.md` |
| `yolo` | YOLO 推理、自动标注流程实验，包含 COCO128 数据、模型、notebook 和输出 | `yolo/README.md` |
| `yolo_world_reproduce` | YOLO-World 复现实验，包含图片、模型、脚本和输出 | `yolo_world_reproduce/README.md` |
| `test` | 前端 HTML 演示，包括普通页面和摄像头交互页面 | `test/README.md` |
| `races` | C++/CMake 最小示例项目 | `races/README.md` |
| `X-AnyLabeling-dev.zip` | 主项目压缩包备份或分发包 | 保留即可 |

## 推荐基础环境

- 操作系统：Windows 10/11 优先，Python 项目也可在 Linux/macOS 上按官方依赖自行适配。
- Python：建议 Python 3.12；`X-AnyLabeling-dev` 要求 Python 3.11 及以上。
- 环境管理：推荐每个项目单独使用 `venv` 或 conda 环境，避免包版本互相影响。
- GPU：不是所有流程都必须用 GPU。YOLO 类实验可以用 CPU 先跑通，速度会慢一些。
- 浏览器：`test/camera-interact.html` 需要现代浏览器、网络 CDN 访问和摄像头权限。
- C++：`races` 需要 CMake 3.30 及以上和支持 C++20 的编译器。

## 主项目快速启动

```powershell
cd X-AnyLabeling-dev
py -3.12 -m venv .venv-cpu
.\.venv-cpu\Scripts\Activate.ps1
python -m pip install -U pip uv
uv pip install -e ".[cpu]"
xanylabeling checks
xanylabeling
```

GPU 版本、开发依赖和常见问题见 `X-AnyLabeling-dev/LOCAL_SETUP.md`。

## YOLO 实验快速启动

```powershell
cd yolo
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
python scripts\infer_coco128.py --device cpu
```

输出会写到 `yolo/outputs/stage1_coco128_inference`。更多 notebook 和 X-AnyLabeling 自动标注流程见 `yolo/README.md`。

## YOLO-World 复现实验快速启动

```powershell
cd yolo_world_reproduce
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
python scripts\run_yolo_world.py --device cpu
```

输出会写到 `yolo_world_reproduce/outputs`。

## 前端演示快速启动

普通页面可以直接双击 `test/index.html`。摄像头交互页面建议通过本地服务访问：

```powershell
cd test
python -m http.server 8000
```

然后打开 `http://localhost:8000/camera-interact.html`。

## 分享给别人前的检查清单

1. 保持目录结构不变，尤其是 `datasets`、`models`、`outputs` 这些相对路径。
2. 大模型文件如 `.pt`、`.onnx`、`.ts` 需要一起提供，否则推理脚本或自动标注会缺模型。
3. 接收方先创建虚拟环境，再安装对应目录的依赖，不要直接在系统 Python 里混装。
4. notebook 建议从项目根目录或 `scripts` 目录打开，避免旧的绝对路径兜底逻辑被触发。
5. `X-AnyLabeling-dev` 当前是 Git 仓库，分享源码时可以保留 `.git`，只发可运行包时也可以按需压缩整个目录。
# SE-Hunter-local
