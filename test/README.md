# 前端演示说明

这个目录是静态 HTML 演示，不需要安装 Node.js。

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `index.html` | 普通静态页面，可直接双击打开 |
| `camera-interact.html` | 摄像头交互页面，依赖 MediaPipe CDN 和摄像头权限 |
| `fix_debug.py` | 一次性调试补丁脚本，正常使用时不用运行 |

## 打开普通页面

直接双击 `index.html`，或用浏览器打开它。

## 打开摄像头交互页面

摄像头权限在部分浏览器里不支持 `file://` 直接访问，建议启动本地服务：

```powershell
cd test
python -m http.server 8000
```

然后访问：

```text
http://localhost:8000/camera-interact.html
```

首次打开时需要允许浏览器摄像头权限，并保持网络可访问 CDN 和 MediaPipe 模型地址。
