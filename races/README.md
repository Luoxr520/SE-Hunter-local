# races C++ 示例说明

这是一个 C++/CMake 最小示例项目。现有文件没有移动或删除。

## 环境要求

- CMake 3.30 及以上
- 支持 C++20 的编译器
- Windows 上可使用 Visual Studio Build Tools、MinGW 或 CLion 自带工具链

## 构建和运行

```powershell
cd races
cmake -S . -B build
cmake --build build
.\build\Debug\races.exe
```

如果使用的是单配置生成器，运行文件可能在：

```powershell
.\build\races.exe
```

`1.exe` 是已有的历史编译产物，源码入口以 `main.cpp` 和 `CMakeLists.txt` 为准。
