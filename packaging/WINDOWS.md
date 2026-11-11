# Windows 独立桌面版

本目录提供 Windows x64 独立桌面应用的构建、启动冒烟测试与 GitHub Releases 发布配置。打包结果只包含 PySide6 桌面版，不包含 Streamlit 网页服务。

## 用户使用

1. 打开项目的 GitHub Releases 页面。
2. 下载 `MatrixCalculator-Windows-x64.zip`。
3. 将 ZIP **完整解压**。
4. 进入 `MatrixCalculator` 文件夹，双击 `MatrixCalculator.exe`。

无需安装 Python、Git、PySide6、NumPy 或 Matplotlib。

**注意：**exe 依赖同目录下的 `_internal` 等内容，不能只复制 exe 文件。当前未配置商业代码签名，Windows SmartScreen 可能提示“未知发布者”；请只从项目的 GitHub Releases 或可信构建产物下载。

## 本地构建

要求：Windows 10/11 x64、PowerShell，以及用于构建的 Python 3.10+。最终用户不需要这些构建工具。

在项目根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\packaging\windows\build.ps1
```

脚本会在被忽略的 `.local/packaging-venv` 中创建独立构建环境，不影响日常开发环境，然后依次：

1. 安装桌面依赖与 PyInstaller。
2. 生成 `dist/MatrixCalculator/MatrixCalculator.exe`。
3. 在独立包内实际执行矩阵行列式计算和微积分绘图自检。
4. 正常启动 exe 约 15 秒，确认没有立即退出。
5. 生成 `dist/MatrixCalculator-Windows-x64.zip`。

后续重复构建可使用 `-SkipInstall` 跳过依赖安装；调试时可使用 `-SkipSmokeTest` 跳过启动检查：

```powershell
.\packaging\windows\build.ps1 -SkipInstall
```

## GitHub 自动构建与发布

`.github/workflows/package-windows.yml` 会在以下情况构建 Windows 包：

- 与本项目源码或打包配置相关的 Pull Request。
- 手动运行 `Windows Package` workflow。
- 推送 `v*` 版本标签。

普通构建会在 Actions 页面保留 ZIP 构件 14 天。推送版本标签时，构建成功后还会创建或更新同名 GitHub Release，并上传 ZIP。

首次发布前，先合并打包配置，再在最新 `master` 上创建与 `pyproject.toml`、`packaging/windows/version_info.txt` 一致的版本标签，例如：

```powershell
git tag v1.2.0
git push origin v1.2.0
```

发布流程仅上传 Windows x64 ZIP，不会把 `dist/`、`build/` 或 exe 提交进源码仓库。

## 验证边界

自动验证会在打包后的 exe 内实际执行一次矩阵行列式计算和微积分绘图，并检查正常启动，能发现缺少依赖、Qt 插件、错误 DLL 或打包入口问题。本地发布前还会从 ZIP 重新解压验证，避免只在构建目录中可用。

这些检查不等于完整的界面人工验收。正式发布前仍应在一台未安装项目开发环境的 Windows 机器上检查：矩阵计算、微积分图示、结果导出以及关闭窗口。当前验证环境为 Windows x64；Windows 10/11 使用系统 ICU，不支持未提供该系统组件的旧版 Windows。
