"""Matrix Calculator 的 Windows 独立桌面打包配置。"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files


PROJECT_ROOT = Path(SPECPATH).resolve().parents[1]

# Matplotlib 的字体、样式与后端资源需要随应用一起分发。
datas = collect_data_files("matplotlib", include_py_files=False)

a = Analysis(
    [str(PROJECT_ROOT / "packaging" / "windows" / "entrypoint.py")],
    pathex=[str(PROJECT_ROOT / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["streamlit", "jupyter", "notebook", "pytest"],
    noarchive=False,
    optimize=0,
)
# Qt for Windows 使用系统 ICU。若构建终端的 PATH 中有其他 ICU（例如
# icuuc.dll + icudt78.dll），PyInstaller 可能将其误收进 _internal 并优先加载；
# 那类 ICU 只导出版本化符号，会让 QtCore 在启动时报 WinError 127。
def _uses_windows_system_icu(binary):
    destination = Path(binary[0]).name.lower()
    is_icu_data = destination == "icudt.dll" or (
        destination.startswith("icudt")
        and destination.endswith(".dll")
        and destination[5:-4].isdigit()
    )
    return destination not in {"icuuc.dll", "icuin.dll"} and not is_icu_data


a.binaries = [binary for binary in a.binaries if _uses_windows_system_icu(binary)]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MatrixCalculator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version=str(PROJECT_ROOT / "packaging" / "windows" / "version_info.txt"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="MatrixCalculator",
)
