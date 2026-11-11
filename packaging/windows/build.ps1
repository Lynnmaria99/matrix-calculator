[CmdletBinding()]
param(
    [string]$Python = 'python',
    [switch]$SkipInstall,
    [switch]$SkipSmokeTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
$Venv = Join-Path $Root '.local/packaging-venv'
$VenvPython = Join-Path $Venv 'Scripts/python.exe'
$Dist = Join-Path $Root 'dist'
$Build = Join-Path $Root 'build'
$Spec = Join-Path $Root 'packaging/windows/MatrixCalculator.spec'

Push-Location $Root
try {
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        & $Python -m venv $Venv
        if ($LASTEXITCODE -ne 0) { throw 'Failed to create the packaging environment.' }
    }

    if (-not $SkipInstall) {
        & $VenvPython -m pip install --upgrade pip
        if ($LASTEXITCODE -ne 0) { throw 'Failed to upgrade pip.' }
        & $VenvPython -m pip install --upgrade 'pyinstaller>=6.0'
        if ($LASTEXITCODE -ne 0) { throw 'Failed to install PyInstaller.' }
        & $VenvPython -m pip install -e '.[desktop]'
        if ($LASTEXITCODE -ne 0) { throw 'Failed to install desktop dependencies.' }
    }

    # 避免当前终端里其他工具的 DLL 被 PyInstaller 误收集（例如另一套 ICU）。
    # Python 包自带的运行库仍会通过 PyInstaller 的依赖分析正常收集。
    $OriginalPath = $env:PATH
    try {
        $env:PATH = @(
            (Join-Path $Venv 'Scripts'),
            (Join-Path $env:SystemRoot 'System32'),
            $env:SystemRoot,
            (Join-Path $env:SystemRoot 'System32/Wbem'),
            (Join-Path $env:SystemRoot 'System32/WindowsPowerShell/v1.0')
        ) -join [IO.Path]::PathSeparator
        & $VenvPython -m PyInstaller --noconfirm --clean --distpath $Dist --workpath $Build $Spec
        if ($LASTEXITCODE -ne 0) { throw 'PyInstaller build failed.' }
    }
    finally {
        $env:PATH = $OriginalPath
    }

    $Exe = Join-Path $Dist 'MatrixCalculator/MatrixCalculator.exe'
    if (-not $SkipSmokeTest) {
        & (Join-Path $Root 'packaging/windows/Test-Package.ps1') -ExePath $Exe
    }

    $Zip = Join-Path $Dist 'MatrixCalculator-Windows-x64.zip'
    Compress-Archive -LiteralPath (Join-Path $Dist 'MatrixCalculator') -DestinationPath $Zip -Force
    Write-Host "Package created: $Zip"
}
finally {
    Pop-Location
}
