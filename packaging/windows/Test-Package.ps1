[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ExePath,
    [int]$WaitSeconds = 15
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ResolvedExe = (Resolve-Path -LiteralPath $ExePath).Path
$Report = Join-Path ([IO.Path]::GetTempPath()) ("MatrixCalculator-smoke-" + [Guid]::NewGuid().ToString('N') + '.json')

try {
    # 先执行包内矩阵计算与微积分绘图，再检查正常 GUI 启动。
    $Arguments = @('--smoke-test', ('"{0}"' -f $Report))
    $SelfTest = Start-Process -FilePath $ResolvedExe -ArgumentList $Arguments -PassThru -Wait -WindowStyle Hidden
    $ReportContent = if (Test-Path -LiteralPath $Report) { Get-Content -Raw -LiteralPath $Report } else { 'No report was created.' }
    if ($SelfTest.ExitCode -ne 0) {
        throw "Packaged functional smoke test failed with code $($SelfTest.ExitCode): $ReportContent"
    }
    Write-Host "Packaged functional smoke test passed: $ReportContent"

    $Process = Start-Process -FilePath $ResolvedExe -PassThru -WindowStyle Hidden
    Start-Sleep -Seconds $WaitSeconds
    $Process.Refresh()

    if ($Process.HasExited) {
        throw "Packaged app exited during startup with code $($Process.ExitCode)."
    }

    Write-Host "Startup smoke test passed: $ResolvedExe"
    Stop-Process -Id $Process.Id -Force
}
finally {
    if (Test-Path -LiteralPath $Report) {
        Remove-Item -LiteralPath $Report -Force
    }
}
