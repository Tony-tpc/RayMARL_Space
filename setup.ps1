param(
    [string]$BasePython = 'C:\Users\ykq31\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
)
$ErrorActionPreference = 'Stop'
$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    & $BasePython -m venv (Join-Path $PSScriptRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create venv' }
}
$taskTemp = Join-Path $PSScriptRoot '.tmp'
New-Item -ItemType Directory -Path $taskTemp -Force | Out-Null
$oldTmp = $env:TMP
$oldTemp = $env:TEMP
try {
    $env:TMP = $taskTemp
    $env:TEMP = $taskTemp
    & $venvPython -m pip install --disable-pip-version-check -r (Join-Path $PSScriptRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
    & $venvPython -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency validation failed' }
} finally {
    $env:TMP = $oldTmp
    $env:TEMP = $oldTemp
}
