$ErrorActionPreference = 'Stop'
$rayCli = Join-Path $PSScriptRoot '.venv\Scripts\ray.exe'
$env:RAY_USAGE_STATS_ENABLED = '0'
& (Join-Path $PSScriptRoot '.venv\Scripts\python.exe') (Join-Path $PSScriptRoot 'ray_local.py')
if ($LASTEXITCODE -ne 0) { throw 'Ray authentication setup failed' }
if (-not $env:RAY_AUTH_TOKEN -and -not $env:RAY_AUTH_TOKEN_PATH) {
    $env:RAY_AUTH_TOKEN_PATH = Join-Path $PSScriptRoot '.ray\auth_token'
}
& $rayCli start --head --node-ip-address=127.0.0.1 --dashboard-host=127.0.0.1 --dashboard-port=8265 --port=6379 --num-cpus=4 --disable-usage-stats --temp-dir (Join-Path $PSScriptRoot '.tmp\ray')
if ($LASTEXITCODE -ne 0) { throw "Ray startup failed with exit code $LASTEXITCODE" }
