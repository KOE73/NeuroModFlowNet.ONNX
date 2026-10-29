param([string]$RuntimeRoot = 'C:\LocalProject\NeuroModFlowNet.WebDebug\Runtime')
$ErrorActionPreference = 'Stop'
# Explicit same-version build selection; never invoked as error recovery or a provider fallback.
$archiveName = 'onnxruntime-win-x64-gpu_cuda12-1.30.0.zip'
$expectedSha256 = 'd4667ea48eb0a10bc9b96b838f7b8975a6bf18de3bc5edd403a22e15c1458b23'
$url = "https://github.com/microsoft/onnxruntime/releases/download/v1.30.0/$archiveName"
New-Item -ItemType Directory -Force -Path $RuntimeRoot | Out-Null
$archivePath = Join-Path $RuntimeRoot $archiveName
if(!(Test-Path -LiteralPath $archivePath)) { Invoke-WebRequest -Uri $url -OutFile $archivePath }
if((Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedSha256) { throw 'ONNX Runtime archive SHA-256 mismatch.' }
Expand-Archive -LiteralPath $archivePath -DestinationPath $RuntimeRoot -Force
Join-Path $RuntimeRoot 'onnxruntime-win-x64-gpu_cuda12-1.30.0\lib'
