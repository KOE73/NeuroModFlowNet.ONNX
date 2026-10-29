param(
    [Parameter(Mandatory)][string]$CamerasConfig,
    [Parameter(Mandatory)][string]$FFmpegLibrariesPath,
    [string]$NativeRuntimeRoot,
    [string]$ModelsRoot,
    [ValidateSet('nvdec','cpu')][string]$Decoder = 'nvdec',
    [ValidateRange(1024,65535)][int]$Port = 5264,
    [string]$ListenHost = '127.0.0.1',
    [string]$InstallRoot = 'C:\LocalProject\NeuroModFlowNet.WebDebug'
)
$ErrorActionPreference = 'Stop'
if($Port -eq 443) { throw 'Port 443 is prohibited.' }
if(!(Test-Path -LiteralPath $CamerasConfig -PathType Leaf)) { throw 'CamerasConfig does not exist.' }
if(!(Test-Path -LiteralPath $FFmpegLibrariesPath -PathType Container)) { throw 'FFmpegLibrariesPath does not exist.' }
if(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) { throw "Port $Port is already occupied. No existing service will be stopped." }
if($ListenHost -notin @('127.0.0.1','localhost','[::1]') -and !$env:WebDebug__AccessToken) { throw 'Set WebDebug__AccessToken before binding to a remote interface.' }
$project = Join-Path $PSScriptRoot 'NeuroModFlowNet.Pipeline.WebDebug.csproj'
$buildArguments = @('build', $project, '--configuration', 'Debug', '--nologo', '--verbosity', 'quiet')
if($NativeRuntimeRoot) { $buildArguments += "-p:WebDebugNativeRoot=$NativeRuntimeRoot" }
& dotnet @buildArguments
if($LASTEXITCODE -ne 0) { throw 'Debugger build failed.' }
$installedApplication = Join-Path $InstallRoot 'App'
New-Item -ItemType Directory -Force -Path $installedApplication | Out-Null
Copy-Item -Path (Join-Path $PSScriptRoot 'bin\Debug\net10.0\*') -Destination $installedApplication -Recurse -Force
if($ModelsRoot) { $env:NEUROMODFLOWNET_ONNX_MODELS = (Resolve-Path -LiteralPath $ModelsRoot).Path }
$url = "http://${ListenHost}:$Port"
Write-Host "VM Debug: $url"
& dotnet (Join-Path $installedApplication 'NeuroModFlowNet.Pipeline.WebDebug.dll') --contentRoot $PSScriptRoot --urls $url --WebDebug:CamerasConfig (Resolve-Path -LiteralPath $CamerasConfig).Path --WebDebug:Decoder $Decoder --WebDebug:FFmpegLibrariesPath $FFmpegLibrariesPath
exit $LASTEXITCODE
