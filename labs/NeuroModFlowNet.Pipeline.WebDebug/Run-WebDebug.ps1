param(
    [Parameter(Mandatory, ParameterSetName='Cameras')][string]$CamerasConfig,
    [Parameter(Mandatory, ParameterSetName='Observation')][string]$ObservationAddress,
    [string]$FFmpegLibrariesPath,
    [string]$NativeRuntimeRoot,
    [string]$ModelsRoot,
    [ValidateSet('nvdec','cpu')][string]$Decoder = 'nvdec',
    [ValidateRange(1024,65535)][int]$Port = 5264,
    [string]$ListenHost = '127.0.0.1',
    [string]$InstallRoot = 'C:\LocalProject\NeuroModFlowNet.WebDebug'
)
$ErrorActionPreference = 'Stop'
if($Port -eq 443) { throw 'Port 443 is prohibited.' }
if($CamerasConfig -and !(Test-Path -LiteralPath $CamerasConfig -PathType Leaf)) { throw 'CamerasConfig does not exist.' }
if($CamerasConfig -and $Decoder -eq 'nvdec' -and (!$FFmpegLibrariesPath -or !(Test-Path -LiteralPath $FFmpegLibrariesPath -PathType Container))) {
    throw 'Decoder nvdec requires FFmpegLibrariesPath.'
}
if(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) { throw "Port $Port is already occupied. No existing service will be stopped." }
if($ListenHost -notin @('127.0.0.1','localhost','[::1]') -and !$env:WebDebug__AccessToken) { throw 'Set WebDebug__AccessToken before binding to a remote interface.' }
$project = Join-Path $PSScriptRoot 'NeuroModFlowNet.Pipeline.WebDebug.csproj'
$buildArguments = @('build', $project, '--configuration', 'Debug', '--nologo', '--verbosity', 'quiet')
if($NativeRuntimeRoot) { $buildArguments += "-p:WebDebugNativeRoot=$NativeRuntimeRoot" }
& dotnet @buildArguments
if($LASTEXITCODE -ne 0) { throw 'Debugger build failed.' }
$installedApplication = Join-Path $InstallRoot $(if($ObservationAddress) { 'App-Observation' } else { 'App' })
New-Item -ItemType Directory -Force -Path $installedApplication | Out-Null
Copy-Item -Path (Join-Path $PSScriptRoot 'bin\Debug\net10.0\*') -Destination $installedApplication -Recurse -Force
if($ModelsRoot) { $env:NEUROMODFLOWNET_ONNX_MODELS = (Resolve-Path -LiteralPath $ModelsRoot).Path }
$url = "http://${ListenHost}:$Port"
Write-Host "VM Debug: $url"
$applicationArguments = @('--contentRoot', $PSScriptRoot, '--urls', $url)
if($ObservationAddress) {
    $applicationArguments += @('--WebDebug:ObservationAddress', $ObservationAddress)
} else {
    $applicationArguments += @('--WebDebug:CamerasConfig', (Resolve-Path -LiteralPath $CamerasConfig).Path, '--WebDebug:Decoder', $Decoder)
}
if($FFmpegLibrariesPath) { $applicationArguments += @('--WebDebug:FFmpegLibrariesPath', $FFmpegLibrariesPath) }
& dotnet (Join-Path $installedApplication 'NeuroModFlowNet.Pipeline.WebDebug.dll') @applicationArguments
exit $LASTEXITCODE
