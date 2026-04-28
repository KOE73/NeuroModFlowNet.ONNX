param(
    [Parameter(Mandatory = $true)]
    [string] $Url,

    [Parameter(Mandatory = $true)]
    [string] $OutputPath,

    [Parameter(Mandatory = $true)]
    [ValidateSet("true", "false")]
    [string] $InjectHead,

    [string] $HeadType = "",

    [string] $ExtraName = "",

    [string] $HeadOutputPath = ""
)

. "$PSScriptRoot/Config.ps1"

$UseInjectHead = $InjectHead -eq "true"
$resolvedModelPath = if ([System.IO.Path]::IsPathRooted($OutputPath)) { $OutputPath } else { Join-Path $ModelsDir $OutputPath }

# Step 1. Resolve the base PaddleOCR model path.
# OutputPath always points to the downloaded clean model, for example det.onnx.
$modelDirectory = Split-Path $resolvedModelPath
if (!(Test-Path $modelDirectory)) {
    New-Item -ItemType Directory -Path $modelDirectory -Force | Out-Null
}

# Step 2. Download the clean PaddleOCR ONNX model only when it is missing.
# Existing models are reused so repeated preparation calls can continue to the next step.
if (Test-Path $resolvedModelPath) {
    Write-Host "Paddle ONNX model exists: $resolvedModelPath" -ForegroundColor Yellow
}
else {
    Write-Host "Downloading Paddle ONNX model from $Url" -ForegroundColor Green
    Invoke-WebRequest -Uri $Url -OutFile $resolvedModelPath -ErrorAction Stop
}

# Step 3. Stop after download/reuse when no preprocessing head is requested.
if (!$UseInjectHead) {
    return
}

# Step 4. Resolve the head type and the output path for the model with embedded preprocessing.
# By default det.onnx becomes det_bytebgr.onnx in the same directory.
if ([string]::IsNullOrWhiteSpace($HeadType)) {
    $HeadType = "ByteBGR_FP32"
}

$resolvedHeadOutputPath = if (![string]::IsNullOrWhiteSpace($HeadOutputPath)) {
    if ([System.IO.Path]::IsPathRooted($HeadOutputPath)) { $HeadOutputPath } else { Join-Path $ModelsDir $HeadOutputPath }
}
else {
    $modelFileName = [System.IO.Path]::GetFileNameWithoutExtension($resolvedModelPath)
    $modelExtension = [System.IO.Path]::GetExtension($resolvedModelPath)
    Join-Path $modelDirectory "$modelFileName`_bytebgr$modelExtension"
}

$headDirectory = Split-Path $resolvedHeadOutputPath
if (!(Test-Path $headDirectory)) {
    New-Item -ItemType Directory -Path $headDirectory -Force | Out-Null
}

if (Test-Path $resolvedHeadOutputPath) {
    Write-Host "Paddle ONNX model with head will be overwritten: $resolvedHeadOutputPath" -ForegroundColor Yellow
}

# Step 5. Build the tool arguments explicitly to keep script behavior easy to inspect.
$injectArgs = @(
    "inject",
    $HeadType,
    $resolvedModelPath,
    "--output",
    $resolvedHeadOutputPath
)

if (![string]::IsNullOrWhiteSpace($ExtraName)) {
    $injectArgs += "--extra"
    $injectArgs += $ExtraName
}

# Step 6. Inject the preprocessing head and fail loudly if the tool did not produce the output.
Write-Host "Injecting $HeadType head into Paddle ONNX model." -ForegroundColor Yellow
& $ToolsExe @injectArgs

if ($LASTEXITCODE -ne 0) {
    throw "Paddle ONNX head injection failed with exit code $LASTEXITCODE."
}

if (!(Test-Path $resolvedHeadOutputPath)) {
    throw "Paddle ONNX head injection did not create output file: $resolvedHeadOutputPath"
}
