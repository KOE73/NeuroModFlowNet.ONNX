param(
    [Parameter(Mandatory = $true)]
    [string[]] $Models,

    [Parameter(Mandatory = $true)]
    [int] $ImgSize,

    [Parameter(Mandatory = $true)]
    [int[]] $Batches,

    [Parameter(Mandatory = $true)]
    [ValidateSet("true", "false")]
    [string] $Half,

    [Parameter(Mandatory = $true)]
    [ValidateSet("true", "false")]
    [string] $Dynamic,

    [Parameter(Mandatory = $true)]
    [ValidateSet("true", "false")]
    [string] $ByteBgr,

    [int] $Opset = 18
)

. "$PSScriptRoot/Config.ps1"

$UseHalf = $Half -eq "true"
$UseDynamic = $Dynamic -eq "true"
$UseByteBgr = $ByteBgr -eq "true"

function Get-ModelName {
    param([string] $Model)

    return [System.IO.Path]::GetFileNameWithoutExtension($Model)
}

function Get-SourceModelPath {
    param([string] $ModelName)

    return Join-Path (Join-Path $ModelsDir $ModelName) "$ModelName.pt"
}

function Get-OnnxModelPath {
    param(
        [string] $ModelName,
        [string] $Suffix,
        [bool] $UseByteBgr
    )

    $fileSuffix = if ($UseByteBgr) { "${Suffix}_bytebgr" } else { $Suffix }
    return Join-Path (Join-Path $ModelsDir $ModelName) "${ModelName}__${fileSuffix}.onnx"
}

function Ensure-SourceModel {
    param([string] $ModelName)

    $modelDir = Join-Path $ModelsDir $ModelName
    if (!(Test-Path $modelDir)) {
        New-Item -ItemType Directory -Path $modelDir -Force | Out-Null
    }

    $sourceModelPath = Get-SourceModelPath $ModelName
    if (Test-Path $sourceModelPath) {
        Write-Host "Source model exists: $sourceModelPath" -ForegroundColor Yellow
        return $sourceModelPath
    }

    $url = "${UltralyticsBaseUrl}${UltralyticsVersion}/${ModelName}.pt"
    Write-Host "Downloading YOLO model $ModelName from $url" -ForegroundColor Green
    Invoke-WebRequest -Uri $url -OutFile $sourceModelPath -ErrorAction Stop
    return $sourceModelPath
}

function Convert-YoloModel {
    param(
        [string] $ModelName,
        [string] $SourceModelPath,
        [int] $Batch
    )

    $precision = if ($UseHalf) { "fp16" } else { "fp32" }
    $suffix = if ($UseDynamic) { $precision } else { "${ImgSize}_b${Batch}_${precision}" }
    $onnxPath = Get-OnnxModelPath $ModelName $suffix $false
    $byteBgrPath = Get-OnnxModelPath $ModelName $suffix $true

    if (!(Test-Path $onnxPath)) {
        $exportArgs = @(
            "export",
            "model=`"$SourceModelPath`"",
            "format=onnx",
            "simplify=True",
            "opset=$Opset",
            "device=0"
        )

        if ($UseDynamic) {
            $exportArgs += "dynamic=True"
        }
        else {
            $exportArgs += "imgsz=$ImgSize"
            $exportArgs += "batch=$Batch"
            $exportArgs += "dynamic=False"
        }

        if ($UseHalf) {
            $exportArgs += "half=True"
        }

        Write-Host "`n>>> Exporting YOLO model: $onnxPath" -ForegroundColor Green
        Start-Process "yolo" -ArgumentList ($exportArgs -join " ") -Wait -NoNewWindow

        $defaultOutput = [System.IO.Path]::ChangeExtension($SourceModelPath, ".onnx")
        if (Test-Path $defaultOutput) {
            Move-Item -Path $defaultOutput -Destination $onnxPath -Force
        }
        elseif (Test-Path "best.onnx") {
            Move-Item -Path "best.onnx" -Destination $onnxPath -Force
        }

        if (!(Test-Path $onnxPath)) {
            throw "YOLO export finished but output was not found for '$ModelName'. Expected '$onnxPath'."
        }
    }
    else {
        Write-Host "ONNX model exists: $onnxPath" -ForegroundColor Yellow
    }

    if ($UseByteBgr -and !(Test-Path $byteBgrPath)) {
        $headType = if ($UseHalf) { "ByteBGR_FP16" } else { "ByteBGR_FP32" }
        Write-Host "Injecting $headType header: $byteBgrPath" -ForegroundColor Yellow
        & $ToolsExe inject $headType "$onnxPath" --extra "_bytebgr"
    }
    elseif ($UseByteBgr) {
        Write-Host "ByteBGR model exists: $byteBgrPath" -ForegroundColor Yellow
    }
}

foreach ($model in $Models) {
    $modelName = Get-ModelName $model
    $sourceModelPath = Ensure-SourceModel $modelName

    if ($UseDynamic) {
        Convert-YoloModel $modelName $sourceModelPath 1
        continue
    }

    foreach ($batch in $Batches) {
        Convert-YoloModel $modelName $sourceModelPath $batch
    }
}
