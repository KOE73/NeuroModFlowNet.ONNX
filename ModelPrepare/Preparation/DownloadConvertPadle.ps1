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
$resolvedOutputPath = if ([System.IO.Path]::IsPathRooted($OutputPath)) { $OutputPath } else { Join-Path $ModelsDir $OutputPath }

$parentDir = Split-Path $resolvedOutputPath
if (!(Test-Path $parentDir)) {
    New-Item -ItemType Directory -Path $parentDir -Force | Out-Null
}

if (Test-Path $resolvedOutputPath) {
    Write-Host "Paddle ONNX model exists: $resolvedOutputPath" -ForegroundColor Yellow
}
else {
    Write-Host "Downloading Paddle ONNX model from $Url" -ForegroundColor Green
    Invoke-WebRequest -Uri $Url -OutFile $resolvedOutputPath -ErrorAction Stop
}

if (!$UseInjectHead) {
    return
}

if ([string]::IsNullOrWhiteSpace($HeadType)) {
    throw "HeadType is required when InjectHead is true."
}

$injectArgs = @(
    "inject",
    $HeadType,
    $resolvedOutputPath
)

if (![string]::IsNullOrWhiteSpace($HeadOutputPath)) {
    $resolvedHeadOutputPath = if ([System.IO.Path]::IsPathRooted($HeadOutputPath)) { $HeadOutputPath } else { Join-Path $ModelsDir $HeadOutputPath }
    $headParentDir = Split-Path $resolvedHeadOutputPath
    if (!(Test-Path $headParentDir)) {
        New-Item -ItemType Directory -Path $headParentDir -Force | Out-Null
    }

    if (Test-Path $resolvedHeadOutputPath) {
        Write-Host "Paddle ONNX model with head exists: $resolvedHeadOutputPath" -ForegroundColor Yellow
        return
    }

    $injectArgs += "--output"
    $injectArgs += $resolvedHeadOutputPath
}
elseif (![string]::IsNullOrWhiteSpace($ExtraName)) {
    $injectArgs += "--extra"
    $injectArgs += $ExtraName
}

Write-Host "Injecting $HeadType head into Paddle ONNX model." -ForegroundColor Yellow
& $ToolsExe @injectArgs
