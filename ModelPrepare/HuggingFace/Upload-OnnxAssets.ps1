param(
    [Parameter(Mandatory = $true)]
    [string] $RepoId,

    [string] $SourceModelsDir = "",

    [string] $StagingDir = "",

    [string] $TargetPathPrefix = "",

    [string] $CommitMessage = "Upload prepared ONNX model assets"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command hf -ErrorAction SilentlyContinue)) {
    throw "Hugging Face CLI 'hf' was not found. Install it with: pip install -U huggingface_hub"
}

$buildScript = Join-Path $PSScriptRoot "Build-OnnxAssetStaging.ps1"

$buildArgs = @{
    SourceModelsDir = $SourceModelsDir
    StagingDir = $StagingDir
    TargetPathPrefix = $TargetPathPrefix
}

& $buildScript @buildArgs

if ([string]::IsNullOrWhiteSpace($StagingDir)) {
    $StagingDir = Join-Path $PSScriptRoot ".publish"
}

$resolvedStagingDir = Resolve-Path $StagingDir
$resolvedStagingPath = $resolvedStagingDir.Path

Write-Host "Uploading prepared ONNX assets to Hugging Face repo '$RepoId'." -ForegroundColor Green
hf upload $RepoId $resolvedStagingPath . --commit-message $CommitMessage
