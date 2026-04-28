param(
    [string] $SourceModelsDir = "",

    [string] $StagingDir = "",

    [string] $TargetPathPrefix = "",

    [string] $ManifestFileName = "models.manifest.json",

    [string] $PublishReadmePath = ""
)

$ErrorActionPreference = "Stop"

$repositoryRoot = Resolve-Path (Join-Path $PSScriptRoot "../..")

if ([string]::IsNullOrWhiteSpace($SourceModelsDir)) {
    $SourceModelsDir = Join-Path $repositoryRoot "models"
}

if ([string]::IsNullOrWhiteSpace($StagingDir)) {
    $StagingDir = Join-Path $PSScriptRoot ".publish"
}

$resolvedSourceModelsDir = Resolve-Path $SourceModelsDir
$resolvedStagingParent = Resolve-Path $PSScriptRoot
$resolvedStagingDir = [System.IO.Path]::GetFullPath($StagingDir)

if (!$resolvedStagingDir.StartsWith($resolvedStagingParent.Path, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "StagingDir must be inside '$($resolvedStagingParent.Path)'. Actual: '$resolvedStagingDir'."
}

if (Test-Path $resolvedStagingDir) {
    Get-ChildItem -LiteralPath $resolvedStagingDir -Force | Remove-Item -Recurse -Force
}

New-Item -ItemType Directory -Path $resolvedStagingDir -Force | Out-Null

$onnxFiles = Get-ChildItem -Path $resolvedSourceModelsDir -Filter "*.onnx" -File -Recurse |
    Sort-Object FullName

if ($onnxFiles.Count -eq 0) {
    throw "No ONNX files found in '$($resolvedSourceModelsDir.Path)'."
}

$assets = foreach ($onnxFile in $onnxFiles) {
    $relativePath = [System.IO.Path]::GetRelativePath($resolvedSourceModelsDir.Path, $onnxFile.FullName)
    $publishPath = if ([string]::IsNullOrWhiteSpace($TargetPathPrefix)) {
        $relativePath
    }
    else {
        Join-Path $TargetPathPrefix $relativePath
    }

    $targetPath = Join-Path $resolvedStagingDir $publishPath
    $targetDirectory = Split-Path $targetPath

    if (!(Test-Path $targetDirectory)) {
        New-Item -ItemType Directory -Path $targetDirectory -Force | Out-Null
    }

    Copy-Item -LiteralPath $onnxFile.FullName -Destination $targetPath -Force

    $hash = Get-FileHash -LiteralPath $onnxFile.FullName -Algorithm SHA256

    [ordered]@{
        path = $publishPath.Replace("\", "/")
        fileName = $onnxFile.Name
        sizeBytes = $onnxFile.Length
        sha256 = $hash.Hash.ToLowerInvariant()
    }
}

$manifest = [ordered]@{
    generatedAtUtc = [System.DateTimeOffset]::UtcNow.ToString("O")
    sourceRoot = "models"
    artifactKind = "prepared-onnx"
    licenseNote = "Prepared model artifacts are not licensed under the repository Apache-2.0 license. Each artifact keeps the license obligations of the source model it was derived from."
    assets = @($assets)
}

$manifestPath = Join-Path $resolvedStagingDir $ManifestFileName
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

$readmePath = Join-Path $resolvedStagingDir "README.md"
if ([string]::IsNullOrWhiteSpace($PublishReadmePath)) {
    $PublishReadmePath = Join-Path $PSScriptRoot "PublishReadme.md"
}

if (!(Test-Path $PublishReadmePath)) {
    throw "Publish README template was not found: $PublishReadmePath"
}

Copy-Item -LiteralPath $PublishReadmePath -Destination $readmePath -Force

Write-Host "Prepared Hugging Face staging directory:" -ForegroundColor Green
Write-Host "  $resolvedStagingDir"
Write-Host "ONNX files:" -ForegroundColor Green
Write-Host "  $($onnxFiles.Count)"
Write-Host "Manifest:" -ForegroundColor Green
Write-Host "  $manifestPath"
