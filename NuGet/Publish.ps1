param(
    [string]$Version,
    [string]$Source = "https://api.nuget.org/v3/index.json",
    [switch]$NoBuild,
    [switch]$Yes,
    [switch]$AllowUncommitted
)

$ErrorActionPreference = "Stop"

function Format-FileSize
{
    param([long]$Bytes)

    if($Bytes -ge 1GB)
    {
        return "{0:N2} GB" -f ($Bytes / 1GB)
    }

    if($Bytes -ge 1MB)
    {
        return "{0:N2} MB" -f ($Bytes / 1MB)
    }

    if($Bytes -ge 1KB)
    {
        return "{0:N2} KB" -f ($Bytes / 1KB)
    }

    return "$Bytes B"
}

function Get-PackageId
{
    param(
        [string]$FileName,
        [string]$PackageVersion
    )

    return $FileName `
        -replace "\.$([regex]::Escape($PackageVersion))\.s?nupkg$", ""
}

function Invoke-Git
{
    param([string[]]$Arguments)

    $output = & git -C $repoRoot @Arguments 2>&1
    if($LASTEXITCODE -ne 0)
    {
        throw "Git command failed: git -C `"$repoRoot`" $($Arguments -join ' ')`n$output"
    }

    return $output
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptRoot
$propsPath = Join-Path $repoRoot "Directory.Build.props"
$srcRoot = Join-Path $repoRoot "src"
$apiKeyPath = Join-Path $scriptRoot "nuget-api-key.local.txt"
$srcProjects = @(Get-ChildItem -LiteralPath $srcRoot -Recurse -File -Filter "*.csproj" | Sort-Object FullName)
$srcPackageIds = @($srcProjects | ForEach-Object { [IO.Path]::GetFileNameWithoutExtension($_.Name) })

if($srcProjects.Count -eq 0)
{
    throw "No .csproj files found in source folder: $srcRoot"
}

$gitBranch = (Invoke-Git @("branch", "--show-current")).Trim()
$gitCommit = (Invoke-Git @("rev-parse", "HEAD")).Trim()
$gitShortCommit = (Invoke-Git @("rev-parse", "--short=12", "HEAD")).Trim()
$gitStatus = @(Invoke-Git @("status", "--porcelain"))
$gitHasUncommittedChanges = $gitStatus.Count -gt 0

if($gitHasUncommittedChanges -and -not $AllowUncommitted)
{
    Write-Host ""
    Write-Host "Git working tree is not clean." -ForegroundColor Red
    Write-Host "NuGet symbols and SourceLink should point to committed source code."
    Write-Host "Commit or stash changes before publishing, otherwise debugger source lookup can mismatch the package."
    Write-Host ""
    $gitStatus | Select-Object -First 30 | ForEach-Object { Write-Host $_ }
    if($gitStatus.Count -gt 30)
    {
        Write-Host "... and $($gitStatus.Count - 30) more entries"
    }
    Write-Host ""
    Write-Host "Use -AllowUncommitted only if you intentionally publish from the current uncommitted tree." -ForegroundColor Yellow
    exit 1
}

if([string]::IsNullOrWhiteSpace($Version))
{
    [xml]$props = Get-Content -LiteralPath $propsPath
    $Version = @($props.Project.PropertyGroup.Version | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })[0]
}

if([string]::IsNullOrWhiteSpace($Version))
{
    throw "Package version was not provided and was not found in Directory.Build.props."
}

if(-not (Test-Path -LiteralPath $apiKeyPath))
{
    throw "NuGet API key file was not found: $apiKeyPath"
}

$apiKey = (Get-Content -LiteralPath $apiKeyPath -Raw).Trim()
if([string]::IsNullOrWhiteSpace($apiKey))
{
    throw "NuGet API key file is empty: $apiKeyPath"
}

if(-not $NoBuild)
{
    foreach($project in $srcProjects)
    {
        dotnet build $project.FullName -c Release
        if($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}

$publishPath = Join-Path $scriptRoot "Publish\$Version"
if(-not (Test-Path -LiteralPath $publishPath))
{
    throw "NuGet publish folder was not found: $publishPath"
}

$packages = Get-ChildItem -LiteralPath $publishPath -File |
    Where-Object { $_.Extension -eq ".nupkg" -and (Get-PackageId -FileName $_.Name -PackageVersion $Version) -in $srcPackageIds }
$symbols = Get-ChildItem -LiteralPath $publishPath -File |
    Where-Object { $_.Extension -eq ".snupkg" -and (Get-PackageId -FileName $_.Name -PackageVersion $Version) -in $srcPackageIds }

if($packages.Count -eq 0)
{
    throw "No .nupkg files found in $publishPath"
}

$publishFiles = @($packages) + @($symbols)
$publishPlan = $publishFiles |
    Sort-Object Name |
    ForEach-Object {
        [pscustomobject]@{
            Kind = if($_.Extension -eq ".snupkg") { "symbols" } else { "package" }
            PackageId = Get-PackageId -FileName $_.Name -PackageVersion $Version
            Version = $Version
            Size = Format-FileSize -Bytes $_.Length
            Updated = $_.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
            Sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.Substring(0, 16)
            File = $_.Name
        }
    }

Write-Host ""
Write-Host "NuGet publish preview" -ForegroundColor Cyan
Write-Host "Source:       $Source"
Write-Host "Version:      $Version"
Write-Host "Folder:       $publishPath"
Write-Host "Scope:        src ($($srcProjects.Count) projects)"
Write-Host "Git branch:   $gitBranch"
Write-Host "Git commit:   $gitShortCommit ($gitCommit)"
Write-Host "Git status:   $(if($gitHasUncommittedChanges) { "uncommitted changes allowed by -AllowUncommitted" } else { "clean" })"
Write-Host "Packages:     $($packages.Count) .nupkg"
Write-Host "Symbols:      $($symbols.Count) .snupkg"
Write-Host "Total size:   $(Format-FileSize -Bytes (($publishFiles | Measure-Object -Property Length -Sum).Sum))"
Write-Host ""
Write-Host ($publishPlan | Format-Table -AutoSize | Out-String -Width 240)
Write-Host "SourceLink reminder: publish only after the commit above is pushed to GitHub; .snupkg/PDB source links rely on that exact commit." -ForegroundColor Yellow
Write-Host ""

if(-not $Yes)
{
    $answer = Read-Host "Publish these files? Type Y to continue"
    if($answer -notin @("Y", "y"))
    {
        Write-Host "Publish cancelled."
        exit 0
    }
}

foreach($package in $packages)
{
    dotnet nuget push $package.FullName --api-key $apiKey --source $Source --skip-duplicate
    if($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

foreach($symbolPackage in $symbols)
{
    dotnet nuget push $symbolPackage.FullName --api-key $apiKey --source $Source --skip-duplicate
    if($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
