[Diagnostics.CodeAnalysis.SuppressMessageAttribute("PSUseDeclaredVarsMoreThanAssignments", "")]
param()

$UltralyticsBaseUrl = "https://github.com/ultralytics/assets/releases/download/"
$UltralyticsVersion = "v8.4.0"
$GitAssetsBaseUrl = "https://github.com/KOE73/NeuroModFlowNet.ONNX/releases/download/v1.0.0/"

$RepositoryRoot = Resolve-Path (Join-Path $PSScriptRoot "../..")
$ModelsDir = Join-Path $RepositoryRoot "models"

$ToolsDir = Join-Path $RepositoryRoot "tools\NeuroModFlowNet.ONNX.Tools\bin\SingleExe"
$ToolsExe = Join-Path $ToolsDir "NeuroModFlowNet.ONNX.Tools.exe"

$DefaultYoloModels = @(
    "yolo26s",
    "yolo26s-obb",
    "yolo26s-seg",
    "yolo26s-pose",
    "yolo26s-cls"
)
