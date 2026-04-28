@echo off
pwsh -ExecutionPolicy Bypass -File "%~dp0Upload-OnnxAssets.ps1" -RepoId "NeuroModFlowNet/NeuroModFlowNet-ONNX-Demo-Models" -SourceModelsDir "%~dp0..\..\models\yolo26n" -TargetPathPrefix "yolo26n" -CommitMessage "Upload yolo26n prepared ONNX assets"
