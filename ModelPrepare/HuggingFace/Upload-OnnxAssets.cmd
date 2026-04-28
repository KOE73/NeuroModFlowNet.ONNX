@echo off
pwsh -ExecutionPolicy Bypass -File "%~dp0Upload-OnnxAssets.ps1" -RepoId "NeuroModFlowNet/NeuroModFlowNet-ONNX-Demo-Models" %*
