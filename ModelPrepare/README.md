# Model Prepare

This directory contains model asset maintenance tooling. The repository keeps scripts and documentation here, but generated model files stay outside Git.

## Directory layout

- `Preparation/` - scripts that download source model weights and export prepared ONNX variants into `models/`.
- `HuggingFace/` - scripts that publish already prepared ONNX files from `models/` to a Hugging Face model repository.

## Normal flow

1. Run preparation scripts only on a maintainer machine that has Python, Ultralytics and the ONNX tooling installed.
2. Check the generated files under `models/`.
3. Publish prepared `.onnx` artifacts through `HuggingFace/Upload-OnnxAssets.ps1`.
4. Keep `.pt`, `.onnx`, TensorRT engines and generated staging folders out of Git.

The public repository should contain reproducible scripts and manifests, not the heavy model artifacts themselves.

