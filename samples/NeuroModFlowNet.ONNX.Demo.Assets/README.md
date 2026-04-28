# NeuroModFlowNet.ONNX.Demo.Assets

Shared helper project for demo and lab model assets.

This project does not prepare or export models. It only resolves prepared model files for runtime code:

1. Look for the requested file under the shared repository-level `models/` directory.
2. If the file is missing, download it from the configured public model storage.
3. Save the downloaded file back under `models/` using the same relative path.

The public model storage is:

```text
https://huggingface.co/NeuroModFlowNet/NeuroModFlowNet-ONNX-Demo-Models
```

YOLO-style model paths must follow `ModelPrepare/MODEL_NAMING_CONVENTION.md`.
PaddleOCR and other ready ONNX files may use explicit paths such as `paddleocr/detection/v3/det.onnx`.
