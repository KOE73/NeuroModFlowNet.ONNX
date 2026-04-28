# ONNX Model Naming Convention

This document describes the naming convention for exported YOLO-style ONNX models used in the NeuroModFlowNet.ONNX ecosystem. This format is used by `DownloadConvertYolo.ps1` and `DownloadConvertImgTextToObb.ps1` and should be followed by consumers to ensure consistency.

## Format

The general format for exported model paths is:
`models/{BaseName}/{BaseName}__{Suffix}.onnx`

- **BaseName**: The original model filename without extension (e.g., `yolo26n`).
- **Model folder**: Each network stores its source weights and exported variants in `models/{BaseName}`.
- **Delimiter**: A double underscore `__` separates the base name from the suffix.
- **Suffix**: Contains details about the export parameters.

## Suffix Structure

The suffix varies depending on whether the model is static or dynamic.

### Static Models
Format: `{ImgSize}_b{BatchSize}_{Precision}`
- **ImgSize**: Integer size (e.g., `640`).
- **BatchSize**: Integer batch size (e.g., `1`, `4`).
- **Precision**: `fp32` or `fp16`.

Example: `models/yolo26n/yolo26n__640_b1_fp32.onnx`

### Dynamic Models
Format: `{Precision}`
- **Precision**: `fp32` or `fp16`.
- *Note*: Dynamic models imply variable image size, so the `ImgSize` parameter is omitted from both the filename and the export arguments.

Example: `models/yolo11n/yolo11n__fp16.onnx`

## Post-Processing
If the model has been processed (e.g., ByteBGR header injection), an additional suffix is appended:
- **ByteBGR**: `_bytebgr` (appended to the suffix before the extension).

Example: `models/yolo11n/yolo11n__640_b1_fp16_bytebgr.onnx`

## Script Usage

`DownloadConvertYolo.ps1` and `DownloadConvertImgTextToObb.ps1` describe one explicit preparation row:

```powershell
.\ModelPrepare\Preparation\DownloadConvertYolo.ps1 -Models yolo26s -ImgSize 640 -Batches 1,4 -Half false -Dynamic false -ByteBgr true
```

- `-Half false` exports FP32.
- `-Half true` exports FP16.
- `-Dynamic false` uses `-ImgSize` and every value from `-Batches`.
- `-Dynamic true` exports a dynamic-shape model and keeps `-ImgSize` / `-Batches` present only for call-table readability.
- `-ByteBgr true` injects the ByteBGR head after the base ONNX exists.

PaddleOCR uses ready ONNX files and does not follow this naming convention. Use `DownloadConvertPadle.ps1` with explicit `-Url` and `-OutputPath`, and set `-InjectHead true` only for models that need a preprocessing head.
