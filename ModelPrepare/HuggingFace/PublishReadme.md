---
license: other
library_name: onnx
tags:
- onnx
- computer-vision
- yolo
- paddleocr
- ocr
- text-detection
- object-detection
- image-segmentation
- image-classification
- pose-estimation
- oriented-object-detection
- neuromodflownet
---

# NeuroModFlowNet.ONNX Prepared Models

Prepared ONNX model artifacts for NeuroModFlowNet.ONNX.

These files are generated artifacts. They are not covered by the Apache-2.0 license of the NeuroModFlowNet.ONNX source code repository unless explicitly stated.

See `models.manifest.json` for file paths, sizes and SHA256 hashes.

## Model Licenses

This repository contains prepared artifacts from different model families. The Hugging Face metadata uses `license: other` because a single repository-level license value would be misleading.

### YOLO-Derived Models

YOLO weights and YOLO-derived prepared artifacts are provided under AGPL-3.0, consistent with the Ultralytics YOLO licensing terms.

This applies to YOLO-family folders such as:

- `yolo26n/`;
- `yolo26n-cls/`;
- `yolo26n-obb/`;
- `yolo26n-pose/`;
- `yolo26n-seg/`;
- other artifacts exported from or derived from Ultralytics YOLO models.

Users are responsible for ensuring that their YOLO model use complies with the licenses of:

- Ultralytics YOLO;
- the base/pretrained model, if any;
- the training dataset;
- any deployment/runtime libraries.

### PaddleOCR Models

PaddleOCR / PP-OCR artifacts are provided under Apache-2.0, consistent with the PaddleOCR project license.

The PaddleOCR artifacts in this repository are prepared ONNX files for NeuroModFlowNet.ONNX demos. The clean detection models are sourced from the ONNX exports in `monkt/paddleocr-onnx`, which describe the original models as PaddleOCR / PP-OCR models from the PaddlePaddle team and mark the ONNX exports as Apache-2.0.

Current PaddleOCR paths include:

- `paddleocr/detection/v3/det.onnx`;
- `paddleocr/detection/v3/det_bytebgr.onnx`;
- `paddleocr/detection/v5/det.onnx`;
- `paddleocr/detection/v5/det_bytebgr.onnx`.

Source references:

- PaddleOCR project: https://github.com/PaddlePaddle/PaddleOCR
- PaddleOCR / PP-OCR ONNX source used by the preparation scripts: https://huggingface.co/monkt/paddleocr-onnx

### Prepared Variants

Files with embedded preprocessing heads, for example `*_bytebgr.onnx`, are modified variants prepared for lower-overhead NeuroModFlowNet.ONNX input pipelines. They keep the license obligations of the source model family they were derived from.

## Warranty

The author provides these artifacts "as is", without warranty. Users are responsible for checking that their use complies with all applicable model, dataset, export, and deployment/runtime licenses.
