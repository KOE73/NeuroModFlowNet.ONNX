# Hugging Face Publishing

This folder contains maintainer scripts for publishing prepared ONNX artifacts from the repository-level `models/` directory to a Hugging Face model repository.

The scripts preserve the folder layout from `models/`, but upload only `.onnx` files plus generated metadata. Source `.pt` weights, TensorRT engines, local caches and secrets are not published by these scripts.

## One-time setup

1. Create a Hugging Face account.
2. Create a public model repository, for example:

```text
NeuroModFlowNet/NeuroModFlowNet-ONNX-Demo-Models
```

3. Install the Hugging Face CLI on the maintainer machine:

```powershell
pip install -U huggingface_hub
```

4. Log in:

```powershell
hf auth login
```

Use a token with write access to the model repository. Do not store the token in this Git repository.

## Build local staging

From the repository root:

```powershell
.\ModelPrepare\HuggingFace\Build-OnnxAssetStaging.ps1
```

or:

```cmd
ModelPrepare\HuggingFace\Build-OnnxAssetStaging.cmd
```

This creates:

```text
ModelPrepare/HuggingFace/.publish/
  README.md
  models.manifest.json
  ...
```

The generated `.publish` directory mirrors all `.onnx` files from `models/` with the same relative folder paths. Its `README.md` is copied from `PublishReadme.md`, so the published license text is explicit and versioned.

## Upload to Hugging Face

From the repository root:

```powershell
.\ModelPrepare\HuggingFace\Upload-OnnxAssets.ps1 -RepoId "NeuroModFlowNet/NeuroModFlowNet-ONNX-Demo-Models"
```

or:

```cmd
ModelPrepare\HuggingFace\Upload-OnnxAssets.cmd
```

## Upload only yolo26n

For a first upload check, use:

```cmd
ModelPrepare\HuggingFace\Upload-Yolo26n.cmd
```

This command uses the same staging pipeline as the full upload:

```text
models/yolo26n -> ModelPrepare/HuggingFace/.publish/yolo26n -> Hugging Face yolo26n/
```

The uploaded Hugging Face repository will keep the same folder structure as `models/`:

```text
yolo26n/yolo26n__640_b1_fp16.onnx
yolo26n/yolo26n__640_b1_fp16_bytebgr.onnx
yolo26n-seg/yolo26n-seg__640_b1_fp16.onnx
paddleocr/detection/v3/det.onnx
models.manifest.json
README.md
```

## License note

Prepared model artifacts are separate from the Apache-2.0 source code license of NeuroModFlowNet.ONNX. Each artifact keeps the license obligations of the model it was derived from. For Ultralytics YOLO-derived files, assume the Ultralytics model license applies unless another license is documented.

YOLO model weights and YOLO-derived prepared artifacts are provided under AGPL-3.0, consistent with the Ultralytics YOLO licensing terms.

The author provides these weights "as is", without warranty. Users are responsible for ensuring that their use complies with the licenses of Ultralytics YOLO, the base/pretrained model if any, the training dataset, and any deployment/runtime libraries.
