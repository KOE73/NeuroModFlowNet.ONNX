# Model Preparation Scripts

These scripts prepare local model artifacts under the repository-level `models/` directory.

## Contents

- `DownloadConvertAll.ps1` - full preparation call table for the currently prepared model set.
- `DownloadConvertYolo.ps1` - downloads Ultralytics YOLO weights and exports ONNX variants.
- `DownloadConvertImgTextToObb.ps1` - downloads project-specific OBB weights and exports ONNX variants.
- `DownloadConvertPadle.ps1` - downloads ready PaddleOCR ONNX models.
- `Config.ps1` - shared paths and source URLs.
- `../MODEL_NAMING_CONVENTION.md` - model artifact naming rules.

## Usage

From the repository root:

```powershell
.\ModelPrepare\Preparation\DownloadConvertAll.ps1
```

or:

```cmd
ModelPrepare\Preparation\DownloadConvertAll.cmd
```

The scripts expect the required Python and model-export tooling to be available on the maintainer machine. They are not required for normal library usage when prepared ONNX artifacts are downloaded from asset storage.
