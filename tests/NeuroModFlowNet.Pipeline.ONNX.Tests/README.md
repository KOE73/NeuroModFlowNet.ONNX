# NeuroModFlowNet.Pipeline.ONNX.Tests

## Purpose

This project tests operations from `NeuroModFlowNet.Pipeline.ONNX` and related base operations from
`NeuroModFlowNet.Pipeline`.

The main rule is that each operation keeps its operation-specific test infrastructure next to the operation test. If an
operation needs diagnostic images, expected metadata, seeded random cases, or visual artifacts, those files belong in
the operation folder rather than in a shared bucket.

## Structure

```text
tests/NeuroModFlowNet.Pipeline.ONNX.Tests/
  Base/
    ...
  Onnx/
    Operators/
      Op_Onnx_Crop/
        Op_Onnx_CropTests.cs
        OpOnnxCropPattern.cs
        OpOnnxCropPatternCase.cs
  Common/
    ...
  appsettings.test.json
```

`Base/` contains tests for `NeuroModFlowNet.Pipeline` operations and infrastructure.

`Onnx/` contains tests for `NeuroModFlowNet.Pipeline.ONNX` operations and infrastructure.

`Common/` contains only shared technical infrastructure: environment paths, visual artifact writing, low-level assert
helpers, `Mat`/`OrtValue` conversion, and `VmRunContext` creation.

Operation-specific logic stays in the operation folder. For example, `OpOnnxCropPattern` decides which pixels prove that
Crop is correct; a common helper should only save an image or compare a pixel.

## Automatic Checks

By default tests do not save images and do not open windows. They create synthetic inputs in memory, execute the
operation, and assert the result automatically.

ONNX operation tests also have a second test-matrix axis: execution backend. The default backend matrix is CPU only so
that normal local and CI runs stay stable. CUDA and TensorRT must be enabled explicitly.

Example for the current Crop tests:

```powershell
dotnet test .\tests\NeuroModFlowNet.Pipeline.ONNX.Tests\NeuroModFlowNet.Pipeline.ONNX.Tests.csproj --filter FullyQualifiedName~Op_Onnx_CropTests
```

## Execution Backend Matrix

Backend selection is part of test configuration, not an implicit fallback. For operation tests that use this matrix, each
test case is expanded as:

```text
operation pattern case x enabled backend
```

Supported matrix entries now:

- `Cpu`
- `Cuda`
- `TensorRt`

When a backend is enabled but the provider cannot be created on the current machine, the affected test cases fail. This
is intentional: an enabled backend is a test contract, so the run must not silently fall back to CPU or be reported as a
successful partial check.

Enable CUDA for one `cmd.exe` session:

```cmd
set NMFN_ONNX_TEST_CUDA=1
dotnet test .\tests\NeuroModFlowNet.Pipeline.ONNX.Tests\NeuroModFlowNet.Pipeline.ONNX.Tests.csproj
```

Enable TensorRT:

```cmd
set NMFN_ONNX_TEST_TENSORRT=1
dotnet test .\tests\NeuroModFlowNet.Pipeline.ONNX.Tests\NeuroModFlowNet.Pipeline.ONNX.Tests.csproj
```

Disable CPU explicitly:

```cmd
set NMFN_ONNX_TEST_CPU=0
```

Values `1` and `true` enable a backend. Values other than `1`/`true` disable it when the environment variable is set.

## Visual Artifacts

Tests can save input and output images as PNG files for manual inspection after a test run.

`Op_Onnx_Crop` already uses this path: when saving is enabled it writes `source` and `actual` PNG files.

General artifact structure:

```text
<VisualArtifactsRoot>\
  <Domain>\
    <OperationName>\
      <OperationName>-<Backend>-<CaseName>-<ArtifactKind>.png
```

Crop example:

```text
<VisualArtifactsRoot>\
  Onnx\
    Op_Onnx_Crop\
      Op_Onnx_Crop-Cpu-center_marker_64x48_crop_24_16_16_16-source.png
      Op_Onnx_Crop-Cpu-center_marker_64x48_crop_24_16_16_16-actual.png
```

`*-source.png` is the synthetic input image.

`*-actual.png` is the operation output.

Future operation folders may also write `*-expected.png`, `*-diff.png`, `*-report.json`, or other operation-specific
files. The important part is that the file name starts with operation and backend, so a flat folder view groups related
results together.

## Default Physical Path

If `NMFN_VISUAL_ROOT` and `visualArtifactsRoot` are not set, artifacts are written under the process temp directory:

```text
%TEMP%\NeuroModFlowNet.Pipeline.ONNX.Tests\VisualTestResults\
```

On the current workstation `%TEMP%` is:

```text
R:\Temp
```

So the current default physical artifact root is:

```text
R:\Temp\NeuroModFlowNet.Pipeline.ONNX.Tests\VisualTestResults\
```

For Crop this becomes:

```text
R:\Temp\NeuroModFlowNet.Pipeline.ONNX.Tests\VisualTestResults\Onnx\Op_Onnx_Crop\Op_Onnx_Crop-Cpu-center_marker_64x48_crop_24_16_16_16-actual.png
```

The xUnit output also shows the backend as a theory argument, for example `executionBackend: Cpu`, `executionBackend:
Cuda`, or `executionBackend: TensorRt`. If CUDA or TensorRT is enabled but unavailable, those cases fail and the output
contains the provider initialization error.

## Save PNG Files

Enable PNG output with `NMFN_VISUAL_SAVE`:

```powershell
$env:NMFN_VISUAL_SAVE = "1"
dotnet test .\tests\NeuroModFlowNet.Pipeline.ONNX.Tests\NeuroModFlowNet.Pipeline.ONNX.Tests.csproj
```

Override the artifact root:

```powershell
$env:NMFN_VISUAL_SAVE = "1"
$env:NMFN_VISUAL_ROOT = "C:\Temp\NmfnVisualTests"
dotnet test .\tests\NeuroModFlowNet.Pipeline.ONNX.Tests\NeuroModFlowNet.Pipeline.ONNX.Tests.csproj
```

Then files are written to:

```text
C:\Temp\NmfnVisualTests\<Domain>\<OperationName>\<OperationName>-<Backend>-<CaseName>-<ArtifactKind>.png
```

The repo-local `VisualTestResults/` folder may be used manually. It is ignored by git.

## Interactive Mode

For local exploration you can show images through OpenCV windows:

```powershell
$env:NMFN_VISUAL_INTERACTIVE = "1"
dotnet test .\tests\NeuroModFlowNet.Pipeline.ONNX.Tests\NeuroModFlowNet.Pipeline.ONNX.Tests.csproj
```

In this mode `VisualTestOutput` calls:

```csharp
Cv2.ImShow(...);
Cv2.WaitKey(0);
```

The test pauses on each shown image and continues after a key press in the OpenCV window.

Do not enable interactive mode in CI or normal automated runs because the test waits for a user.

`NMFN_VISUAL_INTERACTIVE=1` also enables PNG saving even when `NMFN_VISUAL_SAVE` is not set. This keeps artifacts on disk
after manual inspection.

## appsettings.test.json

The same modes can be configured in `appsettings.test.json`:

```json
{
  "nativeLibrarySearchPaths": [],
  "cudaBinPath": "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9\\bin",
  "cudnnBinPath": "C:\\cuDNN\\9.10.2.21_cuda12\\bin",
  "trtLibPath": "C:\\TensorRT\\TensorRT-10.12.0.36-cuda12.9\\lib",
  "visualArtifactsRoot": "C:\\Temp\\NmfnVisualTests",
  "saveVisualArtifacts": true,
  "interactiveVisualArtifacts": false,
  "executionBackends": {
    "cpu": true,
    "cuda": false,
    "tensorRt": false
  }
}
```

Fields:

- `nativeLibrarySearchPaths` adds extra native library directories to `PATH` before tests run.
- `cudaBinPath` adds the CUDA `bin` directory to `PATH`.
- `cudnnBinPath` adds the cuDNN `bin` directory to `PATH`.
- `trtLibPath` adds the TensorRT `lib` directory to `PATH`.
- `visualArtifactsRoot` sets the root directory for PNG artifacts.
- `saveVisualArtifacts` saves `*-source.png`, `*-actual.png`, and future visual outputs.
- `interactiveVisualArtifacts` shows images through OpenCV windows.
- `executionBackends.cpu`, `executionBackends.cuda`, and `executionBackends.tensorRt` control the backend matrix.

Environment variables are better for one-off local runs. `appsettings.test.json` is better for stable local project
configuration.

## Environment Variables

| Variable | Meaning |
| --- | --- |
| `NMFN_VISUAL_SAVE=1` | Save visual artifacts to PNG files. |
| `NMFN_VISUAL_INTERACTIVE=1` | Show images with OpenCV windows and also save PNG files. |
| `NMFN_VISUAL_ROOT=<path>` | Override visual artifacts root directory. |
| `NMFN_CUDA_BIN_PATH=<path>` | Override the CUDA `bin` directory. |
| `NMFN_CUDNN_BIN_PATH=<path>` | Override the cuDNN `bin` directory. |
| `NMFN_TRT_LIB_PATH=<path>` | Override the TensorRT `lib` directory. |
| `NMFN_ONNX_TEST_CPU=0/1` | Disable or enable CPU backend cases. |
| `NMFN_ONNX_TEST_CUDA=0/1` | Disable or enable CUDA backend cases. |
| `NMFN_ONNX_TEST_TENSORRT=0/1` | Disable or enable TensorRT backend cases. |

Values `1` and `true` are treated as enabled.

## Why Images Are Not Saved By Default

Normal test runs should be fast and should not fill the working tree. The default behavior is:

- test images are created in memory;
- assertions run automatically;
- files are written only when `NMFN_VISUAL_SAVE`, `NMFN_VISUAL_INTERACTIVE`, or `appsettings.test.json` explicitly enable
  them.

This keeps the same tests useful both as automated checks and as a manual operation-inspection tool.
