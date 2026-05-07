# Using NeuroModFlowNet.ONNX from a NuGet Package

This document describes the common package usage flow in an application: load an ONNX model, create a runner, prepare input data, run prediction, and read the result.

## General Flow

The workflow has three steps:

1. Create an `OnnxRuntimeContext` for the `.onnx` model.
2. Create an `IRunner<TIn, TOut>` through a factory for the required scenario.
3. Call `Predict(input)` and process the typed result.

```csharp
using NeuroModFlowNet.ONNX;
using OpenCvSharp;

using var context = new OnnxRuntimeContext("model.onnx", InferenceBackend.Cuda);
using IRunner<Mat, IDetectionResult<YoloBox>> runner =
    YoloBoxFactory.CreateRunner<IDetectionResult<YoloBox>>(context);

using Mat image = Cv2.ImRead("image.jpg");
IDetectionResult<YoloBox> result = runner.Predict(image);

for (int index = 0; index < result.GetResultCount(); index++)
{
    YoloBox box = result.GetResult(index);
    Console.WriteLine($"{box.Class}: {box.Score:P1} {box.X},{box.Y} {box.W}x{box.H}");
}
```
## Loading a Model

`OnnxRuntimeContext` loads the model, creates the ONNX Runtime session, selects the backend, and stores model input/output information.

### CPU

```csharp
using var context = new OnnxRuntimeContext("model.onnx", InferenceBackend.Cpu);
```

### CUDA

```csharp
using var context = new OnnxRuntimeContext("model.onnx", InferenceBackend.Cuda);
```

### TensorRT

```csharp
using var context = new OnnxRuntimeContext("model.onnx", InferenceBackend.TensorRt);
```

### DirectML and ROCm

```csharp
using var dmlContext = new OnnxRuntimeContext("model.onnx", InferenceBackend.DML);
using var rocmContext = new OnnxRuntimeContext("model.onnx", InferenceBackend.Rocm);
```

## Fine-Tuning the Execution Provider

The third `OnnxRuntimeContext` parameter accepts `Action<ExecutionProviderConfig>`.

For CUDA, use `CudaConfig`:

```csharp
using var context = new OnnxRuntimeContext(
    "model.onnx",
    InferenceBackend.Cuda,
    config =>
    {
        if (config is CudaConfig cuda)
        {
            cuda.DeviceId = 0;
            cuda.GpuMemLimitGb = 6;
            cuda.CudnnConvAlgoSearch = "HEURISTIC";
            cuda.EnableCudaGraph = true;
        }
    });
```

For TensorRT, use `TrtConfig`:

```csharp
using var context = new OnnxRuntimeContext(
    "model.onnx",
    InferenceBackend.TensorRt,
    config =>
    {
        if (config is TrtConfig trt)
        {
            trt.DeviceId = 0;
            trt.MaxWorkspaceSizeGb = 4;
            trt.EnableFp16 = true;
            trt.EnableEngineCache = true;
            trt.EngineCachePath = "trt-cache";
            trt.BuilderOptimizationLevel = 2;
        }
    });
```

If no TensorRT configuration is passed, built-in defaults are used, including FP16/BF16 and engine cache.

## Native DLLs and PATH

`Cpu` does not require additional NVIDIA DLLs.

`Cuda` requires the process to see CUDA and cuDNN DLLs.
`TensorRt` requires the process to see CUDA, cuDNN, and TensorRT DLLs.

If the required directories are already in the system `PATH`, no extra action is needed.
If paths should be added only for the current process, use:

```csharp
OnnxRuntimePathHelper.AddToSystemPath(@"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin");
OnnxRuntimePathHelper.AddToSystemPath(@"C:\cuDNN\9.10.2.21_cuda12\bin");
OnnxRuntimePathHelper.AddToSystemPath(@"C:\TensorRT\TensorRT-10.12.0.36-cuda12.9\lib");
```

Call this before creating `OnnxRuntimeContext`.

## Runner Principle

All typical scenarios use one shared contract:

```csharp
public interface IRunner<in TIn, out TOut> : IDisposable
{
    TOut Predict(TIn input);
}
```

A runner performs one full pass:

1. Prepares input data for the model format.
2. Runs inference.
3. Converts output tensors into a convenient result type.

User code normally does not work directly with `InferenceSession`, tensor metadata, or raw output buffers.

## Runner Factories

Factories create ready-to-use runners for common scenarios.

There are two approaches:

1. Exact variant: the method explicitly fixes input, preprocessing, precision, and result type.
2. Automatic variant: `CreateRunner` selects converter and extractor from model metadata.

### YOLO Box

Automatic single-image runner:

```csharp
using var context = new OnnxRuntimeContext("yolo-box.onnx", InferenceBackend.Cuda);
using IRunner<Mat, IDetectionResult<YoloBox>> runner =
    YoloBoxFactory.CreateRunner<IDetectionResult<YoloBox>>(context);
```

Exact variants:

```csharp
var fp32Runner = YoloBoxFactory.Single_PosCvdnn_FP32(context);
var fp16Runner = YoloBoxFactory.Single_PosCvdnn_FP16(context);
var byteBgrRunner = YoloBoxFactory.Single_BgrDirect_FP32(context);
var batchRunner = YoloBoxFactory.List_PosCvdnn_FP32(context);
```

Typical result: `IDetectionResult<YoloBox>`.

### YOLO OBB

```csharp
using IRunner<Mat, IDetectionResult<YoloObb>> runner =
    YoloObbFactory.CreateRunner<IDetectionResult<YoloObb>>(context);
```

Exact variants:

```csharp
var fp32Runner = YoloObbFactory.Single_PosCvdnn_FP32(context);
var byteBgrRunner = YoloObbFactory.Single_BgrDirect_FP32(context);
var batchRunner = YoloObbFactory.List_PosCvdnn_FP32(context);
```

Typical result: `IDetectionResult<YoloObb>`.

### YOLO Pose

```csharp
using IRunner<Mat, IDetectionResult<YoloPose>> runner =
    YoloPoseFactory.CreateRunner(context);
```

For a 17-keypoint model:

```csharp
using IRunner<Mat, IDetectionResult<YoloPose_FP32_Size57_Keypoint17>> runner =
    YoloPoseFactory.CreateRunner17(context);
```

### YOLO Classification

```csharp
using IRunner<Mat, IBatchedResult> runner = YoloClsFactory.CreateRunner(context);
IBatchedResult result = runner.Predict(image);
```

If you need the exact type:

```csharp
var fp32Runner = YoloClsFactory.Single_PosCvdnn_FP32(context);
YoloCls result = fp32Runner.Predict(image);
```

### YOLO Segmentation

```csharp
using IRunner<Mat, IBatchedResult> runner = YoloSegFactory.CreateRunner(context);
IBatchedResult result = runner.Predict(image);
```

Exact variants:

```csharp
var fp32Runner = YoloSegFactory.Single_PosCvdnn_FP32(context);
var fp16Runner = YoloSegFactory.Single_PosCvdnn_FP16(context);
var byteBgrRunner = YoloSegFactory.Single_BgrDirect_FP32(context);
```

### PaddleOCR Detection

Single image, output as a grayscale mask:

```csharp
using IRunner<Mat, Mat> runner =
    PaddleOCRDetFactory.CreateRunner<Mat, Mat>(context, MatType.CV_8UC1);

using Mat probabilityMap = runner.Predict(image);
```

Batch:

```csharp
using IRunner<List<Mat>, List<Mat>> runner =
    PaddleOCRDetFactory.CreateRunner<List<Mat>, List<Mat>>(context, MatType.CV_8UC1);

List<Mat> maps = runner.Predict(images);
```

Exact variants:

```csharp
var fp32MapRunner = PaddleOCRDetFactory.Single_FP32_32FC1_Safe(context);
var fp32MaskRunner = PaddleOCRDetFactory.Single_FP32_8UC1(context);
var fp16MaskRunner = PaddleOCRDetFactory.Single_FP16_8UC1(context);
```

### PaddleOCR Recognition

For recognition, a runner can be created directly from the model path:

```csharp
using var runner = PaddleOCRRecFactory.CreateRecSingle("rec.onnx", InferenceBackend.Cuda);
List<PaddleOCRRecExtractor.OcrResult> results = runner.Predict(textLineImage);
```

Batch:

```csharp
using var runner = PaddleOCRRecFactory.CreateRecList("rec.onnx", InferenceBackend.Cuda);
List<PaddleOCRRecExtractor.OcrResult> results = runner.Predict(textLineImages);
```

`OcrResult` contains:

| Field | Meaning |
| --- | --- |
| `Standard` | Main recognized text. |
| `WithSpaces` | Variant with restored spaces. |
| `FullCandidates` | Diagnostic candidate list. |

## Prediction

Before `Predict`, the input must match the selected runner:

| Runner input | What to pass |
| --- | --- |
| `Mat` | One `OpenCvSharp.Mat`. |
| `List<Mat>` | A batch of images. |

Single-image detection example:

```csharp
using Mat image = Cv2.ImRead("image.jpg");
IDetectionResult<YoloBox> result = runner.Predict(image);
```

Batch example:

```csharp
List<Mat> images = [image1, image2, image3];
IDetectionResult<YoloBox> result = batchRunner.Predict(images);

for (int batchIndex = 0; batchIndex < result.BatchCount; batchIndex++)
{
    ReadOnlySpan<YoloBox> boxes = result.GetBatch(batchIndex);
    foreach (YoloBox box in boxes)
    {
        Console.WriteLine($"{batchIndex}: {box.Class} {box.Score:P1}");
    }
}
```

## Using Detection Results

`IDetectionResult<T>` provides:

```csharp
int batchCount = result.BatchCount;
int count = result.GetResultCount(batchIndex: 0);
T item = result.GetResult(index: 0, batchIndex: 0);
ReadOnlySpan<T> batch = result.GetBatch(batchIndex: 0);
```

For `YoloBox`, coordinates, size, score, and class are available:

```csharp
YoloBox box = result.GetResult(0);

float x = box.X;
float y = box.Y;
float width = box.W;
float height = box.H;
float score = box.Score;
int classId = box.Class;
```

## Resource Management

`OnnxRuntimeContext` and runners implement `IDisposable`.
Use `using`, especially with CUDA/TensorRT and `Mat`.

```csharp
using var context = new OnnxRuntimeContext("model.onnx", InferenceBackend.Cuda);
using var runner = YoloBoxFactory.CreateRunner<IDetectionResult<YoloBox>>(context);
using Mat image = Cv2.ImRead("image.jpg");

IDetectionResult<YoloBox> result = runner.Predict(image);
```
