# Использование NeuroModFlowNet.ONNX из NuGet-пакета

Документ описывает типовой путь использования пакета в приложении: загрузить ONNX-модель, создать runner, подготовить входные данные, выполнить prediction и прочитать результат.

## Общая схема

Работа строится в три шага:

1. Создать `OnnxRuntimeContext` для `.onnx` модели.
2. Создать `IRunner<TIn, TOut>` через фабрику под нужный сценарий.
3. Вызвать `Predict(input)` и обработать типизированный результат.

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
## Загрузка модели

`OnnxRuntimeContext` загружает модель, создает ONNX Runtime session, выбирает backend и хранит сведения о входах и выходах модели.

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

### DirectML и ROCm

```csharp
using var dmlContext = new OnnxRuntimeContext("model.onnx", InferenceBackend.DML);
using var rocmContext = new OnnxRuntimeContext("model.onnx", InferenceBackend.Rocm);
```

## Тонкая настройка Execution Provider

Третий параметр `OnnxRuntimeContext` принимает `Action<ExecutionProviderConfig>`.

Для CUDA используйте `CudaConfig`:

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

Для TensorRT используйте `TrtConfig`:

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

Если для TensorRT не передать настройку, будут использованы встроенные значения по умолчанию, включая FP16/BF16 и engine cache.

## Native DLL и PATH

Для `Cpu` дополнительные NVIDIA DLL не нужны.

Для `Cuda` процесс должен видеть CUDA и cuDNN DLL.
Для `TensorRt` процесс должен видеть CUDA, cuDNN и TensorRT DLL.

Если нужные директории уже есть в системном `PATH`, дополнительных действий не требуется.
Если нужно добавить пути только для текущего процесса, можно использовать:

```csharp
OnnxRuntimePathHelper.AddToSystemPath(@"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin");
OnnxRuntimePathHelper.AddToSystemPath(@"C:\cuDNN\9.10.2.21_cuda12\bin");
OnnxRuntimePathHelper.AddToSystemPath(@"C:\TensorRT\TensorRT-10.12.0.36-cuda12.9\lib");
```

Вызов должен быть выполнен до создания `OnnxRuntimeContext`.

## Основной принцип runner

Все типовые сценарии используют общий контракт:

```csharp
public interface IRunner<in TIn, out TOut> : IDisposable
{
    TOut Predict(TIn input);
}
```

Runner выполняет один полный проход:

1. Подготавливает входные данные под формат модели.
2. Запускает inference.
3. Преобразует выходные тензоры в удобный тип результата.

Поэтому пользовательский код обычно не работает напрямую с `InferenceSession`, tensor metadata и raw output buffer.

## Фабрики runner

Фабрики создают готовые runner под распространенные сценарии.

Есть два подхода:

1. Точный вариант: метод явно фиксирует вход, preprocessing, precision и тип результата.
2. Автоматический вариант: `CreateRunner` выбирает converter и extractor по metadata модели.

### YOLO Box

Автоматический single-image runner:

```csharp
using var context = new OnnxRuntimeContext("yolo-box.onnx", InferenceBackend.Cuda);
using IRunner<Mat, IDetectionResult<YoloBox>> runner =
    YoloBoxFactory.CreateRunner<IDetectionResult<YoloBox>>(context);
```

Точные варианты:

```csharp
var fp32Runner = YoloBoxFactory.Single_PosCvdnn_FP32(context);
var fp16Runner = YoloBoxFactory.Single_PosCvdnn_FP16(context);
var byteBgrRunner = YoloBoxFactory.Single_BgrDirect_FP32(context);
var batchRunner = YoloBoxFactory.List_PosCvdnn_FP32(context);
```

Типовой результат: `IDetectionResult<YoloBox>`.

### YOLO OBB

```csharp
using IRunner<Mat, IDetectionResult<YoloObb>> runner =
    YoloObbFactory.CreateRunner<IDetectionResult<YoloObb>>(context);
```

Точные варианты:

```csharp
var fp32Runner = YoloObbFactory.Single_PosCvdnn_FP32(context);
var byteBgrRunner = YoloObbFactory.Single_BgrDirect_FP32(context);
var batchRunner = YoloObbFactory.List_PosCvdnn_FP32(context);
```

Типовой результат: `IDetectionResult<YoloObb>`.

### YOLO Pose

```csharp
using IRunner<Mat, IDetectionResult<YoloPose>> runner =
    YoloPoseFactory.CreateRunner(context);
```

Для модели с 17 keypoints:

```csharp
using IRunner<Mat, IDetectionResult<YoloPose_FP32_Size57_Keypoint17>> runner =
    YoloPoseFactory.CreateRunner17(context);
```

### YOLO Classification

```csharp
using IRunner<Mat, IBatchedResult> runner = YoloClsFactory.CreateRunner(context);
IBatchedResult result = runner.Predict(image);
```

Если нужен точный тип:

```csharp
var fp32Runner = YoloClsFactory.Single_PosCvdnn_FP32(context);
YoloCls result = fp32Runner.Predict(image);
```

### YOLO Segmentation

```csharp
using IRunner<Mat, IBatchedResult> runner = YoloSegFactory.CreateRunner(context);
IBatchedResult result = runner.Predict(image);
```

Точные варианты:

```csharp
var fp32Runner = YoloSegFactory.Single_PosCvdnn_FP32(context);
var fp16Runner = YoloSegFactory.Single_PosCvdnn_FP16(context);
var byteBgrRunner = YoloSegFactory.Single_BgrDirect_FP32(context);
```

### PaddleOCR Detection

Single image, output as grayscale mask:

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

Точные варианты:

```csharp
var fp32MapRunner = PaddleOCRDetFactory.Single_FP32_32FC1_Safe(context);
var fp32MaskRunner = PaddleOCRDetFactory.Single_FP32_8UC1(context);
var fp16MaskRunner = PaddleOCRDetFactory.Single_FP16_8UC1(context);
```

### PaddleOCR Recognition

Для recognition можно создать runner сразу по пути к модели:

```csharp
using var runner = PaddleOCRRecFactory.CreateRecSingle("rec.onnx", InferenceBackend.Cuda);
List<PaddleOCRRecExtractor.OcrResult> results = runner.Predict(textLineImage);
```

Batch:

```csharp
using var runner = PaddleOCRRecFactory.CreateRecList("rec.onnx", InferenceBackend.Cuda);
List<PaddleOCRRecExtractor.OcrResult> results = runner.Predict(textLineImages);
```

`OcrResult` содержит:

| Поле | Значение |
| --- | --- |
| `Standard` | Основной распознанный текст. |
| `WithSpaces` | Вариант с восстановленными пробелами. |
| `FullCandidates` | Диагностический список кандидатов. |

## Prediction

Перед `Predict` вход должен соответствовать выбранному runner:

| Runner input | Что передавать |
| --- | --- |
| `Mat` | Один `OpenCvSharp.Mat`. |
| `List<Mat>` | Batch изображений. |

Пример для single-image detection:

```csharp
using Mat image = Cv2.ImRead("image.jpg");
IDetectionResult<YoloBox> result = runner.Predict(image);
```

Пример для batch:

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

## Использование detection результата

`IDetectionResult<T>` предоставляет:

```csharp
int batchCount = result.BatchCount;
int count = result.GetResultCount(batchIndex: 0);
T item = result.GetResult(index: 0, batchIndex: 0);
ReadOnlySpan<T> batch = result.GetBatch(batchIndex: 0);
```

Для `YoloBox` доступны координаты, размер, score и class:

```csharp
YoloBox box = result.GetResult(0);

float x = box.X;
float y = box.Y;
float width = box.W;
float height = box.H;
float score = box.Score;
int classId = box.Class;
```

## Управление ресурсами

`OnnxRuntimeContext` и runner реализуют `IDisposable`.
Используйте `using`, особенно при работе с CUDA/TensorRT и `Mat`.

```csharp
using var context = new OnnxRuntimeContext("model.onnx", InferenceBackend.Cuda);
using var runner = YoloBoxFactory.CreateRunner<IDetectionResult<YoloBox>>(context);
using Mat image = Cv2.ImRead("image.jpg");

IDetectionResult<YoloBox> result = runner.Predict(image);
```
