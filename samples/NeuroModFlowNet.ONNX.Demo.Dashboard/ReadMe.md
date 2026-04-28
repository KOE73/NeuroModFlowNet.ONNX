# NeuroModFlowNet.ONNX Demo Dashboard

Demo dashboard запускает несколько YOLO pipeline над одним video source и отрисовывает результаты в одном OpenCV-окне.

## Матрица вариантов

- Исполнители: `CPU`, `CUDA`, `TensorRT`, `DML/AMD`.
- Входы: `FP32`, `FP16`, `ByteBGR`.

Текущая конфигурация demo выбирается в `Program.cs`:

- `Backend` задает ONNX Runtime execution provider.
- `Precision` выбирает precision model assets.
- `InputSize` задает входной размер модели.
- `IsByteBgr` выбирает, принимает ли модель прямой `ByteBGR` input.

## Пути к native runtime

Dashboard вызывает `OnnxRuntimePathHelper.InitFromConfig()` до загрузки моделей. Для запуска через CUDA или TensorRT нужно либо добавить native-директории NVIDIA в системный `PATH`, либо указать их в `App.config`:

```xml
<appSettings>
  <add key="CudaBinPath" value="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin" />
  <add key="CudnnBinPath" value="C:\cuDNN\9.10.2.21_cuda12\bin" />
  <add key="TrtLibPath" value="C:\TensorRT\TensorRT-10.12.0.36-cuda12.9\lib" />
</appSettings>
```

Для CUDA `CudaBinPath` и `CudnnBinPath` должны указывать на установленные runtime DLL CUDA и cuDNN. Для TensorRT нужны те же CUDA/cuDNN пути, плюс `TrtLibPath` до директории TensorRT `lib`. Версии должны быть совместимы с ONNX Runtime GPU package, который используется приложением.

Для CPU-only запуска NVIDIA-пути не требуются.
