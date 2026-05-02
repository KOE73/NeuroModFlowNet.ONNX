# Локальная настройка CUDA/cuDNN/TensorRT для ONNX Runtime

Этот документ нужен для пользователей, которые запускают demo/lab/sample и получают ошибку загрузки CUDA/cuDNN/TensorRT DLL. 
Типичный случай: CUDA Toolkit установлен, cuDNN распакован отдельно, но ONNX Runtime не видит нужные native DLL.

## `App.local.config`

`App.local.config` нужен для настройки на конкретной машине и что бы не менять основной `App.config`.
Он неаходится в .gitignore.

- Если установленно несколько версий CUDA, cuDNN, TensorRT и нужная версия не прописанна в `PATH`.
- Что бы переключить режимы работы.

Файл локальный, не переносимая настройка проекта. Он описывает конкретную машину.

Настройки из этих файлов используются в demo/lab проектах.

Для чистого CUDA backend `TrtLibPath` не нужен. Для TensorRT он нужен вместе с CUDA и cuDNN.


## Минимальный пример для CUDA

Создайте рядом с `App.config` исполняемого проекта файл `App.local.config`.
Скопируйте в него следующие строки, отредактировав пути к CUDA, cuDNN и TensorRT под вашу машину.
Настройте режим работы и модели по желанию.

```xml
<?xml version="1.0" encoding="utf-8" ?>
<appSettings>
  <add key="CudaBinPath" value="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin" />
  <add key="CudnnBinPath" value="C:\cuDNN\9.10.2.21_cuda12\bin" />
  <add key="TrtLibPath" value="C:\TensorRT\TensorRT-10.12.0.36-cuda12.9\lib" />

  <!--<add key="InferenceBackend" value="TensorRt" />-->
  <!--<add key="InferenceBackend" value="CPU" />-->
  <add key="InferenceBackend" value="Cuda" />

  <add key="ModelPrecision" value="fp16" />
  <add key="UseByteBgr" value="true" />
  <add key="InputSize" value="640" />
  <add key="BoxModelName" value="yolo26n" />
  <!--<add key="ObbModelName" value="yolo26n-obb" />-->
  <add key="ObbModelName" value="img-text-to-obb" />
  <add key="PoseModelName" value="yolo26n-pose" />
  <add key="SegModelName" value="yolo26n-seg" />

  <!--
    VideoSource examples:
    - 0, 1, 2: local USB camera index.
    - C:\video\sample.mp4: local video file.
    - rtsp://host/stream: stream URL without credentials.
    - @VideoSource.local.txt: read the source from an ignored local file.
      Use this form for RTSP URLs with credentials.
  -->
  <add key="VideoSource" value="0" />
</appSettings>
```

## csproj

Для переключения на CPU в cspoj нужно заменить `Microsoft.ML.OnnxRuntime.Gpu` на `Microsoft.ML.OnnxRuntime`.

```xml
<!--NVIDIA-->
<PackageReference Include="Microsoft.ML.OnnxRuntime.Gpu" Version="1.25.1" />
		
<!--CPU-->
<!--<PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.25.1" />-->
```

# Проверка загрузки DLL

Если ONNX Runtime падает с ошибкой загрузки DLL, первым делом проверьте:

- путь `CudaBinPath` существует;
- путь `CudnnBinPath` существует;
- в `CudnnBinPath` лежат DLL cuDNN 9.x, а не cuDNN 8.x;
- `OnnxRuntimePathHelper.InitFromConfig()` вызван до создания `OnnxRuntimeContext` или `InferenceSession`;
- версия `Microsoft.ML.OnnxRuntime.Gpu` соответствует CUDA/cuDNN major-линейкам.
