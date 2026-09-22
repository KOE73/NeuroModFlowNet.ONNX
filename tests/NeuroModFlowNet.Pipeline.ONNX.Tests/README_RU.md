# NeuroModFlowNet.Pipeline.ONNX.Tests

## Назначение проекта

Этот проект предназначен для тестирования операций из `NeuroModFlowNet.Pipeline.ONNX` и связанных базовых операций из
`NeuroModFlowNet.Pipeline`.

Главное правило: каждая операция хранит свою operation-specific test infrastructure рядом с тестом операции. Если
операции нужны диагностические изображения, expected metadata, seeded random cases или visual artifacts, эти файлы
лежат в папке операции, а не в общей куче.

## Структура

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

`Base/` - тесты для операций и инфраструктуры из `NeuroModFlowNet.Pipeline`.

`Onnx/` - тесты для операций и инфраструктуры из `NeuroModFlowNet.Pipeline.ONNX`.

`Common/` - только общая техническая инфраструктура: пути окружения, сохранение visual artifacts, низкоуровневые assert
helpers, конвертация `Mat`/`OrtValue`, создание `VmRunContext`.

Operation-specific логика должна оставаться в папке операции. Например, `OpOnnxCropPattern` решает, какие pixels
доказывают корректность Crop; общий helper должен только сохранить картинку или сравнить pixel.

## Автоматическая проверка

Обычный режим не сохраняет картинки и не открывает окна. Тесты создают synthetic inputs в памяти, выполняют операцию и
проверяют результат автоматически.

У ONNX operation tests есть второе измерение test matrix: execution backend. По умолчанию включен только CPU, чтобы
обычные локальные и CI-прогоны были стабильными. CUDA и TensorRT включаются явно.

Пример запуска текущих Crop-тестов:

```powershell
dotnet test .\tests\NeuroModFlowNet.Pipeline.ONNX.Tests\NeuroModFlowNet.Pipeline.ONNX.Tests.csproj --filter FullyQualifiedName~Op_Onnx_CropTests
```

## Матрица execution backends

Выбор backend является частью тестовой конфигурации, а не неявным fallback. Для operation tests, которые используют эту
матрицу, каждый test case разворачивается так:

```text
operation pattern case x enabled backend
```

Сейчас поддерживаются:

- `Cpu`
- `Cuda`
- `TensorRt`

Если backend включен, но provider не создается на текущей машине, соответствующие test cases падают с ошибкой. Это
намеренное поведение: включенный backend является test contract, поэтому прогон не должен молча откатываться на CPU или
выглядеть как успешная частичная проверка.

Включить CUDA для текущего окна `cmd.exe`:

```cmd
set NMFN_ONNX_TEST_CUDA=1
dotnet test .\tests\NeuroModFlowNet.Pipeline.ONNX.Tests\NeuroModFlowNet.Pipeline.ONNX.Tests.csproj
```

Включить TensorRT:

```cmd
set NMFN_ONNX_TEST_TENSORRT=1
dotnet test .\tests\NeuroModFlowNet.Pipeline.ONNX.Tests\NeuroModFlowNet.Pipeline.ONNX.Tests.csproj
```

Явно выключить CPU:

```cmd
set NMFN_ONNX_TEST_CPU=0
```

Значения `1` и `true` включают backend. Если environment variable задана, любые значения кроме `1`/`true` выключают
соответствующий backend.

## Visual artifacts

Тесты могут сохранять входные и выходные изображения в PNG, чтобы после прогона посмотреть результат глазами.

`Op_Onnx_Crop` уже использует этот механизм: при включенном сохранении он пишет PNG-файлы `source` и `actual`.

## Матрица real-image visual tests

Кроме синтетических deterministic cases, часть visual tests использует изображения из `images/`. Базовая проверочная
картинка `for_tests.png` должна оставаться у всех графических операций, кроме `Perspective`: у `Perspective` отдельные
картинки, потому что там важны осмысленные quad-точки и видимая перспективная плоскость.

| Ops | Images | Где настраивать |
| --- | --- | --- |
| `Op_Onnx_Crop_U8_NHWC` | `images/for_tests.png` | `Op_Onnx_RealImageVisualTests.CropU8NHWC_SavesRealImageVisualArtifact` |
| `Op_Onnx_Resize_U8_NHWC` | `images/for_tests.png` | `Op_Onnx_RealImageVisualTests.ResizeU8NHWC_SavesRealImageVisualArtifact` |
| `Op_Onnx_PadResize_U8_NHWC` | `images/for_tests.png` | `Op_Onnx_RealImageVisualTests.PadResizeU8NHWC_SavesRealImageVisualArtifact` |
| `Op_Onnx_Rotate90_U8_NHWC` | `images/for_tests.png` | `Op_Onnx_RealImageVisualTests.Rotate90U8NHWC_SavesRealImageVisualArtifact` |
| `Op_Onnx_Undistort_U8_NHWC` | `images/for_tests.png` | `Op_Onnx_RealImageVisualTests.UndistortU8NHWC_SavesRealImageVisualArtifact` |
| `Op_Onnx_Perspective_U8_NHWC`, `Op_Onnx_Perspective_FP32_NCHW`, `Op_Onnx_Perspective_FP16_NCHW` | `images/for_tests_perspective.png`, `images/for_tests_perspective_real_0.png` | `ConveyorDrawingPerspective`, `ConveyorReal0Perspective` в `Op_Onnx_RealImageVisualTests` |

Общая структура artifacts:

```text
<VisualArtifactsRoot>\
  <Domain>\
    <OperationName>\
      <OperationName>-<Backend>-<CaseName>-<ArtifactKind>.png
```

Пример для Crop:

```text
<VisualArtifactsRoot>\
  Onnx\
    Op_Onnx_Crop\
      Op_Onnx_Crop-Cpu-center_marker_64x48_crop_24_16_16_16-source.png
      Op_Onnx_Crop-Cpu-center_marker_64x48_crop_24_16_16_16-actual.png
```

`*-source.png` - синтетическое входное изображение.

`*-actual.png` - результат после выполнения операции.

В будущем рядом могут появиться `*-expected.png`, `*-diff.png`, `*-report.json` или другие operation-specific файлы.
Важно, что имя начинается с операции и backend: тогда в плоском просмотре папки связанные результаты группируются
рядом.

## Реальный путь по умолчанию

Если `NMFN_VISUAL_ROOT` и `visualArtifactsRoot` не заданы, artifacts пишутся в temp-папку процесса:

```text
%TEMP%\NeuroModFlowNet.Pipeline.ONNX.Tests\VisualTestResults\
```

На текущей машине `%TEMP%` равен:

```text
R:\Temp
```

Значит текущий физический путь по умолчанию:

```text
R:\Temp\NeuroModFlowNet.Pipeline.ONNX.Tests\VisualTestResults\
```

Для Crop это будет:

```text
R:\Temp\NeuroModFlowNet.Pipeline.ONNX.Tests\VisualTestResults\Onnx\Op_Onnx_Crop\Op_Onnx_Crop-Cpu-center_marker_64x48_crop_24_16_16_16-actual.png
```

В xUnit output backend также виден как аргумент theory: например `executionBackend: Cpu`, `executionBackend: Cuda` или
`executionBackend: TensorRt`. Если CUDA или TensorRT включены, но provider недоступен, такие cases падают, а output
содержит ошибку инициализации provider.

## Сохранение PNG

Чтобы после тестов остались PNG-файлы, включи `NMFN_VISUAL_SAVE`:

```powershell
$env:NMFN_VISUAL_SAVE = "1"
dotnet test .\tests\NeuroModFlowNet.Pipeline.ONNX.Tests\NeuroModFlowNet.Pipeline.ONNX.Tests.csproj
```

Чтобы задать папку явно:

```powershell
$env:NMFN_VISUAL_SAVE = "1"
$env:NMFN_VISUAL_ROOT = "C:\Temp\NmfnVisualTests"
dotnet test .\tests\NeuroModFlowNet.Pipeline.ONNX.Tests\NeuroModFlowNet.Pipeline.ONNX.Tests.csproj
```

Тогда файлы будут сохранены в:

```text
C:\Temp\NmfnVisualTests\<Domain>\<OperationName>\<OperationName>-<Backend>-<CaseName>-<ArtifactKind>.png
```

Repo-local папку `VisualTestResults/` можно использовать вручную. Она добавлена в `.gitignore`, чтобы PNG artifacts не
попадали в commit.

## Интерактивный режим

Для локального изучения можно включить OpenCV window output:

```powershell
$env:NMFN_VISUAL_INTERACTIVE = "1"
dotnet test .\tests\NeuroModFlowNet.Pipeline.ONNX.Tests\NeuroModFlowNet.Pipeline.ONNX.Tests.csproj
```

В этом режиме `VisualTestOutput` вызывает:

```csharp
Cv2.ImShow(...);
Cv2.WaitKey(0);
```

Тест остановится на каждом показанном изображении и продолжит выполнение после нажатия клавиши в окне OpenCV.

Interactive mode нельзя включать в CI или обычном автоматическом прогоне, потому что тест будет ждать пользователя.

`NMFN_VISUAL_INTERACTIVE=1` также автоматически включает сохранение PNG, даже если `NMFN_VISUAL_SAVE` не задан. Это
сделано, чтобы после ручного просмотра artifacts остались на диске.

## appsettings.test.json

Те же режимы можно включить в `appsettings.test.json`:

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

Поля:

- `nativeLibrarySearchPaths` - дополнительные пути к native libraries, которые добавляются в `PATH` перед тестами.
- `cudaBinPath` - путь к CUDA `bin`, который добавляется в `PATH`.
- `cudnnBinPath` - путь к cuDNN `bin`, который добавляется в `PATH`.
- `trtLibPath` - путь к TensorRT `lib`, который добавляется в `PATH`.
- `visualArtifactsRoot` - корневая папка для PNG artifacts.
- `saveVisualArtifacts` - сохранять `*-source.png`, `*-actual.png` и будущие visual outputs.
- `interactiveVisualArtifacts` - показывать изображения через OpenCV windows.
- `executionBackends.cpu`, `executionBackends.cuda`, `executionBackends.tensorRt` - управление backend matrix.

Environment variables удобнее для разового локального запуска. `appsettings.test.json` удобнее для устойчивой локальной
конфигурации проекта.

## Environment variables

| Variable | Meaning |
| --- | --- |
| `NMFN_VISUAL_SAVE=1` | Сохранять visual artifacts в PNG-файлы. |
| `NMFN_VISUAL_INTERACTIVE=1` | Показывать изображения через OpenCV windows и также сохранять PNG-файлы. |
| `NMFN_VISUAL_ROOT=<path>` | Переопределить корневую папку visual artifacts. |
| `NMFN_CUDA_BIN_PATH=<path>` | Переопределить путь к CUDA `bin`. |
| `NMFN_CUDNN_BIN_PATH=<path>` | Переопределить путь к cuDNN `bin`. |
| `NMFN_TRT_LIB_PATH=<path>` | Переопределить путь к TensorRT `lib`. |
| `NMFN_ONNX_TEST_CPU=0/1` | Выключить или включить CPU backend cases. |
| `NMFN_ONNX_TEST_CUDA=0/1` | Выключить или включить CUDA backend cases. |
| `NMFN_ONNX_TEST_TENSORRT=0/1` | Выключить или включить TensorRT backend cases. |
| `NMFN_PRIVATE_TRACKING_CONFIG_JSON=<path>` | Включить приватный tracking integration test через один внешний config. |
| `NMFN_PRIVATE_TRACKING_SEQUENCE_JSON=<path>` | Включить приватный tracking integration test на внешнем JSON fixture. |
| `NMFN_PRIVATE_TRACKING_VIDEO_PATH=<path>` | Опциональная проверяемая ссылка на приватное видео, из которого получен fixture. |
| `NMFN_PRIVATE_TRACKING_MODEL_PATH=<path>` | Опциональная проверяемая ссылка на приватную модель/сеть, из которой получен fixture. |

Значения `1` и `true` считаются включенным режимом.

## Private tracking sequence

`PrivateTrackingSequenceIntegrationTests` не хранит приватное видео или модель в репозитории. По умолчанию тест
пропускается. Для локального прогона нужно один раз получить JSON с детекциями из приватного видео/сети и указать путь:

```powershell
$env:NMFN_PRIVATE_TRACKING_CONFIG_JSON = "C:\GitKOE\NeuroModFlowNet.ONNX.Private\tracking\tracking-private-config.json"
dotnet test .\tests\NeuroModFlowNet.Pipeline.ONNX.Tests\NeuroModFlowNet.Pipeline.ONNX.Tests.csproj --filter FullyQualifiedName~PrivateTrackingSequenceIntegrationTests
```

Или напрямую:

```powershell
$env:NMFN_PRIVATE_TRACKING_SEQUENCE_JSON = "R:\PrivateTracking\sequence.json"
$env:NMFN_PRIVATE_TRACKING_VIDEO_PATH = "R:\PrivateTracking\source.mp4"
$env:NMFN_PRIVATE_TRACKING_MODEL_PATH = "R:\PrivateTracking\model.onnx"
dotnet test .\tests\NeuroModFlowNet.Pipeline.ONNX.Tests\NeuroModFlowNet.Pipeline.ONNX.Tests.csproj --filter FullyQualifiedName~PrivateTrackingSequenceIntegrationTests
```

Минимальный формат:

```json
{
  "maxInFlight": 1,
  "trackerOptions": {
    "iouThreshold": 0.5,
    "minAge": 3,
    "maxMissedFrames": 10,
    "maxTrailLength": 50,
    "startZoneContainmentThreshold": 1.0
  },
  "startZone": { "x": 0, "y": 0, "width": 1920, "height": 1080 },
  "expect": {
    "minProducedTrackFrames": 1,
    "minConfirmedTracks": 1,
    "maxTrackId": 20
  },
  "frames": [
    {
      "frameIndex": 0,
      "detections": [
        { "x": 100, "y": 120, "w": 40, "h": 30, "angle": 0, "classId": 0, "score": 0.91 }
      ]
    }
  ]
}
```

## Почему картинки не сохраняются всегда

Обычный тестовый прогон должен быть быстрым и не засорять рабочую папку. Поэтому default behavior такой:

- test images создаются в памяти;
- assertions выполняются автоматически;
- файлы пишутся только при явном `NMFN_VISUAL_SAVE`, `NMFN_VISUAL_INTERACTIVE` или настройке в `appsettings.test.json`.

Так один и тот же тест остается и автоматической проверкой, и инструментом для ручного понимания операции.
