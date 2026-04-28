
# NeuroModFlowNet.ONNX.Bench

Этот файл - рабочий реестр измерений для проекта `NeuroModFlowNet.ONNX.Bench`.
Сюда можно добавлять новые пункты "что надо померить", после этого под них удобно дописывать код бенчмарков.

## Статус набора

- Активный компилируемый набор сейчас находится в `Bench/Rec`.
- `Bench/Converters/BenchmarkConverter.cs` и `Bench/Det/BenchmarkDetOrtValue.cs` сохранены в проекте, но исключены из компиляции через `.csproj`. Это legacy/миграционный набор для OBB/детекции, его надо поднимать отдельно после приведения к текущим API.
- Все бенчмарки используют `BenchmarkDotNet`, `MemoryDiagnoser`, `ThreadingDiagnoser` и группировку по категориям.

## OCR Recognition

Файлы:

- `Bench/Rec/BenchmarkCommon.cs`
- `Bench/Rec/BenchmarkOrtValue.cs`

Категория BenchmarkDotNet: `OCR`.

Измерения:

- `Inference_Run`
  - Цель: обычный запуск OCR recognition через `InferenceSession.Run`.
  - Сравнивает полный путь выполнения без `IoBinding`.
  - Сейчас метод объявлен как активный benchmark.

- `Inference_RunWithBinding`
  - Цель: запуск OCR recognition через `IoBinding`.
  - Нужен для сравнения с обычным `Run` и оценки выигрыша от биндинга входов/выходов.
  - Сейчас метод объявлен как активный benchmark.

Параметры:

- `batch`: `1`, `4`.
- `_InferenceBackend`: `TensorRt`, `Cuda`.
- `EnableFp16`: `false`, `true`.
- `EnableBf16`: `false`, `true`.

Входные изображения:

- `Images/image_48_320_5_65.png`
- `Images/image_48_240_5_65.png`

## OCR Input Conversion / Mat To ORT

Файл: `Bench/Rec/BenchmarkOrtValue.cs`.

Категория BenchmarkDotNet: `Ort`.

Сохраненные, но сейчас закомментированные измерения:

- `Mats2Ort_DivHalfTensor_ReorderPtr`
  - Цель: измерить конвертацию списка `Mat` в ORT input buffer через деление, `HalfTensor` и pointer reorder.

- `InferenceOrt_ByMatBlob`
  - Цель: старый путь инференса через `Mat`/blob-подготовку.
  - Сейчас benchmark-атрибут закомментирован.

В комментариях также сохранены имена вариантов для будущего восстановления:

- `Mats2Ort_ReorderDivPtr_HalfTensor`
- `Mats2Ort_ReorderPtr_DivHalfTensor`
- `Mats2Ort_ReorderDivHalfPtr`
- `Mats2Ort_OpenCV_Blob`

Ручные методы проверки, не BenchmarkDotNet:

- `MainCuda_Run`
- `MainTensorRt_Run`
- `MainTensorRt_RunWithBinding`
- `MainTensorRt_RunWithBinding_Context`

## Detection / OBB Image To RBox

Файлы:

- `Bench/Converters/BenchmarkConverter.cs`
- `Bench/Det/BenchmarkDetOrtValue.cs`

Категория BenchmarkDotNet: `Image2RBox`.

Статус: файлы сохранены, но исключены из компиляции в `NeuroModFlowNet.ONNX.Bench.csproj`.
Причина: это старый набор вокруг прежних API/типов, который надо мигрировать на текущую структуру `Inputs`, `Algorithms`, `Runners` и актуальные модели.

Сохраненные измерения:

- `Inference_StdFloat`
  - Цель: стандартный FP32 путь модели.
  - Измеряет полный путь подготовки входа, инференса и получения OBB/RBox результата.

- `Inference_StdFP16_OpenCV_Blob`
  - Цель: стандартный FP16 путь с подготовкой входа через OpenCV blob.
  - Нужен как базовая точка сравнения для оптимизированных input algorithms.

- `Inference_StdFP16_ReorderDivPtr_HalfTensor`
  - Цель: FP16 путь с ручной подготовкой входа через reorder/div pointer algorithm и `HalfTensor`.
  - Ценный benchmark для проверки выигрыша от прямой работы с памятью.

- `Inference_ModHead_InferenceRunner`
  - Цель: путь модифицированной головы модели через текущий runner-подход.
  - Нужен для сравнения standard model vs modified head.

Отключенные/закомментированные измерения:

- `Inference_ModHead`
  - Старый прямой путь modified head.

- `Inference_ModHeadSeq`
  - Задел под `ByDirectSequence`.
  - Сейчас benchmark-атрибут закомментирован, а метод явно бросает `NotSupportedException`, пока не переработана поддержка sequence-input.

Параметры:

- `Batch`: `1`, `4`.
- `_InferenceBackend`: `TensorRt`, `Cuda`.
- `FixedModel`: `false`, `true`.
- `EnableFp16`: сохранен в комментариях.
- `EnableBf16`: сохранен в комментариях.
- `DML`: сохранен в комментариях как возможный backend.

Входные изображения:

- `Images/Image_Text1.jpg`
- `Images/Image_Text2.jpg`
- `Images/Image_Text3.jpg`
- `Images/Image_Text4.jpg`
- `Images/frame_000046.png`
- `Images/frame_000062.png`
- `Images/frame_000198.png`

## Runtime Providers / ORT Settings

Эти измерения проходят через параметры и setup-код benchmark-классов.

Измеряемые направления:

- `Cuda` execution provider.
- `TensorRt` execution provider.
- TensorRT cache / engine build behavior.
- TensorRT `FP16` режим.
- TensorRT `BF16` режим.
- TensorRT timing cache.
- TensorRT engine cache.
- Влияние `IoBinding` на OCR recognition.

Сохраненные, но не активные направления:

- `DML` backend.
- CPU fallback / CPU baseline.
- Сравнение нескольких вариантов Mat-to-ORT конвертации.
- Sequence-input путь `ByDirectSequence`.

## Общие Диагностические Метрики

BenchmarkDotNet сейчас настроен на:

- время выполнения;
- аллокации памяти через `MemoryDiagnoser`;
- threading diagnostics через `ThreadingDiagnoser`;
- группировку результатов по `BenchmarkCategory`;
- запуск отдельных методов через `BenchmarkRunner.Run<Benchmark>()`.

## Очередь На Добавление

Новые пункты для будущих измерений добавлять сюда, по группам:

- OCR Recognition:
- OCR Input Conversion:
- Detection / OBB:
- Runtime Providers:
- Memory / Allocation:
- End-to-end Runner Pipeline:
