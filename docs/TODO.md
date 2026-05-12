# TODO: Планы 

## TODO: анализатор локальной машины

Нужна отдельная библиотека/утилита, которая сможет помочь пользователю до запуска модели:

- определить GPU и compute capability;
- прочитать версию NVIDIA driver через `nvidia-smi`;
- найти установленный CUDA Toolkit и проверить `bin`;
- найти cuDNN DLL и определить major-линейку;
- найти TensorRT и проверить CUDA-линейку;
- определить версию `Microsoft.ML.OnnxRuntime.Gpu` в приложении;
- сверить найденное с `doc/GpuRuntimeTestedConfigurations_ru.md`;
- дать понятный совет: что уже подходит, чего не хватает, какой путь добавить в `App.local.config`;
- опционально выполнить минимальную авто-проверку CUDA Execution Provider на простом ONNX-графе;
- опционально выполнить проверку TensorRT с учетом времени сборки engine cache.

Вывод такой утилиты должен быть рассчитан на пользователя, который просто хочет попробовать demo: минимум терминологии, максимум конкретных действий.

## TODO: OCR ROI realtime performance

Нужно отдельно измерить и оптимизировать новый библиотечный путь:

```text
YoloObb -> OcrQuadRegion -> coordinate mapper -> NaiveTextRegionExtractor -> PaddleOCR Rec
```

Что проверить:

- сравнить старый путь через `RotatedRect` и новый путь через `OcrQuadRegion`;
- измерить цену `YoloObbOcrRegionMapper.MapToSourceRegions(...)` отдельно от `WarpPerspective` / `Resize`;
- проверить порог `stackalloc` для временных точек и регионов;
- для больших батчей проверить `ArrayPool<T>` или отдельный reusable workspace вместо обычных аллокаций;
- проверить вариант обработки координат как плоского `[N, 2]` массива через `MemoryMarshal.Cast<Point2f, float>`;
- проверить `TensorSpan<float>` / broadcasting-style преобразования для массовых offset/scale операций;
- оставить простой `Span` loop как baseline, потому что для десятков/сотен OCR boxes он может быть быстрее tensor-обвязки.

Экспериментальный код для проверки tensor-подхода:

```csharp
using System;
using System.Numerics.Tensors;

float[] xyData =
{
    10, 20,   // XY[0]
    30, 40,   // XY[1]
    50, 60    // XY[2]
};

float[] offsetData = { 5, -2 };     // offsetX, offsetY
float[] scaleData = { 2, 10 };      // scaleX, scaleY
float uniformScale = 0.5f;          // одинаковый scale для X и Y

Tensor<float> xy = Tensor.Create(xyData, [3, 2]);
Tensor<float> offsetXY = Tensor.Create(offsetData, [2]);
Tensor<float> scaleXY = Tensor.Create(scaleData, [2]);

Tensor<float> offsetBroadcasted = Tensor.Broadcast<float>(offsetXY, [3, 2]);
Tensor<float> scaleBroadcasted = Tensor.Broadcast<float>(scaleXY, [3, 2]);

Tensor<float> result = (xy + offsetBroadcasted) * scaleBroadcasted;
Tensor<float> result2 = (xy + offsetXY) * uniformScale;

float[] output = new float[3 * 2];
result.FlattenTo(output);

Console.WriteLine("Per-axis scale:");
Console.WriteLine(string.Join(", ", output));

float[] output2 = new float[3 * 2];
result2.FlattenTo(output2);

Console.WriteLine("Per-axis scale + uniform scale:");
Console.WriteLine(string.Join(", ", output2));

float[] tempData = new float[3 * 2];
float[] resultData = new float[3 * 2];

TensorSpan<float> temp = new TensorSpan<float>(tempData, [3, 2]);
TensorSpan<float> destination = new TensorSpan<float>(resultData, [3, 2]);

Tensor.Add(xy, offsetXY, temp);
Tensor.Multiply(temp, uniformScale, destination);

Console.WriteLine(string.Join(", ", resultData));
```
