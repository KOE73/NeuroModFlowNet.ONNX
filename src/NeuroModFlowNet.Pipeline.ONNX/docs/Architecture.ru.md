# NeuroModFlowNet.Pipeline.ONNX Architecture

Этот проект является ONNX-слоем для VM-пайплайна. Он отделяет VM-инструкции от деталей конкретной модели, ONNX Runtime
binding, batching, native outputs и decode layout.

## Разделение ответственности

```text
NeuroModFlowNet.Pipeline
  transaction, RunId, instructions, jumps, sync gates, global memory

NeuroModFlowNet.Pipeline.ONNX
  ONNX tensor operators, inference endpoints, batching, service decode, coordinate payload mapping

NeuroModFlowNet.ONNX
  OnnxModel, OnnxExecutionContext, model metadata, existing runners/extractors
```

## Почему VM-команда не знает модель напрямую

Канонический шаг инференса в VM - это тонкая команда `Model_Inference<TInput, TOutput>` или
`Model_OrtValueInference<TOutput>`. Команда знает только:

- какой register прочитать;
- какой endpoint вызвать;
- в какой register положить typed result.

Endpoint/resource владеет моделью, ONNX Runtime context, batching policy, native output access и decode. Это позволяет
не вшивать в `Op_*` инструкции имена выходов ONNX-модели, YOLO/Paddle layout, правила batch slicing или recognition
decode.

Для GPU hot path VM передает в endpoint уже подготовленный `OrtValue`. Сервис может собрать batch через ONNX Runtime
graph, выполнить модель на выбранном backend/provider и вернуть только typed result для конкретной VM-заявки.

## Batching policy

Общий batching resource фиксирует:

```text
MaxBatchSize
MaxWaitTime
MaxPendingRequests
```

Один worker внутри ресурса владеет runner/session binding state. Многие VM executions могут отправлять заявки
параллельно, но сам model run внутри ресурса остается сериализованным и возвращает результат каждой заявке по ее
очередному месту в batch.

## Legacy runner wrappers

`Model_Yolo*`, `Model_Paddle*`, `OnnxRunnerResource<TInput,TOutput>` и `ReloadableOnnxRunnerResource<TInput,TOutput>`
остаются совместимым путем для существующих Mat-based `IRunner<TIn,TOut>` активов. Они не являются основным контрактом
GPU-resident VM hot path.

`ReloadableOnnxRunnerResource` нужен для старого PaddleOCR Rec сценария: UI может менять batch/width, persistent shapes
пересоздаются, а VM-программа и имя ресурса остаются стабильными.

## Что возвращается в CPU

VM-visible результат сервиса должен быть typed массивом: например `YoloObb[]`, `YoloBox[]` или
`PaddleOCRRecExtractor.OcrResult[]`. Даже если конкретная сеть обычно возвращает один объект, register результата
сервиса проектируется как массив.

Большие raw tensors и временные ONNX payload не должны выходить в CPU без отдельной причины. Между tensor-операторами и
следующей моделью промежуточные `OrtValue` должны оставаться в provider memory, если инструкция не помечена как
финальная.

## Tracking stage

Трекинг не является ONNX-операцией и не знает layout конкретной модели. Нейтральная CPU-математика находится в
`NeuroModFlowNet.CV.Tracking`: `TrackDetection`, `TrackedObject`, `ITracker`, `IouTracker`.

ONNX-слой содержит только границу:

- `Op_DetectionsFromYoloObb` и `Op_DetectionsFromYoloBox` нормализуют typed YOLO-результаты в `TrackDetection[]`.
- `Op_Track` читает `TrackDetection[]`, берёт per-source state/config из `VmGlobalMemory` и пишет `TrackedObject[]`.
- `Op_Track` не берёт lock внутри `ExecuteAsync`; упорядоченная эксклюзия задаётся через `OpDescriptor.SyncGate`, а
  `VmProgram` исполняет такую инструкцию как LOCK-префикс: `AcquireInOrderAsync -> ExecuteAsync -> Release`.

Входные детекции и tracking-зоны должны быть в одном координатном пространстве. Если нужно перейти из model-space в
source-space или обратно, это делается отдельной стадией coordinate mapping до или после трекера.
