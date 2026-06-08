# NeuroModFlowNet.Pipeline.ONNX Architecture

Этот проект является адаптерным слоем между новой VM-парадигмой и текущей библиотекой `NeuroModFlowNet.ONNX`.

## Разделение ответственности

```text
NeuroModFlowNet.Pipeline
  transaction, RunId, instructions, jumps, sync gates, global memory

NeuroModFlowNet.Pipeline.ONNX
  ONNX resources, batching, YOLO/Paddle доменные команды

NeuroModFlowNet.ONNX
  OnnxRuntimeContext, IRunner<TIn,TOut>, converters, extractors
```

## Почему команда не дергает модель напрямую

Команда `Model_YoloObb` является шагом VM. Она читает input из transaction, отправляет typed request в общий
`YoloObbBatchedResource`, ждет свой результат и пишет compact result обратно в transaction.

Ресурс владеет runner/model context и решает batching policy:

```text
MaxBatchSize
MaxWaitTime
MaxPendingRequests
```

Это явно фиксирует компромисс throughput / latency / memory.

Для текущих batch-1 моделей есть `OnnxRunnerResource` и `Model_YoloSingleDetection`. Это не отменяет batching
модель, а дает честный путь поверх существующих `IRunner<TIn,TOut>` без искусственного изменения контрактов.

`ReloadableOnnxRunnerResource` нужен для PaddleOCR Rec: UI может менять batch/width, persistent shapes пересоздаются,
а VM-программа и имя ресурса остаются стабильными.

## Что возвращается в CPU

В transaction возвращается только доменный результат: список OBB/Box с координатами, классом и confidence. Большие raw
тензоры и временные ONNX payload не должны выходить наружу без отдельной причины.
