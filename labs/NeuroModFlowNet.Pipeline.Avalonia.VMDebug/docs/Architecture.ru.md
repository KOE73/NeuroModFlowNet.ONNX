# NeuroModFlowNet.Pipeline.Avalonia.VMDebug Architecture

Назначение проекта - визуальная отладка VM-парадигмы. Это не витрина сетей и не перенос старого Avalonia UI.

Главная граница: источник существует снаружи, VM получает входные переменные и возвращает выбранные output-переменные.
В текущей лаборатории вход один:

```text
configured source
  -> source.frame
  -> PipelineVmController
  -> PipelineRunInputs
  -> VM instructions
  -> PipelineRunOutput
  -> trace/watch UI
```

Каждый кадр, который был принят в обработку, получает `RunId`. Если VM занята, новый кадр не ставится в FIFO
очередь, а отбрасывается до старта transaction. Это соответствует будущей realtime-модели: обработка не должна догонять
старые кадры пачками.

## Почему preview идет через PipelineVmController

Даже простой вывод первоисточника уже проверяет базовый контракт:

- transaction имеет строковые ключи;
- `source.frame` живет только в пределах одного accepted run;
- временные `Mat`/ROI освобождаются через transaction ownership;
- UI получает отдельные `Mat` и сам отвечает за их Dispose;
- ONNX-модели живут в `PipelineModelResources`, а не внутри visual controls;
- `PipelineRunTrace` показывает время каждой инструкции и снимок переменных после нее;
- memory placement в trace нужен для контроля главной цели: минимизировать переносы данных обратно в CPU.

## Почему нет старого UI

Старые панели выбора сетей, несколько источников, thumbnails и отдельный PaddleDet preview убраны. В этой лаборатории
левая панель управляет VM-программой, центр показывает один результат, правая панель готовится под динамические watch
выводы из переменных VM.
