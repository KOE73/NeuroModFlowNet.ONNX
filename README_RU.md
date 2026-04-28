# NeuroModFlowNet.ONNX

## Описание

**NeuroModFlowNet.ONNX** — это модульная высокопроизводительная библиотека поверх [ONNX Runtime](https://onnxruntime.ai/) для задач компьютерного зрения. Основной фокус проекта — real-time инференс, где важны предсказуемые задержки, минимальные аллокации и строгая типизация всего пути от входного изображения до готового результата.

Библиотека закрывает типовой, но обычно очень ручной сценарий работы с ONNX Runtime: подготовить входные данные, выполнить инференс, разобрать сырые выходные тензоры и получить удобные доменные структуры. Вместо того чтобы в каждом проекте заново писать обвязку вокруг `InferenceSession`, `OrtValue`, layout-тензоров и постобработки, здесь эти части разнесены по отдельным слоям с понятными зонами ответственности.

Основной поток данных выглядит так:

```text
Input data -> Converter -> OnnxRuntimeContext / Runner -> Extractor -> Typed result
```

## Основные компоненты

### Core

`Core` содержит базовый runtime-слой. Центральная сущность здесь — `OnnxRuntimeContext`: он владеет `InferenceSession`, настройками backend, метаданными модели и ресурсами, которые нужны для выполнения инференса.

В этом же слое находятся типизированные конфигурации execution providers, включая CUDA и TensorRT. Это позволяет держать настройки рядом с кодом, который реально создает runtime-контекст, а не размазывать их по строковым параметрам в вызывающем коде.

### Inputs

`Inputs` отвечает за подготовку входных данных. Конвертеры принимают внешние данные, например `OpenCvSharp.Mat` или список `Mat`, и приводят их к формату, который ожидает модель.

Внутри этого слоя находятся:

- универсальные контракты конвертеров;
- реализации для `BGR`, `NCHW`, positive normalization и PaddleOCR;
- низкоуровневые алгоритмы подготовки данных в `Inputs/Algorithms`;
- helper-классы для проверки и автоподбора алгоритмов.

Для горячего пути важна идея: выбор стратегии и все дорогие проверки должны происходить заранее, а не внутри обработки каждого кадра.

### Runners

`Runners` связывает входной конвертер, runtime-контекст и extractor в исполняемый pipeline. Базовый контракт — `IRunner<TIn, TOut>`.

Runner получает входные данные, подготавливает буферы, запускает ONNX Runtime и возвращает типизированный результат. Для image-based сценариев используется `ImageRunner`, а общая стратегия выполнения вынесена в `StrategyRunner`.

### Outputs

`Outputs` отвечает за интерпретацию выходов модели. Здесь находятся extractor-классы, структуры для packed layout, структуры результата и фабрики.

Для YOLO выходы разложены по семействам:

- `Box`;
- `Cls`;
- `Obb`;
- `Pose`;
- `Seg`.

Для каждого семейства, где это применимо, используются папки:

- `InnerData` — структуры, соответствующие layout выходного тензора и пригодные для `MemoryMarshal.Cast`;
- `OutData` — независимые структуры результата;
- `Extractors` — логика преобразования сырых выходов модели;
- `Factories` — готовые factory-методы и metadata-based создание runner.

Отдельно поддерживаются выходы PaddleOCR: detection, recognition и связанные задачи.

### Visualizer

`NeuroModFlowNet.ONNX.Visualizer` — отдельный проект для отрисовки результатов инференса. Визуализация намеренно вынесена из core-библиотеки, чтобы runtime-код не зависел от UI/debug-задач.

Здесь находятся painter-классы для YOLO Box, OBB, Seg, Pose, Cls и вспомогательные методы для работы с `Mat`.

### Diagnostics

`NeuroModFlowNet.ONNX.Diagnostics` — отдельный проект для отображения информации о runtime-контексте и структуре модели. Например, через `WriteInfo(...)` можно вывести backend, имя модели, входы, выходы и при необходимости выбранные metadata.

Этот код вынесен отдельно по той же причине: в рабочей библиотеке не должно быть лишнего форматирования и console/UI-зависимостей, но во время разработки нужна быстрая и наглядная диагностика.

## Поддерживаемые сценарии

Основные рабочие сценарии сейчас связаны с YOLO-моделями:

- detection / box;
- oriented bounding boxes;
- segmentation;
- pose;
- classification.

Также в проекте есть поддержка PaddleOCR-задач, включая detection и recognition.

Для YOLO26 есть отдельные документы:

- `src/NeuroModFlowNet.ONNX/Doc/Yolo26_2.md` — реально существующие варианты экспортов и выходов моделей;
- `src/NeuroModFlowNet.ONNX/Doc/Factores.md` — сверка текущих factory/extractor-реализаций с этими реальными вариантами.

## Runtime-требования

Библиотека таргетит `.NET 10` и использует ONNX Runtime как движок инференса. Core-проект подключает `Microsoft.ML.OnnxRuntime.Gpu`, когда `UseGpu=true`, и переключается на CPU-пакет ONNX Runtime, если GPU-поддержка отключена на этапе сборки.

Для CPU-инференса NVIDIA-библиотеки не нужны. Для CUDA и TensorRT нужны native-зависимости NVIDIA, совместимые с ONNX Runtime GPU package, который используется в конечном приложении.

Для запуска через CUDA должны быть установлены:

- совместимый NVIDIA driver;
- runtime-библиотеки CUDA Toolkit;
- runtime-библиотеки cuDNN;
- `Microsoft.ML.OnnxRuntime.Gpu` или `Microsoft.ML.OnnxRuntime.Gpu.Windows` в конечном приложении.

Для запуска через TensorRT нужно всё, что требуется для CUDA, плюс TensorRT-библиотеки, собранные под ту же CUDA major/minor линейку. TensorRT также чувствительнее к поддержке shape и operator coverage модели, а первый запуск может занять больше времени из-за сборки engine.

На Windows директории с native DLL должны быть доступны до создания `InferenceSession`. Можно прописать их в системный `PATH`, либо добавить пути в `App.config` исполняемого проекта и вызвать `OnnxRuntimePathHelper.InitFromConfig()` до создания runtime-контекста:

```xml
<appSettings>
  <add key="CudaBinPath" value="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin" />
  <add key="CudnnBinPath" value="C:\cuDNN\9.10.2.21_cuda12\bin" />
  <add key="TrtLibPath" value="C:\TensorRT\TensorRT-10.12.0.36-cuda12.9\lib" />
</appSettings>
```

Эти значения являются путями локальной машины, а не переносимыми настройками проекта. Их нужно держать в соответствии с версиями CUDA/cuDNN/TensorRT, установленными на целевой машине.

При выборе `InferenceBackend.TensorRt` без пользовательской настройки provider в `OnnxRuntimeContext` по умолчанию включается TensorRT engine cache. Путь cache определяется в таком порядке:

- `TrtConfigDefaults.CustomEngineCachePath`;
- environment variable `NMFN_ONNX_TRT_CACHE_PATH`;
- `%LOCALAPPDATA%\NeuroModFlowNet\TRT_Cache`.

## Factory и автоматический выбор

Для типовых сочетаний входов, выходов и precision существуют factory-методы. Они позволяют явно создать runner под конкретный сценарий, например для FP32, FP16, BGR direct input или конкретного YOLO-задачного семейства.

Там, где это возможно, используется metadata-based создание через `CreateRunner(...)`. Такой путь читает метаданные модели и выбирает подходящий converter и extractor автоматически. Это особенно удобно для набора близких ONNX-экспортов, где отличия находятся в типе входного тензора, precision выходов или форме результата.

## Производительность

Проект проектируется вокруг нескольких практических ограничений real-time систем.

Во-первых, горячий путь должен быть коротким и предсказуемым. Подбор алгоритма, анализ metadata, создание буферов и выбор extractor должны происходить при инициализации. Во время обработки кадра runner должен выполнять уже выбранную последовательность действий.

Во-вторых, библиотека старается избегать лишних аллокаций. Для этого используются заранее подготовленные буферы, `Span<T>`, `ReadOnlySpan<T>`, packed-структуры и прямое отображение данных там, где формат модели это позволяет.

В-третьих, когда модель принимает `U8 BGR` input, можно существенно упростить CPU-side preprocessing. В таком сценарии часть подготовки переносится внутрь ONNX-графа, а внешний код работает ближе к zero-copy режиму.

## Структура репозитория

- `src/NeuroModFlowNet.ONNX` — основная библиотека.
- `src/NeuroModFlowNet.ONNX.Visualizer` — визуализация результатов инференса.
- `src/NeuroModFlowNet.ONNX.Diagnostics` — диагностический вывод информации о моделях и runtime-контексте.
- `models/` — общие локальные model assets для samples, labs и ручных проверок библиотеки. Каждая сеть лежит в своей папке, например `models/yolo26s/yolo26s__640_b1_fp32.onnx`.
- `ModelPrepare/` — служебные инструменты для общих model assets. `Preparation/` содержит скрипты скачивания и экспорта, а `HuggingFace/` содержит скрипты публикации готовых ONNX-артефактов во внешнее хранилище моделей.
- `samples/NeuroModFlowNet.ONNX.Demo.Dashboard` — demo dashboard для параллельной обработки кадров разными pipeline-обработчиками.
- `samples/NeuroModFlowNet.ONNX.Demo.Assets` — общие assets для demo-проектов.
- `labs/NeuroModFlowNet.ONNX.Lab` — интерактивные эксперименты и сценарии с камерой.
- `labs/NeuroModFlowNet.ONNX.Lab.Algorithms` — наглядная проверка и демонстрация input algorithms.
- `labs/NeuroModFlowNet.ONNX.Bench` — benchmark-проект и измерения.
- `tests/NeuroModFlowNet.ONNX.Tests` — unit и integration tests.
- `tools/NeuroModFlowNet.ONNX.Tools` — CLI и helper-библиотека для просмотра структуры ONNX-графа, встройки preprocessing-голов и регистрации `.onnx` в системе.

## Tools

`NeuroModFlowNet.ONNX.Tools` — отдельный CLI и helper-библиотека для работы с ONNX-моделями на уровне графа.

Текущий набор инструментов поддерживает:

- удобный просмотр структуры модели: входы, выходы, промежуточные значения, тензоры, атрибуты и узлы графа;
- настраиваемый вывод графа: первые/последние узлы, полный вывод, фильтры и полные имена входов/выходов;
- встройку preprocessing-голов, например конвертацию byte BGR input в FP16 или FP32 RGB tensor с нормализацией;
- регистрацию в системе для интеграции с `.onnx` файлами и command aliases.

Сценарий с preprocessing-головами полезен, когда простую подготовку входа можно перенести внутрь модели. Тогда runtime pipeline может принимать данные ближе к исходному формату, меньше заниматься CPU-side preprocessing и сохранять converter/extractor путь проще.

Это один из важных путей оптимизации проекта: модель с правильно встроенной входной головой убирает повторяющийся per-frame conversion code из приложения и позволяет ONNX Runtime получать данные в формате, который внешнему pipeline дешевле подготовить.

Примеры команд находятся в `tools/NeuroModFlowNet.ONNX.Tools/README.md`.

## Где смотреть дальше

- `src/NeuroModFlowNet.ONNX/Doc/Architecture_ru.md` — архитектурные заметки.
- `src/NeuroModFlowNet.ONNX/Doc/ConvertersTranslators_ru.md` — исторические и технические заметки по converters/algorithms.
- `src/NeuroModFlowNet.ONNX/Doc/Yolo26_2.md` — реальные YOLO26 экспорты.
- `src/NeuroModFlowNet.ONNX/Doc/Factores.md` — сверка factory/extractor-реализаций.
- `labs/NeuroModFlowNet.ONNX.Bench/Doc/ReadMe.md` — список текущих benchmark-измерений.

## Лицензия

Исходный код NeuroModFlowNet.ONNX распространяется по лицензии Apache License 2.0. См. `LICENSE`.

Сторонние зависимости сохраняют свои собственные лицензии. CUDA, cuDNN и TensorRT не распространяются вместе с этим репозиторием и должны устанавливаться отдельно в соответствии с лицензионными условиями NVIDIA.

## Лицензия моделей

Веса моделей и подготовленные model artifacts не покрываются Apache-2.0 лицензией исходного кода этого репозитория, если это не указано явно.

YOLO-веса и подготовленные YOLO-derived artifacts предоставляются по AGPL-3.0, в соответствии с лицензионными условиями Ultralytics YOLO.

Автор предоставляет эти веса "as is", без каких-либо гарантий. Пользователь сам отвечает за то, чтобы его использование соответствовало лицензиям:

- Ultralytics YOLO;
- базовой/pretrained модели, если она использовалась;
- training dataset;
- deployment/runtime-библиотек.
