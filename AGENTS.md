# Agent Guidance

## Purpose
NeuroModFlowNet.ONNX is a modular high-performance C# framework for ONNX inference in computer vision pipelines.

Primary workloads:
- using YOLO models:
  - detection
  - segmentation
  - classification
  - other YOLO-based CV tasks
- using PaddleOCR models:
  - text detection and recognition
  
Core goals: 
- low-overhead inference execution
- strongly typed postprocessing pipeline

## Runner Workflow
1. Convert external input data into the model input format.
2. Execute inference.
3. Transform raw model outputs into convenient strongly typed result structures.


## Project Structure
- `src/NeuroModFlowNet.ONNX/` — core library.

- `src/NeuroModFlowNet.ONNX.Visualizer/` — isolated project for rendering inference results.
- `samples/` — library usage examples. After core or public API changes, keep samples aligned and update them when needed.
- `labs/` — experiments, benchmarking, and architectural exploration. Do not treat this folder as the main architectural source of truth unless explicitly requested.
- `tests/` —  unit and integration tests for pipeline correctness and regression safety.
- `doc/` —  common documentation.


## Folder structure by models
- InnerData - Packed structures matching the model output and allowing MemoryMarshal.Cast<float, T>(data) or MemoryMarshal.Cast<Half, T>(data) 
- OutData - structures for storing independent format inference results
- Extractors - classes for extracting inference results
- Factories - classes for creating extractors


## Factories
Static methods for creating typical extractors.
CreateRunner - static method for automatic selection of converter and extractor based on model metadata.

## Архитектурные фундаменты (НЕ МЕНЯТЬ)
*   **Универсальные раннеры**: Весь инференс завязан на базовый интерфейс `IRunner<TIn, TOut>`. Никаких специализированных маркерных интерфейсов вроде `IImageRunner`.


## Код-стайл и Оформление
*   **Регионы (#region)**: Обязательны для больших файлов (от 10 методов и более). Используются для логической группировки методов по смыслу.
*   **Версия языка**: Только самые последние фичи C# (C# 12/13+). Максимальное использование современных возможностей языка.

## Процесс работы
- Любые изменения фундаментальных интерфесов или базовой архитектуры требуют предварительного согласования.

# Правила именования
При именовании используются следующие префиксы приставки корнии суффиуксы
Типы для классов и структур связанных с инференсом
  - FP32 - большими буквами. для float
  - FP16 - большими буквами. для System.Half

Для классов и структур связанных с инференсом на вход. 
  - Single - для одного элемента
  - List - для списка элементов

Для классов и структур связанных с инференсом на выход.
  - Single - для одного результата
  - List - для списка результатов

# Правила создания кода
- Один класс один файл. Название файла совпадает с названием класса.
- Исключение возможно для взаимосвязанных record
- Не разделяй классы по разным файлам и не объединяй их в один файл без явной необходимости или отдельного запроса.
- Не добавляй, не удаляй и не переупорядочивай `using`, если это не требуется для компиляции изменённого кода.
- Не экономь на названиях пременных, не используй сокращения (не b а batch).
- Используй в первую очередь auto-properties там, где это не вредит производительности, ясности кода и модели владения данными.
- Используй современные возможности C# и .NET, если они реально упрощают код, улучшают безопасность или производительность.
- Используй `Span<>`, `ReadOnlySpan<>`, `Memory<>` и related low-allocation patterns там, где это оправдано по hot path, производительности и читаемости.
- Не используй название Anchors для количества элементов после нейросети, используй ItemCount
- Не выполняй stylistic cleanup вне области запрошенных изменений.
- порядок полей в структурах с атрибутом StructLayout менять нельзя
- Не убирай readonly из структур и полей структур, если они не меняются.
- Не сокращай смысл комментариев, если они несут важную информацию, если в них описанны важные, не очевидные вещи.

# Рефакторинг
При рефакторинге проверяется все решение, в том числе и демо проекты, Labs и т.д.

# NeuroModFlowNet.ONNX.Visualizer
Вся визуализация находится в `src/NeuroModFlowNet.ONNX.Visualizer`.

Все классы, связанные с визуализацией, должны оставаться в этом проекте.  
Не допускается перенос визуализаторов, рендереров и вспомогательных классов отображения в другие части решения.

## Validation
- После изменений проверяй всё решение целиком.
- Если изменено ядро, проверь `samples`, `labs` и `tests`.
- При изменениях в hot path отдельно обращай внимание на аллокации и layout данных.

# Модели для программы
Код надо согласовывать между собой
- ModelPrepare/MODEL_NAMING_CONVENTION.md - правила именования
- samples/NeuroModFlowNet.ONNX.Demo.Assets/ общий проект загрузки подготовленных моделей
- ModelPrepare/Preparation - загрузка исходников и подготовка
- ModelPrepare/HuggingFace - выгрузка в хранилище 
- Модели тут https://huggingface.co/NeuroModFlowNet/NeuroModFlowNet-ONNX-Demo-Models
