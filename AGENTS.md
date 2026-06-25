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
- `docs/` —  common documentation.


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
*   **Абстракция расположения памяти (CPU/GPU)**: В пайплайнах не должно быть прямого ручного управления перемещением памяти (например, низкоуровневых CUDA-копирований). Все переносы между CPU и GPU осуществляются косвенно через ONNX-модели (например, динамический шаг `Identity` с соответствующим Execution Provider) или через механизмы самого ONNX Runtime. Это обеспечивает переносимость пайплайна между CUDA, TensorRT, DirectML (AMD/Intel) и CPU.
*   **Контроль и минимизация неявных копирований**: Неявные переносы данных (когда ONNX Runtime автоматически и скрытно копирует тензор с CPU на GPU при несоответствии провайдеров) крайне негативно влияют на производительность. Разработчики должны явно готовить данные (шаг `UploadToDevice`). В пайплайне должны присутствовать механизмы валидации/диагностики, контролирующие размещение тензоров и запрещающие неявный перенос.
*   **Fallback запрещен**: Для VM, pipeline, ONNX-операций, сервисов инференса и backend-specific путей недопустимы скрытые или автоматические fallback-режимы. Если явно выбранный backend, opset, оператор, layout, тип данных или provider path недоступен, код должен завершаться ошибкой, а не переходить на CPU, другой Execution Provider, другую реализацию или приближенную семантику. Это assembler-style контракт: либо выбранная команда работает быстро и точно в заявленном режиме, либо не работает вообще.

## VM / Pipeline решения
- **Ops работают с готовыми значениями**: `Op_*` инструкции не должны знать внутренности конкретной нейросети, batching service, ONNX output names или decode layout. Они читают и пишут именованные VM-регистры и преобразуют уже готовые типизированные значения.
- **Вызов внешнего сервиса отделен от Ops**: inference, MPEG, OCR recognition, batching, native output access и decode должны жить в service/endpoint слое. VM-инструкция вызова сервиса должна быть тонкой: взять входные регистры, вызвать service, положить typed results в выходные регистры.
- **Frontend один, kernel/backend выбирается строго на init**: для ONNX-операций допустимы разные реализации под opset/backend/provider, но выбор делается при создании/инициализации операции и фиксируется. Во время `ExecuteAsync` нельзя пробовать другие реализации или provider-ы.
- **Обратные координаты через именованные регистры**: графические операции над `OrtValue` пишут результат `OrtValue` и, если запрошено, отдельный регистр с данными обратного преобразования (`ICoordinateBackTransform`). В регистр кладутся данные/тип transform, а не delegate/замыкание.
- **Одна команда применения обратного преобразования**: `Op_Map_Coordinates` применяет один named back-transform к одному payload-регистру и пишет новый payload-регистр. Разный код crop/resize/rotate/perspective живет в типах `ICoordinateBackTransform`; знание структуры `YoloBox`, `YoloObb`, `OcrQuadRegion`, `List<T>`, массивов и batch-result живет в payload mapper-ах.
- **ONNX-типы не загрязнять VM-интерфейсами**: типы из `NeuroModFlowNet.ONNX` используются как payload напрямую. VM/coordinate mapping знания размещаются в `NeuroModFlowNet.Pipeline` и `NeuroModFlowNet.Pipeline.ONNX`, а не добавляются в модельные структуры без отдельной необходимости.
- **Политика формы должна быть явной**: если transform может изменить геометрический класс фигуры, нельзя молча врать исходным типом. Для будущих shape policy использовать явные режимы вроде `PreserveShape`, `BoundingBox`, `BoundingOBB`, `Quad`, `RejectIfShapeChanges`.
- **Runtime-generated ONNX operators**: операции вроде Crop/Resize строят маленькие ONNX-графы под конкретный контракт. Для TensorRT у таких runtime operator graphs нельзя использовать общий engine cache, если cache identity не включает точную форму/семантику графа.
- **Resize contract**: внешний tensor-контракт Resize остается NHWC `UInt8`, но backend-friendly graph может внутри выполнять `Cast`, `Transpose NHWC->NCHW`, `Resize`, `Transpose NCHW->NHWC`, `Cast`. Это деталь реализации, а не изменение VM-регистрового контракта.
- **Lifetime GPU intermediates**: если инструкция выделяет intermediate `OrtValue` через provider allocator/session, VM run context с этими значениями должен освобождаться раньше долгоживущей инструкции и ее allocator/session. Тесты обязаны повторять этот порядок владения.

## Код-стайл и Оформление
*   **Регионы (#region)**: Обязательны для больших файлов (от 10 методов и более). Используются для логической группировки методов по смыслу.
*   **Версия языка**: Только самые последние фичи C# (C# 12/13+). Максимальное использование современных возможностей языка.

## Comment Style
- Write comments only when they explain intent, ownership, lifecycle, performance behavior, data layout, non-obvious constraints, or public contracts. Do not restate what the member name already says.
- For important public or architectural XML documentation, use the project bilingual block style:
  - Start with `EN:` and explain the behavior in clear English.
  - Add a blank XML-doc line.
  - Add `RU:` and provide the same meaning in Russian.
- Use `<remarks>` for constraints, lifecycle rules, performance notes, resource ownership, validation timing, or other details that are important but secondary to the main summary.
- Use `<param>` and `<typeparam>` only when the parameter meaning is not obvious from the signature or when it carries a contract. Do not fill parameter docs with generic text.
- Preserve meaningful existing comments. Do not shorten comments that describe non-obvious behavior, compatibility decisions, memory ownership, or hot-path reasoning.
- Keep implementation comments short and local. Prefer XML documentation for API contracts and short `//` comments only for complex blocks inside a method.

## Процесс работы
- Любые изменения фундаментальных интерфесов или базовой архитектуры требуют предварительного согласования.
- Periodically remind the user to update `docs/usage/` when project usage patterns change, but do not create or update usage documentation unless explicitly requested.

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
- В библиотечных проектах (`src/*`) не используй `Console.Write*`. Диагностика должна идти через trace, callback, logger abstraction или explicit debug hook. В `labs/*` временный `Console.Write*` допустим.
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
