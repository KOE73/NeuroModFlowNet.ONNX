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
- **OrtValue inference endpoint для GPU pipeline**: новый сервисный путь не должен впихиваться в старый `RunnerResource<Mat,...>`. VM передает в сервис один уже подготовленный `OrtValue` из регистра; сервис сам собирает batch из заявок разных VM через ONNX-операции, запускает модель, декодирует native outputs и возвращает typed результат только для этой заявки.
- **Результат сервиса как массив**: даже если конкретная сеть обычно возвращает один объект, VM-регистр сервисного результата должен быть рассчитан на `T[]`/массив typed значений. Это убирает частные случаи для детекторов, OCR, classification/top-k и будущих моделей с несколькими результатами.
- **Batch assembly без CPU round-trip**: если входные `OrtValue` уже находятся на GPU, объединение в batch должно выполняться ONNX Runtime graph-ом на выбранном backend/provider и отдавать tensor в provider memory для следующего model run. CPU/device fallback или скрытое копирование вместо выбранного backend недопустимы.
- **Fixed batch padding в сервисе**: если модель собрана на фиксированный batch больше числа реальных VM-заявок, padding-слоты не должны быть видимы VM. Для hot path допустимо привязать padding-слот к уже существующему реальному input `OrtValue`, потому что создание отдельного black/zero tensor является дополнительной подготовительной операцией; decoder обязан возвращать только результаты реальных заявок.
- **Frontend один, kernel/backend выбирается строго на init**: для ONNX-операций допустимы разные реализации под opset/backend/provider, но выбор делается при создании/инициализации операции и фиксируется. Во время `ExecuteAsync` нельзя пробовать другие реализации или provider-ы.
- **Обратные координаты через именованные регистры**: графические операции над `OrtValue` пишут результат `OrtValue` и, если запрошено, отдельный регистр с данными обратного преобразования (`ICoordinateBackTransform`). В регистр кладутся данные/тип transform, а не delegate/замыкание.
- **Одна команда применения обратного преобразования**: `Op_Map_Coordinates` применяет один named back-transform к одному payload-регистру и пишет новый payload-регистр. Разный код crop/resize/rotate/perspective живет в типах `ICoordinateBackTransform`; знание структуры `YoloBox`, `YoloObb`, `OcrQuadRegion`, `List<T>`, массивов и batch-result живет в payload mapper-ах.
- **ONNX-типы не загрязнять VM-интерфейсами**: типы из `NeuroModFlowNet.ONNX` используются как payload напрямую. VM/coordinate mapping знания размещаются в `NeuroModFlowNet.Pipeline` и `NeuroModFlowNet.Pipeline.ONNX`, а не добавляются в модельные структуры без отдельной необходимости.
- **Политика формы должна быть явной**: если transform может изменить геометрический класс фигуры, нельзя молча врать исходным типом. Для будущих shape policy использовать явные режимы вроде `PreserveShape`, `BoundingBox`, `BoundingOBB`, `Quad`, `RejectIfShapeChanges`.
- **Runtime-generated ONNX operators**: операции вроде Crop/Resize строят маленькие ONNX-графы под конкретный контракт. Для TensorRT у таких runtime operator graphs нельзя использовать общий engine cache, если cache identity не включает точную форму/семантику графа.
- **Geometry tensor ops именуются по типу и layout**: для операций, которые двигают элементы/плоскости и не интерпретируют цвет (`Crop`, `Resize`, `PadResize`, `Rotate90`, `Perspective`, `Undistort`), публичное имя должно включать element type и layout, например `Op_Onnx_Crop_FP16_NCHW`, `Op_Onnx_PadResize_U8_NHWC`. `RGB/BGR` в такие имена не включать, потому что geometry не меняет и не читает цветовую семантику.
- **Color/normalize macro-ops именуются полным форматом**: если операция меняет цветовой порядок, layout, element type или диапазон значений, это должно быть видно в имени, например `Op_Onnx_BgrU8Hwc_To_RgbFP16Nchw_Div255`. Такие комплексные hot-path операции допустимы рядом с RISC-инструкциями, потому что уменьшают количество ORT calls, промежуточных tensor-ов и синхронизаций.
- **Основной внутренний image tensor target**: OpenCV `BGR U8 HWC/NHWC` считать bridge-форматом на границе с `Mat`. Для model pipeline в первую очередь развивать `NCHW` варианты (`FP16`, `FP32`) и только затем дополнительные layout/type варианты по реальной необходимости.
- **OCR ROI prepare CISC-op**: для hot path после text OBB допускается специализированная CISC-команда вида `Op_Onnx_ExtractObbToPaddleRec_FP32_NCHW`: вход `RGB FP32 NCHW 0..1` из промежуточной GPU-картинки, payload с OBB/quad регионами, выход `FP32 NCHW` batch `[N, 3, targetHeight, targetWidth]` уже в Paddle Rec нормализации. `targetWidth`, `targetHeight`, `paddingPixels`, `paddingScale` и алгоритм построения ONNX-графа являются параметрами, а не константами. Такая команда не заменяет RISC `Crop/Perspective/PadResize/Normalize`, а собирает проверенные ONNX-примитивы в один граф ради меньшего числа ORT calls, tensor intermediates и синхронизаций.
- **OCR ROI prepare algorithms измеряются явно**: варианты вроде `PerRegionGridSampleConcat`, `BatchedGridSample`, будущие OBB-specific affine/axis-aligned paths выбираются параметром и benchmark-ятся отдельно. Production-команда не должна сама переключаться на другой алгоритм при ошибке backend/opset/provider; выбранный вариант фиксируется на init и при неподдержке падает. Это не fallback, а явный выбор реализации.
- **OCR ROI fixed capacity**: `maxRoiCount` является параметром команды/benchmark, а не константой. Он нужен для TensorRT и других backend-ов, где выгоднее или надежнее фиксировать форму выхода `[maxRoiCount, 3, H, W]`. Фактическое число ROI пишется отдельно (`actualCountOutputKey`), лишние ROI обрабатываются только по явной политике `Fail` или `Truncate`; скрытое добивание, silent drop или runtime-переключение алгоритма недопустимы.
- **PaddleOCR Rec OrtValue service path**: recognition service принимает уже подготовленный `OrtValue` batch `[N, 3, 48, W]`, объединяет request batches через ONNX concat, запускает `rec.onnx`, декодирует native output в `PaddleOCRRecExtractor.OcrResult[]` и режет результат обратно по фактическому числу ROI каждого VM-запроса. Старый `RunnerResource<List<Mat>, ...>` не использовать для этого GPU hot path.
- **PaddleOCR Rec backend choice**: strict CUDA для текущего `models/paddleocr/languages/english/rec.onnx` не является подтвержденным рабочим путем: ONNX Runtime назначает часть nodes на CPU EP, а fallback запрещен. Для GPU integration сейчас явно выбирать TensorRT для Rec service. Это не fallback: backend сервиса задается явно на init.
- **BF16 отложен**: `BF16` достаточно узкий для текущего этапа и пока не реализуется в geometry/preprocess ops. Если позже появится реальная модель/backend-необходимость, добавлять отдельные `BF16` инструкции и тесты только после доказанной поддержки ONNX Runtime Execution Provider, без fallback.
- **Backend kernel cache**: runtime-generated ONNX operator kernels готовить lazy для операций реально используемой VM-программы, но кэшировать глобально между программами по полному ключу: operation kind, input/output shape, element type, layout, interpolation/mode/pad values, backend, opset/backend variant и provider options. Не готовить все комбинации заранее.
- **Kernel cache ownership**: kernel cache хранит только immutable `OnnxModel`/`InferenceSession`. `RunOptions` и `IoBinding` остаются отдельными на каждый `OnnxExecutionContext`, потому что это mutable native state с привязками к конкретным `OrtValue`. Любой параметр конструктора операции, который влияет на graph bytes или output semantics (`cropRect`, `targetSize`, `rotate mode`, `quad points`, distortion coefficients, padding, stride, normalization values), обязан входить в cache key.
- **Почему так**: VM рассматривается как assembler. Формат в имени geometry-команд нужен, чтобы в trace сразу видеть смешение `FP16`/`FP32` и `NCHW`/`NHWC`. `RGB/BGR` исключен из geometry-имен, потому что `Crop/Resize/PadResize/Rotate/Perspective` работают с осями и элементами, а не с цветом. Fused macro-ops оставлены для типичных hot path, где один ONNX graph быстрее цепочки мелких VM-инструкций. Lazy cache выбран потому, что полная матрица `operation * type * layout * backend * shape` взрывается комбинаторно, но одинаковые kernels между программами должны переиспользоваться.
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
  - BF16 - большими буквами. для BFloat16/bfloat16 tensor element type

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
- В visual/operation tests генерируемые изображения должны быть достаточно крупными для ручной проверки: ориентир 500-1000 px по основной стороне. Не добавлять новые "микро-картинки" 2x3, 8x6 и т.п. как единственный visual artifact; маленькие tensors допустимы только для чисто численных unit-тестов без визуального вывода.

## ONNX visual/integration artifacts

Default visual artifact root is resolved by tests: `NMFN_VISUAL_ROOT`, then `visualArtifactsRoot` from
`appsettings.test.json`, otherwise `%TEMP%\NeuroModFlowNet.Pipeline.ONNX.Tests\VisualTestResults\`. Set
`NMFN_VISUAL_ROOT=R:\Tests` when the physical `R:\Tests` root is required.

| Test / Ops | Test images | Artifact folder | Outputs to inspect | Purpose |
| --- | --- | --- | --- | --- |
| `Op_Onnx_RealImageVisualTests` for Crop / Resize / PadResize / Rotate90 / Perspective / Undistort | `images/for_tests.png`, `images/for_tests_perspective.png`, `images/for_tests_perspective_real_0.png` | `<artifactRoot>\Onnx\<TypeFolder>\<OperationName>\` | `*-actual.png`, perspective quad previews where applicable | Manual visual validation for individual ONNX image operators on real images. |
| `ImgTextToObbOrtValueIntegrationTests` | `images/for_tests.png`, `images/for_tests_perspective_real_0.png` | `<artifactRoot>\Onnx\UnknownType\ImgTextToObbOrtValueIntegration\` | `*_source_obb.png`, `*_perspective_obb.png`, `*_model_640_padresize_obb.png`, `*_source_obb_crop_###.png`, `*_perspective_obb_crop_###.png`, `*_model_obb_crop_###.png`, `*_source_recognition.txt`, `*_perspective_recognition.txt` | Full integration path: source image -> optional perspective -> PadResize -> img-text-to-obb service -> coordinate remap -> OBB ROI prepare -> PaddleOCR Rec OrtValue service -> typed text results. |
| `Op_Onnx_ExtractObbToPaddleRecTests` | generated large visual test tensors, plus real-image integration through `ImgTextToObbOrtValueIntegrationTests` | `<artifactRoot>\Onnx\FP32_NCHW\Op_Onnx_PaddleRecRoiPrepare\` or integration folder depending on test | prepared Paddle Rec ROI batch images / numerical asserts | Validate OBB/ROI extraction, padding, target size, fixed capacity and Paddle Rec normalization. |

# Модели для программы
Код надо согласовывать между собой
- ModelPrepare/MODEL_NAMING_CONVENTION.md - правила именования
- samples/NeuroModFlowNet.ONNX.Demo.Assets/ общий проект загрузки подготовленных моделей
- ModelPrepare/Preparation - загрузка исходников и подготовка
- ModelPrepare/HuggingFace - выгрузка в хранилище 
- Модели тут https://huggingface.co/NeuroModFlowNet/NeuroModFlowNet-ONNX-Demo-Models
