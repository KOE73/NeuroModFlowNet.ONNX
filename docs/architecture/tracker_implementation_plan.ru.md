# План реализации: трекинг-стадия, per-source config, ordered critical section

Статус: **MVP реализован**. Проектные решения и обоснования: `docs/architecture/tracker_and_source_config.ru.md`
(читать перед стартом). Этот файл — пошаговый план, достаточный для реализации без домысливания архитектуры.

Solution: `NeuroModFlowNet.ONNX.slnx`. Целевой TFM: `net10.0` (как у остальных проектов).
Стиль/договоры: корневой `AGENTS.md` (регионы, «no fallback», «VM как ассемблер», «ops работают с готовыми значениями»).

Порядок фаз обязателен: каждая фаза компилируется и покрывается тестами до перехода к следующей. Не смешивать фазы в
одном коммите.

---

## Фаза 0. Новый проект `NeuroModFlowNet.CV`

**Цель**: чистая CPU-библиотека CV-алгоритмов для пайплайна, без ONNX и без VM.

1. Создать `src/NeuroModFlowNet.CV/NeuroModFlowNet.CV.csproj`, `net10.0`, nullable enable, те же свойства, что у
   `src/NeuroModFlowNet.Pipeline` (свериться с `Directory.Build.props`).
2. Зависимости: **никаких** на `NeuroModFlowNet.ONNX`, `NeuroModFlowNet.Pipeline`, ONNX Runtime. Разрешён только
   `OpenCvSharp`, если реально нужен `RectF`/`Point2f`; **предпочтительно** свои лёгкие структуры геометрии, чтобы не
   тянуть OpenCV в нейтральную либу. Решить на месте: если геометрия минимальна — свой `struct RectF`/`Obb`.
3. Добавить проект в `NeuroModFlowNet.ONNX.slnx` в папку `/src/`.
4. Acceptance: `dotnet build` solution зелёный, новый проект собран, ни на что тяжёлое не ссылается.

---

## Фаза 1. Нейтральные типы + контракт трекера (в `NeuroModFlowNet.CV`)

Namespace: `NeuroModFlowNet.CV.Tracking`.

1. `TrackDetection` — вход трекера (нейтральная детекция):
   - геометрия: `float X, Y, W, H, Angle` (OBB; axis-aligned = `Angle 0`);
   - `int ClassId`, `float Score`;
   - `int SourceIndex` — индекс в исходном массиве детекций кадра (для переассоциации downstream).
   - readonly struct/record struct.
2. `TrackedObject` — результат трекера (чистый, без OCR/словарей):
   - `int TrackId`, `int ConfirmedTrackId` (или `-1`), `bool IsConfirmed`;
   - `int Age`, `int MissedFrames`;
   - геометрия текущего бокса (те же поля, что в `TrackDetection`);
   - `int SourceDetectionIndex` (`-1`, если в этом кадре detection пропущен);
   - траектория `IReadOnlyList<PointF>` `Path` (ограничение длины — параметр трекера).
3. `FrameContext` — нейтральный контекст кадра: `long RunId`, `DateTimeOffset Timestamp`, `long? SourceFrameId`.
   (Заполняется из `VmRunIdentity` на стороне `Op_Track`.)
4. `ITracker` — **не generic**:
   ```csharp
   public interface ITracker
   {
       IReadOnlyList<TrackedObject> Process(
           IReadOnlyList<TrackDetection> detections,
           in FrameContext frame,
           in TrackerFrameOptions options);
   }
   ```
5. `ITrackerDebugSnapshot` (read-only, для задней двери отладки):
   `int ConfirmedTrackCount`, `int NextId`, `IReadOnlyList<TrackedObject> ActiveTracks`.
6. Acceptance: типы компилируются, покрыты минимальными тестами конструирования/равенства.

---

## Фаза 2. Порт алгоритма IoU (в `NeuroModFlowNet.CV`)

Источник для переноса семантики (**внешний старый проект**, только читать как референс):
`C:/Proect/Git/NN/NeuroModFlowNet/src/NeuroModFlowNet.Core/Common/Algoritms/IoUTrackerVector.cs`,
`.../Common/Data/TrackedBox.cs`.

Перенести **только алгоритм**, без `TrackedBox`-god-object и без OCR-полей.

1. `IouTracker : ITracker, ITrackerDebugSnapshot`. Конфиг конструктора (структурный, на init):
   `IouThreshold`, `MinAge`, `MaxMissedFrames`, `MaxTrailLength`, `startBoxIoU` (containment), плюс приём зон
   `StartZone`/`EndZone` **на каждый кадр** (см. ниже), а не в конструктор.
2. Семантика (сохранить из оригинала):
   - матчинг по IoU **только внутри одного `ClassId`**;
   - новый трек создаётся только если detection проходит `StartZone` через `ContainmentRatio >= startBoxIoU`;
   - подтверждение трека после `Age >= MinAge` → выставляется `ConfirmedTrackId`;
   - при пропуске detection геометрия трека помечается пропущенной, `MissedFrames++`; удаление после
     `MaxMissedFrames`;
   - обрезка траектории по `MaxTrailLength`.
3. **Hot-reload зон**: `StartZone`/`EndZone` — параметры метода `Process` (или отдельного `Configure(zones)` перед
   каждым кадром), НЕ поля конструктора. Так живая подмена конфига работает.
4. Векторизацию/`RectF.Zero`-сбросы из оригинала переносить по мере надобности; приоритет — корректность семантики,
   затем производительность.
5. Acceptance: юнит-тесты в `tests/NeuroModFlowNet.Pipeline.Tests` (или новый `tests/NeuroModFlowNet.CV.Tests`):
   - создание трека только внутри `StartZone`;
   - подтверждение после `MinAge` кадров;
   - потеря трека после `MaxMissedFrames`;
   - раздельность по `ClassId`;
   - детерминизм при повторе одной последовательности.

---

## Фаза 3. Ordered critical section (S2) в `NeuroModFlowNet.Pipeline`

**Цель**: упорядоченная эксклюзивная секция, покрывающая тело инструкции, с сохранением drop-tolerance.

1. Расширить `VmSyncGate` (или добавить `VmOrderedCriticalSection` рядом) режимом «ordered mutex»:
   - `AcquireInOrderAsync(runId, ct)` — ждёт своей очереди по `RunId` **и** держит эксклюзию до `Release`;
   - `Release(runId)` — отпускает секцию и продвигает `nextRunIdToRelease`, освобождая следующий ожидающий run;
   - `NotifyRunClosed(runId)` — как сейчас: если run закрылся/упал не дойдя или не отдав секцию, секция продвигается,
     как будто он прошёл (не виснем на дропнутых кадрах).
   - Инвариант: одновременно секцию держит максимум один run; порядок входа строго по возрастанию `RunId` (пропуская
     закрытые).
2. Способ применения к инструкции: **только LOCK-префикс в VM loop**.
   - Добавить в `IOp`/`OpDescriptor` необязательное имя ordered critical section, например `SyncGate`.
   - Цикл `VmProgram.ExecuteLoopInternalAsync` для инструкции с `SyncGate` оборачивает шаг:
     `AcquireInOrderAsync → stepExecutor(...) → Release` в `finally`.
   - Sync должен быть виден в descriptor/trace как часть VM-контракта, а не спрятан внутри конкретной инструкции.
   - `Op_Track` не должен сам брать lock/синхронизацию внутри `ExecuteAsync`: это смешивает stateful-доменную команду
     и политику исполнения. Исключение возможно только после отдельного согласования, если правка VM loop окажется
     технически невозможной.
3. `OrderedSyncInstruction` (barrier) **оставить** для случаев «только порядок».
4. Acceptance (в `tests/NeuroModFlowNet.Pipeline.Tests`):
   - при `MaxInFlight > 1` секция исполняется строго по одному run одновременно (проверка счётчиком concurrency);
   - порядок входа по `RunId` строгий;
   - падение/отмена run в секции не вешает следующий (drop-tolerance);
   - без sync-gate поведение инструкций не изменилось (регресс).

---

## Фаза 4. `Op_Track` + адаптер в `NeuroModFlowNet.Pipeline.ONNX`

Namespace: `NeuroModFlowNet.Pipeline.ONNX`.

1. Адаптер границы `Op_DetectionsFromYoloObb` (и/или `...FromYoloBox`):
   - reads: регистр `YoloObb[]` (или `YoloBox[]`) в tracking-space;
   - writes: регистр `TrackDetection[]` (проставить `SourceIndex = i`, `ClassId`, `Score`, геометрию);
   - тонкий, без состояния; по образцу существующих payload-mapper-ов.
2. `Op_Track : OpBase` — тонкая VM-команда по образцу `Model_Inference`:
   - конструктор: `inputKey` (`TrackDetection[]`), `outputKey` (`TrackedObject[]`), `trackerStateKey`,
     `syncGate` (имя секции S2), ключи per-source config (зоны/пороги) или ссылку на config-key,
     фабрика/структурный конфиг трекера;
   - `ExecuteAsync`:
     1. НЕ берёт lock: ordered critical section держит VM loop по `OpDescriptor.SyncGate`;
     2. `var tracker = context.GlobalMemory.GetOrAdd(trackerStateKey, _ => new IouTracker(structuralConfig));`
     3. прочитать зоны/пороги из `GlobalMemory` config-ключей (hot-reload на каждый кадр);
     4. собрать `FrameContext` из `context.Identity`;
     5. `var tracks = tracker.Process(detections, frame);`
     6. `context.Set(outputKey, tracks);`
   - op **не знает** IoU/StartBox; всё в `IouTracker`.
   - `OpDescriptor`: объявить reads/writes; при наличии scope в `VarRequirement` (см. фаза 5, опц.) — задекларировать
     и global config/state.
3. Ссылки проекта: `Pipeline.ONNX` уже видит `Pipeline` и `ONNX`; добавить ref на `NeuroModFlowNet.CV`.
4. Acceptance (в `tests/NeuroModFlowNet.Pipeline.ONNX.Tests`):
   - end-to-end мини-программа: `YoloObb[]` → адаптер → `Op_Track` → `TrackedObject[]`;
   - при `MaxInFlight > 1` и sync-gate треки корректны и порядок-детерминированы;
   - `SourceDetectionIndex` корректно связывает трек с исходным `YoloObb`.

---

## Фаза 5. Per-source config seeding + debug back-door (опционально, но желательно)

1. Хост-путь: показать в sample/тесте, как в `VmGlobalMemory` при инициализации контроллера кладутся per-source
   значения (зоны как `RectF`, пороги) плоскими ключами, а `Op_Track`/`IouTracker` читают их по ключу.
2. `VmGlobalMemory`: добавить read-only перечисление/snapshot содержимого (сейчас нет `Keys`/enumerate) для задней
   двери отладки. Не ломать существующее API.
3. (Опц.) `VarRequirement` + scope `RunLocal`/`Global`, чтобы дескрипторы честно декларировали global-доступ.
4. (Опц.) Учесть teardown: решить, диспозить ли `GlobalMemory` в `VmController.DisposeAsync` (сейчас нет). Для
   managed-трекера не требуется; зафиксировать решение комментарием.
5. Acceptance: тест на seed + чтение config из global memory; тест snapshot-а.

---

## Фаза 6. Документация и sample

1. Обновить `docs/architecture/tracker_and_source_config.ru.md` (снять «код ещё не написан», проставить реальные имена
   типов/файлов, если разошлись).
2. Короткий раздел в `src/NeuroModFlowNet.Pipeline.ONNX/docs/Architecture.ru.md`: трекинг-стадия, адаптер, S2.
3. При наличии — минимальный sample-пайплайн с трекером. В текущем MVP не добавлялся: покрытие сделано тестовыми
   мини-программами VM.
4. Обновить `docs/TODO.md`, если там есть релевантные пункты.

---

## Границы (повторить перед стартом — НЕ делать)

- НЕ делать `Op_Track` generic и НЕ вводить `object -> object`/рефлексию/schema-driven детекции.
- НЕ тащить ONNX/`YoloBox` в `NeuroModFlowNet.CV`.
- НЕ переносить `TrackedBox`-god-object, `OcrResults`, словарь-индексатор.
- НЕ добавлять fallback (backend/opset/провайдер/алгоритм) — при неподдержке падать (корневой AGENTS.md).
- НЕ класть per-source config/state в поля инструкций — только ключи + `VmGlobalMemory`.

## Definition of Done

- Solution собирается; все тесты зелёные.
- `Op_Track` domain-blind; вся семантика трекинга — в `NeuroModFlowNet.CV`.
- При `MaxInFlight > 1` трекинг корректен и детерминирован (S2).
- Config/state per-source вынесены в `VmGlobalMemory`; один набор инструкций работает на разные камеры.
- Существующие доки не противоречат реализации.
