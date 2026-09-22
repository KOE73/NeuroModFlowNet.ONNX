# Трекинг-стадия и per-source config в VM-пайплайне — проектные решения

Статус: **MVP реализован**: нейтральный CV-трекер, S2 LOCK-префикс, `Op_Track`, YOLO-адаптеры и регрессионные тесты.
Область: `NeuroModFlowNet.Pipeline`, `NeuroModFlowNet.Pipeline.ONNX`, новый `NeuroModFlowNet.CV`.

Документ фиксирует решения из проектного разговора. План реализации отдельно:
`docs/architecture/tracker_implementation_plan.ru.md`.

---

## 1. Задача

Перенести семантику трекера из старого проекта (`IoUTracker` + `TrackedBox` + `TrackingBoxModule`) в новый
VM-пайплайн так, чтобы:

- трекер был **stateful VM-стадией**, а не ONNX-операцией и не частью `Model_*`;
- один и тот же **набор инструкций** (рецепт) работал на много камер; различия физического мира (перспектива,
  дисторсия, `StartZone`/`EndZone`, пороги) не были вшиты в инструкции;
- координатный договор вокруг трекера был явным (в каком пространстве живут входные боксы и зоны);
- сам трекер был **сменяемым**: сегодня IoU, завтра ByteTrack/Kalman, без переписывания VM-команды;
- не возникло over-инжиниринга ради абстрактной «любой трекер».

---

## 2. Три категории данных

Ключ ко всему — разделять не «state vs code», а три вещи:

| Категория | Пример | Изменчивость | Где живёт |
|---|---|---|---|
| Структура программы | порядок инструкций | одна на все камеры | код builder-а |
| **Per-source config** | `StartZone`, `EndZone`, перспектива, дисторсия, пороги | разная по камерам, стабильна годами | `VmGlobalMemory` (данные) |
| **Per-source state** | внутреннее состояние трекера | меняется каждый кадр | `VmGlobalMemory` (объект) |

Первая версия проекта ошибочно клала config в поля op-а. Это возвращает связку «экземпляр программы = камера».
Правильно — **и config, и state косвенно через память; op держит только ключи**.

---

## 3. Решение G2 — state трекера в `VmGlobalMemory`

`VmGlobalMemory` уже принадлежит контроллеру источника (см. `VmController`: по умолчанию каждый контроллер создаёт
свою `VmGlobalMemory`). Значит она **и есть та самая поперечная per-source корзина** — изоляция по источнику даётся
автоматически, без ключа по `SourceId`.

- Трекер (объект `state + code` целиком) кладётся через `GlobalMemory.GetOrAdd("tracker.<name>", factory)`.
- Внутри одного контроллера ключ = имя стадии (`SourceId` неявный).
- НЕ используем отдельный store, НЕ используем `IVmResource` — трекер это не shared external service, а per-source
  вычислительное состояние. Плодить сущности незачем.

Известный нюанс жизненного цикла: `VmController.DisposeAsync` диспозит инструкции программы, но **не** трогает
`GlobalMemory`. Для чисто managed-состояния трекера это безвредно; если у трекера появятся unmanaged-ресурсы —
предусмотреть явный teardown global memory (см. план, фаза 5).

---

## 4. Решение по per-source config

Config настраивается **один раз** при пуско-наладке камеры и потом живёт годами; при настройке гранулярность
непринципиальна. Поэтому — просто и плоско:

- Плоские нейтральные значения (`RectF`, коэффициенты, пороги) кладутся в `VmGlobalMemory` при инициализации
  контроллера (host seed из конфигурационного файла камеры).
- `Op_*` держит **строковый ключ** параметра, читает значение на исполнении. Никаких blob-церемоний.
- Config иммутабелен в пределах жизни источника → конкурентное чтение при `MaxInFlight > 1` безопасно без синхронизации.
- Bonus — **hot-reload**: параметр можно атомарно подменить в памяти. Для значений, которые обязаны меняться вместе
  (перспектива + дисторсия + зоны одной калибровки), подменять их одним immutable под-объектом, чтобы не поймать
  несогласованную смесь. Для независимых ручек — по одному ключу.

Почему в общей памяти, а не в спец-памяти: отдельный config-store даёт только жёсткое enforcement иммутабельности,
которое и так гарантируется immutable-значениями. Второй тип хранилища не окупается.

---

## 5. Решение S2 — ordered critical section вместо barrier

### Находка

Текущий `OrderedSyncInstruction` — это **barrier, а не mutex**, и он **не покрывает тело следующей инструкции**.
Разбор `VmSyncGate.ArriveAndWaitAsync` + `TryReleaseReadyRuns`:

- run N приходит, он `nextRunIdToRelease` → его waiter завершается **сразу**, `nextRunIdToRelease := N+1`. Run N идёт в
  трекер.
- run N+1 приходит → он теперь next → отпускается **сразу**. Тоже идёт в трекер.

Итог: гейт упорядочивает **момент отпускания**, но N и N+1 могут оказаться в трекере **одновременно**; при накоплении
waiter-ов они отпускаются плотным циклом и вместе рвутся к любому внутреннему локу op-а → строгий порядок входа тоже не
гарантирован. Для трекинга, где важен временной порядок кадров (age, missed frames, траектория), этого мало.

### Решение

Ввести **упорядоченную критическую секцию** («LOCK-префикс» на инструкции — ложится на философию «VM как ассемблер»):

- Любой инструкции можно задать необязательное имя sync-gate.
- Цикл VM оборачивает `ExecuteAsync` такой инструкции в: дождаться своей очереди по `RunId` → выполнить эксклюзивно →
  отпустить следующий run.
- Это даёт **и порядок, и взаимоисключение, покрывая тело op-а** одним механизмом; op остаётся sync-agnostic; факт
  синхронизации виден в дескрипторе/trace (explicit).
- **Drop-tolerance сохраняется**: если run упал/отменён, не отдав секцию, следующий run должен пройти. Механизм
  `NotifyRunClosed` уже делает это для barrier — расширить семантику «отпускания» на «отдал секцию / закрылся».

`OrderedSyncInstruction` (barrier) остаётся для случаев, где нужен только порядок без эксклюзии.

Синхронизация нужна только при `MaxInFlight > 1` (широкий конвейер + последовательный хвост-трекер). При `MaxInFlight = 1`
всё серийно и гейт не нужен.

---

## 6. Трекер как универсальный компонент

### 6.1. Трекер — голая CPU-математика, без ONNX

Трекер работает на геометрии (IoU, `ClassId`, containment), которая уже посчитана и лежит в CPU. Он **не знает** про
`YoloBox`, ONNX Runtime, provider memory. Значит:

- Трекер и его типы — в **новом нейтральном проекте `NeuroModFlowNet.CV`** (старт CV-библиотеки для пайплайна).
- Никакой зависимости от `NeuroModFlowNet.ONNX`. Никакого имени вроде `IoUFrameTracker` в ONNX-слое.

### 6.2. Нейтральные типы вместо generic-протаскивания

По поводу аналогии с `Model_Inference<TInput, TOutput>` — «и да, и нет». Да, паттерн «тонкая команда + интерфейс»
правильный. Но у инференса выходные типы **объективно разные** (`YoloBox`/`YoloObb`/`OcrResult`) и endpoint
доменно-специфичен, поэтому там оправдан generic. Трекер наоборот — **универсальная математика, которая должна
нормализовать вход к нейтральной геометрии**. Полное сокрытие типов через generic иллюзорно (типы всё равно текут).
Поэтому:

- Определяем нейтральные типы: `TrackDetection` (геометрия бокса/OBB + `ClassId` + `Score` + индекс исходной детекции)
  и `TrackedObject` (`TrackId`, `ConfirmedTrackId`, `IsConfirmed`, `Age`, `MissedFrames`, траектория, геометрия,
  `SourceDetectionIndex`).
- `ITracker` — **не generic**:
  `Process(IReadOnlyList<TrackDetection>, in FrameContext, in TrackerFrameOptions) -> IReadOnlyList<TrackedObject>`.
- Знание про `YoloBox` изолируется в **тонком адаптере на границе** (`YoloBox[]`/`YoloObb[]` → `TrackDetection[]`),
  который живёт в `Pipeline.ONNX`.
- `TrackedObject.SourceDetectionIndex` сохраняет связь трека с исходной типизированной детекцией, чтобы downstream
  (crop/OCR/overlay) переассоциировал трек с нужным `YoloObb` из исходного регистра.

### 6.3. Что переносим из старого проекта, что режем

- **Алгоритм `IoUTrackerVector`** (IoU-матчинг внутри `ClassId`, `StartBox` через `ContainmentRatio >= StartBoxIoU`,
  подтверждение по `Age >= MinAge`, сброс в `MissedFrames`) — **переносим и развиваем в `NeuroModFlowNet.CV`**. Это
  чистая переносимая геометрия.
- **`TrackedBox` (god-object)** — **НЕ переносим как есть**. `TrackId/ConfirmedTrackId/Age/MissedFrames/Path/IsConfirmed`
  оставляем в чистом `TrackedObject`. `OcrResults` и произвольный словарь-индексатор — это доменная грязь и источник
  связанности; downstream-обогащение навешивается отдельными регистрами/mapper-ами, а не полями трека (принцип
  «VM/ONNX-типы не загрязнять» из корневого AGENTS.md).

### 6.4. Размещение по сборкам

| Тип | Сборка | Зависимости |
|---|---|---|
| `TrackDetection`, `TrackedObject`, `FrameContext`, `TrackerFrameOptions`, `ITracker`, `IouTracker`, box/obb-математика | **`NeuroModFlowNet.CV`** (новый) | нет (чистый CPU) |
| ordered critical section (расширение sync) | `NeuroModFlowNet.Pipeline` | VM-ядро |
| `Op_Track` (тонкая VM-команда) + адаптер `Op_DetectionsFromYolo*` | `NeuroModFlowNet.Pipeline.ONNX` | CV + Pipeline + ONNX |

`Op_Track` сам по себе ONNX-агностичен; временно размещается в `Pipeline.ONNX` (единственные потребители сейчас там),
позже может переехать в нейтральный VM+CV glue. `NeuroModFlowNet.Pipeline` остаётся чистым от домена.

---

## 7. Координатный договор

- Вход `Op_Track` и зоны (`StartZone`/`EndZone`) должны быть в **одном** координатном пространстве. Это явный договор
  через имена регистров и параметр `coordinateSpace` (только для trace/диагностики, не логика).
- Если tracking-space ≠ source-space, tracks маппятся обратно отдельной командой `Op_Map_Coordinates` (механизм
  `ICoordinateBackTransform` уже есть).

Схема:

```
det.boxes.model
  -> Op_Map_Coordinates(... -> det.boxes.tracking)      // при необходимости
  -> Op_DetectionsFromYolo(det.boxes.tracking -> track.det.in)   // адаптер YoloObb[] -> TrackDetection[]
  -> Op_Track(track.det.in -> track.out) [syncGate="tracker"]    // S2, state в GlobalMemory
  -> Op_Map_Coordinates(track.out -> track.out.source)  // опционально обратно в source
  -> OCR / overlay / counter (через SourceDetectionIndex)
```

---

## 8. Границы: чего НЕ делаем (чтобы не было over-инжиниринга)

- НЕ делаем команду агностичной к **типам**: никаких `object -> object`, рефлексии, config-driven schema детекций.
- НЕ вводим канонический тип детекции с адаптерами «под всё» заранее. `TrackDetection` — минимальный нейтральный тип
  ровно под трекинг, а не универсальная шина.
- НЕ строим generic `Op_Track<TIn,TOut>`: нормализация к нейтральным типам + адаптер на границе честнее и проще.
- НЕ переносим `TrackedBox`-god-object и его расширяемый словарь.

---

## 9. Текущее состояние реализации

- Нейтральные типы и `IouTracker` размещены в `src/NeuroModFlowNet.CV/Tracking/`.
- S2 реализован как `OpDescriptor.SyncGate`: `VmProgram` оборачивает тело инструкции в
  `VmSyncGate.AcquireInOrderAsync(...)` / `Release(...)`; `Op_Track` lock не берёт.
- `VmSyncGateRegistry` запоминает закрытые `RunId` и replay-ит их новым gates, чтобы поздно созданная секция не ждала
  кадр, который уже упал до этой точки.
- `Op_DetectionsFromYoloObb`, `Op_DetectionsFromYoloBox` и `Op_Track` размещены в
  `src/NeuroModFlowNet.Pipeline.ONNX/Tracking/`.
- `StartZone`/`EndZone` читаются через `TrackerFrameOptions` из `VmGlobalMemory` по ключам на каждом кадре.
- `SourceDetectionIndex` берётся из `TrackDetection.SourceIndex`, то есть сохраняет связь с исходной typed-детекцией.
- `VmGlobalMemory` получил read-only `Keys` и `Snapshot()` для debug back-door.

## 10. Оставшиеся мелочи

- OBB vs axis-aligned: `TrackDetection` несёт OBB (`cx,cy,w,h,angle`), axis-aligned = `angle 0`. Достаточно ли IoU по
  OBB на первом этапе или начать с AABB — уточнить в бенче.
- `VarRequirement` scope (`RunLocal` / `Global`): расширить, чтобы дескриптор честно декларировал global-чтения
  config/state (explicit-налог). Опционально в первой версии.
- Debug back-door: `VmGlobalMemory` сейчас не умеет перечислять содержимое — добавить read-only snapshot + маленький
  `ITrackerDebugSnapshot` (confirmed count, next id, треки с age/missed).
