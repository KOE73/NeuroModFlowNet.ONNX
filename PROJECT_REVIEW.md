# PROJECT_REVIEW

## Контекст ревизии
- Режим работы: маленькими шагами, без удержания всего проекта в контексте.
- Правило на текущую серию шагов: код не менять без отдельного разрешения, накапливать устойчивые выводы здесь.

## Прогресс ревизии
- Пройдено 7 рабочих шагов ревизии.
- Покрытые крупные зоны:
  - базовые контракты раннера;
  - active `Converters`;
  - active `Extractors` base/PaddleOCR/YOLO Box;
  - active `CreateRunner`/factory orchestration;
  - нижний lifecycle слой `OnnxModel`;
  - dynamic-shape active path `PaddleOCRDet`;
  - dynamic-shape/batch path `YoloSeg`;
  - быстрый usage-check по `tests`/`samples`.
- Грубая оценка покрытия базовой архитектурной ревизии: около 75-85%.
- Что ещё остаётся как логичные белые пятна:
  - `YoloCls` / `YoloPose` / `YoloObb` на уровне конкретных extractor-base реализаций;
  - `labs/` как место, где могут всплывать реальные usage-path для спорных API;
  - полноценная верификация найденных рисков запуском тестов/демо-сценариев после будущих исправлений.

## Шаг 1 - Базовые контракты раннера (`Abstract` + связанная реализация)

### Что проверено
- Прочитан `README.md` для сверки заявленной архитектуры.
- Прочитаны файлы:
  - `src/NeuroModFlowNet.ONNX/Abstract/IRunner.cs`
  - `src/NeuroModFlowNet.ONNX/Abstract/IInputConverter.cs`
  - `src/NeuroModFlowNet.ONNX/Abstract/IResultExtractor.cs`
  - `src/NeuroModFlowNet.ONNX/Abstract/ResultExtractorBase.cs`
  - `src/NeuroModFlowNet.ONNX/Abstract/IAdapterInfo.cs`
  - `src/NeuroModFlowNet.ONNX/Abstract/IImageAdapter.cs`
  - `src/NeuroModFlowNet.ONNX/Abstract/IImageInfo.cs`
  - `src/NeuroModFlowNet.ONNX/Inference/Base/Runner.cs`
  - `src/NeuroModFlowNet.ONNX/Inference/Base/StrategyRunner.cs`
  - `src/NeuroModFlowNet.ONNX/Inference/Base/ImageRunner.cs`
  - `src/NeuroModFlowNet.ONNX/Factories/PaddleOCRFactory.cs`
- Точечно проверено использование `IAdapterInfo` поиском по `src/NeuroModFlowNet.ONNX`.
- Точечно проверен найденный legacy-след:
  - `src/NeuroModFlowNet.ONNX/Obsolete/Adapters/PaddleOCR/PaddleOCRDetAdapter.cs`

### Что понял
- Архитектурное правило из `AGENTS.md` про базовый интерфейс `IRunner<TIn, TOut>` соблюдается: основная generic-модель раннера действительно построена вокруг него.
- Текущий рабочий pipeline в базовом раннере выглядит так:
  1. `Adapter.SetModel(model)`
  2. `Extractor.SetModel(model)`
  3. на каждом `Predict`: `Adapter.Prepare(input)` -> `Model.Run()` -> `Extractor.Extract()` -> `Model.Cleanup()`
- `ImageRunner<...>` не вводит отдельный специализированный runner-интерфейс, а только добавляет `IImageInfo` поверх общего `IRunner<TIn, TOut>`. Это согласуется с ограничением "не вводить специализированные маркерные интерфейсы".
- `ResultExtractorBase<TOut>` задаёт двухфазную инициализацию `Init()` + `Check()`, что похоже на нормализованный шаблон для extractor-ов.
- В активной части проекта фабрики сейчас в основном возвращают жёстко типизированные `ImageRunner<...>` и не опираются на `IAdapterInfo`.

### Найденные риски
- Риск 1. Несогласованность контракта `IRunner.As<T>()` с реализацией.
  - В `src/NeuroModFlowNet.ONNX/Abstract/IRunner.cs:14-17` комментарий говорит, что поиск capability идёт по входному и выходному пути.
  - В `src/NeuroModFlowNet.ONNX/Inference/Base/StrategyRunner.cs:42-49` реальная реализация ищет ещё и в самом раннере (`this as T`).
  - Риск: потребитель API, читающий только интерфейс/документацию, получает неполное описание поведения. Это не ломает выполнение, но повышает шанс неверных ожиданий и скрытой зависимости на поведение `StrategyRunner`, а не на контракт.
- Риск 2. `IAdapterInfo` выглядит как "висящий" контракт в активной архитектуре.
  - Интерфейс объявлен в `src/NeuroModFlowNet.ONNX/Abstract/IAdapterInfo.cs:5-12`.
  - По точечному поиску в production-части `src/NeuroModFlowNet.ONNX` не найдено его использования в текущих фабриках/раннерах.
  - Найденная реализация относится к obsolete-коду: `src/NeuroModFlowNet.ONNX/Obsolete/Adapters/PaddleOCR/PaddleOCRDetAdapter.cs:12-14`.
  - Риск: контракт создаёт ложный сигнал о поддерживаемом сценарии introspection/select-by-input-type, хотя в текущем активном коде этот механизм, похоже, не участвует в сборке раннера. Нужно подтвердить на более широком участке, прежде чем считать это техническим долгом наверняка.

### Устаревшие выводы
- Пока нет.

### Что делать дальше
- Следующий узкий шаг: проверить активный слой `Converters`, начиная с базовых классов/иерархии, чтобы понять:
  - как именно конвертеры получают shape/model metadata;
  - есть ли единый base-class вместо разрозненных реализаций;
  - жив ли где-то сценарий, ради которого нужен `IAdapterInfo`.
- После этого отдельно пройтись по `Extractors` base-классам, чтобы проверить, насколько симметрична архитектура "Converters -> Model -> Extractors".

### Вопросы к человеку
- Нужно ли в рамках ревизии считать "несовпадение комментария и реализации" полноценным finding, если само поведение вас устраивает и менять планируется только документацию/контракт?
- Есть ли в проекте внешний потребитель, который использует reflection/introspection по input type для автоматического выбора converter/extractor, но этот код находится вне текущего репозитория? Это важно для оценки реальной ценности `IAdapterInfo`.

## Шаг 2 - Активный слой `Converters`

### Что проверено
- Перечитан `PROJECT_REVIEW.md` перед новым проходом.
- Прочитаны файлы:
  - `src/NeuroModFlowNet.ONNX/Converters/Base/ConverterBase.cs`
  - `src/NeuroModFlowNet.ONNX/Converters/Base/ConverterNchwBase.cs`
  - `src/NeuroModFlowNet.ONNX/Converters/Abstract/AdapterBuilder.cs`
  - `src/NeuroModFlowNet.ONNX/Converters/Mat/MatAdapters.cs`
  - `src/NeuroModFlowNet.ONNX/Converters/Mat/ListMatAdapters.cs`
  - `src/NeuroModFlowNet.ONNX/Converters/Nchw/ConverterMatSingleNchw.cs`
  - `src/NeuroModFlowNet.ONNX/Converters/Nchw/ConverterMatListNchw.cs`
  - `src/NeuroModFlowNet.ONNX/Converters/Bgr/ConverterMatSingleBgrDirectU8.cs`
  - `src/NeuroModFlowNet.ONNX/Converters/NchwPos/ConverterMatSingleNchwPosCvdnnFP32.cs`
  - `src/NeuroModFlowNet.ONNX/Converters/NchwPos/ConverterMatSingleNchwPosCvdnnFP16.cs`
  - `src/NeuroModFlowNet.ONNX/Converters/NchwPos/ConverterMatListNchwPosCvdnnFP32.cs`
  - `src/NeuroModFlowNet.ONNX/Converters/NchwPos/ConverterMatListNchwPosCvdnnFP16.cs`
  - `src/NeuroModFlowNet.ONNX/Converters/PaddleOCR/PaddleOCRRecSingleConverter.cs`
  - `src/NeuroModFlowNet.ONNX/Converters/PaddleOCR/PaddleOCRRecListConverter.cs`
  - `src/NeuroModFlowNet.ONNX/Converters/PaddleOCR/PaddleUVDocConverter.cs`
- Дополнительно выполнен поиск по активной части `src/NeuroModFlowNet.ONNX` без `bin/obj/Obsolete`:
  - по `IAdapterInfo` / `InputType`
  - по `AdapterName`, `SetModel`, `Prepare`

### Что понял
- У активного слоя `Converters` есть нормальная базовая архитектура:
  - `ConverterBase<TIn>` задаёт общий lifecycle `SetModel -> Init -> Check`.
  - `ConverterNchwBase<TIn>` стандартизует извлечение `Batch/Channels/Height/Width` из `ModelInputShapes`.
  - generic-конвертеры `ConverterMatSingleNchw<,>` и `ConverterMatListNchw<,>` инжектируют алгоритм через struct-policy (`TAlgo`) и выглядят как основной современный путь.
- Вывод из шага 1 по `IAdapterInfo` уточнён:
  - интерфейс не только в obsolete-коде;
  - в активных converter-ах есть классы с `InputType`, например:
    - `Converters/Bgr/ConverterMatSingleBgrDirectU8.cs`
    - `Converters/NchwPos/ConverterMatSingleNchwPosCvdnnFP32.cs`
    - `Converters/NchwPos/ConverterMatSingleNchwPosCvdnnFP16.cs`
    - `Converters/NchwPos/ConverterMatListNchwPosCvdnnFP32.cs`
    - `Converters/NchwPos/ConverterMatListNchwPosCvdnnFP16.cs`
- Но сценарий использования `IAdapterInfo` всё ещё не подтверждён:
  - найдено наличие `InputType` у части active-конвертеров;
  - не найдено активного кода, который действительно использует `IAdapterInfo` или `InputType` для выбора/связывания converter-а.
- Внутри активного слоя есть две разные линии реализации:
  - современная: через `ConverterBase` / `ConverterNchwBase`;
  - отдельная PaddleOCR-линия: `PaddleOCRRecSingleConverter`, `PaddleOCRRecListConverter`, `PaddleUVDocConverter` реализуют `IImageAdapter` напрямую и не используют общий base lifecycle.

### Найденные риски
- Риск 1. Несогласованность контракта `IRunner.As<T>()` с реализацией.
  - Статус: подтверждение не изменилось.
- Риск 2. `IAdapterInfo` выглядит как частично висящий контракт в активной архитектуре.
  - Уточнение: в шаге 1 вывод был слишком узким.
  - Актуальная формулировка:
    - `IAdapterInfo` и `InputType` всё ещё представлены в active converter-ах.
    - Но активного orchestration-кода, который реально использует эту capability, пока не найдено.
  - Старую формулировку ниже считаю устаревшей, см. раздел "Устаревшие выводы".
- Риск 3. В активных PaddleOCR-конвертерах пустой input не очищает входной буфер модели, в отличие от базовой converter-линии.
  - `src/NeuroModFlowNet.ONNX/Converters/PaddleOCR/PaddleOCRRecSingleConverter.cs:24-28`
  - `src/NeuroModFlowNet.ONNX/Converters/PaddleOCR/PaddleOCRRecListConverter.cs:24-29`
  - `src/NeuroModFlowNet.ONNX/Converters/PaddleOCR/PaddleUVDocConverter.cs:22-40`
  - В этих классах `Prepare(...)` просто делает `return` на пустом входе.
  - В `StrategyRunner.Predict(...)` после `Adapter.Prepare(input)` всё равно вызывается `Model.Run()`.
  - Следствие: модель может получить stale input buffer от предыдущего вызова и вернуть результат не для текущего входа, а для прошлых данных.
  - Это уже выглядит как реальный поведенческий defect, а не только архитектурная шероховатость.
- Риск 4. PaddleOCR-конвертеры обходят общий lifecycle `ConverterBase`, из-за чего поведение drift-ит относительно остального слоя.
  - `PaddleOCRRecSingleConverter`, `PaddleOCRRecListConverter`, `PaddleUVDocConverter` дублируют `SetModel` и shape extraction вручную.
  - Они не получают централизованные `Init()/Check()` и уже расходятся с общей линией хотя бы по обработке empty input.
  - Это повышает шанс дальнейшего drift-а: разные правила валидации, разные empty-input semantics, разные гарантии по buffer hygiene.

### Устаревшие выводы
- Устарел фрагмент из шага 1:
  - "Найденная реализация относится к obsolete-коду"
  - Почему устарело:
    - в шаге 2 найдены active-конвертеры с `InputType` в `Converters/Bgr` и `Converters/NchwPos`;
    - значит, проблема не в том, что `IAdapterInfo` живёт только в obsolete, а в том, что его реальное использование в orchestration-слое пока не найдено.

### Что делать дальше
- Следующий узкий шаг: пройти active `Extractors` base-классы и 2-3 конкретных extractor-а.
- Цель следующего шага:
  - проверить, есть ли такой же разрыв между "общей базовой архитектурой" и отдельными специальными реализациями, как в `Converters`;
  - понять, насколько симметричен lifecycle относительно converter-ов;
  - поискать похожие проблемы с stale state / отсутствием cleanup / различающимися empty-result semantics.

### Вопросы к человеку
- Ожидается ли в этом проекте, что пустой `Mat` или пустой `List<Mat>` являются допустимым входом для PaddleOCR runner-ов и должны обрабатываться детерминированно?
- Используется ли где-то вне репозитория reflection по `InputType` у converter-ов, или это просто оставшийся API surface от старой схемы выбора адаптера?

## Шаг 3 - Активный слой `Extractors` (base + PaddleOCR + один срез YOLO)

### Что проверено
- Перечитан `PROJECT_REVIEW.md` перед новым проходом.
- Прочитаны файлы:
  - `src/NeuroModFlowNet.ONNX/Abstract/ResultExtractorBase.cs`
  - `src/NeuroModFlowNet.ONNX/Extractors/Common/IOutAsT.cs`
  - `src/NeuroModFlowNet.ONNX/Extractors/Common/IExtractorThreshold.cs`
  - `src/NeuroModFlowNet.ONNX/Extractors/Common/IBatchedResult.cs`
  - `src/NeuroModFlowNet.ONNX/Extractors/Common/BatchedResultBase.cs`
  - `src/NeuroModFlowNet.ONNX/Extractors/Common/BatchedResult.cs`
  - `src/NeuroModFlowNet.ONNX/Extractors/Common/BatchedResultPooled.cs`
  - `src/NeuroModFlowNet.ONNX/Extractors/PaddleOCR/Rec/PaddleOCRRecExtractor.cs`
  - `src/NeuroModFlowNet.ONNX/Extractors/PaddleOCR/UVDoc/PaddleUVDocExtractor.cs`
  - `src/NeuroModFlowNet.ONNX/Extractors/PaddleOCR/PaddleOCRMaskExtractor.cs`
  - `src/NeuroModFlowNet.ONNX/Extractors/PaddleOCR/Det/Extractors/PaddleOCRDetExtractorBase.cs`
  - `src/NeuroModFlowNet.ONNX/Extractors/PaddleOCR/Det/Extractors/FP32/PaddleOCRDetFP32_ExtractorBase.cs`
  - `src/NeuroModFlowNet.ONNX/Extractors/PaddleOCR/Det/Extractors/FP16/PaddleOCRDetFP16_ExtractorBase.cs`
  - `src/NeuroModFlowNet.ONNX/Extractors/PaddleOCR/Det/Extractors/FP32/PaddleOCRDetFP32_32FC1_SafeExtractor.cs`
  - `src/NeuroModFlowNet.ONNX/Extractors/PaddleOCR/Det/Factories/PaddleOCRDetFactory.cs`
  - `src/NeuroModFlowNet.ONNX/Extractors/Yolo/Box/Extractors/YoloBoxNmsExtractorBase.cs`
  - `src/NeuroModFlowNet.ONNX/Extractors/Yolo/Box/Extractors/YoloBoxNmsFP32ExtractorBase.cs`
  - `src/NeuroModFlowNet.ONNX/Extractors/Yolo/Box/Extractors/YoloBoxNmsFP32Extractor.cs`
- Дополнительно выполнены поиски по active `Extractors`:
  - по `return default!;`, `TODO`, `Console.WriteLine`, `File.Exists`, `File.ReadAllLines`
  - по классам, наследующимся от `ResultExtractorBase<>`
  - по использованию `PaddleOCRMaskExtractor`

### Что понял
- В active `Extractors` базовая архитектура в целом выдержана лучше, чем в `Converters`:
  - PaddleOCR, YOLO Box/Cls/Obb/Pose/Seg ветки в основном действительно строятся поверх `ResultExtractorBase<TOut>`.
  - Проверка type metadata через `Check()` и извлечение shape metadata через `Init()` выглядят как основной единый паттерн.
- Ветка PaddleOCR Detection выглядит архитектурно аккуратнее, чем PaddleOCR converters:
  - есть общий `PaddleOCRDetExtractorBase<TOut>`;
  - есть типоспециализированные FP32/FP16 базы;
  - factory `PaddleOCRDetFactory.CreateRunner<TIn, TOut>` реально использует metadata модели для выбора extractor-а и converter-а.
- Ветка YOLO Box также выглядит консистентной:
  - `YoloBoxNmsExtractorBase<TOut>` хранит `BatchCount`/`ItemCount`;
  - `YoloBoxNmsFP32ExtractorBase<TOut>` проверяет тип выходного тензора и собирает batched result через pooled container.
- `PaddleOCRMaskExtractor` сейчас не найден в active factory/orchestration-пути; по текущему проходу он выглядит как незавершённый активный artefact, но не как подтверждённый runtime path.

### Найденные риски
- Риск 1. Несогласованность контракта `IRunner.As<T>()` с реализацией.
  - Статус: подтверждение не изменилось.
- Риск 2. `IAdapterInfo` выглядит как частично висящий контракт в активной архитектуре.
  - Статус: подтверждение не изменилось.
- Риск 3. В активных PaddleOCR-конвертерах пустой input не очищает входной буфер модели, в отличие от базовой converter-линии.
  - Статус: подтверждение не изменилось.
- Риск 4. PaddleOCR-конвертеры обходят общий lifecycle `ConverterBase`, из-за чего поведение drift-ит относительно остального слоя.
  - Статус: подтверждение не изменилось.
- Риск 5. `PaddleOCRMaskExtractor` находится в active-коде, но `Extract()` фактически не реализован и всегда возвращает `default!`.
  - `src/NeuroModFlowNet.ONNX/Extractors/PaddleOCR/PaddleOCRMaskExtractor.cs:5-8`
  - Это не просто TODO-комментарий: публичный extractor компилируется и выглядит готовым к использованию, но при вызове отдаст `null` под видом `Mat`.
  - По текущему проходу active factory usage не найден, поэтому риск пока классифицируется как latent API hazard, а не подтверждённый продакшн-path defect.
- Риск 6. `PaddleOCRRecExtractor` смешивает extraction с filesystem/console side effects.
  - `src/NeuroModFlowNet.ONNX/Extractors/PaddleOCR/Rec/PaddleOCRRecExtractor.cs:23-25`, `28-45`
  - На `Init()` extractor автоматически читает `dict.txt` рядом с моделью.
  - При отсутствии файла или успешной загрузке он пишет сообщения через `Console.WriteLine`.
  - Риск: библиотечный extractor получает неожиданные side effects при инициализации, становится менее предсказуемым для сервисных/GUI-hosting сценариев и труднее тестируется.
  - Это не обязательно bug, но уже smell уровня API behavior.

### Устаревшие выводы
- Новых устаревших выводов на этом шаге нет.

### Что делать дальше
- Следующий узкий шаг: пройти только active factory/orchestration слой вокруг YOLO или PaddleOCR `CreateRunner`, чтобы понять:
  - насколько последовательно используется metadata-driven selection;
  - где именно проходит граница между "современной" generic-архитектурой и legacy/manual paths;
  - есть ли ещё публичные типы, присутствующие в active API, но не подключённые к фактическому runtime path.

### Вопросы к человеку
- Допустимы ли для этого репозитория side effects уровня `Console.WriteLine` и автоматического чтения файлов внутри library extractor-а, или это стоит считать отдельным smell/finding?
- Нужно ли в ревизии отдельно помечать "latent hazards" вроде `PaddleOCRMaskExtractor`, если тип публичный и активный, но factory-path его сейчас не использует?

## Шаг 4 - Active factory/orchestration (`CreateRunner` и reflection-based сборка)

### Что проверено
- Перечитан `PROJECT_REVIEW.md` перед новым проходом.
- По active `src/NeuroModFlowNet.ONNX` выполнен поиск по:
  - `CreateRunner`
  - `Activator.CreateInstance`
  - `MakeGenericType`
  - `GetConverterType`
  - `GetExtractorType`
- Прочитаны файлы:
  - `src/NeuroModFlowNet.ONNX/Extractors/PaddleOCR/Det/Factories/PaddleOCRDetFactory.cs`
  - `src/NeuroModFlowNet.ONNX/Extractors/Yolo/Box/Factories/YoloBoxFactory.cs`
  - `src/NeuroModFlowNet.ONNX/Extractors/Yolo/Cls/Factories/YoloClsFactory.cs`
  - `src/NeuroModFlowNet.ONNX/Extractors/Yolo/Obb/Factories/YoloObbFactory.cs`
  - `src/NeuroModFlowNet.ONNX/Extractors/Yolo/Pose/Factories/YoloPoseFactory.cs`
  - `src/NeuroModFlowNet.ONNX/Extractors/Yolo/Seg/Factories/YoloSegFactory.cs`

### Что понял
- В active orchestration действительно есть единая линия: несколько модулей используют metadata-driven `CreateRunner`, который:
  - читает `InputMetadata` / `OutputMetadata`;
  - выбирает converter/extractor типы;
  - собирает `ImageRunner<,,,>` через reflection.
- Это подтверждённый active runtime path, а не только задумка в API.
- Но реализация между модулями непоследовательна:
  - `PaddleOCRDetFactory`, `YoloClsFactory`, `YoloObbFactory`, `YoloPoseFactory`, `YoloSegFactory` создают closed generic тип и передают `model` в конструктор.
  - `YoloBoxFactory` содержит две разные перегрузки `CreateRunner`, и одна из них выбивается из общего паттерна.
- Граница между "современной" и "ручной" логикой действительно плавает:
  - часть factory-методов честно выводит типы из metadata;
  - часть всё ещё кодирует доменные assumptions прямо внутри фабрики;
  - есть перегрузки, которые выглядят как transitional API и не дотянуты до общего стандарта.

### Найденные риски
- Риск 1. Несогласованность контракта `IRunner.As<T>()` с реализацией.
  - Статус: подтверждение не изменилось.
- Риск 2. `IAdapterInfo` выглядит как частично висящий контракт в активной архитектуре.
  - Статус: подтверждение не изменилось.
- Риск 3. В активных PaddleOCR-конвертерах пустой input не очищает входной буфер модели, в отличие от базовой converter-линии.
  - Статус: подтверждение не изменилось.
- Риск 4. PaddleOCR-конвертеры обходят общий lifecycle `ConverterBase`, из-за чего поведение drift-ит относительно остального слоя.
  - Статус: подтверждение не изменилось.
- Риск 5. `PaddleOCRMaskExtractor` находится в active-коде, но `Extract()` фактически не реализован и всегда возвращает `default!`.
  - Статус: подтверждение не изменилось.
- Риск 6. `PaddleOCRRecExtractor` смешивает extraction с filesystem/console side effects.
  - Статус: подтверждение не изменилось.
- Риск 7. Перегрузка `YoloBoxFactory.CreateRunner<TOut>(OnnxModel model, bool isByteBgr)` сломана: она создаёт `ImageRunner` без передачи `model` в конструктор.
  - `src/NeuroModFlowNet.ONNX/Extractors/Yolo/Box/Factories/YoloBoxFactory.cs:69-85`
  - В `ImageRunner`/`StrategyRunner` нет parameterless constructor; рабочий путь в остальных фабриках использует `Activator.CreateInstance(closedType, model)`.
  - Здесь вызван `Activator.CreateInstance(closedType)!` без аргумента `model`.
  - Следствие: при реальном вызове эта перегрузка должна падать на этапе создания раннера, то есть это подтверждённый runtime defect в active factory API.
- Риск 8. В `YoloBoxFactory.CreateRunner<TOut>(OnnxModel model, bool isByteBgr)` generic-параметр `TOut` фактически игнорируется при выборе закрываемого runtime-типа.
  - `src/NeuroModFlowNet.ONNX/Extractors/Yolo/Box/Factories/YoloBoxFactory.cs:73-82`
  - Метод всегда закрывает `ImageRunner<Mat, IDetectionResult<YoloBox>, ...>` через `typeOut = typeof(IDetectionResult<YoloBox>)`, хотя сигнатура обещает `IRunner<Mat, TOut>`.
  - Даже если бы проблема с конструктором была исправлена, API уже здесь вводит в заблуждение и потенциально приводит к invalid cast / ложным ожиданиям о поддерживаемых типах результата.

### Устаревшие выводы
- Новых устаревших выводов на этом шаге нет.

### Что делать дальше
- Следующий узкий шаг: пройти один небольшой public API surface вне factory-слоя, например `OnnxModel`/`OnnxModelExtensions`, чтобы проверить:
  - как управляются `SetInput`, `Run`, `Cleanup`, persistent values и временные `OrtValue`;
  - нет ли несоответствий между safe/unsafe extractor paths и реальным lifecycle model buffers;
  - не протекают ли ошибки orchestration уровня уже на границе с ONNX Runtime.

### Вопросы к человеку
- Считать ли для этой ревизии перегрузки вроде `YoloBoxFactory.CreateRunner<TOut>(OnnxModel, bool)` high-priority finding даже если они могут быть редко используемыми, но публично доступны?
- Нужно ли отдельно отмечать API drift между фабриками как самостоятельный smell, или фиксировать только конкретные поломки выполнения?

## Шаг 5 - Нижний lifecycle слой (`OnnxModel` / buffers / cleanup)

### Что проверено
- Перечитан `PROJECT_REVIEW.md` перед новым проходом.
- По active `src/NeuroModFlowNet.ONNX` выполнен поиск по:
  - `SetInput`, `Run`, `Cleanup`
  - `GetInputBuffer`, `GetOutputValue`, `GetTensorDataAsSpan`
  - `OrtValue`, `IoBinding`, `PersistentValue`
- Прочитаны файлы:
  - `src/NeuroModFlowNet.ONNX/ModelWokers/Base/OnnxModel.cs`
  - `src/NeuroModFlowNet.ONNX/Inference/Extensions/OnnxModelExtensions.cs`
- Дополнительно проверено использование:
  - `GetRealInputShape`
  - `GetRealOutputShape`

### Что понял
- Центральный runtime lifecycle действительно сосредоточен в `OnnxModel`:
  - `RunInputValues` / `RunOutputValues` держат временные `OrtValue`, переданные через `SetInput/SetOutput`;
  - при `Run()` для каждого input/output либо используется временный `OrtValue`, либо lazily-created persistent buffer;
  - `StrategyRunner` потом вызывает `Model.Cleanup()`, который dispose-ит только временные run values, а persistent buffers живут дольше.
- Архитектурно это объясняет две линии работы:
  - обычный путь: converter пишет в persistent input buffer через `GetInputBuffer<T>()`;
  - zero-copy путь: converter создаёт внешний `OrtValue` и кладёт его через `SetInput(...)`.
- `GetRealOutputShape` и `GetRealInputShape` возвращают shape уже инициализированных persistent buffers, а не shape из metadata модели. Это логично, но завязано на то, что persistent value действительно был создан.

### Найденные риски
- Риск 1. Несогласованность контракта `IRunner.As<T>()` с реализацией.
  - Статус: подтверждение не изменилось.
- Риск 2. `IAdapterInfo` выглядит как частично висящий контракт в активной архитектуре.
  - Статус: подтверждение не изменилось.
- Риск 3. В активных PaddleOCR-конвертерах пустой input не очищает входной буфер модели, в отличие от базовой converter-линии.
  - Статус: подтверждение не изменилось.
- Риск 4. PaddleOCR-конвертеры обходят общий lifecycle `ConverterBase`, из-за чего поведение drift-ит относительно остального слоя.
  - Статус: подтверждение не изменилось.
- Риск 5. `PaddleOCRMaskExtractor` находится в active-коде, но `Extract()` фактически не реализован и всегда возвращает `default!`.
  - Статус: подтверждение не изменилось.
- Риск 6. `PaddleOCRRecExtractor` смешивает extraction с filesystem/console side effects.
  - Статус: подтверждение не изменилось.
- Риск 7. Перегрузка `YoloBoxFactory.CreateRunner<TOut>(OnnxModel model, bool isByteBgr)` сломана: она создаёт `ImageRunner` без передачи `model` в конструктор.
  - Статус: подтверждение не изменилось.
- Риск 8. В `YoloBoxFactory.CreateRunner<TOut>(OnnxModel model, bool isByteBgr)` generic-параметр `TOut` фактически игнорируется при выборе закрываемого runtime-типа.
  - Статус: подтверждение не изменилось.
- Риск 9. `OnnxModel` может попытаться автоматически создать persistent input/output buffer для partially-dynamic shape, хотя такой буфер ещё нельзя корректно аллоцировать.
  - `src/NeuroModFlowNet.ONNX/ModelWokers/Base/OnnxModel.cs:220-224`
  - `src/NeuroModFlowNet.ONNX/ModelWokers/Base/OnnxModel.cs:256-260`
  - В обоих путях используется `Debug.Assert(shape.Any(d => d > 0), ...)`.
  - Эта проверка пропускает формы вида `[-1, 3, 640, 640]`, потому что в них "хотя бы одно измерение > 0" истинно.
  - После этого код идёт в `InitInputPersistentValue/InitOutputPersistentValue` и пытается вызвать `OrtValue.CreateAllocatedTensorValue(..., shape)` с shape, содержащим динамические размеры.
  - В release-сборке `Debug.Assert` вообще не защищает выполнение, поэтому риск не ограничен debug-only поведением.
  - Корректная логика здесь, похоже, должна требовать полностью статическую shape либо явную инициализацию через `SetInput`/`Init...PersistentValue` с реальными runtime размерами.

### Устаревшие выводы
- Новых устаревших выводов на этом шаге нет.

### Что делать дальше
- Следующий узкий шаг: пройти только один конкретный модуль с dynamic-shape зависимостью, лучше `PaddleOCRDet` или `YoloSeg`, чтобы проверить:
  - насколько реально используются partially-dynamic outputs;
  - есть ли в active коде путь, который уже может упереться в проблему из риска 9;
  - как safe/unsafe extractors полагаются на `GetRealOutputShape`.

### Вопросы к человеку
- В проекте ожидаются модели с частично динамическими input/output shape в production-path, или сейчас это скорее задел на будущее?
- Считать ли для этой ревизии ошибки базового lifecycle-слоя вроде риска 9 приоритетнее, чем модульные API-smells, даже если они проявятся только на определённых моделях?

## Шаг 6 - Dynamic-shape active path: `PaddleOCRDet`

### Что проверено
- Перечитан `PROJECT_REVIEW.md` перед новым проходом.
- Прочитаны файлы:
  - `src/NeuroModFlowNet.ONNX/Extractors/PaddleOCR/Det/Extractors/PaddleOCRDetExtractorBase.cs`
  - `src/NeuroModFlowNet.ONNX/Extractors/PaddleOCR/Det/Extractors/FP32/PaddleOCRDetFP32_32FC1_UnsafeExtractor.cs`
  - `src/NeuroModFlowNet.ONNX/Extractors/PaddleOCR/Det/Extractors/FP32/PaddleOCRDetFP32_32FC1_SafeListExtractor.cs`
  - `src/NeuroModFlowNet.ONNX/Extractors/PaddleOCR/Det/Extractors/FP16/PaddleOCRDetFP16_16FC1_UnsafeExtractor.cs`
  - `src/NeuroModFlowNet.ONNX/Extractors/PaddleOCR/Det/Extractors/FP16/PaddleOCRDetFP16_16FC1_SafeListExtractor.cs`
- Дополнительно выполнены поиски по:
  - использованию `GetRealOutputShape`
  - условиям `ModelOutputImageHeight < 0` / `ModelOutputImageWidth < 0`
  - использованию `new List<Mat>(BatchCount)` и циклов `for (i < BatchCount)` внутри `PaddleOCRDet`

### Что понял
- Риск 9 из шага 5 не теоретический: у `PaddleOCRDet` есть явный active-path, который уже рассчитывает на dynamic output support.
- `PaddleOCRDetExtractorBase<TOut>` делает partial fallback:
  - для `Height/Width` использует `Model.GetRealOutputShape(...)`, если в metadata стоит отрицательное значение;
  - но `BatchCount` всегда берётся напрямую из `Model.ModelOutputShapes[PrimaryOutputName][0]`.
- Значит модуль conceptually поддерживает dynamic spatial dimensions, но не поддерживает dynamic batch dimension как first-class scenario.
- Safe-list extractors (`...SafeListExtractor`) используют `BatchCount` напрямую и завязаны на то, что он уже конкретный и неотрицательный.

### Найденные риски
- Риск 1. Несогласованность контракта `IRunner.As<T>()` с реализацией.
  - Статус: подтверждение не изменилось.
- Риск 2. `IAdapterInfo` выглядит как частично висящий контракт в активной архитектуре.
  - Статус: подтверждение не изменилось.
- Риск 3. В активных PaddleOCR-конвертерах пустой input не очищает входной буфер модели, в отличие от базовой converter-линии.
  - Статус: подтверждение не изменилось.
- Риск 4. PaddleOCR-конвертеры обходят общий lifecycle `ConverterBase`, из-за чего поведение drift-ит относительно остального слоя.
  - Статус: подтверждение не изменилось.
- Риск 5. `PaddleOCRMaskExtractor` находится в active-коде, но `Extract()` фактически не реализован и всегда возвращает `default!`.
  - Статус: подтверждение не изменилось.
- Риск 6. `PaddleOCRRecExtractor` смешивает extraction с filesystem/console side effects.
  - Статус: подтверждение не изменилось.
- Риск 7. Перегрузка `YoloBoxFactory.CreateRunner<TOut>(OnnxModel model, bool isByteBgr)` сломана: она создаёт `ImageRunner` без передачи `model` в конструктор.
  - Статус: подтверждение не изменилось.
- Риск 8. В `YoloBoxFactory.CreateRunner<TOut>(OnnxModel model, bool isByteBgr)` generic-параметр `TOut` фактически игнорируется при выборе закрываемого runtime-типа.
  - Статус: подтверждение не изменилось.
- Риск 9. `OnnxModel` может попытаться автоматически создать persistent input/output buffer для partially-dynamic shape, хотя такой буфер ещё нельзя корректно аллоцировать.
  - Статус: подтверждение усилено active-path анализом `PaddleOCRDet`.
- Риск 10. `PaddleOCRDet` поддерживает dynamic spatial output only partially: dynamic `BatchCount` остаётся неразрешённым metadata value и может сломать list-extractor path.
  - `src/NeuroModFlowNet.ONNX/Extractors/PaddleOCR/Det/Extractors/PaddleOCRDetExtractorBase.cs:13-17`, `20`, `26-28`
  - `src/NeuroModFlowNet.ONNX/Extractors/PaddleOCR/Det/Extractors/FP32/PaddleOCRDetFP32_32FC1_SafeListExtractor.cs:12-15`
  - `src/NeuroModFlowNet.ONNX/Extractors/PaddleOCR/Det/Extractors/FP16/PaddleOCRDetFP16_16FC1_SafeListExtractor.cs:15-18`
  - Если batch dimension у output shape динамический и приходит как `-1`, код не делает fallback через `GetRealOutputShape()[0]`.
  - В результате list extractors используют `new List<Mat>(BatchCount)` и цикл `for (i < BatchCount)`, что даёт некорректное поведение или исключение ещё до реального чтения результатов.
  - Это уже прямое продолжение риска 9 на уровне конкретного active модуля.

### Устаревшие выводы
- Новых устаревших выводов на этом шаге нет.

### Что делать дальше
- Следующий узкий шаг: пройти аналогичный dynamic-shape путь в `YoloSeg`, чтобы сравнить:
  - использует ли он dynamic output более последовательно, чем `PaddleOCRDet`;
  - завязан ли он на те же assumptions `OnnxModel`;
  - есть ли ещё active modules, где fallback реализован только для части измерений.

### Вопросы к человеку
- Нужна ли тебе в `PROJECT_REVIEW.md` ещё и отдельная секция "Прогресс ревизии" с процентной или этапной оценкой, чтобы потом было проще продолжать ревизию в новых сессиях?
- Есть ли приоритет между двумя оставшимися большими зонами: `YoloSeg`/dynamic outputs и слой `samples/tests`, который может показать, какие из найденных путей реально используются?

## Шаг 7 - Dynamic/batch path `YoloSeg` + usage-check в `tests`/`samples`

### Что проверено
- Перечитан `PROJECT_REVIEW.md` перед новым проходом.
- Прочитаны файлы:
  - `src/NeuroModFlowNet.ONNX/Extractors/Yolo/Seg/Extractors/YoloSegExtractorBase.cs`
  - `src/NeuroModFlowNet.ONNX/Extractors/Yolo/Seg/Extractors/YoloSegFP32ExtractorBase.cs`
  - `src/NeuroModFlowNet.ONNX/Extractors/Yolo/Seg/Extractors/YoloSegFP16ExtractorBase.cs`
  - `src/NeuroModFlowNet.ONNX/Extractors/Yolo/Seg/Factories/YoloSegFactory.cs`
  - `samples/NeuroModFlowNet.ONNX.Demo.Dashboard/Program.cs`
  - `tests/NeuroModFlowNet.ONNX.Tests/ExtractorOBBNMSTests.cs`
- Дополнительно выполнены поиски по исходникам `tests` и `samples` без `bin/obj`:
  - использования `CreateRunner(...)` у проблемных фабрик;
  - использования `PaddleOCRMaskExtractor`.

### Что понял
- `YoloSeg` в active коде conceptually хранит `BatchCount`, `ItemCount` и shape прототипов, но concrete FP32/FP16 extractor-ветки фактически обрабатывают только batch `0`.
- Это не просто ограничение single-image demo:
  - `YoloSegExtractorBase<TOut>` хранит batch-aware metadata;
  - но FP32 и FP16 base implementations читают только первый батч detection/prototype данных.
- Быстрый usage-check по исходникам `tests` и `samples` показал:
  - demo в `samples` в основном использует конкретные `Single_*` фабрики или прямой `new ImageRunner<...>`, а не проблемные generic `CreateRunner(...)` перегрузки;
  - test-покрытие orchestration/factory/lifecycle слоёв очень слабое;
  - явного использования `PaddleOCRMaskExtractor` в исходниках `tests`/`samples` не найдено.
- Это значит, что часть найденных рисков пока лучше трактовать как публичные active API hazards, а не как дефекты, уже защищённые тестами или воспроизводимые по sample-code внутри репозитория.

### Найденные риски
- Риск 1. Несогласованность контракта `IRunner.As<T>()` с реализацией.
  - Статус: подтверждение не изменилось.
- Риск 2. `IAdapterInfo` выглядит как частично висящий контракт в активной архитектуре.
  - Статус: подтверждение не изменилось.
- Риск 3. В активных PaddleOCR-конвертерах пустой input не очищает входной буфер модели, в отличие от базовой converter-линии.
  - Статус: подтверждение не изменилось.
- Риск 4. PaddleOCR-конвертеры обходят общий lifecycle `ConverterBase`, из-за чего поведение drift-ит относительно остального слоя.
  - Статус: подтверждение не изменилось.
- Риск 5. `PaddleOCRMaskExtractor` находится в active-коде, но `Extract()` фактически не реализован и всегда возвращает `default!`.
  - Статус: подтверждение не изменилось.
- Риск 6. `PaddleOCRRecExtractor` смешивает extraction с filesystem/console side effects.
  - Статус: подтверждение не изменилось.
- Риск 7. Перегрузка `YoloBoxFactory.CreateRunner<TOut>(OnnxModel model, bool isByteBgr)` сломана: она создаёт `ImageRunner` без передачи `model` в конструктор.
  - Статус: подтверждение не изменилось.
- Риск 8. В `YoloBoxFactory.CreateRunner<TOut>(OnnxModel model, bool isByteBgr)` generic-параметр `TOut` фактически игнорируется при выборе закрываемого runtime-типа.
  - Статус: подтверждение не изменилось.
- Риск 9. `OnnxModel` может попытаться автоматически создать persistent input/output buffer для partially-dynamic shape, хотя такой буфер ещё нельзя корректно аллоцировать.
  - Статус: подтверждение не изменилось.
- Риск 10. `PaddleOCRDet` поддерживает dynamic spatial output only partially: dynamic `BatchCount` остаётся неразрешённым metadata value и может сломать list-extractor path.
  - Статус: подтверждение не изменилось.
- Риск 11. `YoloSeg` хранит batch-aware metadata, но concrete FP32/FP16 extractor logic фактически обрабатывает только первый батч.
  - `src/NeuroModFlowNet.ONNX/Extractors/Yolo/Seg/Extractors/YoloSegExtractorBase.cs:18-30`, `33-46`
  - `src/NeuroModFlowNet.ONNX/Extractors/Yolo/Seg/Extractors/YoloSegFP32ExtractorBase.cs:30`, `35-37`, `43`, `75`, `80-82`, `88`
  - `src/NeuroModFlowNet.ONNX/Extractors/Yolo/Seg/Extractors/YoloSegFP16ExtractorBase.cs:33`, `43`
  - FP32 path использует `tensorSeg[0, itemIndex]` и `prototypeData.Slice(0, ...)`.
  - FP16 path использует `lineSeg.Slice(0, ItemCount)` и `maskData.Slice(0, ...)`.
  - При `BatchCount > 1` extractor не итерирует по батчам и не смещает prototype tensors по batch offset.
  - Риск: metadata и API создают ожидание batch-aware работы, но реальная реализация silently игнорирует все батчи, кроме первого.

### Устаревшие выводы
- Новых устаревших выводов на этом шаге нет.

### Что делать дальше
- Базовую архитектурную ревизию на этом этапе можно считать почти закрытой.
- Если продолжать следующей сессией, наиболее полезные направления:
  - подтвердить usage-path через `labs/` и живые демо;
  - после разрешения на правки чинить findings начиная с P1;
  - затем прогонять всё решение, `samples`, `tests` и при необходимости benchmark/lab сценарии.

### Вопросы к человеку
- Нужен ли следующий шаг именно как "исправить P1/P2 findings по приоритету", или пока оставить всё на уровне ревизионной карты?
- Если переходить к фиксам, хочешь ли ты сначала минимальные безопасные исправления runtime-defect'ов, или сперва зачистку архитектурного drift-а между модулями?

## Шаг 8 - Первые исправления P1 runtime-defect'ов + попытка верификации

### Что проверено
- Перечитан `PROJECT_REVIEW.md` перед началом исправлений.
- Прочитаны и изменены файлы:
  - `src/NeuroModFlowNet.ONNX/Converters/PaddleOCR/PaddleOCRRecSingleConverter.cs`
  - `src/NeuroModFlowNet.ONNX/Converters/PaddleOCR/PaddleOCRRecListConverter.cs`
  - `src/NeuroModFlowNet.ONNX/Converters/PaddleOCR/PaddleUVDocConverter.cs`
  - `src/NeuroModFlowNet.ONNX/Extractors/Yolo/Box/Factories/YoloBoxFactory.cs`
  - `src/NeuroModFlowNet.ONNX/ModelWokers/Base/OnnxModel.cs`
  - `src/NeuroModFlowNet.ONNX/Extractors/PaddleOCR/Det/Extractors/PaddleOCRDetExtractorBase.cs`
  - `src/NeuroModFlowNet.ONNX/Extractors/Yolo/Seg/Extractors/YoloSegExtractorBase.cs`
  - `src/NeuroModFlowNet.ONNX/Extractors/Yolo/Seg/Extractors/YoloSegFP32ExtractorBase.cs`
  - `src/NeuroModFlowNet.ONNX/Extractors/Yolo/Seg/Extractors/YoloSegFP16ExtractorBase.cs`
- Дополнительно перечитаны, чтобы выбрать безопасную стратегию фикса для `YoloSeg`:
  - `src/NeuroModFlowNet.ONNX/Extractors/Yolo/Seg/OutData/YoloSegResult_FP32_Mask32.cs`
  - `src/NeuroModFlowNet.ONNX/Extractors/Yolo/Seg/OutData/YoloSegResult_FP16_Mask32.cs`
  - `src/NeuroModFlowNet.ONNX/Extractors/Yolo/Seg/OutData/YoloSeg_FP32.cs`
  - `src/NeuroModFlowNet.ONNX/Extractors/Common/IDetectionResult.cs`
- Выполнена попытка верификации:
  - `dotnet build "NeuroModFlowNet.ONNX.slnx" -v minimal`
  - Roslyn diagnostics по solution.

### Что понял
- Часть P1 находок можно закрыть локальными безопасными исправлениями без изменения архитектурных фундаменталов:
  - stale input buffer в PaddleOCR converters;
  - сломанная generic factory перегрузка в `YoloBoxFactory`;
  - слишком слабая dynamic-shape защита в `OnnxModel`;
  - partial dynamic batch support в `PaddleOCRDet`.
- По `YoloSeg` полноценная batch-aware поддержка не выглядит как "маленький безопасный фикс":
  - текущие result-типы и active API уже завязаны на single-result semantics;
  - в result-типах явно зашито `BatchCount => 1`;
  - поэтому на этом шаге безопаснее не пытаться тихо доделывать multi-batch extraction, а сделать fail-fast с явной ошибкой при `BatchCount != 1`.
- Полная сборочная верификация сейчас заблокирована не содержимым патчей, а проблемой доступа к `obj`/cache файлам в нескольких проектах решения.

### Что изменено
- Исправлен stale input buffer в active PaddleOCR converters:
  - `PaddleOCRRecSingleConverter`, `PaddleOCRRecListConverter`, `PaddleUVDocConverter` теперь очищают input buffer при пустом входе, вместо молчаливого `return`.
- Исправлена public generic factory перегрузка:
  - `YoloBoxFactory.CreateRunner<TOut>(OnnxModel model, bool isByteBgr)` теперь действительно использует `TOut`, корректно выбирает extractor и создаёт runner с передачей `model` в конструктор.
- Усилена защита от partially-dynamic shape:
  - `OnnxModel.GetInputPersistentValue(...)` и `GetOutputPersistentValue(...)` теперь бросают `InvalidOperationException`, если shape содержит неконкретные размеры, вместо слабого `Debug.Assert`.
- Исправлен dynamic batch fallback в `PaddleOCRDetExtractorBase`:
  - `BatchCount` теперь берётся из `GetRealOutputShape(...)[0]`, если metadata batch dimension динамический.
- Смягчён риск silent data loss в `YoloSeg`:
  - base extractor теперь явно отклоняет модели с `BatchCount != 1`;
  - FP32/FP16 extractor-ветки вызывают `base.Check()`.

### Найденные риски
- Риск 1. Несогласованность контракта `IRunner.As<T>()` с реализацией.
  - Статус: не исправлялся на этом шаге.
- Риск 2. `IAdapterInfo` выглядит как частично висящий контракт в активной архитектуре.
  - Статус: не исправлялся на этом шаге.
- Риск 3. В активных PaddleOCR-конвертерах пустой input не очищает входной буфер модели, в отличие от базовой converter-линии.
  - Статус: исправлено на этом шаге.
- Риск 4. PaddleOCR-конвертеры обходят общий lifecycle `ConverterBase`, из-за чего поведение drift-ит относительно остального слоя.
  - Статус: не исправлялся архитектурно; конкретный runtime-defect из риска 3 закрыт, но сам drift остался.
- Риск 5. `PaddleOCRMaskExtractor` находится в active-коде, но `Extract()` фактически не реализован и всегда возвращает `default!`.
  - Статус: не исправлялся на этом шаге.
- Риск 6. `PaddleOCRRecExtractor` смешивает extraction с filesystem/console side effects.
  - Статус: не исправлялся на этом шаге.
- Риск 7. Перегрузка `YoloBoxFactory.CreateRunner<TOut>(OnnxModel model, bool isByteBgr)` сломана: она создаёт `ImageRunner` без передачи `model` в конструктор.
  - Статус: исправлено на этом шаге.
- Риск 8. В `YoloBoxFactory.CreateRunner<TOut>(OnnxModel model, bool isByteBgr)` generic-параметр `TOut` фактически игнорируется при выборе закрываемого runtime-типа.
  - Статус: исправлено на этом шаге.
- Риск 9. `OnnxModel` может попытаться автоматически создать persistent input/output buffer для partially-dynamic shape, хотя такой буфер ещё нельзя корректно аллоцировать.
  - Статус: исправлено на этом шаге через явный runtime guard.
- Риск 10. `PaddleOCRDet` поддерживает dynamic spatial output only partially: dynamic `BatchCount` остаётся неразрешённым metadata value и может сломать list-extractor path.
  - Статус: исправлено на этом шаге.
- Риск 11. `YoloSeg` хранит batch-aware metadata, но concrete FP32/FP16 extractor logic фактически обрабатывает только первый батч.
  - Статус: не исправлено полностью; риск смягчён fail-fast проверкой, чтобы исключить silent truncation при `BatchCount > 1`.
- Риск 12. Полная solution-верификация сейчас заблокирована инфраструктурно.
  - `dotnet build` упёрся в `MSB3491` / `Access to the path is denied` при записи в `obj`/cache файлы нескольких проектов решения.
  - Это не даёт пока честно подтвердить сборку всех проектов после фиксов именно стандартным build-путём.
- Риск 13. В `samples` и `tests` уже есть независимые compile errors, не связанные напрямую с текущими исправлениями.
  - По Roslyn diagnostics обнаружены ошибки в:
    - `samples/NeuroModFlowNet.ONNX.Demo.Dashboard/Program.cs`
    - `tests/NeuroModFlowNet.ONNX.Tests/ExtractorOBBNMSTests.cs`
  - Это означает, что даже после снятия проблемы с доступом к `obj`, solution, вероятно, не будет "зелёной" без отдельного прохода по sample/test-коду.

### Устаревшие выводы
- Устарел риск 3 в исходной формулировке про stale input buffer в active PaddleOCR converters.
  - Почему устарело:
    - на этом шаге конвертеры стали очищать input buffer при пустом входе;
    - сам архитектурный drift не устранён, но конкретный runtime defect больше не актуален.
- Устарели риски 7 и 8 в исходной формулировке по `YoloBoxFactory.CreateRunner<TOut>(OnnxModel model, bool isByteBgr)`.
  - Почему устарело:
    - перегрузка теперь передаёт `model` в конструктор `ImageRunner`;
    - runtime output type больше не захардкожен в `IDetectionResult<YoloBox>`.
- Устарел риск 9 в исходной формулировке по weak dynamic-shape guard в `OnnxModel`.
  - Почему устарело:
    - теперь partially-dynamic shape не проходит в lazy persistent allocation path молча.
- Устарел риск 10 в исходной формулировке по `PaddleOCRDet`.
  - Почему устарело:
    - `BatchCount` теперь умеет брать runtime batch dimension из real output shape.

### Что делать дальше
- Следующий логичный шаг уже не в обзор ширины, а в добивку оставшихся findings:
  - `PaddleOCRMaskExtractor` как публичный active stub;
  - `PaddleOCRRecExtractor` side effects;
  - комментарий/контракт `IRunner.As<T>()`;
  - подтверждение роли `IAdapterInfo` или его перевод в legacy/obsolete surface.
- Отдельный технический шаг для верификации:
  - снять проблему с доступом к `obj`/cache файлам и потом прогнать всё решение целиком;
  - после этого отдельно разобрать pre-existing compile errors в `samples` и `tests`.

### Вопросы к человеку
- Хочешь следующим шагом именно продолжать фиксы по оставшимся finding'ам, или сначала добить сборочную верификацию и расчистить уже существующие ошибки в `samples/tests`?

## Шаг 9 - Legacy/obsolete след `IAdapterInfo`

### Что проверено
- Перечитан `PROJECT_REVIEW.md` перед новым проходом.
- Прочитаны файлы:
  - `src/NeuroModFlowNet.ONNX/Abstract/IAdapterInfo.cs`
  - `src/NeuroModFlowNet.ONNX/Abstract/IInputConverter.cs`
  - `src/NeuroModFlowNet.ONNX/Converters/Base/ConverterBase.cs`
  - `src/NeuroModFlowNet.ONNX/Converters/Abstract/AdapterBuilder.cs`
  - `src/NeuroModFlowNet.ONNX/Converters/Bgr/ConverterMatSingleBgrDirectU8.cs`
  - `src/NeuroModFlowNet.ONNX/Converters/NchwPos/ConverterMatSingleNchwPosCvdnnFP32.cs`
  - `src/NeuroModFlowNet.ONNX/Obsolete/Adapters/PaddleOCR/PaddleOCRDetAdapter.cs`
  - `src/NeuroModFlowNet.ONNX/Obsolete/Adapters/PaddleOCR/PaddleOCRRecAdapter.cs`
- Дополнительно выполнен поиск по `src/NeuroModFlowNet.ONNX`:
  - по `IAdapterInfo`
  - по `InputType`
  - по `AdapterName`
  - по возможному runtime/reflection-использованию `IAdapterInfo` / `InputType`

### Что понял
- Акцент на `obsolete` действительно меняет трактовку риска 2:
  - проблема выглядит не как "active orchestration сломан из-за `IAdapterInfo`";
  - проблема выглядит как legacy API drift, где старый сценарий introspection/select-by-input-type уже почти исчез, но его следы остались в public surface.
- `IAdapterInfo` сейчас живёт очень слабо как настоящий контракт:
  - интерфейс объявлен отдельно в `Abstract/IAdapterInfo.cs`;
  - но по текущему проходу активный код почти не использует сам интерфейс ни в `is/as`, ни через reflection, ни в factory/builder слое.
- В active-коде живёт скорее не интерфейс, а "несвязанные свойства":
  - `AdapterName` является частью общего `IInputConverter<TIn>`;
  - `InputType` присутствует только у части конвертеров как ad hoc свойство;
  - эти свойства не образуют подтверждённый orchestration-механизм.
- `obsolete`-ветка хорошо объясняет происхождение этого следа:
  - `PaddleOCRDetAdapter` в obsolete ещё явно оформлен как адаптер со своим `InputType`;
  - `PaddleOCRRecAdapter` уже не содержит `InputType`, то есть даже внутри legacy-линии поведение было не единым.
- Builder/factory слой подтверждает сдвиг архитектуры:
  - `AdapterBuilder` работает через generic delegates и compile-time типы;
  - active `CreateRunner(...)` в factory-коде собирают runner по metadata модели и generic-типам, а не по `IAdapterInfo`.

### Найденные риски
- Риск 1. Несогласованность контракта `IRunner.As<T>()` с реализацией.
  - Статус: не перепроверялся на этом шаге.
- Риск 2. `IAdapterInfo` выглядит как частично висящий контракт в активной архитектуре.
  - Статус: уточнён.
  - Актуальная формулировка после explicit obsolete-прохода:
    - `IAdapterInfo` сегодня больше похож на legacy/obsolete residue, чем на реально действующий active contract;
    - active pipeline и factory/builder слой не показали подтверждённого runtime-использования этого интерфейса;
    - `InputType` сохранился только у части active converter-ов и выглядит как остаток старой модели introspection.
- Риск 4. PaddleOCR-конвертеры обходят общий lifecycle `ConverterBase`, из-за чего поведение drift-ит относительно остального слоя.
  - Статус: подтверждение не изменилось.
- Риск 5. `PaddleOCRMaskExtractor` находится в active-коде, но `Extract()` фактически не реализован и всегда возвращает `default!`.
  - Статус: подтверждение не изменилось.
- Риск 6. `PaddleOCRRecExtractor` смешивает extraction с filesystem/console side effects.
  - Статус: подтверждение не изменилось.
- Риск 11. `YoloSeg` хранит batch-aware metadata, но concrete FP32/FP16 extractor logic фактически обрабатывает только первый батч.
  - Статус: смягчён fail-fast, но полностью не закрыт.
- Риск 12. Полная solution-верификация сейчас заблокирована инфраструктурно.
  - Статус: подтверждение не изменилось.
- Риск 13. В `samples` и `tests` уже есть независимые compile errors, не связанные напрямую с текущими исправлениями.
  - Статус: подтверждение не изменилось.

### Устаревшие выводы
- Устарела ранняя интерпретация риска 2 как проблемы преимущественно active-конвертеров.
  - Почему устарело:
    - explicit obsolete-проход показал, что корень проблемы скорее в legacy drift;
    - active pipeline не демонстрирует, что `IAdapterInfo` реально участвует в выборе адаптера;
    - сам интерфейс сегодня ближе к "остаточному публичному обещанию", чем к рабочему механизму.

### Что делать дальше
- Следующий логичный шаг по этой линии:
  - проверить, нужен ли `IAdapterInfo` вообще для внешних потребителей вне репозитория;
  - если нет, то это кандидат на перевод в legacy/obsolete surface или на полное удаление в будущей cleanup-волне;
  - если да, то надо явно восстановить его роль в active pipeline, а не оставлять как полумёртвый контракт.
- Самый полезный следующий маленький шаг внутри текущего репозитория:
  - перейти к `IRunner.As<T>()`, потому что это второй похожий случай contract drift, но уже без legacy-объяснения и ближе к активному public API.

### Вопросы к человеку
- Есть ли внешний код вне этого репозитория, который читает `InputType` у converter-ов или ожидает наличие `IAdapterInfo` как public capability?

## Шаг 10 - `OnnxModel` как `sealed`: код, obsolete-следы и документация

### Что проверено
- Перечитан `PROJECT_REVIEW.md` перед новым проходом.
- Прочитаны файлы:
  - `src/NeuroModFlowNet.ONNX/ModelWokers/Base/OnnxModel.cs`
  - `src/NeuroModFlowNet.ONNX/Inference/Extensions/OnnxModelExtensions.cs`
  - `src/NeuroModFlowNet.ONNX/Obsolete/AdapterBuilder.cs.xxx`
- Дополнительно выполнен поиск по исходникам без `bin/obj`:
  - по упоминаниям `OnnxModel`
  - по формулировкам про "base class"
  - по generic constraints `where TModel : OnnxModel`
  - по возможным следам наследования/расширения `OnnxModel`

### Что понял
- В active-коде `OnnxModel` уже однозначно воспринимается как concrete runtime object:
  - `IInputConverter<TIn>.SetModel(OnnxModel model)`
  - `IResultExtractor.SetModel(OnnxModel model)`
  - `ResultExtractorBase<TOut>.Model`
  - `OnnxModelExtensions`
  - active factories и раннеры
  Все эти места работают с конкретным типом `OnnxModel`, а не с его расширяемой иерархией.
- Следов реального active-наследования от `OnnxModel` в текущем репозитории не найдено.
- Основной drift после перевода в `sealed` остался в двух формах:
  - документация в самом `OnnxModel` всё ещё называет его `Base class`;
  - obsolete helper-код всё ещё использует generic constraints `where TModel : OnnxModel`, хотя при `sealed` это уже не про расширяемость, а фактически про "ровно `OnnxModel`".

### Найденные риски
- Риск 1. Несогласованность контракта `IRunner.As<T>()` с реализацией.
  - Статус: не перепроверялся на этом шаге.
- Риск 2. `IAdapterInfo` выглядит как legacy/obsolete residue, а не как active contract.
  - Статус: не менялся на этом шаге.
- Риск 12. Полная solution-верификация сейчас заблокирована инфраструктурно.
  - Статус: подтверждение не изменилось.
- Риск 13. В `samples` и `tests` уже есть независимые compile errors, не связанные напрямую с текущими исправлениями.
  - Статус: подтверждение не изменилось.
- Риск 14. Документация `OnnxModel` больше не соответствует реальному статусу типа.
  - `src/NeuroModFlowNet.ONNX/ModelWokers/Base/OnnxModel.cs:13`
  - XML-summary говорит `Base class for high-performance ONNX model inference`, хотя сам тип объявлен как `public sealed class OnnxModel`.
  - Риск: читатель API получает неверный архитектурный сигнал, будто `OnnxModel` предназначен для наследования или служит базой для специализированных model-классов.
- Риск 15. В obsolete helper-коде остались generic constraints, подразумевающие расширяемость `OnnxModel`.
  - `src/NeuroModFlowNet.ONNX/Obsolete/AdapterBuilder.cs.xxx:24`, `37`, `50`
  - `InputAdapter<TModel, T>`, `InputAdapterViaConverter<TModel, T>` и `AdapterBuilder<TConverters, TModel, TOut>` используют `where TModel : OnnxModel`.
  - При `sealed` это ограничение больше не описывает полезную полиморфность и выглядит как след прежней архитектуры.
  - Так как файл obsolete, это не active runtime defect, но это вводящий в заблуждение legacy artefact и плохой reference point для будущей ревизии/рефакторинга.

### Устаревшие выводы
- Новых полностью опровергнутых выводов на этом шаге нет.
- Но уточнилась трактовка вокруг `OnnxModel`:
  - на active пути проблема не в том, что кто-то всё ещё реально наследуется от `OnnxModel`;
  - проблема в том, что документация и obsolete-code всё ещё хранят старое архитектурное сообщение.

### Что делать дальше
- Следующий маленький шаг по этой линии:
  - проверить `IRunner.As<T>()` на том же уровне "код + комментарий + public contract";
  - если потом будет разрешение на правки, то `OnnxModel`-линия просится на очень локальный cleanup:
    - поправить XML-summary в `OnnxModel.cs`;
    - решить, нужен ли `Obsolete/AdapterBuilder.cs.xxx` как архив, или его хотя бы стоит явно пометить как исторический слой, не отражающий текущую архитектуру.

### Вопросы к человеку
- Для тебя `Obsolete/*.xxx` нужно рассматривать как просто архив, который может содержать устаревшие архитектурные сигналы, или ты хочешь, чтобы даже этот слой был приведён в согласованное состояние по документации и generic-ограничениям?
