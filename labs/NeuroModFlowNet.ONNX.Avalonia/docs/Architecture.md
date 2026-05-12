# NeuroModFlowNet.ONNX.Avalonia

Локальная карта проекта. Это lab UI, а не архитектурный источник истины для основной библиотеки.

## Entry Point

| Файл | Ответственность |
| --- | --- |
| `Program.cs` | Инициализирует ONNX Runtime пути, задает `MODELS_ROOT_PATH` по умолчанию и запускает Avalonia desktop lifetime. |
| `App.axaml` | Подключает базовую Fluent тему Avalonia. |
| `App.axaml.cs` | Создает главное окно при старте desktop lifetime. |
| `App.config` | Локальные настройки моделей, backend-ов и `VideoSource`, аналогично консольной lab-демке. |

## Main Window

| Файл | Ответственность |
| --- | --- |
| `MainWindow.axaml` | Главная компоновка окна: левое меню режимов, центральная video/ROI/recognition область, регулируемые вертикальные splitters, правые настройки, нижние метрики. |
| `MainWindow.axaml.cs` | Склеивает UI controls с `RealTimeInferenceEngine`, принимает кадры из background loop и передает их в visual controls на UI thread. |

## Controls

| Файл | Ответственность |
| --- | --- |
| `Controls/ModeNavigationView.axaml` | Левый inference selector: OCR/Box/OBB/Segmentation/Classification/Pose чекбоксы плюс compact model metadata. |
| `Controls/ModeNavigationView.axaml.cs` | Синхронизирует inference slot чекбоксы с `RecognitionOptions` и отображает model input/output shapes. |
| `Controls/VideoSceneView.axaml` | Центральная область: Skia frame layer, Skia overlay layer, регулируемый splitter и список OCR-результатов. |
| `Controls/VideoSceneView.axaml.cs` | Методы обновления кадра, overlay и OCR list. Каждая OCR-строка строится как ROI-картинка слева и текст справа; preview показывает полный подготовленный ROI без обрезки темных полей. |
| `Controls/OcrSettingsView.axaml` | Правая панель настройки OCR: Start/Stop, frame width, batch, ROI processing, brightness, contrast, gamma, порядок stage-ов, режим текста. |
| `Controls/OcrSettingsView.axaml.cs` | Обработчики кнопок и синхронизация OCR-настроек с `RecognitionOptions`. |
| `Controls/MetricsPanelView.axaml` | Нижняя компактная диагностическая строка FPS/timing/items с моноширинным шрифтом. |
| `Controls/MetricsPanelView.axaml.cs` | Отображает `RealTimeMetricsSnapshot`. |

## Rendering

| Файл | Ответственность |
| --- | --- |
| `Rendering/SkiaFrameView.cs` | Avalonia control для отображения `Mat` кадра. Хранит последнюю копию кадра и вызывает custom Skia draw operation. |
| `Rendering/SkiaFrameDrawOperation.cs` | `ICustomDrawOperation`: берет `ISkiaSharpApiLeaseFeature`, получает `SKCanvas` и вызывает отрисовку кадра. |
| `Rendering/SkiaMatPainter.cs` | Рисует BGRA `Mat` напрямую как `SKImage.FromPixels`, без PNG/JPG/stream/Bitmap. |
| `Rendering/SkiaOverlayView.cs` | Avalonia control для overlay поверх кадра. Хранит snapshot боксов и подписей. |
| `Rendering/SkiaOverlayDrawOperation.cs` | `ICustomDrawOperation` для overlay layer. |
| `Rendering/SkiaOverlayPainter.cs` | Рисует OBB, centers и OCR labels через Skia с пересчетом координат в fitted viewport. |

## Runtime

| Файл | Ответственность |
| --- | --- |
| `Runtime/RealTimeInferenceEngine.cs` | Background loop: capture, resize, выбранные inference slots, OCR ROI preparation, recognition, OpenCV visualization для Seg/Cls/Pose, Skia overlay snapshot, metrics, событие `FrameReady`. |
| `Runtime/InferenceResources.cs` | Загрузка OCR/Box/OBB/Seg/Cls/Pose моделей, создание ONNX contexts/runners, переинициализация recognition batch. |
| `Runtime/RealTimeAvaloniaSettings.cs` | Чтение `App.config` в отдельный settings object, включая имена моделей Box/OBB/Seg/Cls/Pose. |
| `Runtime/RecognitionOptions.cs` | Live-настройки OCR, ROI processing pipeline и selection flags: frame width, Rec input width/height, ROI height scale, ROI display scale, batch, text mode, brightness, contrast, gamma, порядок stage-ов. |
| `Runtime/InferenceSelectionOptions.cs` | Live-флаги включения OCR/Box/OBB/Seg/Cls/Pose inference slots. Выключенный slot не вызывает `Predict` в realtime loop. |
| `Runtime/RuntimeModelInfo.cs` | Компактное описание model input/output shapes для левого inference-блока. |
| `Runtime/RealTimeOneFrameData.cs` | Передача данных одного готового кадра из background loop в UI. Владеет `Mat` кадра и списком ROI-результатов. |
| `Runtime/RealTimeRecognitionItemData.cs` | Один OCR-result для UI: ROI image `Mat` плюс распознанный текст. |
| `Runtime/RealTimeMetricsSnapshot.cs` | Снимок FPS и timing metrics. |
| `Runtime/FrameOverlaySnapshot.cs` | Immutable snapshot overlay-данных для Skia layer. |
| `Runtime/OverlayObb.cs` | Данные одного OBB для overlay. |
| `Runtime/OverlayText.cs` | Данные одной текстовой подписи для overlay. |

## Shared Lab Code

| Подключение | Ответственность |
| --- | --- |
| `src/NeuroModFlowNet.ONNX/Outputs/PaddleOCR/Roi` | ROI extractor, coordinate mapper и processing stages находятся в основной библиотеке. Avalonia и консольная lab-демка используют один библиотечный код без `Compile LinkBase`. |
| `NeuroModFlowNet.ONNX.Visualizer` | Сейчас используется для `Letterbox` и цветовой логики. Текущие painter-ы в основном рисуют в `Mat`, поэтому Avalonia overlay имеет отдельный Skia painter. |

## Known Runtime Risks

| Риск | Что проверить |
| --- | --- |
| Нажатие `Start` не дает видео | Смотреть статус справа. Сейчас модели инициализируются до первого camera frame, поэтому долгий старт может выглядеть как зависание. |
| PaddleOCR Rec падает на отрицательном tensor shape | Проверить, что `InferenceResources.EnsureRecognitionBatch(...)` вызван до создания `RunnerRec`. Для Rec модели dynamic output нельзя оставлять на автоинициализацию persistent output. |
| Recognition classes hardcoded | Не использовать `438`. Размер третьей оси output берется из `ModelRec.ModelOutputShapes[PrimaryOutputName][2]`. |
| OCR-строка без картинки | `SkiaFrameView` принимает BGRA/BGR/GRAY и приводит к BGRA перед отрисовкой. Если картинка пустая, проверять `NaiveTextRegionExtractor`. |
| Кажется, что `Start` не нажали | Окно запускает engine автоматически после инициализации UI. Правую панель статуса смотреть первой: `Starting...`, `Initializing models...`, `Running` или ошибка camera-only. |
| Overlay не совпадает с кадром | Проверять mapping `LetterboxInfo -> LetterboxCoordinateMapper -> source frame`. |
| `Visualizer` не покрывает Avalonia | Нужен backend-neutral visualizer contract: geometry/style отдельно, OpenCV/Skia/Avalonia renderers отдельно. |
