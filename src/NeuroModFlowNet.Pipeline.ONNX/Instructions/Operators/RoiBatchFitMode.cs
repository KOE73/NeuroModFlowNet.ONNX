namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// EN: How a region is fitted into a fixed ROI batch canvas: <see cref="Letterbox"/> keeps the aspect ratio and centers
/// (YOLO letterbox), <see cref="Stretch"/> maps the region onto the whole canvas (plain resize, as the CGP bag detector
/// was trained). Explicit choice, no automatic switching.
///
/// RU: Как регион укладывается в холст фиксированного ROI-батча: <see cref="Letterbox"/> сохраняет пропорции и
/// центрирует (YOLO letterbox), <see cref="Stretch"/> растягивает регион на весь холст (обычный resize, как обучался
/// детектор мешков CGP). Выбор явный, без автоматического переключения.
/// </summary>
public enum RoiBatchFitMode
{
    Letterbox,
    Stretch
}
