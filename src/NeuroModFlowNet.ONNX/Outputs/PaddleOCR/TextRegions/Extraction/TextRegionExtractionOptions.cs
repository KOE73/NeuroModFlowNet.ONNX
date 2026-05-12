namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Per-call extraction options for the OCR recognition crop size and optional post-processing.
/// RU: Параметры одного вызова вырезания: размер OCR-кропа и необязательная постобработка.
/// </summary>
/// <remarks>
/// EN: Options are a small readonly value because extraction is a hot path and should not depend on mutable
/// application state. The processing stage is injected here so crop geometry stays separate from brightness,
/// contrast, gamma, and other image-normalization experiments.
/// RU: Опции сделаны маленьким readonly-значением, потому что вырезание находится в hot path и не должно
/// зависеть от изменяемого состояния приложения. Processing stage передается отдельно, чтобы геометрия
/// вырезания не смешивалась с яркостью, контрастом, гаммой и другими экспериментами нормализации изображения.
/// </remarks>
public readonly record struct TextRegionExtractionOptions(
    int TargetWidth,
    int TargetHeight,
    ITextRegionProcessingStage? ProcessingStage = null);
