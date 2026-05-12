using OpenCvSharp;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Extracts OCR-ready image crops from text regions already expressed in source-image coordinates.
/// RU: Вырезает готовые для OCR изображения из текстовых областей, уже приведенных к координатам исходного кадра.
/// </summary>
/// <remarks>
/// EN: The extractor deliberately works with <see cref="OcrQuadRegion"/> instead of YOLO-specific boxes.
/// This keeps the crop step reusable for OBB detectors, column analyzers, manual regions, and future ONNX-based
/// text-region predictors while preserving the fast application path through a small mapper.
/// RU: Вырезалка намеренно принимает <see cref="OcrQuadRegion"/>, а не YOLO-специфичный бокс.
/// Так один и тот же шаг можно использовать для OBB-детекторов, анализаторов колонок, ручных областей
/// и будущих ONNX-вариантов выделения текста, не ломая быстрый прикладной путь через небольшой mapper.
/// </remarks>
public interface ITextRegionExtractor
{
    bool TryExtract(Mat sourceMat, in OcrQuadRegion sourceRegion, in TextRegionExtractionOptions options, out Mat? recognitionRoi);

    int ExtractMany(
        Mat sourceMat,
        ReadOnlySpan<OcrQuadRegion> sourceRegions,
        in TextRegionExtractionOptions options,
        List<Mat> destination);
}
