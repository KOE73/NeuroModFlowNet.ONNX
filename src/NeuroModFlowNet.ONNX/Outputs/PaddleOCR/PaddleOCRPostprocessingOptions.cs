namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: JSON-friendly postprocessing block for PaddleOCR detection, OCR region analysis, and recognition ROI preparation.
/// RU: JSON-совместимый блок постобработки PaddleOCR detection, анализа OCR-регионов и подготовки ROI для recognition.
/// </summary>
public sealed class PaddleOCRPostprocessingOptions
{
    public PaddleOCRDetMaskRegionExtractorOptions DetMask { get; set; } = new();

    public OcrRegionPostprocessorJsonOptions Analyzer { get; set; } = new();

    public OcrRecognitionRoiOptions RecognitionRoi { get; set; } = new();

    public OcrRegionPostprocessorOptions CreateAnalyzerRuntimeOptions() =>
        Analyzer.ToRuntimeOptions(RecognitionRoi.NormalizedTargetWidth / (float)RecognitionRoi.NormalizedTargetHeight);
}
