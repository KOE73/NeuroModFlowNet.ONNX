namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: JSON-friendly overlap suppression options for OCR text regions.
/// RU: JSON-совместимые настройки удаления сильно пересекающихся OCR-регионов.
/// </summary>
public sealed class OcrOverlapSuppressionOptions
{
    public bool Enabled { get; set; } = true;

    public float Ratio { get; set; } = 0.8f;
}
