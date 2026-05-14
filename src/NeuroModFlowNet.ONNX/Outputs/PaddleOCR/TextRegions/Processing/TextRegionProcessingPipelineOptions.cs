namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: JSON-friendly OCR ROI processing pipeline definition.
/// RU: JSON-совместимое описание pipeline обработки OCR ROI.
/// </summary>
public sealed class TextRegionProcessingPipelineOptions
{
    public bool Enabled { get; set; } = true;

    public List<TextRegionProcessingStageOptions> Stages { get; set; } = [];
}
