namespace NeuroModFlowNet.ONNX.Avalonia.Runtime;

/// <summary>
/// EN: Stores which realtime inference slots are enabled by the operator.
/// RU: Хранит, какие realtime inference слоты включены оператором.
/// </summary>
/// <remarks>
/// EN: Disabled slots are not only hidden from the UI; the engine skips their Predict call. OCR is a pipeline slot and
/// internally uses OBB detection for ROI extraction.
/// RU: Выключенные слоты не просто скрываются в UI; engine пропускает их Predict-вызов. OCR является pipeline-слотом и
/// внутри использует OBB detection для вырезания ROI.
/// </remarks>
public sealed class InferenceSelectionOptions
{
    public bool OcrEnabled { get; set; } = true;
    public bool BoxDetectionEnabled { get; set; }
    public bool ObbDetectionEnabled { get; set; }
    public bool SegmentationEnabled { get; set; }
    public bool ClassificationEnabled { get; set; }
    public bool PoseEnabled { get; set; }
}
