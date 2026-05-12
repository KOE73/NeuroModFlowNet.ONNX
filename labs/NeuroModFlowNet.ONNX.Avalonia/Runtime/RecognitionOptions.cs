using NeuroModFlowNet.ONNX;

namespace NeuroModFlowNet.ONNX.Avalonia.Runtime;

/// <summary>
/// EN: Stores live OCR and ROI processing options edited by the Avalonia settings panel.
/// RU: Хранит live-настройки OCR и ROI processing, которые редактируются через Avalonia-панель настроек.
/// </summary>
/// <remarks>
/// EN: The object owns the ROI processing pipeline and exposes small adjustment methods for buttons. Changing recognition
/// input width or height increments <see cref="RecognitionShapeVersion"/> so the inference resources can rebuild PaddleOCR
/// Rec persistent buffers.
/// RU: Объект владеет ROI processing pipeline и предоставляет небольшие методы регулировки для кнопок. Изменение ширины
/// или высоты входа recognition увеличивает <see cref="RecognitionShapeVersion"/>, чтобы inference resources пересоздали
/// persistent buffers PaddleOCR Rec.
/// </remarks>
public sealed class RecognitionOptions : IDisposable
{
    const int BatchMin = 1;
    const int BatchMax = 16;
    const int RecognitionInputHeightMin = 16;
    const int RecognitionInputHeightMax = 128;
    const int RecognitionInputWidthMin = 64;
    const int RecognitionInputWidthMax = 2048;
    const float RoiHeightScaleMin = 0.5f;
    const float RoiHeightScaleMax = 6f;
    const double RoiDisplayScaleMin = 0.5;
    const double RoiDisplayScaleMax = 6.0;

    readonly TextRegionBrightnessContrastStage brightnessContrastStage = new(40, 115);
    readonly TextRegionGammaCorrectionStage gammaCorrectionStage = new(5.0);
    readonly TextRegionProcessingPipeline processingPipeline;

    public RecognitionOptions()
    {
        processingPipeline = new TextRegionProcessingPipeline(
            brightnessContrastStage,
            gammaCorrectionStage);
    }

    public int FrameWidth { get; private set; } = 640;
    public int BatchSize { get; private set; } = 1;
    public int RecognitionInputHeight { get; private set; } = 48;
    public int RecognitionInputWidth { get; private set; } = 640;
    public float RoiHeightScale { get; private set; } = 2f;
    public double RoiDisplayScale { get; private set; } = 1.0;
    public int RecognitionShapeVersion { get; private set; }
    public int RecognitionMode { get; set; } = 1;
    public bool ProcessingEnabled { get; set; } = true;
    public InferenceSelectionOptions InferenceSelection { get; } = new();
    public double Brightness => brightnessContrastStage.Brightness;
    public double ContrastPercent => brightnessContrastStage.ContrastPercent;
    public double Gamma => gammaCorrectionStage.Gamma;
    public ITextRegionProcessingStage? ProcessingStage => ProcessingEnabled ? processingPipeline : null;
    public int RecognitionOutputItemCount => RecognitionInputWidth / PaddleOCRRecExtractor.OutputWidthStride;

    public void AdjustFrameWidth(int delta) =>
        FrameWidth = Math.Clamp(FrameWidth + delta, 320, 1920);

    public void AdjustBatchSize(int delta) =>
        BatchSize = Math.Clamp(BatchSize + delta, BatchMin, BatchMax);

    public void AdjustRecognitionInputWidth(int delta)
    {
        int alignedWidth = AlignToStride(RecognitionInputWidth + delta);
        int newWidth = Math.Clamp(alignedWidth, RecognitionInputWidthMin, RecognitionInputWidthMax);
        if(newWidth == RecognitionInputWidth) return;

        RecognitionInputWidth = newWidth;
        RecognitionShapeVersion++;
    }

    public void AdjustRecognitionInputHeight(int delta)
    {
        int newHeight = Math.Clamp(RecognitionInputHeight + delta, RecognitionInputHeightMin, RecognitionInputHeightMax);
        if(newHeight == RecognitionInputHeight) return;

        RecognitionInputHeight = newHeight;
        RecognitionShapeVersion++;
    }

    public void AdjustRoiHeightScale(float delta) =>
        RoiHeightScale = Math.Clamp(RoiHeightScale + delta, RoiHeightScaleMin, RoiHeightScaleMax);

    public void AdjustRoiDisplayScale(double delta) =>
        RoiDisplayScale = Math.Clamp(RoiDisplayScale + delta, RoiDisplayScaleMin, RoiDisplayScaleMax);

    public void AdjustBrightness(double delta) =>
        brightnessContrastStage.Brightness = Math.Clamp(brightnessContrastStage.Brightness + delta, -255, 255);

    public void AdjustContrast(double delta) =>
        brightnessContrastStage.ContrastPercent = Math.Clamp(brightnessContrastStage.ContrastPercent + delta, 0, 300);

    public void AdjustGamma(double delta) =>
        gammaCorrectionStage.Gamma = Math.Clamp(gammaCorrectionStage.Gamma + delta, 0.1, 10);

    public void RotateProcessingStages()
    {
        if(processingPipeline.Count < 2) return;

        processingPipeline.Move(0, processingPipeline.Count - 1);
    }

    public string GetProcessingStageList() =>
        string.Join(" -> ", processingPipeline.GetStages().Select(stage => stage.Name));

    public string FormatRecognitionText(PaddleOCRRecExtractor.OcrResult recognitionResult) =>
        RecognitionMode switch
        {
            1 => recognitionResult.Standard,
            2 => recognitionResult.WithSpaces,
            3 => recognitionResult.FullCandidates,
            _ => recognitionResult.Standard,
        };

    public void Dispose()
    {
        processingPipeline.Dispose();
    }

    private static int AlignToStride(int value)
    {
        int stride = PaddleOCRRecExtractor.OutputWidthStride;
        return Math.Max(stride, (int)Math.Round(value / (double)stride) * stride);
    }
}
