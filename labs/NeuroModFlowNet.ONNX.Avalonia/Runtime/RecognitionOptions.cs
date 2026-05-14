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
    const float RoiAdaptiveBasePadMin = 0f;
    const float RoiAdaptiveBasePadMax = 32f;
    const float RoiAdaptivePadRatioMin = 0f;
    const float RoiAdaptivePadRatioMax = 2f;
    const float RoiAdaptiveMaxPadMin = 0f;
    const float RoiAdaptiveMaxPadMax = 128f;
    const double RoiDisplayScaleMin = 0.5;
    const double RoiDisplayScaleMax = 6.0;
    const int MaxRegionsPerFrameMin = 1;
    const int MaxRegionsPerFrameMax = 512;
    const float RegionAspectRatioMin = 0f;
    const float RegionAspectRatioMax = 50f;

    readonly TextRegionBrightnessContrastStage brightnessContrastStage = new(0, 100);
    readonly TextRegionGammaCorrectionStage gammaCorrectionStage = new(1.0);
    readonly TextRegionProcessingPipeline processingPipeline;
    readonly List<ITextRegionProcessingStage> removedProcessingStages = [];
    int selectedProcessingStageIndex;

    public RecognitionOptions()
    {
        processingPipeline = new TextRegionProcessingPipeline(
            brightnessContrastStage,
            gammaCorrectionStage);
    }

    public int FrameWidth { get; private set; } = 640;
    public int BatchSize { get; private set; } = 1;
    // Local rec.onnx metadata reports [batch, 3, 48, width]; the external PaddleOCR config says height=32.
    // Runtime follows the ONNX contract, so keep 48 unless the model file is replaced.
    public int RecognitionInputHeight { get; private set; } = 48;
    public int RecognitionInputWidth { get; private set; } = 320;
    public float RoiHeightScale { get; private set; } = 2f;
    public bool AdaptiveRoiHeightEnabled { get; set; } = true;
    public float RoiAdaptiveBasePad { get; private set; } = 1f;
    public float RoiAdaptivePadRatio { get; private set; } = 0.25f;
    public float RoiAdaptiveMaxPad { get; private set; } = 8f;
    public double RoiDisplayScale { get; private set; } = 1.0;
    public float MinRegionHeight { get; private set; }
    public float MaxRegionHeight { get; private set; } = float.PositiveInfinity;
    public float MinRegionWidth { get; private set; }
    public float MaxRegionWidth { get; private set; } = float.PositiveInfinity;
    public float MinRegionAspectRatio { get; private set; }
    public float MaxRegionAspectRatio { get; private set; } = float.PositiveInfinity;
    public int MaxRegionsPerFrame { get; private set; } = 64;
    public bool OverlapSuppressionEnabled { get; set; } = true;
    public float OverlapSuppressionRatio { get; private set; } = 0.8f;
    public bool LineMergeEnabled { get; set; }
    public float MergeAngleDeltaDegrees { get; private set; } = 12f;
    public float MergeNormalOffsetInHeights { get; private set; } = 0.75f;
    public float MergeHeightRatio { get; private set; } = 1.8f;
    public float MergeGapInHeights { get; private set; } = 2.5f;
    public float MinimumMergedCoverageRatio { get; private set; } = 0.45f;
    public float MaxMergedRegionWidth { get; private set; } = float.PositiveInfinity;
    public float MaxMergedRegionAspectRatio { get; private set; } = float.PositiveInfinity;
    public int RecognitionShapeVersion { get; private set; }
    public int RecognitionMode { get; set; } = 1;
    public bool ProcessingEnabled { get; set; } = true;
    public InferenceSelectionOptions InferenceSelection { get; } = new();
    public double Brightness => brightnessContrastStage.Brightness;
    public double ContrastPercent => brightnessContrastStage.ContrastPercent;
    public double Gamma => gammaCorrectionStage.Gamma;
    public ITextRegionProcessingStage? ProcessingStage => ProcessingEnabled ? processingPipeline : null;
    public int SelectedProcessingStageIndex => selectedProcessingStageIndex;
    public ITextRegionProcessingStage? SelectedProcessingStage => GetProcessingStage(selectedProcessingStageIndex);
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

    public void ToggleAdaptiveRoiHeight() =>
        AdaptiveRoiHeightEnabled = !AdaptiveRoiHeightEnabled;

    public void AdjustRoiAdaptiveBasePad(float delta) =>
        RoiAdaptiveBasePad = Math.Clamp(RoiAdaptiveBasePad + delta, RoiAdaptiveBasePadMin, RoiAdaptiveBasePadMax);

    public void AdjustRoiAdaptivePadRatio(float delta) =>
        RoiAdaptivePadRatio = Math.Clamp(RoiAdaptivePadRatio + delta, RoiAdaptivePadRatioMin, RoiAdaptivePadRatioMax);

    public void AdjustRoiAdaptiveMaxPad(float delta) =>
        RoiAdaptiveMaxPad = Math.Clamp(RoiAdaptiveMaxPad + delta, RoiAdaptiveMaxPadMin, RoiAdaptiveMaxPadMax);

    public void ResetAdaptiveRoiHeight()
    {
        RoiAdaptiveBasePad = 1f;
        RoiAdaptivePadRatio = 0.25f;
        RoiAdaptiveMaxPad = 8f;
    }

    public RoiHeightDebugData CalculateRoiHeightDebug(float sourceRegionHeight)
    {
        if(!AdaptiveRoiHeightEnabled)
            return new RoiHeightDebugData(sourceRegionHeight, MathF.Max(0, sourceRegionHeight * (RoiHeightScale - 1f) * 0.5f), RoiHeightScale);

        if(sourceRegionHeight <= 0)
            return new RoiHeightDebugData(sourceRegionHeight, 0, RoiHeightScale);

        float pad = Math.Min(RoiAdaptiveMaxPad, RoiAdaptiveBasePad + sourceRegionHeight * RoiAdaptivePadRatio);
        float scaledHeight = sourceRegionHeight + pad * 2f;
        float scale = Math.Clamp(scaledHeight / sourceRegionHeight, RoiHeightScaleMin, RoiHeightScaleMax);
        return new RoiHeightDebugData(sourceRegionHeight, pad, scale);
    }

    public float CalculateRoiHeightScale(float sourceRegionHeight) =>
        CalculateRoiHeightDebug(sourceRegionHeight).Scale;

    public void AdjustRoiDisplayScale(double delta) =>
        RoiDisplayScale = Math.Clamp(RoiDisplayScale + delta, RoiDisplayScaleMin, RoiDisplayScaleMax);

    public void AdjustMinRegionHeight(float delta) =>
        MinRegionHeight = Math.Max(0, MinRegionHeight + delta);

    public void AdjustMaxRegionHeight(float delta) =>
        MaxRegionHeight = NormalizeOptionalMaximum(MaxRegionHeight, delta);

    public void ResetMaxRegionHeight() =>
        MaxRegionHeight = float.PositiveInfinity;

    public void ResetRegionHeightLimits()
    {
        MinRegionHeight = 0;
        MaxRegionHeight = float.PositiveInfinity;
    }

    public void AdjustMinRegionWidth(float delta) =>
        MinRegionWidth = Math.Max(0, MinRegionWidth + delta);

    public void AdjustMaxRegionWidth(float delta) =>
        MaxRegionWidth = NormalizeOptionalMaximum(MaxRegionWidth, delta);

    public void ResetMaxRegionWidth() =>
        MaxRegionWidth = float.PositiveInfinity;

    public void ResetRegionWidthLimits()
    {
        MinRegionWidth = 0;
        MaxRegionWidth = float.PositiveInfinity;
    }

    public void AdjustMinRegionAspectRatio(float delta) =>
        MinRegionAspectRatio = Math.Clamp(MinRegionAspectRatio + delta, RegionAspectRatioMin, RegionAspectRatioMax);

    public void AdjustMaxRegionAspectRatio(float delta) =>
        MaxRegionAspectRatio = NormalizeOptionalMaximum(MaxRegionAspectRatio, delta);

    public void ResetMaxRegionAspectRatio() =>
        MaxRegionAspectRatio = float.PositiveInfinity;

    public void ResetRegionAspectRatioLimits()
    {
        MinRegionAspectRatio = 0;
        MaxRegionAspectRatio = float.PositiveInfinity;
    }

    public void AdjustMaxRegionsPerFrame(int delta) =>
        MaxRegionsPerFrame = Math.Clamp(MaxRegionsPerFrame + delta, MaxRegionsPerFrameMin, MaxRegionsPerFrameMax);

    public void ToggleOverlapSuppression() =>
        OverlapSuppressionEnabled = !OverlapSuppressionEnabled;

    public void AdjustOverlapSuppressionRatio(float delta) =>
        OverlapSuppressionRatio = Math.Clamp(OverlapSuppressionRatio + delta, 0f, 1f);

    public void ToggleLineMerge() =>
        LineMergeEnabled = !LineMergeEnabled;

    public void AdjustMergeAngleDeltaDegrees(float delta) =>
        MergeAngleDeltaDegrees = Math.Clamp(MergeAngleDeltaDegrees + delta, 0f, 90f);

    public void AdjustMergeNormalOffsetInHeights(float delta) =>
        MergeNormalOffsetInHeights = Math.Clamp(MergeNormalOffsetInHeights + delta, 0f, 10f);

    public void AdjustMergeHeightRatio(float delta) =>
        MergeHeightRatio = Math.Clamp(MergeHeightRatio + delta, 1f, 10f);

    public void AdjustMergeGapInHeights(float delta) =>
        MergeGapInHeights = Math.Clamp(MergeGapInHeights + delta, 0f, 20f);

    public void AdjustMinimumMergedCoverageRatio(float delta) =>
        MinimumMergedCoverageRatio = Math.Clamp(MinimumMergedCoverageRatio + delta, 0f, 1f);

    public void AdjustMaxMergedRegionWidth(float delta) =>
        MaxMergedRegionWidth = NormalizeOptionalMaximum(MaxMergedRegionWidth, delta);

    public void ResetMaxMergedRegionWidth() =>
        MaxMergedRegionWidth = float.PositiveInfinity;

    public void AdjustMaxMergedRegionAspectRatio(float delta) =>
        MaxMergedRegionAspectRatio = NormalizeOptionalMaximum(MaxMergedRegionAspectRatio, delta);

    public void ResetMaxMergedRegionAspectRatio() =>
        MaxMergedRegionAspectRatio = float.PositiveInfinity;

    public void AdjustBrightness(double delta) =>
        brightnessContrastStage.Brightness = Math.Clamp(brightnessContrastStage.Brightness + delta, -255, 255);

    public void ResetBrightness() =>
        brightnessContrastStage.Brightness = 0;

    public void AdjustContrast(double delta) =>
        brightnessContrastStage.ContrastPercent = Math.Clamp(brightnessContrastStage.ContrastPercent + delta, 0, 300);

    public void ResetContrast() =>
        brightnessContrastStage.ContrastPercent = 100;

    public void AdjustGamma(double delta) =>
        gammaCorrectionStage.Gamma = Math.Clamp(gammaCorrectionStage.Gamma + delta, 0.1, 10);

    public void ResetGamma() =>
        gammaCorrectionStage.Gamma = 1.0;

    public void RotateProcessingStages()
    {
        if(processingPipeline.Count < 2) return;

        processingPipeline.Move(0, processingPipeline.Count - 1);
        selectedProcessingStageIndex = Math.Clamp(selectedProcessingStageIndex, 0, processingPipeline.Count - 1);
    }

    public string GetProcessingStageList() =>
        string.Join(" -> ", processingPipeline.GetStages().Select(stage => stage.Name));

    public IReadOnlyList<string> GetProcessingStageNames() =>
        processingPipeline.GetStages().Select(FormatProcessingStageName).ToArray();

    public IReadOnlyList<ITextRegionProcessingStage> GetProcessingStages() =>
        processingPipeline.GetStages();

    public void SelectProcessingStage(int index)
    {
        if(processingPipeline.Count == 0)
        {
            selectedProcessingStageIndex = 0;
            return;
        }

        selectedProcessingStageIndex = Math.Clamp(index, 0, processingPipeline.Count - 1);
    }

    public void AddProcessingStage(TextRegionProcessingStageKind stageKind)
    {
        processingPipeline.Add(CreateProcessingStage(stageKind));
        selectedProcessingStageIndex = processingPipeline.Count - 1;
    }

    public void RemoveSelectedProcessingStage()
    {
        if(processingPipeline.Count == 0) return;

        ITextRegionProcessingStage removedStage = processingPipeline.RemoveAt(selectedProcessingStageIndex);
        removedProcessingStages.Add(removedStage);
        selectedProcessingStageIndex = processingPipeline.Count == 0
            ? 0
            : Math.Clamp(selectedProcessingStageIndex, 0, processingPipeline.Count - 1);
    }

    public void MoveSelectedProcessingStageUp()
    {
        if(selectedProcessingStageIndex <= 0) return;

        processingPipeline.Move(selectedProcessingStageIndex, selectedProcessingStageIndex - 1);
        selectedProcessingStageIndex--;
    }

    public void MoveSelectedProcessingStageDown()
    {
        if(selectedProcessingStageIndex >= processingPipeline.Count - 1) return;

        processingPipeline.Move(selectedProcessingStageIndex, selectedProcessingStageIndex + 1);
        selectedProcessingStageIndex++;
    }

    public void AdjustSelectedBrightness(double delta)
    {
        if(SelectedProcessingStage is ITextRegionBrightnessContrastSettings stage)
            stage.Brightness = Math.Clamp(stage.Brightness + delta, -255, 255);
    }

    public void ResetSelectedBrightness()
    {
        if(SelectedProcessingStage is ITextRegionBrightnessContrastSettings stage)
            stage.Brightness = 0;
    }

    public void AdjustSelectedContrast(double delta)
    {
        if(SelectedProcessingStage is ITextRegionBrightnessContrastSettings stage)
            stage.ContrastPercent = Math.Clamp(stage.ContrastPercent + delta, 0, 300);
    }

    public void ResetSelectedContrast()
    {
        if(SelectedProcessingStage is ITextRegionBrightnessContrastSettings stage)
            stage.ContrastPercent = 100;
    }

    public void AdjustSelectedGamma(double delta)
    {
        if(SelectedProcessingStage is ITextRegionGammaCorrectionSettings stage)
            stage.Gamma = Math.Clamp(stage.Gamma + delta, 0.1, 10);
    }

    public void ResetSelectedGamma()
    {
        if(SelectedProcessingStage is ITextRegionGammaCorrectionSettings stage)
            stage.Gamma = 1.0;
    }

    public void ToggleSelectedRedChannel()
    {
        if(SelectedProcessingStage is ITextRegionGrayscaleSettings stage)
            stage.UseRed = !stage.UseRed;
    }

    public void ToggleSelectedGreenChannel()
    {
        if(SelectedProcessingStage is ITextRegionGrayscaleSettings stage)
            stage.UseGreen = !stage.UseGreen;
    }

    public void ToggleSelectedBlueChannel()
    {
        if(SelectedProcessingStage is ITextRegionGrayscaleSettings stage)
            stage.UseBlue = !stage.UseBlue;
    }

    public void AdjustSelectedThreshold(double delta)
    {
        if(SelectedProcessingStage is ITextRegionThresholdSettings stage)
            stage.Threshold = Math.Clamp(stage.Threshold + delta, 0, 255);
    }

    public void ResetSelectedThreshold()
    {
        if(SelectedProcessingStage is ITextRegionThresholdSettings stage)
            stage.Threshold = 128;
    }

    public void ToggleSelectedOtsu()
    {
        if(SelectedProcessingStage is ITextRegionThresholdSettings stage)
            stage.UseOtsu = !stage.UseOtsu;
    }

    public void AdjustSelectedGaussianKernel(int delta)
    {
        if(SelectedProcessingStage is ITextRegionGaussianBlurSettings stage)
            stage.KernelSize = Math.Clamp(stage.KernelSize + delta, 1, 31);
    }

    public void AdjustSelectedGaussianSigma(double delta)
    {
        if(SelectedProcessingStage is ITextRegionGaussianBlurSettings stage)
            stage.Sigma = Math.Clamp(stage.Sigma + delta, 0, 20);
    }

    public void ResetSelectedGaussian()
    {
        if(SelectedProcessingStage is ITextRegionGaussianBlurSettings stage)
        {
            stage.KernelSize = 3;
            stage.Sigma = 0;
        }
    }

    public void AdjustSelectedSharpenKernel(int delta)
    {
        if(SelectedProcessingStage is ITextRegionSharpenSettings stage)
            stage.KernelSize = Math.Clamp(stage.KernelSize + delta, 1, 31);
    }

    public void AdjustSelectedSharpenSigma(double delta)
    {
        if(SelectedProcessingStage is ITextRegionSharpenSettings stage)
            stage.Sigma = Math.Clamp(stage.Sigma + delta, 0, 20);
    }

    public void AdjustSelectedSharpenAmount(double delta)
    {
        if(SelectedProcessingStage is ITextRegionSharpenSettings stage)
            stage.Amount = Math.Clamp(stage.Amount + delta, 0, 5);
    }

    public void ResetSelectedSharpen()
    {
        if(SelectedProcessingStage is ITextRegionSharpenSettings stage)
        {
            stage.KernelSize = 3;
            stage.Sigma = 0;
            stage.Amount = 1.0;
        }
    }

    public string FormatRecognitionText(PaddleOCRRecExtractor.OcrResult recognitionResult) =>
        RecognitionMode switch
        {
            1 => recognitionResult.Standard,
            2 => recognitionResult.WithSpaces,
            3 => recognitionResult.FullCandidates,
            _ => recognitionResult.Standard,
        };

    public OcrRegionPostprocessorOptions CreateRegionPostprocessorOptions() =>
        new()
        {
            MinRegionHeight = MinRegionHeight,
            MaxRegionHeight = MaxRegionHeight,
            MinRegionWidth = MinRegionWidth,
            MaxRegionWidth = MaxRegionWidth,
            MinRegionAspectRatio = MinRegionAspectRatio,
            MaxRegionAspectRatio = MaxRegionAspectRatio,
            MaxRegions = MaxRegionsPerFrame,
            EnableOverlapSuppression = OverlapSuppressionEnabled,
            OverlapSuppressionRatio = OverlapSuppressionRatio,
            EnableLineMerge = LineMergeEnabled,
            MergeAngleDeltaDegrees = MergeAngleDeltaDegrees,
            MergeNormalOffsetInHeights = MergeNormalOffsetInHeights,
            MergeHeightRatio = MergeHeightRatio,
            MergeGapInHeights = MergeGapInHeights,
            MinimumMergedCoverageRatio = MinimumMergedCoverageRatio,
            MaxMergedRegionWidth = MaxMergedRegionWidth,
            MaxMergedRegionAspectRatio = float.IsPositiveInfinity(MaxMergedRegionAspectRatio)
                ? RecognitionInputWidth / (float)RecognitionInputHeight
                : MaxMergedRegionAspectRatio,
        };

    public void Dispose()
    {
        processingPipeline.Dispose();

        foreach(ITextRegionProcessingStage removedStage in removedProcessingStages)
        {
            if(removedStage is IDisposable disposableStage)
                disposableStage.Dispose();
        }
    }

    private static int AlignToStride(int value)
    {
        int stride = PaddleOCRRecExtractor.OutputWidthStride;
        return Math.Max(stride, (int)Math.Round(value / (double)stride) * stride);
    }

    private static float NormalizeOptionalMaximum(float value, float delta)
    {
        if(float.IsPositiveInfinity(value) && delta < 0)
            return Math.Abs(delta);

        float adjustedValue = value + delta;
        return adjustedValue <= 0 ? float.PositiveInfinity : adjustedValue;
    }

    private ITextRegionProcessingStage? GetProcessingStage(int index)
    {
        IReadOnlyList<ITextRegionProcessingStage> stages = processingPipeline.GetStages();
        return index >= 0 && index < stages.Count ? stages[index] : null;
    }

    private static ITextRegionProcessingStage CreateProcessingStage(TextRegionProcessingStageKind stageKind) =>
        TextRegionProcessingStageFactory.CreateStage(new TextRegionProcessingStageOptions
        {
            Kind = stageKind,
        });

    private static string FormatProcessingStageName(ITextRegionProcessingStage stage) =>
        stage switch
        {
            ITextRegionBrightnessContrastSettings settings => $"{stage.Name} b {settings.Brightness:F0} c {settings.ContrastPercent:F0}%",
            ITextRegionGammaCorrectionSettings settings => $"{stage.Name} {settings.Gamma:F2}",
            ITextRegionGrayscaleSettings settings => $"{stage.Name} RGB {FormatFlag(settings.UseRed)}{FormatFlag(settings.UseGreen)}{FormatFlag(settings.UseBlue)}",
            ITextRegionThresholdSettings settings => settings.UseOtsu ? $"{stage.Name} Otsu" : $"{stage.Name} {settings.Threshold:F0}",
            ITextRegionGaussianBlurSettings settings => $"{stage.Name} k {settings.KernelSize} s {settings.Sigma:F1}",
            ITextRegionSharpenSettings settings => $"{stage.Name} k {settings.KernelSize} s {settings.Sigma:F1} a {settings.Amount:F1}",
            _ => stage.Name,
        };

    private static string FormatFlag(bool value) => value ? "1" : "0";
}
