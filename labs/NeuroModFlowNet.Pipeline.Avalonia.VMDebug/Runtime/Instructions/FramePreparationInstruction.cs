using NeuroModFlowNet.ONNX.Visualizer;
using NeuroModFlowNet.Pipeline;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.Avalonia.VMDebug.Runtime;

/// <summary>
/// Prepares the display and model-input frames for one VM run.
/// </summary>
/// <remarks>
/// This keeps resize/letterbox geometry in explicit run context registers. Later extractors and overlay builders read
/// the geometry instead of relying on hidden Mat+metadata containers.
/// </remarks>
internal sealed class FramePreparationInstruction : OpBase
{
    readonly RealTimeAvaloniaSettings settings;
    readonly RecognitionOptions recognitionOptions;

    public FramePreparationInstruction(RealTimeAvaloniaSettings settings, RecognitionOptions recognitionOptions)
        : base(OpDescriptor.Create(
            "prepare-frame",
            "avalonia.prepareFrame",
            [VarRequirement.Read<Mat>(PipelineAvaloniaKeys.SourceFrame)],
            [
                VarRequirement.Write<Mat>(PipelineAvaloniaKeys.ResizedFrame),
                VarRequirement.Write<Mat>(PipelineAvaloniaKeys.ModelInputFrame),
                VarRequirement.Write<LetterboxInfo>(PipelineAvaloniaKeys.LetterboxInfo),
                VarRequirement.Write<LetterboxInfo>(PipelineAvaloniaKeys.SourceLetterboxInfo),
            ]))
    {
        this.settings = settings;
        this.recognitionOptions = recognitionOptions;
    }

    public override ValueTask<OpResult> ExecuteAsync(VmRunContext context, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        Mat sourceFrame = context.Get<Mat>(PipelineAvaloniaKeys.SourceFrame);
        Mat resizedFrame = ResizeFrame(sourceFrame, recognitionOptions.FrameWidth);
        Mat letterboxedFrame = resizedFrame.Letterbox(settings.InputSize, settings.InputSize, out LetterboxInfo letterboxInfo);
        LetterboxInfo sourceLetterboxInfo = CreateSourceLetterboxInfo(letterboxInfo, resizedFrame, sourceFrame);

        context.Set(PipelineAvaloniaKeys.ResizedFrame, resizedFrame, disposeWithContext: true);
        context.Set(PipelineAvaloniaKeys.ModelInputFrame, letterboxedFrame, disposeWithContext: true);
        context.Set(PipelineAvaloniaKeys.LetterboxInfo, letterboxInfo);
        context.Set(PipelineAvaloniaKeys.SourceLetterboxInfo, sourceLetterboxInfo);

        return ValueTask.FromResult(OpResult.Continue);
    }

    static Mat ResizeFrame(Mat input, int targetWidth)
    {
        double aspectRatio = input.Height / (double)input.Width;
        int targetHeight = Math.Max(1, (int)(targetWidth * aspectRatio));
        var resizedMat = new Mat();
        Cv2.Resize(input, resizedMat, new Size(targetWidth, targetHeight));
        return resizedMat;
    }

    static LetterboxInfo CreateSourceLetterboxInfo(LetterboxInfo displayLetterboxInfo, Mat displayFrame, Mat sourceFrame)
    {
        if(displayFrame.Width == sourceFrame.Width && displayFrame.Height == sourceFrame.Height)
            return displayLetterboxInfo;

        float displayScale = displayFrame.Width / (float)sourceFrame.Width;

        return new LetterboxInfo
        {
            Ratio = displayLetterboxInfo.Ratio * displayScale,
            OffsetX = displayLetterboxInfo.OffsetX,
            OffsetY = displayLetterboxInfo.OffsetY,
            SourceWidth = sourceFrame.Width,
            SourceHeight = sourceFrame.Height,
            TargetWidth = displayLetterboxInfo.TargetWidth,
            TargetHeight = displayLetterboxInfo.TargetHeight,
        };
    }
}
