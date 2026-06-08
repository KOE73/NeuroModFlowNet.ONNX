using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.ONNX.Visualizer;
using NeuroModFlowNet.Pipeline;
using NeuroModFlowNet.Pipeline.ONNX;
using OpenCvSharp;
using SkiaSharp;

namespace NeuroModFlowNet.Pipeline.Avalonia.VMDebug.Runtime;

/// <summary>
/// Builds frame debug DTO from run context registers and leaves it as a VM output variable.
/// </summary>
/// <remarks>
/// Rendering is deliberately the last instruction in the debug VM program. Heavyweight model and ROI buffers stay
/// context-owned, while the result DTO receives independent Mats that survive run context disposal.
/// </remarks>
internal sealed class CollectFrameResultInstruction : OpBase
{
    readonly PipelineModelResources? resources;
    readonly RecognitionOptions recognitionOptions;
    readonly string outputKey;

    public CollectFrameResultInstruction(
        PipelineModelResources? resources,
        RecognitionOptions recognitionOptions,
        string outputKey)
        : base(OpDescriptor.Create(
            "collect-frame-result",
            "debug.collectFrameResult"))
    {
        this.resources = resources;
        this.recognitionOptions = recognitionOptions;
        this.outputKey = outputKey;
    }

    public override ValueTask<OpResult> ExecuteAsync(VmRunContext context, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        Mat sourceFrame = context.Get<Mat>(PipelineAvaloniaKeys.SourceFrame);
        LetterboxInfo sourceLetterboxInfo = context.TryGet(PipelineAvaloniaKeys.SourceLetterboxInfo, out LetterboxInfo letterboxInfo)
            ? letterboxInfo
            : new LetterboxInfo { Ratio = 1, SourceWidth = sourceFrame.Width, SourceHeight = sourceFrame.Height, TargetWidth = sourceFrame.Width, TargetHeight = sourceFrame.Height };

        YoloBox[] boxDetections = context.TryGet<YoloDetectionBatchResult<YoloBox>>(PipelineAvaloniaKeys.YoloBoxResult, out var boxResult)
            ? boxResult.Detections.ToArray()
            : [];
        YoloObb[] obbDetections = context.TryGet<YoloDetectionBatchResult<YoloObb>>(PipelineAvaloniaKeys.YoloObbResult, out var obbResult)
            ? obbResult.Detections.ToArray()
            : [];
        IReadOnlyList<OcrQuadRegion> detRegions = context.TryGet<List<OcrQuadRegion>>(PipelineAvaloniaKeys.PaddleDetRegions, out var regions)
            ? regions
            : [];
        IReadOnlyList<RecognitionTextRow> recognitionRows = context.TryGet<List<RecognitionTextRow>>(PipelineAvaloniaKeys.RecognitionRows, out var rows)
            ? rows
            : [];

        using Mat visualizedFrame = sourceFrame.Clone();
        DrawAdditionalInferenceResults(visualizedFrame, context, sourceLetterboxInfo);

        Mat displayFrame = ToBgra(visualizedFrame);
        FrameOverlaySnapshot overlay = resources is null
            ? new FrameOverlaySnapshot(sourceFrame.Width, sourceFrame.Height, [], [])
            : BuildOverlay(sourceFrame, boxDetections, obbDetections, detRegions, sourceLetterboxInfo, recognitionRows);

        string sourceId = context.TryGet<string>(PipelineAvaloniaKeys.SourceId, out var transactionSourceId)
            ? transactionSourceId
            : context.Identity.SourceId;
        var frameData = new PipelineFrameResultData(
            sourceId,
            displayFrame,
            overlay);

        context.Set(outputKey, frameData);

        return ValueTask.FromResult(OpResult.Continue);
    }

    void DrawAdditionalInferenceResults(Mat target, VmRunContext context, LetterboxInfo letterboxInfo)
    {
        if(resources is null)
            return;

        if(context.TryGet<YoloSegFrameResult>(PipelineAvaloniaKeys.YoloSegResult, out var segmentationResult))
            DrawSegmentationResult(target, segmentationResult, letterboxInfo);

        if(context.TryGet<YoloPoseFrameResult>(PipelineAvaloniaKeys.YoloPoseResult, out var poseResult))
        {
            YoloPosePainter.DrawPose(
                target,
                poseResult.Detections,
                letterboxInfo,
                1f,
                1f,
                resources.ModelPose.GetYoloClassName);
        }

        if(context.TryGet<YoloClassificationFrameResult>(PipelineAvaloniaKeys.YoloClsResult, out var classificationResult))
            DrawCompactClassification(target, classificationResult, resources.ModelCls.GetYoloClassName);
    }

    void DrawSegmentationResult(Mat target, YoloSegFrameResult segmentationResult, LetterboxInfo letterboxInfo)
    {
        if(resources is null)
            return;

        YoloSegResult_FP32_Mask32 fp32Result = segmentationResult.FP32 ?? (segmentationResult.FP16 is not null
            ? ConvertSegmentationToFP32(segmentationResult.FP16)
            : throw new NotSupportedException("Unsupported empty YOLO Seg frame result."));

        SegPainter.DrawSeg(
            target,
            fp32Result.Values,
            fp32Result.Masks,
            letterboxInfo,
            1f,
            1f,
            resources.ModelSeg.GetYoloClassName);
    }

    FrameOverlaySnapshot BuildOverlay(
        Mat frame,
        ReadOnlySpan<YoloBox> boxDetections,
        ReadOnlySpan<YoloObb> obbDetections,
        IReadOnlyList<OcrQuadRegion> detRegions,
        LetterboxInfo letterboxInfo,
        IReadOnlyList<RecognitionTextRow> recognitionRows)
    {
        if(resources is null)
            return new FrameOverlaySnapshot(frame.Width, frame.Height, [], []);

        var mapper = LetterboxCoordinateMapper.Create(letterboxInfo.Ratio, letterboxInfo.OffsetX, letterboxInfo.OffsetY);
        List<OverlayObb> overlayBoxes = new(boxDetections.Length + obbDetections.Length + detRegions.Count);

        foreach(OcrQuadRegion detRegion in detRegions)
            overlayBoxes.Add(CreateOverlayBox(detRegion, new SKColor(0, 190, 80, 72), string.Empty, fill: true));

        if(recognitionOptions.InferenceSelection.BoxDetectionEnabled)
        {
            foreach(YoloBox box in boxDetections)
            {
                string className = resources.ModelBox.GetYoloClassName(box.Class);
                overlayBoxes.Add(CreateOverlayBox(box, letterboxInfo, BoxPainter.ClassColorSkia(box.Class), FormatDetectionLabel(className, box.Score)));
            }
        }

        if(recognitionOptions.InferenceSelection.OcrEnabled || recognitionOptions.InferenceSelection.ObbDetectionEnabled)
        {
            foreach(YoloObb box in obbDetections)
            {
                OcrQuadRegion sourceRegion = YoloObbOcrRegionMapper.MapToSourceRegion(box, mapper);
                string className = resources.ModelObb.GetYoloClassName(box.Class);
                overlayBoxes.Add(CreateOverlayBox(sourceRegion, SKColors.Red, FormatDetectionLabel(className, box.Score)));
            }
        }

        // Recognition text is intentionally not drawn over the main frame. The ROI panel keeps tuning legible.
        return new FrameOverlaySnapshot(frame.Width, frame.Height, overlayBoxes, []);
    }

    static Mat ToBgra(Mat source)
    {
        var bgra = new Mat();
        if(source.Channels() == 4)
            source.CopyTo(bgra);
        else if(source.Channels() == 3)
            Cv2.CvtColor(source, bgra, ColorConversionCodes.BGR2BGRA);
        else if(source.Channels() == 1)
            Cv2.CvtColor(source, bgra, ColorConversionCodes.GRAY2BGRA);
        else
            source.CopyTo(bgra);

        return bgra;
    }

    static string FormatDetectionLabel(string className, float score)
    {
        string compactClassName = className.Length <= 8 ? className : className[..8];
        return $"{compactClassName} {score * 100:F0}%";
    }

    static OverlayObb CreateOverlayBox(YoloBox box, LetterboxInfo letterboxInfo, SKColor color, string label)
    {
        Point2f topLeft = letterboxInfo.MapBack(box.X, box.Y);
        Point2f bottomRight = letterboxInfo.MapBack(box.W, box.H);
        float left = Math.Min(topLeft.X, bottomRight.X);
        float top = Math.Min(topLeft.Y, bottomRight.Y);
        float right = Math.Max(topLeft.X, bottomRight.X);
        float bottom = Math.Max(topLeft.Y, bottomRight.Y);

        SKPoint[] points =
        [
            new(left, top),
            new(right, top),
            new(right, bottom),
            new(left, bottom),
        ];

        return new OverlayObb(
            new SKPoint((left + right) * 0.5f, (top + bottom) * 0.5f),
            points,
            color,
            label);
    }

    static OverlayObb CreateOverlayBox(OcrQuadRegion sourceRegion, SKColor color, string label, bool fill = false)
    {
        Point2f[] points =
        [
            sourceRegion.Point0,
            sourceRegion.Point1,
            sourceRegion.Point2,
            sourceRegion.Point3,
        ];

        Rect bounds = GetBoundingRect(sourceRegion);
        return new OverlayObb(
            new SKPoint(bounds.X + bounds.Width * 0.5f, bounds.Y + bounds.Height * 0.5f),
            points.Select(point => new SKPoint(point.X, point.Y)).ToArray(),
            color,
            label,
            fill);
    }

    static Rect GetBoundingRect(OcrQuadRegion sourceRegion)
    {
        float left = Math.Min(Math.Min(sourceRegion.X0, sourceRegion.X1), Math.Min(sourceRegion.X2, sourceRegion.X3));
        float top = Math.Min(Math.Min(sourceRegion.Y0, sourceRegion.Y1), Math.Min(sourceRegion.Y2, sourceRegion.Y3));
        float right = Math.Max(Math.Max(sourceRegion.X0, sourceRegion.X1), Math.Max(sourceRegion.X2, sourceRegion.X3));
        float bottom = Math.Max(Math.Max(sourceRegion.Y0, sourceRegion.Y1), Math.Max(sourceRegion.Y2, sourceRegion.Y3));

        int x = (int)Math.Floor(left);
        int y = (int)Math.Floor(top);
        return new Rect(
            x,
            y,
            Math.Max(1, (int)Math.Ceiling(right) - x),
            Math.Max(1, (int)Math.Ceiling(bottom) - y));
    }

    static void DrawCompactClassification(Mat target, YoloClassificationFrameResult result, Func<int, string>? nameResolver)
    {
        string className = nameResolver?.Invoke(result.ClassId) ?? $"Class #{result.ClassId}";
        string label = $"CLS {className} {result.Score:P1}";
        Size textSize = Cv2.GetTextSize(label, HersheyFonts.HersheySimplex, 0.65, 1, out int baseline);
        int panelWidth = Math.Min(target.Width - 16, textSize.Width + 24);
        var panelRect = new Rect(8, 8, panelWidth, textSize.Height + baseline + 18);

        using var overlay = target.Clone();
        Cv2.Rectangle(overlay, panelRect, new Scalar(30, 36, 48), -1, LineTypes.AntiAlias);
        Cv2.AddWeighted(overlay, 0.68, target, 0.32, 0, target);
        Cv2.Rectangle(target, panelRect, new Scalar(40, 220, 150), 1, LineTypes.AntiAlias);
        Cv2.PutText(
            target,
            label,
            new Point(panelRect.X + 12, panelRect.Y + textSize.Height + 8),
            HersheyFonts.HersheySimplex,
            0.65,
            new Scalar(80, 240, 180),
            1,
            LineTypes.AntiAlias);
    }

    static YoloSegResult_FP32_Mask32 ConvertSegmentationToFP32(YoloSegResult_FP16_Mask32 result)
    {
        var values = new YoloSeg_FP32_XYWHSC_Mask32[result.Values.Length];
        for(int index = 0; index < result.Values.Length; index++)
        {
            YoloSeg_FP16_XYWHSC_Mask32 item = result.Values[index];
            var maskCoefficients = new InlineArray_FP32_Mask32();
            for(int maskIndex = 0; maskIndex < InlineArray_FP16_Mask_Count32.Length; maskIndex++)
                maskCoefficients[maskIndex] = (float)item.MaskCoefficients[maskIndex];

            values[index] = new YoloSeg_FP32_XYWHSC_Mask32(
                (float)item.X,
                (float)item.Y,
                (float)item.W,
                (float)item.H,
                (float)item.Score,
                (float)item.ClassId,
                maskCoefficients);
        }

        var masks = new float[result.Masks.Length][];
        for(int index = 0; index < result.Masks.Length; index++)
        {
            Half[] sourceMask = result.Masks[index];
            float[] targetMask = new float[sourceMask.Length];
            for(int maskIndex = 0; maskIndex < sourceMask.Length; maskIndex++)
                targetMask[maskIndex] = (float)sourceMask[maskIndex];

            masks[index] = targetMask;
        }

        return new YoloSegResult_FP32_Mask32
        {
            Values = values,
            Masks = masks,
            PrototypeShape = result.PrototypeShape,
        };
    }
}
