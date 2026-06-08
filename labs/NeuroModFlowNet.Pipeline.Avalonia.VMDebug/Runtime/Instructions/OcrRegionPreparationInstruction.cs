using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.ONNX.Visualizer;
using NeuroModFlowNet.Pipeline;
using NeuroModFlowNet.Pipeline.ONNX;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.Avalonia.VMDebug.Runtime;

/// <summary>
/// Converts detector outputs into OCR regions and recognition input crops.
/// </summary>
/// <remarks>
/// YOLO OBB, PaddleOCR Det, postprocessing, and ROI extraction are kept in one instruction because they form one
/// CPU-side bridge: compact detections become small crop tensors for recognition. The heavy source frame remains a
/// context-owned payload and every ROI Mat is registered for disposal.
/// </remarks>
internal sealed class OcrRegionPreparationInstruction : OpBase
{
    readonly RecognitionOptions recognitionOptions;
    readonly AvaloniaJsonConfig jsonConfig;

    public OcrRegionPreparationInstruction(RecognitionOptions recognitionOptions, AvaloniaJsonConfig jsonConfig)
        : base(OpDescriptor.Create(
            "prepare-ocr-regions",
            "ocr.prepareRegions",
            [
                VarRequirement.Read<Mat>(PipelineAvaloniaKeys.SourceFrame),
                VarRequirement.Read<YoloDetectionBatchResult<YoloObb>>(PipelineAvaloniaKeys.YoloObbResult, required: false),
                VarRequirement.Read<Mat>(PipelineAvaloniaKeys.PaddleDetScoreMap, required: false),
                VarRequirement.Read<LetterboxInfo>(PipelineAvaloniaKeys.SourceLetterboxInfo),
            ],
            [
                VarRequirement.Write<List<OcrQuadRegion>>(PipelineAvaloniaKeys.PaddleDetRegions),
                VarRequirement.Write<List<PreparedRecognitionRoi>>(PipelineAvaloniaKeys.RecognitionRois),
            ],
            hasSideEffects: true))
    {
        this.recognitionOptions = recognitionOptions;
        this.jsonConfig = jsonConfig;
    }

    public override ValueTask<OpResult> ExecuteAsync(VmRunContext context, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        PipelineFrameTiming timing = context.Get<PipelineFrameTiming>(PipelineAvaloniaKeys.FrameTiming);
        timing.StartRoi();

        Mat sourceFrame = context.Get<Mat>(PipelineAvaloniaKeys.SourceFrame);
        LetterboxInfo sourceLetterboxInfo = context.Get<LetterboxInfo>(PipelineAvaloniaKeys.SourceLetterboxInfo);
        YoloObb[] obbBoxes = context.TryGet<YoloDetectionBatchResult<YoloObb>>(PipelineAvaloniaKeys.YoloObbResult, out var obbResult)
            ? obbResult.Detections.ToArray()
            : [];

        List<OcrQuadRegion> detRegions = context.TryGet<Mat>(PipelineAvaloniaKeys.PaddleDetScoreMap, out var detScoreMap)
            ? PrepareDetSourceRegions(detScoreMap, sourceLetterboxInfo)
            : [];

        List<PreparedRecognitionRoi> preparedImages = PrepareRecognitionImages(sourceFrame, obbBoxes, sourceLetterboxInfo, context);
        timing.RecognitionItemCount = preparedImages.Count;
        timing.StopRoi();

        context.Set(PipelineAvaloniaKeys.PaddleDetRegions, detRegions);
        context.Set(PipelineAvaloniaKeys.RecognitionRois, preparedImages);
        return ValueTask.FromResult(OpResult.Continue);
    }

    List<PreparedRecognitionRoi> PrepareRecognitionImages(
        Mat sourceFrame,
        ReadOnlySpan<YoloObb> boxes,
        LetterboxInfo letterboxInfo,
        VmRunContext context)
    {
        var mapper = LetterboxCoordinateMapper.Create(letterboxInfo.Ratio, letterboxInfo.OffsetX, letterboxInfo.OffsetY);

        Span<OcrQuadRegion> mappedSourceRegions = boxes.Length <= 1000
            ? stackalloc OcrQuadRegion[boxes.Length]
            : new OcrQuadRegion[boxes.Length];
        Span<RoiHeightDebugData> mappedHeightDebug = boxes.Length <= 1000
            ? stackalloc RoiHeightDebugData[boxes.Length]
            : new RoiHeightDebugData[boxes.Length];

        for(int index = 0; index < boxes.Length; index++)
        {
            OcrQuadRegion unscaledSourceRegion = YoloObbOcrRegionMapper.MapToSourceRegion(boxes[index], mapper);
            float sourceRegionHeight = GetRegionHeight(unscaledSourceRegion);
            RoiHeightDebugData heightDebug = recognitionOptions.CalculateRoiHeightDebug(sourceRegionHeight);
            mappedSourceRegions[index] = YoloObbOcrRegionMapper.MapToSourceRegion(boxes[index], mapper, heightDebug.Scale);
            mappedHeightDebug[index] = heightDebug;
        }

        List<OcrQuadRegion> sourceRegions = [];
        OcrRegionPostprocessor.Shared.Process(
            mappedSourceRegions,
            recognitionOptions.CreateRegionPostprocessorOptions(),
            sourceRegions);

        var options = new TextRegionExtractionOptions(
            recognitionOptions.RecognitionInputWidth,
            recognitionOptions.RecognitionInputHeight,
            recognitionOptions.ProcessingStage);

        List<PreparedRecognitionRoi> preparedImages = [];

        foreach(OcrQuadRegion sourceRegion in sourceRegions)
        {
            if(!NaiveTextRegionExtractor.Shared.TryExtract(sourceFrame, sourceRegion, options, out Mat? recognitionRoi))
                continue;
            if(recognitionRoi is null)
                continue;

            context.AddOwnedResource(recognitionRoi);
            RoiHeightDebugData heightDebug = FindNearestHeightDebug(sourceRegion, mappedSourceRegions, mappedHeightDebug);
            preparedImages.Add(new PreparedRecognitionRoi(recognitionRoi, GetRecognitionLabelPoint(sourceRegion, sourceFrame), heightDebug));
        }

        return preparedImages;
    }

    List<OcrQuadRegion> PrepareDetSourceRegions(Mat detScoreMap, LetterboxInfo sourceLetterboxInfo)
    {
        List<OcrQuadRegion> modelRegions = [];
        PaddleOCRDetMaskRegionExtractor.Shared.Extract(detScoreMap, jsonConfig.Postprocessing.DetMask, modelRegions);

        if(modelRegions.Count == 0)
            return [];

        var mapper = LetterboxCoordinateMapper.Create(sourceLetterboxInfo.Ratio, sourceLetterboxInfo.OffsetX, sourceLetterboxInfo.OffsetY);

        Span<Point2f> modelPoints = stackalloc Point2f[4];
        Span<Point2f> sourcePoints = stackalloc Point2f[4];
        List<OcrQuadRegion> mappedRegions = new(modelRegions.Count);

        foreach(OcrQuadRegion modelRegion in modelRegions)
        {
            modelRegion.CopyTo(modelPoints);
            mapper.MapPointsToSource(modelPoints, sourcePoints);
            mappedRegions.Add(OcrQuadRegion.FromPoints(sourcePoints));
        }

        List<OcrQuadRegion> sourceRegions = [];
        OcrRegionPostprocessor.Shared.Process(
            System.Runtime.InteropServices.CollectionsMarshal.AsSpan(mappedRegions),
            recognitionOptions.CreateRegionPostprocessorOptions(),
            sourceRegions);

        return sourceRegions;
    }

    static RoiHeightDebugData FindNearestHeightDebug(
        OcrQuadRegion displayRegion,
        ReadOnlySpan<OcrQuadRegion> sourceRegions,
        ReadOnlySpan<RoiHeightDebugData> heightDebugData)
    {
        if(sourceRegions.Length == 0 || heightDebugData.Length == 0)
            return RoiHeightDebugData.Empty;

        Point2f displayCenter = GetRegionCenter(displayRegion);
        float bestDistanceSquared = float.PositiveInfinity;
        int bestIndex = 0;

        for(int index = 0; index < sourceRegions.Length; index++)
        {
            Point2f sourceCenter = GetRegionCenter(sourceRegions[index]);
            float dx = displayCenter.X - sourceCenter.X;
            float dy = displayCenter.Y - sourceCenter.Y;
            float distanceSquared = dx * dx + dy * dy;
            if(distanceSquared >= bestDistanceSquared) continue;

            bestDistanceSquared = distanceSquared;
            bestIndex = index;
        }

        return heightDebugData[Math.Min(bestIndex, heightDebugData.Length - 1)];
    }

    static Point2f GetRegionCenter(OcrQuadRegion region) =>
        new(
            (region.X0 + region.X1 + region.X2 + region.X3) * 0.25f,
            (region.Y0 + region.Y1 + region.Y2 + region.Y3) * 0.25f);

    static float GetRegionHeight(OcrQuadRegion region) =>
        MathF.Max(
            Distance(region.Point0, region.Point3),
            Distance(region.Point1, region.Point2));

    static float Distance(Point2f left, Point2f right)
    {
        float dx = left.X - right.X;
        float dy = left.Y - right.Y;
        return MathF.Sqrt(dx * dx + dy * dy);
    }

    static Point GetRecognitionLabelPoint(OcrQuadRegion sourceRegion, Mat sourceMat)
    {
        Rect bounds = ClipRect(GetBoundingRect(sourceRegion), sourceMat.Width, sourceMat.Height);
        return new Point(bounds.X, Math.Max(0, bounds.Y - 6));
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

    static Rect ClipRect(Rect rect, int width, int height)
    {
        int x = Math.Clamp(rect.X, 0, width - 1);
        int y = Math.Clamp(rect.Y, 0, height - 1);
        int right = Math.Clamp(rect.Right, x + 1, width);
        int bottom = Math.Clamp(rect.Bottom, y + 1, height);
        return new Rect(x, y, right - x, bottom - y);
    }
}
