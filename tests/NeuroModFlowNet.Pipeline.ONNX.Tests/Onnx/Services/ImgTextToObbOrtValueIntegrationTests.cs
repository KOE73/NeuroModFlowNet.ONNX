using Microsoft.ML.OnnxRuntime;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline;
using NeuroModFlowNet.Pipeline.ONNX.Tests.Common;
using OpenCvSharp;
using CvSize = OpenCvSharp.Size;

namespace NeuroModFlowNet.Pipeline.ONNX.Tests.Onnx.Services;

public sealed class ImgTextToObbOrtValueIntegrationTests
{
    const string ModelPath = "models/img-text-to-obb/img-text-to-obb__640_b4_fp32.onnx";
    const int ModelInputSize = 640;
    const float ScoreThreshold = 0.25f;
    const string CatImageFileName = "for_tests.png";
    const string ConveyorImageFileName = "for_tests_perspective_real_0.png";

    static readonly CvSize ConveyorPerspectiveOutputSize = new(900, 4500);

    static readonly Point2f[] ConveyorPerspectiveSourceQuad =
    [
        new(918, 0),
        new(1142, 0),
        new(1315, 1558),
        new(732, 1558)
    ];

    public static IEnumerable<object[]> Backends =>
        OnnxExecutionBackendMatrix.EnabledBackends.Select(static backend => new object[] { backend });

    [Theory]
    [MemberData(nameof(Backends))]
    public async Task TwoVmRuns_PrepareImagesBatch4ModelAndReceiveOwnObbResults(InferenceBackend executionBackend)
    {
        OnnxExecutionBackendAvailability.AssertAvailable(executionBackend);
        string modelPath = ResolveRepositoryPath(ModelPath);
        Assert.True(File.Exists(modelPath), $"Required integration model was not found: {modelPath}");


        await using var endpoint = new OrtValueBatchedInferenceEndpoint<YoloObb>(
            name: "img-text-to-obb.fp32.b4",
            modelPath: modelPath,
            executionBackend: executionBackend,
            options: new OnnxBatchedResourceOptions(MaxBatchSize: 2, MaxWaitTime: TimeSpan.FromMilliseconds(500), MaxPendingRequests: 4),
            batchInputAssembler: new ConcatOrtValueBatchInputAssembler(fixedBatchSize: 4),
            outputShapeResolver: new YoloNmsOutputShapeResolver(defaultItemCount: 300, fieldCount: 7),
            outputDecoder: new YoloObbOrtValueOutputDecoder(scoreThreshold: ScoreThreshold));

        var inference = new Model_OrtValueInference<YoloObb>(
            name: "Model_ImgTextToObb_OrtValue_FP32_B4",
            inputKey: "image.modelInput",
            outputKey: "obb.results",
            endpoint: endpoint);

        var mapToCropImageCoordinates = new Op_Map_Coordinates(
            inputKey: "obb.results",
            transformKey: "image.resized.toCropImage",
            outputKey: "obb.cropImage",
            shapePolicy: CoordinateMappingShapePolicy.BoundingOBB);

        var firstPrepared = await PrepareContext(CatImageFileName, executionBackend, perspective: null);
        using IDisposable firstPreparationLifetime = firstPrepared.RuntimeLifetime;
        await using VmRunContext firstContext = firstPrepared.Context;

        var secondPrepared = await PrepareContext(
            ConveyorImageFileName,
            executionBackend,
            new PerspectivePreparation(ConveyorPerspectiveSourceQuad, ConveyorPerspectiveOutputSize));
        using IDisposable secondPreparationLifetime = secondPrepared.RuntimeLifetime;
        await using VmRunContext secondContext = secondPrepared.Context;

        ValueTask<OpResult> firstTask = inference.ExecuteAsync(firstContext, CancellationToken.None);
        ValueTask<OpResult> secondTask = inference.ExecuteAsync(secondContext, CancellationToken.None);

        Assert.Equal(OpResult.Continue, await firstTask);
        Assert.Equal(OpResult.Continue, await secondTask);
        Assert.Equal(OpResult.Continue, await mapToCropImageCoordinates.ExecuteAsync(firstContext, CancellationToken.None));
        Assert.Equal(OpResult.Continue, await mapToCropImageCoordinates.ExecuteAsync(secondContext, CancellationToken.None));

        YoloObb[] firstResult = firstContext.Get<YoloObb[]>("obb.results");
        YoloObb[] secondResult = secondContext.Get<YoloObb[]>("obb.results");
        YoloObb[] firstCropImageResult = firstContext.Get<YoloObb[]>("obb.cropImage");
        YoloObb[] secondCropImageResult = secondContext.Get<YoloObb[]>("obb.cropImage");

        Assert.NotNull(firstResult);
        Assert.NotNull(secondResult);
        Assert.NotNull(firstCropImageResult);
        Assert.NotNull(secondCropImageResult);
        Assert.Equal(firstResult.Length, firstCropImageResult.Length);
        Assert.Equal(secondResult.Length, secondCropImageResult.Length);
        Assert.NotEmpty(firstResult);
        Assert.NotEmpty(secondResult);
        Assert.All(firstResult, static box => Assert.InRange(box.Score, ScoreThreshold, 1f));
        Assert.All(secondResult, static box => Assert.InRange(box.Score, ScoreThreshold, 1f));
        Assert.All(firstCropImageResult, static box => Assert.InRange(box.Score, ScoreThreshold, 1f));
        Assert.All(secondCropImageResult, static box => Assert.InRange(box.Score, ScoreThreshold, 1f));

        SaveArtifacts(CatImageFileName, firstResult, firstCropImageResult, executionBackend, perspective: null);
        SaveArtifacts(
            ConveyorImageFileName,
            secondResult,
            secondCropImageResult,
            executionBackend,
            new PerspectivePreparation(ConveyorPerspectiveSourceQuad, ConveyorPerspectiveOutputSize));
    }

    static async Task<(
        VmRunContext Context,
        IDisposable RuntimeLifetime)> PrepareContext(
            string imageFileName,
            InferenceBackend executionBackend,
            PerspectivePreparation? perspective)
    {
        using Mat source = RealImageTestSource.LoadBgr(imageFileName);
        using Mat sourceClone = source.Clone();
        OrtValue input = OrtTestTensorFactory.CreateBgrU8NhwcTensor(sourceClone);

        var context = VmRunContextFactory.Create();
        context.Set("image.input", input, disposeWithContext: true);

        string padResizeInputKey = "image.input";
        var disposables = new List<IDisposable>();

        if(perspective is not null)
        {
            var perspectiveOp = new Op_Onnx_Perspective_U8_NHWC(
                inputKey: "image.input",
                outputKey: "image.cropImage",
                sourcePoints: perspective.SourceQuad,
                outputSize: perspective.OutputSize,
                outputTransformKey: "image.cropImage.toSource",
                isFinal: false,
                executionBackend: executionBackend);

            Assert.Equal(OpResult.Continue, await perspectiveOp.ExecuteAsync(context, CancellationToken.None));
            disposables.Add(perspectiveOp);
            padResizeInputKey = "image.cropImage";
        }

        var padResize = new Op_Onnx_PadResize_U8_NHWC(
            inputKey: padResizeInputKey,
            outputKey: "image.resized",
            targetSize: new CvSize(ModelInputSize, ModelInputSize),
            outputTransformKey: "image.resized.toCropImage",
            stride: 32,
            mode: PadResizeMode.FixedCanvas,
            isFinal: false,
            executionBackend: executionBackend);
        disposables.Add(padResize);

        var prepare = new Op_Onnx_BgrU8Hwc_To_RgbFP32Nchw_Div255(
            inputKey: "image.resized",
            outputKey: "image.modelInput",
            isFinal: false,
            executionBackend: executionBackend);
        disposables.Add(prepare);

        Assert.Equal(OpResult.Continue, await padResize.ExecuteAsync(context, CancellationToken.None));
        Assert.Equal(OpResult.Continue, await prepare.ExecuteAsync(context, CancellationToken.None));

        return (context, new CompositeDisposable(disposables));
    }

    static void SaveArtifacts(
        string imageFileName,
        YoloObb[] modelBoxes,
        YoloObb[] cropImageBoxes,
        InferenceBackend executionBackend,
        PerspectivePreparation? perspective)
    {
        using Mat source = RealImageTestSource.LoadBgr(imageFileName);
        using Mat cropImage = perspective is null ? source.Clone() : CreatePerspectivePreview(source, perspective);
        using Mat cropImageOverlay = cropImage.Clone();
        DrawObb(cropImageOverlay, cropImageBoxes, Scalar.Lime);

        using Mat modelPreview = CreatePadResizePreview(cropImage);
        using Mat modelOverlay = modelPreview.Clone();
        DrawObb(modelOverlay, modelBoxes, Scalar.Cyan);

        string caseName = Path.GetFileNameWithoutExtension(imageFileName);
        SaveArtifact(executionBackend, caseName, perspective is null ? "source_obb" : "perspective_obb", cropImageOverlay);
        SaveArtifact(executionBackend, caseName, "model_640_padresize_obb", modelOverlay);
        SaveObbCrops(executionBackend, caseName, perspective is null ? "source_obb_crop" : "perspective_obb_crop", cropImage, cropImageBoxes);
        SaveObbCrops(executionBackend, caseName, "model_obb_crop", modelPreview, modelBoxes);
    }

    static Mat CreatePerspectivePreview(Mat source, PerspectivePreparation perspective)
    {
        Point2f[] targetQuad =
        [
            new(0, 0),
            new(perspective.OutputSize.Width - 1, 0),
            new(perspective.OutputSize.Width - 1, perspective.OutputSize.Height - 1),
            new(0, perspective.OutputSize.Height - 1)
        ];
        using Mat matrix = Cv2.GetPerspectiveTransform(perspective.SourceQuad, targetQuad);
        var output = new Mat(perspective.OutputSize.Height, perspective.OutputSize.Width, source.Type(), Scalar.All(0));
        Cv2.WarpPerspective(source, output, matrix, perspective.OutputSize, InterpolationFlags.Linear, BorderTypes.Constant, Scalar.All(0));
        return output;
    }

    static Mat CreatePadResizePreview(Mat source)
    {
        PadResizeLayout layout = PadResizeLayout.Create(
            source.Width,
            source.Height,
            ModelInputSize,
            ModelInputSize,
            stride: 32,
            PadResizeMode.FixedCanvas);

        var output = new Mat(layout.OutputHeight, layout.OutputWidth, MatType.CV_8UC3, new Scalar(114, 114, 114));
        using var resized = new Mat();
        Cv2.Resize(source, resized, new CvSize(layout.ResizedWidth, layout.ResizedHeight), 0, 0, InterpolationFlags.Linear);
        using Mat targetRoi = new(output, new Rect(layout.PadLeft, layout.PadTop, layout.ResizedWidth, layout.ResizedHeight));
        resized.CopyTo(targetRoi);
        return output;
    }

    static void DrawObb(Mat image, ReadOnlySpan<YoloObb> boxes, Scalar color)
    {
        foreach(YoloObb box in boxes)
        {
            var rotatedRect = new RotatedRect(
                new Point2f(box.X, box.Y),
                new Size2f(Math.Max(2, box.W), Math.Max(2, box.H)),
                box.Angle * 180f / MathF.PI);
            Point[] vertices = rotatedRect.Points().Select(static point => point.ToPoint()).ToArray();
            Cv2.Polylines(image, [vertices], isClosed: true, color, thickness: 3, lineType: LineTypes.AntiAlias);
            Cv2.Circle(image, new Point((int)MathF.Round(box.X), (int)MathF.Round(box.Y)), 4, Scalar.Red, -1);
            Cv2.PutText(
                image,
                $"{box.Score:0.00}",
                vertices[0],
                HersheyFonts.HersheySimplex,
                0.7,
                color,
                2,
                LineTypes.AntiAlias);
        }
    }

    static void SaveObbCrops(InferenceBackend executionBackend, string caseName, string artifactKind, Mat source, ReadOnlySpan<YoloObb> boxes)
    {
        for(int index = 0; index < boxes.Length; index++)
        {
            using Mat crop = CreateObbCrop(source, boxes[index]);
            SaveArtifact(executionBackend, caseName, $"{artifactKind}_{index:D3}", crop);
        }
    }

    static Mat CreateObbCrop(Mat source, YoloObb box)
    {
        int outputWidth = Math.Max(2, (int)MathF.Ceiling(box.W));
        int outputHeight = Math.Max(2, (int)MathF.Ceiling(box.H));
        var rotatedRect = new RotatedRect(
            new Point2f(box.X, box.Y),
            new Size2f(Math.Max(2, box.W), Math.Max(2, box.H)),
            box.Angle * 180f / MathF.PI);
        Point2f[] sourcePoints = rotatedRect.Points();
        OrderRotatedRectPoints(sourcePoints);

        Point2f[] targetPoints =
        [
            new(0, 0),
            new(outputWidth - 1, 0),
            new(outputWidth - 1, outputHeight - 1),
            new(0, outputHeight - 1)
        ];
        using Mat matrix = Cv2.GetPerspectiveTransform(sourcePoints, targetPoints);
        var output = new Mat(outputHeight, outputWidth, source.Type(), Scalar.All(0));
        Cv2.WarpPerspective(source, output, matrix, new CvSize(outputWidth, outputHeight), InterpolationFlags.Linear, BorderTypes.Constant, Scalar.All(0));
        return output;
    }

    static void OrderRotatedRectPoints(Point2f[] points)
    {
        Point2f[] ordered = [.. points.OrderBy(static point => point.Y)];
        Point2f topLeft = ordered[0].X <= ordered[1].X ? ordered[0] : ordered[1];
        Point2f topRight = ordered[0].X > ordered[1].X ? ordered[0] : ordered[1];
        Point2f bottomLeft = ordered[2].X <= ordered[3].X ? ordered[2] : ordered[3];
        Point2f bottomRight = ordered[2].X > ordered[3].X ? ordered[2] : ordered[3];
        points[0] = topLeft;
        points[1] = topRight;
        points[2] = bottomRight;
        points[3] = bottomLeft;
    }

    static void SaveArtifact(InferenceBackend executionBackend, string caseName, string artifactKind, Mat image)
    {
        string path = PipelineOnnxTestEnvironment.Current.CreateOperationArtifactPath(
            "Onnx",
            "ImgTextToObbOrtValueIntegration",
            executionBackend.ToString(),
            caseName,
            artifactKind);
        VisualTestOutput.SaveOrShow($"{caseName}-{artifactKind}", image, path);
    }

    static string ResolveRepositoryPath(string relativePath)
    {
        string? directory = AppContext.BaseDirectory;
        while(!string.IsNullOrWhiteSpace(directory))
        {
            string candidate = Path.GetFullPath(Path.Combine(directory, relativePath));
            if(File.Exists(candidate))
                return candidate;

            directory = Directory.GetParent(directory)?.FullName;
        }

        return Path.GetFullPath(relativePath);
    }

    sealed record PerspectivePreparation(Point2f[] SourceQuad, CvSize OutputSize);

    sealed class CompositeDisposable(IReadOnlyList<IDisposable> disposables) : IDisposable
    {
        public void Dispose()
        {
            for(int index = disposables.Count - 1; index >= 0; index--)
                disposables[index].Dispose();
        }
    }

}
