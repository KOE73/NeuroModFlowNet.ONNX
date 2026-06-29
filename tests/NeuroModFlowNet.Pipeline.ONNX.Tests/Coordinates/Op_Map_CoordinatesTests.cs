using Microsoft.ML.OnnxRuntime;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline;
using NeuroModFlowNet.Pipeline.ONNX.Tests.Common;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.ONNX.Tests.Coordinates;

public sealed class Op_Map_CoordinatesTests
{
    [Fact]
    public async Task ExecuteAsync_MapsYoloDetectionBatchResultThroughExplicitRegisters()
    {
        await using var context = VmRunContextFactory.Create();
        var detections = new YoloDetectionBatchResult<YoloBox>(
            [
                new YoloBox
                {
                    X = 10,
                    Y = 20,
                    W = 30,
                    H = 40,
                    Score = 0.9f,
                    Class = 3,
                },
            ]);

        context.Set("obb.result.inResizedCoords", detections);
        context.Set("transform.resized.toCrop", new ResizeCoordinateBackTransform(200, 100, 100, 50));
        context.Set("transform.crop.toSource", new CropCoordinateBackTransform(5, 7));

        var resizedToCrop = new Op_Map_Coordinates(
            "obb.result.inResizedCoords",
            "transform.resized.toCrop",
            "obb.result.inCropCoords");
        var cropToSource = new Op_Map_Coordinates(
            "obb.result.inCropCoords",
            "transform.crop.toSource",
            "obb.result.inSourceCoords");

        OpResult firstResult = await resizedToCrop.ExecuteAsync(context, CancellationToken.None);
        OpResult secondResult = await cropToSource.ExecuteAsync(context, CancellationToken.None);

        Assert.Equal(OpResultKind.Continue, firstResult.Kind);
        Assert.Equal(OpResultKind.Continue, secondResult.Kind);

        YoloDetectionBatchResult<YoloBox> cropResult = context.Get<YoloDetectionBatchResult<YoloBox>>("obb.result.inCropCoords");
        YoloBox cropBox = Assert.Single(cropResult.Detections);
        Assert.Equal(20, cropBox.X);
        Assert.Equal(40, cropBox.Y);
        Assert.Equal(60, cropBox.W);
        Assert.Equal(80, cropBox.H);

        YoloDetectionBatchResult<YoloBox> sourceResult = context.Get<YoloDetectionBatchResult<YoloBox>>("obb.result.inSourceCoords");
        YoloBox sourceBox = Assert.Single(sourceResult.Detections);
        Assert.Equal(25, sourceBox.X);
        Assert.Equal(47, sourceBox.Y);
        Assert.Equal(65, sourceBox.W);
        Assert.Equal(87, sourceBox.H);
        Assert.Equal(0.9f, sourceBox.Score);
        Assert.Equal(3, sourceBox.Class);
    }

    [Fact]
    public async Task ExecuteAsync_MapsOcrQuadRegionList()
    {
        await using var context = VmRunContextFactory.Create();
        var regions = new List<OcrQuadRegion>
        {
            new(1, 2, 3, 2, 3, 4, 1, 4),
        };

        context.Set("regions.inCropCoords", regions);
        context.Set("transform.crop.toSource", new CropCoordinateBackTransform(10, 20));

        var instruction = new Op_Map_Coordinates(
            "regions.inCropCoords",
            "transform.crop.toSource",
            "regions.inSourceCoords");

        OpResult result = await instruction.ExecuteAsync(context, CancellationToken.None);

        Assert.Equal(OpResultKind.Continue, result.Kind);

        List<OcrQuadRegion> mappedRegions = context.Get<List<OcrQuadRegion>>("regions.inSourceCoords");
        OcrQuadRegion mapped = Assert.Single(mappedRegions);

        Assert.Equal(new OcrQuadRegion(11, 22, 13, 22, 13, 24, 11, 24), mapped);
    }

    [Fact]
    public async Task ExecuteAsync_MapsOcrQuadRegionWithExplicitQuadPolicy()
    {
        await using var context = VmRunContextFactory.Create();
        context.Set("region.inCropCoords", new OcrQuadRegion(1, 2, 3, 2, 3, 4, 1, 4));
        context.Set("transform.crop.toSource", new CropCoordinateBackTransform(10, 20));

        var instruction = new Op_Map_Coordinates(
            "region.inCropCoords",
            "transform.crop.toSource",
            "region.inSourceCoords",
            CoordinateMappingShapePolicy.Quad);

        OpResult result = await instruction.ExecuteAsync(context, CancellationToken.None);

        Assert.Equal(OpResultKind.Continue, result.Kind);
        Assert.Equal(
            new OcrQuadRegion(11, 22, 13, 22, 13, 24, 11, 24),
            context.Get<OcrQuadRegion>("region.inSourceCoords"));
    }

    [Fact]
    public async Task ExecuteAsync_MapsYoloBoxArrayThroughResizeTransform()
    {
        await using var context = VmRunContextFactory.Create();
        var detections =
            new[]
            {
                new YoloBox
                {
                    X = 3,
                    Y = 4,
                    W = 13,
                    H = 24,
                    Score = 0.7f,
                    Class = 5,
                },
                new YoloBox
                {
                    X = 10,
                    Y = 8,
                    W = 18,
                    H = 14,
                    Score = 0.4f,
                    Class = 2,
                },
            };

        context.Set("boxes.inResizedCoords", detections);
        context.Set("transform.resized.toSource", new ResizeCoordinateBackTransform(
            sourceWidth: 200,
            sourceHeight: 100,
            targetWidth: 100,
            targetHeight: 50));

        var instruction = new Op_Map_Coordinates(
            "boxes.inResizedCoords",
            "transform.resized.toSource",
            "boxes.inSourceCoords");

        OpResult result = await instruction.ExecuteAsync(context, CancellationToken.None);

        Assert.Equal(OpResultKind.Continue, result.Kind);

        YoloBox[] mapped = context.Get<YoloBox[]>("boxes.inSourceCoords");
        Assert.Equal(2, mapped.Length);
        Assert.Equal(6, mapped[0].X);
        Assert.Equal(8, mapped[0].Y);
        Assert.Equal(26, mapped[0].W);
        Assert.Equal(48, mapped[0].H);
        Assert.Equal(0.7f, mapped[0].Score);
        Assert.Equal(5, mapped[0].Class);
        Assert.Equal(20, mapped[1].X);
        Assert.Equal(16, mapped[1].Y);
        Assert.Equal(36, mapped[1].W);
        Assert.Equal(28, mapped[1].H);
        Assert.Equal(0.4f, mapped[1].Score);
        Assert.Equal(2, mapped[1].Class);
    }

    [Fact]
    public async Task ExecuteAsync_MapsYoloBoxWithExplicitBoundingBoxPolicy()
    {
        await using var context = VmRunContextFactory.Create();
        context.Set("box.inResizedCoords", new YoloBox
        {
            X = 5,
            Y = 7,
            W = 15,
            H = 17,
            Score = 0.6f,
            Class = 4,
        });
        context.Set("transform.resized.toSource", new ResizeCoordinateBackTransform(
            sourceWidth: 200,
            sourceHeight: 100,
            targetWidth: 100,
            targetHeight: 50));

        var instruction = new Op_Map_Coordinates(
            "box.inResizedCoords",
            "transform.resized.toSource",
            "box.inSourceCoords",
            CoordinateMappingShapePolicy.BoundingBox);

        OpResult result = await instruction.ExecuteAsync(context, CancellationToken.None);

        Assert.Equal(OpResultKind.Continue, result.Kind);

        YoloBox mapped = context.Get<YoloBox>("box.inSourceCoords");
        Assert.Equal(10, mapped.X);
        Assert.Equal(14, mapped.Y);
        Assert.Equal(30, mapped.W);
        Assert.Equal(34, mapped.H);
        Assert.Equal(0.6f, mapped.Score);
        Assert.Equal(4, mapped.Class);
    }

    [Fact]
    public async Task ExecuteAsync_ReturnsFailForUnsupportedShapePolicy()
    {
        await using var context = VmRunContextFactory.Create();
        context.Set("box.inCropCoords", new YoloBox
        {
            X = 1,
            Y = 2,
            W = 3,
            H = 4,
            Score = 0.5f,
            Class = 1,
        });
        context.Set("transform.crop.toSource", new CropCoordinateBackTransform(10, 20));

        var instruction = new Op_Map_Coordinates(
            "box.inCropCoords",
            "transform.crop.toSource",
            "box.inSourceCoords",
            CoordinateMappingShapePolicy.Quad);

        OpResult result = await instruction.ExecuteAsync(context, CancellationToken.None);

        Assert.Equal(OpResultKind.Fail, result.Kind);
        Assert.Contains("Shape policy 'Quad' is not supported for YoloBox payloads yet.", result.Reason);
        Assert.False(context.Contains("box.inSourceCoords"));
    }

    [Fact]
    public async Task ExecuteAsync_MapsYoloObbListThroughCropTransform()
    {
        await using var context = VmRunContextFactory.Create();
        var detections = new List<YoloObb>
        {
            new()
            {
                X = 20,
                Y = 10,
                W = 8,
                H = 4,
                Angle = 0,
                Score = 0.8f,
                Class = 1,
            },
        };

        context.Set("obb.inCropCoords", detections);
        context.Set("transform.crop.toSource", new CropCoordinateBackTransform(100, 50));

        var instruction = new Op_Map_Coordinates(
            "obb.inCropCoords",
            "transform.crop.toSource",
            "obb.inSourceCoords");

        OpResult result = await instruction.ExecuteAsync(context, CancellationToken.None);

        Assert.Equal(OpResultKind.Continue, result.Kind);

        List<YoloObb> mapped = context.Get<List<YoloObb>>("obb.inSourceCoords");
        YoloObb item = Assert.Single(mapped);
        Assert.Equal(120, item.X, precision: 4);
        Assert.Equal(60, item.Y, precision: 4);
        Assert.Equal(8, item.W, precision: 4);
        Assert.Equal(4, item.H, precision: 4);
        Assert.Equal(0, item.Angle, precision: 4);
        Assert.Equal(0.8f, item.Score);
        Assert.Equal(1, item.Class);
    }

    [Fact]
    public async Task ExecuteAsync_MapsYoloObbWithExplicitBoundingObbPolicy()
    {
        await using var context = VmRunContextFactory.Create();
        context.Set("obb.inCropCoords", new YoloObb
        {
            X = 20,
            Y = 10,
            W = 8,
            H = 4,
            Angle = 0,
            Score = 0.8f,
            Class = 1,
        });
        context.Set("transform.crop.toSource", new CropCoordinateBackTransform(100, 50));

        var instruction = new Op_Map_Coordinates(
            "obb.inCropCoords",
            "transform.crop.toSource",
            "obb.inSourceCoords",
            CoordinateMappingShapePolicy.BoundingOBB);

        OpResult result = await instruction.ExecuteAsync(context, CancellationToken.None);

        Assert.Equal(OpResultKind.Continue, result.Kind);

        YoloObb item = context.Get<YoloObb>("obb.inSourceCoords");
        Assert.Equal(120, item.X, precision: 4);
        Assert.Equal(60, item.Y, precision: 4);
        Assert.Equal(8, item.W, precision: 4);
        Assert.Equal(4, item.H, precision: 4);
        Assert.Equal(0, item.Angle, precision: 4);
        Assert.Equal(0.8f, item.Score);
        Assert.Equal(1, item.Class);
    }

    [Fact]
    public async Task ExecuteAsync_ReturnsFailWhenPayloadMapperIsMissing()
    {
        await using var context = VmRunContextFactory.Create();
        context.Set("unsupported.payload", new object());
        context.Set("transform.crop.toSource", new CropCoordinateBackTransform(1, 2));

        var instruction = new Op_Map_Coordinates(
            "unsupported.payload",
            "transform.crop.toSource",
            "unsupported.output");

        OpResult result = await instruction.ExecuteAsync(context, CancellationToken.None);

        Assert.Equal(OpResultKind.Fail, result.Kind);
        Assert.Contains("No coordinate mapper is registered", result.Reason);
    }

    [Fact]
    public async Task ExecuteAsync_ReturnsFailWhenTransformRegisterIsMissing()
    {
        await using var context = VmRunContextFactory.Create();
        context.Set("regions.inCropCoords", new OcrQuadRegion(1, 2, 3, 2, 3, 4, 1, 4));

        var instruction = new Op_Map_Coordinates(
            "regions.inCropCoords",
            "transform.missing",
            "regions.inSourceCoords");

        OpResult result = await instruction.ExecuteAsync(context, CancellationToken.None);

        Assert.Equal(OpResultKind.Fail, result.Kind);
        Assert.Contains("Transform key 'transform.missing' was not found", result.Reason);
    }

    [Fact]
    public void Op_Onnx_Crop_DeclaresTransformOutputWhenRequested()
    {
        var instruction = new Op_Onnx_Crop_U8_NHWC(
            "input",
            "output",
            new Rect(2, 3, 10, 20),
            outputTransformKey: "transform.crop.toSource",
            isFinal: true);

        VarRequirement transformWrite = Assert.Single(
            instruction.Descriptor.Writes,
            requirement => requirement.Key == "transform.crop.toSource");

        Assert.Equal(typeof(ICoordinateBackTransform), transformWrite.ValueType);
    }

    [Fact]
    public void Op_Onnx_Resize_DeclaresTransformOutputWhenRequested()
    {
        var instruction = new Op_Onnx_Resize_U8_NHWC(
            "input",
            "output",
            new Size(320, 240),
            outputTransformKey: "transform.resized.toCrop",
            isFinal: true);

        VarRequirement transformWrite = Assert.Single(
            instruction.Descriptor.Writes,
            requirement => requirement.Key == "transform.resized.toCrop");

        Assert.Equal(typeof(ICoordinateBackTransform), transformWrite.ValueType);
    }

    [Theory]
    [MemberData(nameof(FullProgramBackends))]
    public async Task VmProgram_MapsDetectorResultBackThroughResizeAndCropRegisters(InferenceBackend executionBackend)
    {
        OnnxExecutionBackendAvailability.AssertAvailable(executionBackend);

        using Mat sourceImage = CreateCoordinateTestImage(width: 32, height: 24);
        using OrtValue sourceTensor = OrtTestTensorFactory.CreateBgrU8NhwcTensor(sourceImage);
        using var crop = new Op_Onnx_Crop_U8_NHWC(
            "image.source",
            "image.crop",
            new Rect(4, 3, 16, 12),
            outputTransformKey: "transform.crop.toSource",
            isFinal: false,
            executionBackend: executionBackend);
        using var resize = new Op_Onnx_Resize_U8_NHWC(
            "image.crop",
            "image.modelInput",
            new Size(8, 6),
            outputTransformKey: "transform.modelInput.toCrop",
            isFinal: true,
            executionBackend: executionBackend);
        await using var context = VmRunContextFactory.Create();
        context.Set("image.source", sourceTensor);

        VmProgram program = new VmProgramBuilder()
            .Step(crop)
            .Step(resize)
            .Step(new OpDelegate(
                OpDescriptor.Create(
                    "fake-detector",
                    "test.fakeDetector",
                    reads: [VarRequirement.Read<OrtValue>("image.modelInput")],
                    writes: [VarRequirement.Write<YoloDetectionBatchResult<YoloBox>>("detections.modelInput")]),
                (runContext, _) =>
                {
                    OrtValue modelInput = runContext.Get<OrtValue>("image.modelInput");
                    long[] shape = modelInput.GetTensorTypeAndShape().Shape;
                    Assert.Equal([1, 6, 8, 3], shape);

                    runContext.Set("detections.modelInput", new YoloDetectionBatchResult<YoloBox>(
                        [
                            new YoloBox
                            {
                                X = 2,
                                Y = 2,
                                W = 6,
                                H = 4,
                                Score = 0.95f,
                                Class = 7,
                            },
                        ]));
                    return ValueTask.FromResult(OpResult.Continue);
                }))
            .Step(new Op_Map_Coordinates(
                "detections.modelInput",
                "transform.modelInput.toCrop",
                "detections.crop"))
            .Step(new Op_Map_Coordinates(
                "detections.crop",
                "transform.crop.toSource",
                "detections.source"))
            .Build();

        await program.ExecuteAsync(context, CancellationToken.None);

        Assert.True(context.TryGet("transform.modelInput.toCrop", out ICoordinateBackTransform _));
        Assert.True(context.TryGet("transform.crop.toSource", out ICoordinateBackTransform _));

        YoloDetectionBatchResult<YoloBox> cropResult = context.Get<YoloDetectionBatchResult<YoloBox>>("detections.crop");
        YoloBox cropBox = Assert.Single(cropResult.Detections);
        Assert.Equal(4, cropBox.X);
        Assert.Equal(4, cropBox.Y);
        Assert.Equal(12, cropBox.W);
        Assert.Equal(8, cropBox.H);

        YoloDetectionBatchResult<YoloBox> sourceResult = context.Get<YoloDetectionBatchResult<YoloBox>>("detections.source");
        YoloBox sourceBox = Assert.Single(sourceResult.Detections);
        Assert.Equal(8, sourceBox.X);
        Assert.Equal(7, sourceBox.Y);
        Assert.Equal(16, sourceBox.W);
        Assert.Equal(11, sourceBox.H);
        Assert.Equal(0.95f, sourceBox.Score);
        Assert.Equal(7, sourceBox.Class);

        Assert.Equal(5, context.Trace.Instructions.Count);
        Assert.Equal(
            ["op.onnx.crop.u8.nhwc", "op.onnx.resize.u8.nhwc", "test.fakeDetector", "op.map.coordinates", "op.map.coordinates"],
            context.Trace.Instructions.Select(static item => item.Operation).ToArray());
    }

    [Theory]
    [MemberData(nameof(FullProgramBackends))]
    public async Task VmProgram_MapsQuadBackThroughPerspectiveUndistortRotateResizePadResizeAndCropRegisters(InferenceBackend executionBackend)
    {
        OnnxExecutionBackendAvailability.AssertAvailable(executionBackend);

        using Mat sourceImage = CreateCoordinateTestImage(width: 1000, height: 750);
        using OrtValue sourceTensor = OrtTestTensorFactory.CreateBgrU8NhwcTensor(sourceImage);
        using var crop = new Op_Onnx_Crop_U8_NHWC(
            "image.source",
            "image.crop",
            new Rect(100, 50, 400, 300),
            outputTransformKey: "transform.crop.toSource",
            isFinal: false,
            executionBackend: executionBackend);
        using var padResize = new Op_Onnx_PadResize_U8_NHWC(
            "image.crop",
            "image.padded",
            new Size(640, 640),
            outputTransformKey: "transform.padded.toCrop",
            isFinal: false,
            executionBackend: executionBackend);
        using var resize = new Op_Onnx_Resize_U8_NHWC(
            "image.padded",
            "image.resized",
            new Size(320, 320),
            outputTransformKey: "transform.resized.toPadded",
            isFinal: false,
            executionBackend: executionBackend);
        using var rotate = new Op_Onnx_Rotate90_U8_NHWC(
            "image.resized",
            "image.rotated",
            Rotate90Mode.Clockwise90,
            outputTransformKey: "transform.rotated.toResized",
            isFinal: false,
            executionBackend: executionBackend);
        using var undistort = new Op_Onnx_Undistort_U8_NHWC(
            "image.rotated",
            "image.undistorted",
            new RadialTangentialDistortionParameters(Fx: 300, Fy: 300, Cx: 160, Cy: 160, K1: 0),
            outputTransformKey: "transform.undistorted.toRotated",
            isFinal: false,
            executionBackend: executionBackend);
        using var perspective = new Op_Onnx_Perspective_U8_NHWC(
            "image.undistorted",
            "image.modelInput",
            [
                new Point2f(20, 30),
                new Point2f(299, 30),
                new Point2f(299, 309),
                new Point2f(20, 309)
            ],
            new Size(280, 280),
            outputTransformKey: "transform.modelInput.toUndistorted",
            isFinal: true,
            executionBackend: executionBackend);
        await using var context = VmRunContextFactory.Create();
        context.Set("image.source", sourceTensor);

        VmProgram program = new VmProgramBuilder()
            .Step(crop)
            .Step(padResize)
            .Step(resize)
            .Step(rotate)
            .Step(undistort)
            .Step(perspective)
            .Step(new OpDelegate(
                OpDescriptor.Create(
                    "fake-quad-detector",
                    "test.fakeQuadDetector",
                    reads: [VarRequirement.Read<OrtValue>("image.modelInput")],
                    writes: [VarRequirement.Write<OcrQuadRegion>("region.modelInput")]),
                (runContext, _) =>
                {
                    OrtValue modelInput = runContext.Get<OrtValue>("image.modelInput");
                    long[] shape = modelInput.GetTensorTypeAndShape().Shape;
                    Assert.Equal([1, 280, 280, 3], shape);

                    runContext.Set("region.modelInput", new OcrQuadRegion(10, 20, 30, 20, 30, 40, 10, 40));
                    return ValueTask.FromResult(OpResult.Continue);
                }))
            .Step(new Op_Map_Coordinates(
                "region.modelInput",
                "transform.modelInput.toUndistorted",
                "region.undistorted",
                CoordinateMappingShapePolicy.Quad))
            .Step(new Op_Map_Coordinates(
                "region.undistorted",
                "transform.undistorted.toRotated",
                "region.rotated",
                CoordinateMappingShapePolicy.Quad))
            .Step(new Op_Map_Coordinates(
                "region.rotated",
                "transform.rotated.toResized",
                "region.resized",
                CoordinateMappingShapePolicy.Quad))
            .Step(new Op_Map_Coordinates(
                "region.resized",
                "transform.resized.toPadded",
                "region.padded",
                CoordinateMappingShapePolicy.Quad))
            .Step(new Op_Map_Coordinates(
                "region.padded",
                "transform.padded.toCrop",
                "region.crop",
                CoordinateMappingShapePolicy.Quad))
            .Step(new Op_Map_Coordinates(
                "region.crop",
                "transform.crop.toSource",
                "region.source",
                CoordinateMappingShapePolicy.Quad))
            .Build();

        await program.ExecuteAsync(context, CancellationToken.None);

        Assert.True(context.TryGet("transform.modelInput.toUndistorted", out ICoordinateBackTransform _));
        Assert.True(context.TryGet("transform.undistorted.toRotated", out ICoordinateBackTransform _));
        Assert.True(context.TryGet("transform.rotated.toResized", out ICoordinateBackTransform _));
        Assert.True(context.TryGet("transform.resized.toPadded", out ICoordinateBackTransform _));
        Assert.True(context.TryGet("transform.padded.toCrop", out ICoordinateBackTransform _));
        Assert.True(context.TryGet("transform.crop.toSource", out ICoordinateBackTransform _));
        OcrQuadRegion sourceRegion = context.Get<OcrQuadRegion>("region.source");
        Assert.Equal(162.5f, sourceRegion.X0, precision: 5);
        Assert.Equal(361.25f, sourceRegion.Y0, precision: 5);
        Assert.Equal(162.5f, sourceRegion.X1, precision: 5);
        Assert.Equal(336.25f, sourceRegion.Y1, precision: 5);
        Assert.Equal(187.5f, sourceRegion.X2, precision: 5);
        Assert.Equal(336.25f, sourceRegion.Y2, precision: 5);
        Assert.Equal(187.5f, sourceRegion.X3, precision: 5);
        Assert.Equal(361.25f, sourceRegion.Y3, precision: 5);

        Assert.Equal(13, context.Trace.Instructions.Count);
        Assert.Equal(
            [
                "op.onnx.crop.u8.nhwc",
                "op.onnx.padResize.u8.nhwc",
                "op.onnx.resize.u8.nhwc",
                "op.onnx.rotate90.u8.nhwc",
                "op.onnx.undistort.u8.nhwc",
                "op.onnx.perspective.u8.nhwc",
                "test.fakeQuadDetector",
                "op.map.coordinates",
                "op.map.coordinates",
                "op.map.coordinates",
                "op.map.coordinates",
                "op.map.coordinates",
                "op.map.coordinates"
            ],
            context.Trace.Instructions.Select(static item => item.Operation).ToArray());
    }

    public static IEnumerable<object[]> FullProgramBackends =>
        OnnxExecutionBackendMatrix.EnabledBackends.Select(static backend => new object[] { backend });

    static Mat CreateCoordinateTestImage(int width, int height)
    {
        var image = new Mat(height, width, MatType.CV_8UC3);

        for(int y = 0; y < height; y++)
        {
            for(int x = 0; x < width; x++)
            {
                image.Set(y, x, new Vec3b(
                    (byte)(x * 3),
                    (byte)(y * 5),
                    (byte)((x + y) * 2)));
            }
        }

        return image;
    }
}
