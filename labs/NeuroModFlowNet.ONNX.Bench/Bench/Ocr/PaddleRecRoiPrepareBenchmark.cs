using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX.Graph.Builders;
using NeuroModFlowNet.Pipeline.ONNX;
using OpenCvSharp;
using OnnxDataType = Onnx.TensorProto.Types.DataType;

namespace NeuroModFlowNet.ONNX.Bench;

[Config(typeof(OcrRoiPrepareConfig))]
[MemoryDiagnoser]
[GroupBenchmarksBy(BenchmarkLogicalGroupRule.ByParams)]
public class PaddleRecRoiPrepareBenchmark
{
    const int SourceWidth = 2048;
    const int SourceHeight = 1024;
    const int Channels = 3;
    const int TargetHeight = 48;
    const float PaddingPixels = 2f;
    const float PaddingScale = 0.12f;

    readonly long[] sourceShape = [1, Channels, SourceHeight, SourceWidth];
    readonly RunOptions runOptions = new();

    OnnxExecutionContext? uploadContext;
    OnnxExecutionContext? matrixUploadContext;
    OnnxExecutionContext? prepareContext;
    OrtMemoryInfo? uploadCudaMemoryInfo;
    OrtAllocator? uploadCudaAllocator;
    OrtMemoryInfo? outputCudaMemoryInfo;
    OrtAllocator? outputCudaAllocator;
    OrtValue? matricesOnDevice;
    OrtMemoryInfo? matricesCudaMemoryInfo;
    OrtAllocator? matricesCudaAllocator;
    OrtValue? sourceOnDevice;
    OrtValue? preparedOutput;

    [Params(InferenceBackend.Cuda, InferenceBackend.TensorRt)]
    public InferenceBackend _InferenceBackend { get; set; } = InferenceBackend.Cuda;

    [Params(
        PaddleRecRoiPrepareAlgorithm.PerRegionGridSampleConcat,
        PaddleRecRoiPrepareAlgorithm.BatchedGridSample,
        PaddleRecRoiPrepareAlgorithm.BatchedGridSampleFromMatrices)]
    public PaddleRecRoiPrepareAlgorithm Algorithm { get; set; } = PaddleRecRoiPrepareAlgorithm.BatchedGridSample;

    [Params(4, 16)]
    public int RegionCount { get; set; } = 4;

    [Params(320, 640)]
    public int TargetWidth { get; set; } = 320;

    [GlobalSetup]
    public void Setup()
    {
        IReadOnlyList<float[]> matrices = CreateRoiMatrices(RegionCount, TargetWidth, TargetHeight);
        byte[] modelBytes = Algorithm switch
        {
            PaddleRecRoiPrepareAlgorithm.PerRegionGridSampleConcat =>
                PaddleRecRoiPrepareBuilder.BuildPerRegionGridSampleConcatFP32Nchw(
                    SourceWidth,
                    SourceHeight,
                    Channels,
                    TargetWidth,
                    TargetHeight,
                    matrices),
            PaddleRecRoiPrepareAlgorithm.BatchedGridSample =>
                PaddleRecRoiPrepareBuilder.BuildBatchedGridSampleFP32Nchw(
                    SourceWidth,
                    SourceHeight,
                    Channels,
                    TargetWidth,
                    TargetHeight,
                    matrices),
            PaddleRecRoiPrepareAlgorithm.BatchedGridSampleFromMatrices =>
                PaddleRecRoiPrepareBuilder.BuildBatchedGridSampleFromMatricesFP32Nchw(
                    SourceWidth,
                    SourceHeight,
                    Channels,
                    TargetWidth,
                    TargetHeight,
                    RegionCount),
            _ => throw new ArgumentOutOfRangeException(nameof(Algorithm), Algorithm, null)
        };

        sourceOnDevice = UploadSourceToDevice(CreateSourceTensor());
        if(Algorithm == PaddleRecRoiPrepareAlgorithm.BatchedGridSampleFromMatrices)
            matricesOnDevice = UploadMatricesToDevice(CreateMatrixTensor(matrices));

        prepareContext = new OnnxExecutionContext(new OnnxModel(
            modelBytes,
            _InferenceBackend,
            ConfigureRuntimeOperatorProvider,
            $"{Algorithm}.paddle-rec-roi-prepare.onnx"), ownsModel: true);

        outputCudaMemoryInfo = new OrtMemoryInfo(
            OrtMemoryInfo.allocatorCUDA,
            OrtAllocatorType.DeviceAllocator,
            0,
            OrtMemType.Default);
        outputCudaAllocator = new OrtAllocator(prepareContext.Model.Session, outputCudaMemoryInfo);
        preparedOutput = OrtValue.CreateAllocatedTensorValue(
            outputCudaAllocator,
            TensorElementType.Float,
            [RegionCount, Channels, TargetHeight, TargetWidth]);

        RunPrepare();
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        runOptions.Dispose();

        preparedOutput?.Dispose();
        preparedOutput = null;

        outputCudaAllocator?.Dispose();
        outputCudaAllocator = null;

        outputCudaMemoryInfo?.Dispose();
        outputCudaMemoryInfo = null;

        prepareContext?.Dispose();
        prepareContext = null;

        matricesOnDevice?.Dispose();
        matricesOnDevice = null;

        matricesCudaAllocator?.Dispose();
        matricesCudaAllocator = null;

        matricesCudaMemoryInfo?.Dispose();
        matricesCudaMemoryInfo = null;

        matrixUploadContext?.Dispose();
        matrixUploadContext = null;

        sourceOnDevice?.Dispose();
        sourceOnDevice = null;

        uploadCudaAllocator?.Dispose();
        uploadCudaAllocator = null;

        uploadCudaMemoryInfo?.Dispose();
        uploadCudaMemoryInfo = null;

        uploadContext?.Dispose();
        uploadContext = null;
    }

    [Benchmark(OperationsPerInvoke = 1), BenchmarkCategory("OCR_ROI_PREP")]
    public void PreparePaddleRecBatch() => RunPrepare();

    void RunPrepare()
    {
        ArgumentNullException.ThrowIfNull(prepareContext);
        ArgumentNullException.ThrowIfNull(sourceOnDevice);
        ArgumentNullException.ThrowIfNull(preparedOutput);

        prepareContext.IoBinding.ClearBoundInputs();
        prepareContext.IoBinding.ClearBoundOutputs();

        try
        {
            prepareContext.IoBinding.BindInput(PaddleRecRoiPrepareBuilder.InputName, sourceOnDevice);
            if(Algorithm == PaddleRecRoiPrepareAlgorithm.BatchedGridSampleFromMatrices)
            {
                ArgumentNullException.ThrowIfNull(matricesOnDevice);
                prepareContext.IoBinding.BindInput(PaddleRecRoiPrepareBuilder.MatrixInputName, matricesOnDevice);
            }

            prepareContext.IoBinding.BindOutput(PaddleRecRoiPrepareBuilder.OutputName, preparedOutput);
            prepareContext.Model.Session.RunWithBinding(runOptions, prepareContext.IoBinding);
            prepareContext.IoBinding.SynchronizeBoundOutputs();
        }
        finally
        {
            prepareContext.IoBinding.ClearBoundInputs();
            prepareContext.IoBinding.ClearBoundOutputs();
        }
    }

    OrtValue UploadSourceToDevice(OrtValue source)
    {
        uploadContext = new OnnxExecutionContext(new OnnxModel(
            IdentityBuilder.Build(OnnxDataType.Float, sourceShape),
            InferenceBackend.Cuda,
            ConfigureRuntimeOperatorProvider,
            "paddle-rec-roi-prepare-upload.onnx"), ownsModel: true);

        uploadCudaMemoryInfo = new OrtMemoryInfo(
            OrtMemoryInfo.allocatorCUDA,
            OrtAllocatorType.DeviceAllocator,
            0,
            OrtMemType.Default);
        uploadCudaAllocator = new OrtAllocator(uploadContext.Model.Session, uploadCudaMemoryInfo);

        OrtValue deviceValue = OrtValue.CreateAllocatedTensorValue(uploadCudaAllocator, TensorElementType.Float, sourceShape);
        uploadContext.IoBinding.BindInput(IdentityBuilder.InputName, source);
        uploadContext.IoBinding.BindOutput(IdentityBuilder.OutputName, deviceValue);
        uploadContext.Model.Session.RunWithBinding(uploadContext.RunOptions, uploadContext.IoBinding);
        uploadContext.IoBinding.SynchronizeBoundOutputs();
        uploadContext.IoBinding.ClearBoundInputs();
        uploadContext.IoBinding.ClearBoundOutputs();
        source.Dispose();
        return deviceValue;
    }

    OrtValue UploadMatricesToDevice(OrtValue matrices)
    {
        matrixUploadContext = new OnnxExecutionContext(new OnnxModel(
            IdentityBuilder.Build(OnnxDataType.Float, RegionCount, 9),
            InferenceBackend.Cuda,
            ConfigureRuntimeOperatorProvider,
            "paddle-rec-roi-prepare-matrix-upload.onnx"), ownsModel: true);

        matricesCudaMemoryInfo = new OrtMemoryInfo(
            OrtMemoryInfo.allocatorCUDA,
            OrtAllocatorType.DeviceAllocator,
            0,
            OrtMemType.Default);
        matricesCudaAllocator = new OrtAllocator(matrixUploadContext.Model.Session, matricesCudaMemoryInfo);

        OrtValue deviceValue = OrtValue.CreateAllocatedTensorValue(matricesCudaAllocator, TensorElementType.Float, [RegionCount, 9]);
        matrixUploadContext.IoBinding.BindInput(IdentityBuilder.InputName, matrices);
        matrixUploadContext.IoBinding.BindOutput(IdentityBuilder.OutputName, deviceValue);
        matrixUploadContext.Model.Session.RunWithBinding(matrixUploadContext.RunOptions, matrixUploadContext.IoBinding);
        matrixUploadContext.IoBinding.SynchronizeBoundOutputs();
        matrixUploadContext.IoBinding.ClearBoundInputs();
        matrixUploadContext.IoBinding.ClearBoundOutputs();
        matrices.Dispose();
        return deviceValue;
    }

    OrtValue CreateSourceTensor()
    {
        OrtValue source = OrtValue.CreateAllocatedTensorValue(
            OrtAllocator.DefaultInstance,
            TensorElementType.Float,
            sourceShape);

        Span<float> data = source.GetTensorMutableDataAsSpan<float>();
        int planeSize = SourceHeight * SourceWidth;
        for(int channel = 0; channel < Channels; channel++)
        {
            int channelOffset = channel * planeSize;
            for(int y = 0; y < SourceHeight; y++)
            {
                for(int x = 0; x < SourceWidth; x++)
                {
                    float xPart = x / (float)(SourceWidth - 1);
                    float yPart = y / (float)(SourceHeight - 1);
                    data[channelOffset + y * SourceWidth + x] = Math.Clamp((xPart + yPart + channel * 0.15f) * 0.5f, 0f, 1f);
                }
            }
        }

        return source;
    }

    static OrtValue CreateMatrixTensor(IReadOnlyList<float[]> matrices)
    {
        OrtValue matrixTensor = OrtValue.CreateAllocatedTensorValue(
            OrtAllocator.DefaultInstance,
            TensorElementType.Float,
            [matrices.Count, 9]);

        Span<float> matrixData = matrixTensor.GetTensorMutableDataAsSpan<float>();
        for(int matrixIndex = 0; matrixIndex < matrices.Count; matrixIndex++)
            matrices[matrixIndex].AsSpan(0, 9).CopyTo(matrixData.Slice(matrixIndex * 9, 9));

        return matrixTensor;
    }

    IReadOnlyList<float[]> CreateRoiMatrices(int regionCount, int targetWidth, int targetHeight)
    {
        var matrices = new float[regionCount][];
        for(int index = 0; index < regionCount; index++)
        {
            float lane = (index % 4 + 1) / 5f;
            float row = (index / 4 + 1) / (float)(Math.Max(1, (regionCount + 3) / 4) + 1);
            float centerX = SourceWidth * lane;
            float centerY = SourceHeight * row;
            float regionWidth = 170f + (index % 3) * 35f;
            float regionHeight = 38f + (index % 2) * 12f;
            float padX = PaddingPixels + regionWidth * PaddingScale;
            float padY = PaddingPixels + regionHeight * PaddingScale;
            float paddedWidth = regionWidth + padX * 2f;
            float paddedHeight = regionHeight + padY * 2f;
            float angleDegrees = -18f + index * 7f;

            matrices[index] = CreateTargetToSourceMatrix(centerX, centerY, paddedWidth, paddedHeight, angleDegrees, targetWidth, targetHeight);
        }

        return matrices;
    }

    static float[] CreateTargetToSourceMatrix(
        float centerX,
        float centerY,
        float width,
        float height,
        float angleDegrees,
        int targetWidth,
        int targetHeight)
    {
        var rectangle = new RotatedRect(new Point2f(centerX, centerY), new Size2f(width, height), angleDegrees);
        Point2f[] sourcePoints = rectangle.Points();
        OrderRotatedRectPoints(sourcePoints);

        Point2f[] targetPoints =
        [
            new(0, 0),
            new(targetWidth - 1, 0),
            new(targetWidth - 1, targetHeight - 1),
            new(0, targetHeight - 1)
        ];

        using Mat matrix = Cv2.GetPerspectiveTransform(targetPoints, sourcePoints);
        return
        [
            (float)matrix.At<double>(0, 0),
            (float)matrix.At<double>(0, 1),
            (float)matrix.At<double>(0, 2),
            (float)matrix.At<double>(1, 0),
            (float)matrix.At<double>(1, 1),
            (float)matrix.At<double>(1, 2),
            (float)matrix.At<double>(2, 0),
            (float)matrix.At<double>(2, 1),
            (float)matrix.At<double>(2, 2)
        ];
    }

    public static void OrderRotatedRectPoints(Point2f[] points)
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

    static void ConfigureRuntimeOperatorProvider(ExecutionProviderConfig config)
    {
        switch(config)
        {
            case TrtConfig trtConfig:
                trtConfig.EnableEngineCache = false;
                trtConfig.EnableFp16 = false;
                trtConfig.EnableBf16 = false;
                trtConfig.BuilderOptimizationLevel = 2;
                break;
            case CudaConfig cudaConfig:
                cudaConfig.EnableCudaGraph = false;
                break;
        }
    }
}
