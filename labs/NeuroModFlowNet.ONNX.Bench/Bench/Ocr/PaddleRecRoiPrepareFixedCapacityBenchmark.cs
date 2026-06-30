using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX.Graph.Builders;
using OnnxDataType = Onnx.TensorProto.Types.DataType;

namespace NeuroModFlowNet.ONNX.Bench;

[Config(typeof(OcrRoiPrepareConfig))]
[MemoryDiagnoser]
[GroupBenchmarksBy(BenchmarkLogicalGroupRule.ByParams)]
public class PaddleRecRoiPrepareFixedCapacityBenchmark
{
    const int SourceWidth = 2048;
    const int SourceHeight = 1024;
    const int Channels = 3;
    const int TargetHeight = 48;

    readonly long[] sourceShape = [1, Channels, SourceHeight, SourceWidth];
    readonly RunOptions runOptions = new();

    OnnxExecutionContext? uploadContext;
    OnnxExecutionContext? matrixUploadContext;
    OnnxExecutionContext? prepareContext;
    OrtMemoryInfo? uploadCudaMemoryInfo;
    OrtAllocator? uploadCudaAllocator;
    OrtMemoryInfo? outputCudaMemoryInfo;
    OrtAllocator? outputCudaAllocator;
    OrtMemoryInfo? matricesCudaMemoryInfo;
    OrtAllocator? matricesCudaAllocator;
    OrtValue? sourceOnDevice;
    OrtValue? matricesOnDevice;
    OrtValue? preparedOutput;

    [Params(InferenceBackend.Cuda, InferenceBackend.TensorRt)]
    public InferenceBackend _InferenceBackend { get; set; } = InferenceBackend.Cuda;

    [Params(0, 2, 6, 8)]
    public int ActualRegionCount { get; set; } = 2;

    [Params(6, 16)]
    public int MaxRoiCount { get; set; } = 6;

    [Params(320, 640)]
    public int TargetWidth { get; set; } = 320;

    [GlobalSetup]
    public void Setup()
    {
        byte[] modelBytes = PaddleRecRoiPrepareBuilder.BuildBatchedGridSampleFromMatricesFP32Nchw(
            SourceWidth,
            SourceHeight,
            Channels,
            TargetWidth,
            TargetHeight,
            MaxRoiCount);

        sourceOnDevice = UploadSourceToDevice(CreateSourceTensor());
        matricesOnDevice = UploadMatricesToDevice(CreateMatrixTensor(CreateCapacityMatrices(ActualRegionCount, MaxRoiCount, TargetWidth, TargetHeight)));

        prepareContext = new OnnxExecutionContext(new OnnxModel(
            modelBytes,
            _InferenceBackend,
            ConfigureRuntimeOperatorProvider,
            $"fixed-capacity-{MaxRoiCount}.paddle-rec-roi-prepare.onnx"), ownsModel: true);

        outputCudaMemoryInfo = new OrtMemoryInfo(
            OrtMemoryInfo.allocatorCUDA,
            OrtAllocatorType.DeviceAllocator,
            0,
            OrtMemType.Default);
        outputCudaAllocator = new OrtAllocator(prepareContext.Model.Session, outputCudaMemoryInfo);
        preparedOutput = OrtValue.CreateAllocatedTensorValue(
            outputCudaAllocator,
            TensorElementType.Float,
            [MaxRoiCount, Channels, TargetHeight, TargetWidth]);

        RunPrepare();
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        runOptions.Dispose();

        preparedOutput?.Dispose();
        outputCudaAllocator?.Dispose();
        outputCudaMemoryInfo?.Dispose();
        prepareContext?.Dispose();

        matricesOnDevice?.Dispose();
        matricesCudaAllocator?.Dispose();
        matricesCudaMemoryInfo?.Dispose();
        matrixUploadContext?.Dispose();

        sourceOnDevice?.Dispose();
        uploadCudaAllocator?.Dispose();
        uploadCudaMemoryInfo?.Dispose();
        uploadContext?.Dispose();
    }

    [Benchmark(OperationsPerInvoke = 1), BenchmarkCategory("OCR_ROI_PREP_FIXED_CAPACITY")]
    public void PrepareFixedCapacityPaddleRecBatch() => RunPrepare();

    void RunPrepare()
    {
        ArgumentNullException.ThrowIfNull(prepareContext);
        ArgumentNullException.ThrowIfNull(sourceOnDevice);
        ArgumentNullException.ThrowIfNull(matricesOnDevice);
        ArgumentNullException.ThrowIfNull(preparedOutput);

        prepareContext.IoBinding.ClearBoundInputs();
        prepareContext.IoBinding.ClearBoundOutputs();

        try
        {
            prepareContext.IoBinding.BindInput(PaddleRecRoiPrepareBuilder.InputName, sourceOnDevice);
            prepareContext.IoBinding.BindInput(PaddleRecRoiPrepareBuilder.MatrixInputName, matricesOnDevice);
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
            "fixed-capacity-paddle-rec-roi-prepare-upload.onnx"), ownsModel: true);

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
            IdentityBuilder.Build(OnnxDataType.Float, MaxRoiCount, 9),
            InferenceBackend.Cuda,
            ConfigureRuntimeOperatorProvider,
            "fixed-capacity-paddle-rec-roi-prepare-matrix-upload.onnx"), ownsModel: true);

        matricesCudaMemoryInfo = new OrtMemoryInfo(
            OrtMemoryInfo.allocatorCUDA,
            OrtAllocatorType.DeviceAllocator,
            0,
            OrtMemType.Default);
        matricesCudaAllocator = new OrtAllocator(matrixUploadContext.Model.Session, matricesCudaMemoryInfo);

        OrtValue deviceValue = OrtValue.CreateAllocatedTensorValue(matricesCudaAllocator, TensorElementType.Float, [MaxRoiCount, 9]);
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

    IReadOnlyList<float[]> CreateCapacityMatrices(int actualRegionCount, int maxRoiCount, int targetWidth, int targetHeight)
    {
        var matrices = new float[maxRoiCount][];
        int materializedRegionCount = Math.Min(actualRegionCount, maxRoiCount);
        float[] dummyMatrix = CreateTargetToSourceMatrix(
            SourceWidth * 0.5f,
            SourceHeight * 0.5f,
            2f,
            2f,
            0f,
            targetWidth,
            targetHeight);

        for(int index = 0; index < maxRoiCount; index++)
        {
            if(index >= materializedRegionCount)
            {
                matrices[index] = materializedRegionCount > 0 ? matrices[materializedRegionCount - 1] : dummyMatrix;
                continue;
            }

            float lane = (index % 4 + 1) / 5f;
            float row = (index / 4 + 1) / (float)(Math.Max(1, (materializedRegionCount + 3) / 4) + 1);
            float centerX = SourceWidth * lane;
            float centerY = SourceHeight * row;
            float regionWidth = 170f + (index % 3) * 35f;
            float regionHeight = 38f + (index % 2) * 12f;
            float angleDegrees = -18f + index * 7f;

            matrices[index] = CreateTargetToSourceMatrix(centerX, centerY, regionWidth, regionHeight, angleDegrees, targetWidth, targetHeight);
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
        var rectangle = new OpenCvSharp.RotatedRect(new OpenCvSharp.Point2f(centerX, centerY), new OpenCvSharp.Size2f(width, height), angleDegrees);
        OpenCvSharp.Point2f[] sourcePoints = rectangle.Points();
        PaddleRecRoiPrepareBenchmark.OrderRotatedRectPoints(sourcePoints);

        OpenCvSharp.Point2f[] targetPoints =
        [
            new(0, 0),
            new(targetWidth - 1, 0),
            new(targetWidth - 1, targetHeight - 1),
            new(0, targetHeight - 1)
        ];

        using OpenCvSharp.Mat matrix = OpenCvSharp.Cv2.GetPerspectiveTransform(targetPoints, sourcePoints);
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
