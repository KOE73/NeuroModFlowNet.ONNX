using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.ONNX.Graph.Builders;
using NeuroModFlowNet.Pipeline;
using OpenCvSharp;
using OnnxDataType = Onnx.TensorProto.Types.DataType;
using CvSize = OpenCvSharp.Size;

namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// EN: Extracts YOLO OBB text regions from one RGB FP32 NCHW image into one PaddleOCR Rec FP32 NCHW batch.
///
/// RU: Вырезает текстовые YOLO OBB области из одного RGB FP32 NCHW изображения в один FP32 NCHW batch для PaddleOCR Rec.
/// </summary>
/// <remarks>
/// EN: Input image values are expected to already be RGB FP32 in the 0..1 range, typically after the YOLO preparation
/// stage. This command performs perspective extraction, fixed output resize, and Paddle Rec normalization to -1..1 in a
/// single generated ONNX graph. ROI matrices are a small CPU-side typed-result artifact and are explicitly uploaded by an
/// Identity graph for GPU backends; the source image itself stays in provider memory.
///
/// RU: Входное изображение ожидается уже как RGB FP32 в диапазоне 0..1, обычно после стадии подготовки YOLO.
/// Команда одним сгенерированным ONNX-графом делает perspective extraction, resize в фиксированный размер и нормализацию
/// Paddle Rec до -1..1. ROI-матрицы являются маленьким CPU-side typed-result артефактом и для GPU backend явно загружаются
/// через Identity graph; исходное изображение остается в provider memory.
/// </remarks>
public sealed class Op_Onnx_ExtractObbToPaddleRec_FP32_NCHW : OpBase, IDisposable, IHasExecutionDevice
{
    readonly string imageInputKey;
    readonly string obbInputKey;
    readonly string outputKey;
    readonly CvSize targetSize;
    readonly float paddingPixels;
    readonly float paddingScale;
    readonly PaddleRecRoiPrepareAlgorithm algorithm;
    readonly int? maxRoiCount;
    readonly PaddleRecRoiOverflowPolicy overflowPolicy;
    readonly string? actualCountOutputKey;
    readonly bool isFinal;
    readonly InferenceBackend executionBackend;

    OnnxExecutionContext? prepareContext;
    OnnxExecutionContext? matrixUploadContext;
    long[]? initializedImageShape;
    int initializedRegionCount;
    OrtMemoryInfo? outputCudaMemoryInfo;
    OrtAllocator? outputCudaAllocator;
    OrtMemoryInfo? matrixCudaMemoryInfo;
    OrtAllocator? matrixCudaAllocator;

    public Op_Onnx_ExtractObbToPaddleRec_FP32_NCHW(
        string imageInputKey,
        string obbInputKey,
        string outputKey,
        CvSize targetSize,
        float paddingPixels = 0f,
        float paddingScale = 0f,
        PaddleRecRoiPrepareAlgorithm algorithm = PaddleRecRoiPrepareAlgorithm.BatchedGridSampleFromMatrices,
        int? maxRoiCount = null,
        PaddleRecRoiOverflowPolicy overflowPolicy = PaddleRecRoiOverflowPolicy.Fail,
        string? actualCountOutputKey = null,
        bool isFinal = false,
        InferenceBackend? executionBackend = null)
        : base(CreateDescriptor(imageInputKey, obbInputKey, outputKey, actualCountOutputKey))
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(imageInputKey);
        ArgumentException.ThrowIfNullOrWhiteSpace(obbInputKey);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputKey);

        if(targetSize.Width <= 0 || targetSize.Height <= 0)
            throw new ArgumentOutOfRangeException(nameof(targetSize), "Target width and height must be positive.");

        if(paddingPixels < 0)
            throw new ArgumentOutOfRangeException(nameof(paddingPixels), "Padding pixels must be non-negative.");

        if(paddingScale < 0)
            throw new ArgumentOutOfRangeException(nameof(paddingScale), "Padding scale must be non-negative.");

        if(maxRoiCount is <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxRoiCount), "Max ROI count must be positive.");

        if(actualCountOutputKey is not null)
            ArgumentException.ThrowIfNullOrWhiteSpace(actualCountOutputKey);

        if(algorithm != PaddleRecRoiPrepareAlgorithm.BatchedGridSampleFromMatrices)
        {
            throw new NotSupportedException(
                $"{nameof(Op_Onnx_ExtractObbToPaddleRec_FP32_NCHW)} supports only {PaddleRecRoiPrepareAlgorithm.BatchedGridSampleFromMatrices} for dynamic OBB payloads.");
        }

        this.imageInputKey = imageInputKey;
        this.obbInputKey = obbInputKey;
        this.outputKey = outputKey;
        this.targetSize = targetSize;
        this.paddingPixels = paddingPixels;
        this.paddingScale = paddingScale;
        this.algorithm = algorithm;
        this.maxRoiCount = maxRoiCount;
        this.overflowPolicy = overflowPolicy;
        this.actualCountOutputKey = actualCountOutputKey;
        this.isFinal = isFinal;
        this.executionBackend = executionBackend ?? InferenceBackend.Cuda;
    }

    bool IHasExecutionDevice.IsGpuExecution =>
        prepareContext?.Model.InferenceBackend is { } backend && backend != InferenceBackend.Cpu;

    string IHasExecutionDevice.ExecutionDeviceName =>
        prepareContext?.Model.InferenceBackend.ToString() ?? "Uninitialized";

    public override ValueTask<OpResult> ExecuteAsync(
        VmRunContext context,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(context);
        cancellationToken.ThrowIfCancellationRequested();

        if(!context.TryGet(imageInputKey, out OrtValue image))
            return ValueTask.FromResult(OpResult.Fail($"Input key '{imageInputKey}' was not found in the pipeline context."));

        if(!context.TryGet(obbInputKey, out YoloObb[] boxes))
            return ValueTask.FromResult(OpResult.Fail($"Input key '{obbInputKey}' was not found in the pipeline context or is not a YoloObb array."));

        if(boxes.Length == 0 && maxRoiCount is null)
            return ValueTask.FromResult(OpResult.Fail("Paddle Rec ROI preparation requires at least one OBB."));

        int outputRegionCount = maxRoiCount ?? boxes.Length;
        int actualRegionCount = boxes.Length;
        if(actualRegionCount > outputRegionCount)
        {
            if(overflowPolicy == PaddleRecRoiOverflowPolicy.Fail)
                return ValueTask.FromResult(OpResult.Fail($"Actual ROI count {actualRegionCount} exceeds fixed max ROI count {outputRegionCount}."));

            actualRegionCount = outputRegionCount;
        }

        var imageInfo = image.GetTensorTypeAndShape();
        string? validationError = ValidateImage(imageInfo.Shape, imageInfo.ElementDataType);
        if(validationError is not null)
            return ValueTask.FromResult(OpResult.Fail(validationError));

        EnsureRuntime(imageInfo.Shape, outputRegionCount);

        using OrtValue matrixHost = CreateMatrixTensor(
            boxes,
            actualRegionCount,
            outputRegionCount,
            checked((int)imageInfo.Shape[3]),
            checked((int)imageInfo.Shape[2]));
        using OrtValue? matrixDevice = executionBackend == InferenceBackend.Cpu
            ? null
            : UploadMatricesToDevice(matrixHost);

        OrtValue output = CreateOutputTensor(outputRegionCount);
        try
        {
            RunWithFreshBinding(image, matrixDevice ?? matrixHost, output);
            context.Set(outputKey, output, disposeWithContext: true);
            output = null!;
        }
        finally
        {
            output?.Dispose();
        }

        if(actualCountOutputKey is not null)
            context.Set(actualCountOutputKey, actualRegionCount);

        return ValueTask.FromResult(OpResult.Continue);
    }

    static string? ValidateImage(long[] imageShape, TensorElementType imageElementType)
    {
        if(imageElementType != TensorElementType.Float)
            return $"Input image tensor must be FP32 RGB NCHW data, actual element type: {imageElementType}.";

        if(imageShape.Length != 4)
            return $"Input image tensor must be 4D NCHW, actual rank: {imageShape.Length}.";

        if(imageShape[0] != 1)
            return $"Input image tensor batch must be 1, actual batch: {imageShape[0]}.";

        if(imageShape[1] != 3)
            return $"Input image tensor must have 3 RGB channels, actual channels: {imageShape[1]}.";

        if(imageShape[2] <= 1 || imageShape[3] <= 1)
            return $"Input image tensor width and height must be greater than 1, actual shape: [{string.Join(", ", imageShape)}].";

        return null;
    }

    void EnsureRuntime(long[] imageShape, int regionCount)
    {
        if(prepareContext is not null &&
            initializedRegionCount == regionCount &&
            initializedImageShape is not null &&
            initializedImageShape.SequenceEqual(imageShape))
        {
            return;
        }

        DisposeRuntimeState();

        byte[] modelBytes = PaddleRecRoiPrepareBuilder.BuildBatchedGridSampleFromMatricesFP32Nchw(
            checked((int)imageShape[3]),
            checked((int)imageShape[2]),
            checked((int)imageShape[1]),
            targetSize.Width,
            targetSize.Height,
            regionCount);

        prepareContext = new OnnxExecutionContext(new OnnxModel(
            modelBytes,
            executionBackend,
            ConfigureRuntimeOperatorProvider,
            $"{algorithm}.extract-obb-to-paddle-rec-fp32-nchw.onnx"), ownsModel: true);

        initializedImageShape = [.. imageShape];
        initializedRegionCount = regionCount;

        if(executionBackend is InferenceBackend.Cuda or InferenceBackend.TensorRt)
        {
            outputCudaMemoryInfo = new OrtMemoryInfo(
                OrtMemoryInfo.allocatorCUDA,
                OrtAllocatorType.DeviceAllocator,
                0,
                OrtMemType.Default);
            outputCudaAllocator = new OrtAllocator(prepareContext.Model.Session, outputCudaMemoryInfo);

            byte[] matrixUploadModelBytes = IdentityBuilder.Build(OnnxDataType.Float, regionCount, 9);
            matrixUploadContext = new OnnxExecutionContext(new OnnxModel(
                matrixUploadModelBytes,
                executionBackend,
                ConfigureRuntimeOperatorProvider,
                "extract-obb-to-paddle-rec-matrix-upload.onnx"), ownsModel: true);

            matrixCudaMemoryInfo = new OrtMemoryInfo(
                OrtMemoryInfo.allocatorCUDA,
                OrtAllocatorType.DeviceAllocator,
                0,
                OrtMemType.Default);
            matrixCudaAllocator = new OrtAllocator(matrixUploadContext.Model.Session, matrixCudaMemoryInfo);
        }
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

    OrtValue CreateMatrixTensor(
        ReadOnlySpan<YoloObb> boxes,
        int actualRegionCount,
        int outputRegionCount,
        int sourceWidth,
        int sourceHeight)
    {
        OrtValue matrixTensor = OrtValue.CreateAllocatedTensorValue(
            OrtAllocator.DefaultInstance,
            TensorElementType.Float,
            [outputRegionCount, 9]);

        Span<float> matrixData = matrixTensor.GetTensorMutableDataAsSpan<float>();
        Span<Point2f> sourcePoints = stackalloc Point2f[4];
        YoloObb dummyBox = CreateDummyBox(sourceWidth, sourceHeight);

        for(int boxIndex = 0; boxIndex < outputRegionCount; boxIndex++)
        {
            YoloObb box = boxIndex < actualRegionCount
                ? boxes[boxIndex]
                : actualRegionCount > 0
                    ? boxes[actualRegionCount - 1]
                    : dummyBox;

            WritePaddedObbPoints(box, sourcePoints);
            PerspectiveHomography.WriteTargetToSourceMatrix(
                sourcePoints,
                targetSize.Width,
                targetSize.Height,
                matrixData.Slice(boxIndex * 9, 9));
        }

        return matrixTensor;
    }

    static YoloObb CreateDummyBox(int sourceWidth, int sourceHeight) =>
        new()
        {
            X = Math.Max(0, sourceWidth - 1) * 0.5f,
            Y = Math.Max(0, sourceHeight - 1) * 0.5f,
            W = 2,
            H = 2,
            Angle = 0,
            Score = 0,
            Class = -1
        };

    void WritePaddedObbPoints(in YoloObb box, Span<Point2f> points)
    {
        float width = Math.Max(2f, box.W);
        float height = Math.Max(2f, box.H);
        float padX = paddingPixels + width * paddingScale;
        float padY = paddingPixels + height * paddingScale;

        var region = new OcrObbRegion(
            box.X,
            box.Y,
            width + padX * 2f,
            height + padY * 2f,
            box.Angle);

        region.GetPoints(points);
    }

    OrtValue UploadMatricesToDevice(OrtValue matrixHost)
    {
        ArgumentNullException.ThrowIfNull(matrixUploadContext);
        ArgumentNullException.ThrowIfNull(matrixCudaAllocator);

        OrtValue matrixDevice = OrtValue.CreateAllocatedTensorValue(
            matrixCudaAllocator,
            TensorElementType.Float,
            [initializedRegionCount, 9]);

        matrixUploadContext.IoBinding.ClearBoundInputs();
        matrixUploadContext.IoBinding.ClearBoundOutputs();

        try
        {
            matrixUploadContext.IoBinding.BindInput(IdentityBuilder.InputName, matrixHost);
            matrixUploadContext.IoBinding.BindOutput(IdentityBuilder.OutputName, matrixDevice);
            matrixUploadContext.Model.Session.RunWithBinding(matrixUploadContext.RunOptions, matrixUploadContext.IoBinding);
            matrixUploadContext.IoBinding.SynchronizeBoundOutputs();
        }
        finally
        {
            matrixUploadContext.IoBinding.ClearBoundInputs();
            matrixUploadContext.IoBinding.ClearBoundOutputs();
        }

        return matrixDevice;
    }

    OrtValue CreateOutputTensor(int regionCount)
    {
        long[] outputShape = [regionCount, 3, targetSize.Height, targetSize.Width];

        if(!isFinal && executionBackend is InferenceBackend.Cuda or InferenceBackend.TensorRt)
            return OrtValue.CreateAllocatedTensorValue(outputCudaAllocator!, TensorElementType.Float, outputShape);

        return OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, TensorElementType.Float, outputShape);
    }

    void RunWithFreshBinding(OrtValue image, OrtValue matrices, OrtValue output)
    {
        ArgumentNullException.ThrowIfNull(prepareContext);

        prepareContext.IoBinding.ClearBoundInputs();
        prepareContext.IoBinding.ClearBoundOutputs();

        try
        {
            prepareContext.IoBinding.BindInput(PaddleRecRoiPrepareBuilder.InputName, image);
            prepareContext.IoBinding.BindInput(PaddleRecRoiPrepareBuilder.MatrixInputName, matrices);
            prepareContext.IoBinding.BindOutput(PaddleRecRoiPrepareBuilder.OutputName, output);
            prepareContext.Model.Session.RunWithBinding(prepareContext.RunOptions, prepareContext.IoBinding);
            prepareContext.IoBinding.SynchronizeBoundOutputs();
        }
        finally
        {
            prepareContext.IoBinding.ClearBoundInputs();
            prepareContext.IoBinding.ClearBoundOutputs();
        }
    }

    static OpDescriptor CreateDescriptor(string imageInputKey, string obbInputKey, string outputKey, string? actualCountOutputKey)
    {
        VarRequirement[] writes = actualCountOutputKey is null
            ? [VarRequirement.Write<OrtValue>(outputKey)]
            : [VarRequirement.Write<OrtValue>(outputKey), VarRequirement.Write<int>(actualCountOutputKey)];

        return OpDescriptor.Create(
            "Op_Onnx_ExtractObbToPaddleRec_FP32_NCHW",
            "op.onnx.extractObbToPaddleRec.fp32.nchw",
            [VarRequirement.Read<OrtValue>(imageInputKey), VarRequirement.Read<YoloObb[]>(obbInputKey)],
            writes);
    }

    void DisposeRuntimeState()
    {
        matrixCudaAllocator?.Dispose();
        matrixCudaAllocator = null;

        matrixCudaMemoryInfo?.Dispose();
        matrixCudaMemoryInfo = null;

        matrixUploadContext?.Dispose();
        matrixUploadContext = null;

        outputCudaAllocator?.Dispose();
        outputCudaAllocator = null;

        outputCudaMemoryInfo?.Dispose();
        outputCudaMemoryInfo = null;

        prepareContext?.Dispose();
        prepareContext = null;
    }

    public void Dispose()
    {
        DisposeRuntimeState();
        initializedImageShape = null;
        initializedRegionCount = 0;
    }
}
