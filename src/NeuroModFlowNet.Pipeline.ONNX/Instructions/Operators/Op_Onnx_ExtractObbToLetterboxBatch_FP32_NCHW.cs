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
/// EN: CISC hot-path op: cuts N regions out of one <c>RGB FP32 NCHW 0..1</c> image on the model device and letterboxes
/// each into a fixed <c>[N, 3, targetHeight, targetWidth]</c> batch that keeps the 0..1 range, so the batch can be fed
/// directly to a YOLO-style model (for example text OBB detection on bag crops). Regions come as <see cref="YoloObb"/>
/// (axis-aligned bounds of the padded OBB are used, aspect ratio is preserved, the letterbox margin samples outside the
/// region and is therefore black). For every batch slot a target-to-source <see cref="ICoordinateBackTransform"/> is
/// written to <c>outputTransformsKey</c> so model results can be mapped back with <c>Op_Map_Coordinates</c>.
///
/// RU: CISC-операция hot path: вырезает N регионов из одного изображения <c>RGB FP32 NCHW 0..1</c> на устройстве модели
/// и укладывает каждый letterbox-ом в фиксированный батч <c>[N, 3, targetHeight, targetWidth]</c> с сохранением
/// диапазона 0..1, чтобы батч можно было подать прямо в YOLO-подобную модель (например, детекция текста на вырезках
/// мешков). Регионы приходят как <see cref="YoloObb"/> (берутся axis-aligned границы OBB с отступом, пропорции
/// сохраняются, поля letterbox выбираются вне региона и потому чёрные). Для каждого слота батча в
/// <c>outputTransformsKey</c> пишется target-to-source <see cref="ICoordinateBackTransform"/> для обратного маппинга
/// результатов через <c>Op_Map_Coordinates</c>.
/// </summary>
/// <remarks>
/// EN: Fixed capacity: <c>maxRoiCount</c> defines the batch shape; missing slots repeat the last real region (no extra
/// tensor preparation) and the real count is written to <c>actualCountOutputKey</c>. Extra regions follow
/// <see cref="PaddleRecRoiOverflowPolicy"/>. Kernels are cached globally by image shape, target size and capacity.
///
/// RU: Фиксированная ёмкость: <c>maxRoiCount</c> задаёт форму батча; недостающие слоты повторяют последний реальный
/// регион (без отдельной подготовки тензора), реальное число пишется в <c>actualCountOutputKey</c>. Лишние регионы
/// обрабатываются по <see cref="PaddleRecRoiOverflowPolicy"/>. Kernels кэшируются глобально по форме изображения,
/// целевому размеру и ёмкости.
/// </remarks>
public sealed class Op_Onnx_ExtractObbToLetterboxBatch_FP32_NCHW : OpBase, IDisposable, IHasExecutionDevice
{
    readonly string imageInputKey;
    readonly string obbInputKey;
    readonly string outputKey;
    readonly string? outputTransformsKey;
    readonly string? actualCountOutputKey;
    readonly CvSize targetSize;
    readonly float paddingPixels;
    readonly float paddingScale;
    readonly int maxRoiCount;
    readonly PaddleRecRoiOverflowPolicy overflowPolicy;
    readonly RoiBatchFitMode fitMode;
    readonly bool isFinal;
    readonly InferenceBackend executionBackend;
    readonly SemaphoreSlim executionLock = new(1, 1);

    OnnxExecutionContext? prepareContext;
    OnnxExecutionContext? matrixUploadContext;
    long[]? initializedImageShape;
    OrtMemoryInfo? outputCudaMemoryInfo;
    OrtAllocator? outputCudaAllocator;
    OrtMemoryInfo? matrixCudaMemoryInfo;
    OrtAllocator? matrixCudaAllocator;
    bool disposed;

    public Op_Onnx_ExtractObbToLetterboxBatch_FP32_NCHW(
        string imageInputKey,
        string obbInputKey,
        string outputKey,
        CvSize targetSize,
        int maxRoiCount,
        string? outputTransformsKey = null,
        string? actualCountOutputKey = null,
        float paddingPixels = 0f,
        float paddingScale = 0f,
        PaddleRecRoiOverflowPolicy overflowPolicy = PaddleRecRoiOverflowPolicy.Fail,
        RoiBatchFitMode fitMode = RoiBatchFitMode.Letterbox,
        bool isFinal = false,
        InferenceBackend? executionBackend = null)
        : base(CreateDescriptor(imageInputKey, obbInputKey, outputKey, outputTransformsKey, actualCountOutputKey))
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(imageInputKey);
        ArgumentException.ThrowIfNullOrWhiteSpace(obbInputKey);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputKey);

        if(targetSize.Width <= 0 || targetSize.Height <= 0)
            throw new ArgumentOutOfRangeException(nameof(targetSize), "Target width and height must be positive.");

        if(maxRoiCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxRoiCount), "Max ROI count must be positive.");

        if(paddingPixels < 0)
            throw new ArgumentOutOfRangeException(nameof(paddingPixels), "Padding pixels must be non-negative.");

        if(paddingScale < 0)
            throw new ArgumentOutOfRangeException(nameof(paddingScale), "Padding scale must be non-negative.");

        if(outputTransformsKey is not null)
            ArgumentException.ThrowIfNullOrWhiteSpace(outputTransformsKey);

        if(actualCountOutputKey is not null)
            ArgumentException.ThrowIfNullOrWhiteSpace(actualCountOutputKey);

        this.imageInputKey = imageInputKey;
        this.obbInputKey = obbInputKey;
        this.outputKey = outputKey;
        this.outputTransformsKey = outputTransformsKey;
        this.actualCountOutputKey = actualCountOutputKey;
        this.targetSize = targetSize;
        this.maxRoiCount = maxRoiCount;
        this.paddingPixels = paddingPixels;
        this.paddingScale = paddingScale;
        this.overflowPolicy = overflowPolicy;
        this.fitMode = fitMode;
        this.isFinal = isFinal;
        this.executionBackend = executionBackend ?? InferenceBackend.Cuda;
    }

    bool IHasExecutionDevice.IsGpuExecution =>
        prepareContext?.Model.InferenceBackend is { } backend && backend != InferenceBackend.Cpu;

    string IHasExecutionDevice.ExecutionDeviceName =>
        prepareContext?.Model.InferenceBackend.ToString() ?? "Uninitialized";

    #region Execution

    public override async ValueTask<OpResult> ExecuteAsync(
        VmRunContext context,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(context);
        cancellationToken.ThrowIfCancellationRequested();
        ThrowIfDisposed();

        if(!context.TryGet(imageInputKey, out OrtValue image))
            return OpResult.Fail($"Input key '{imageInputKey}' was not found in the pipeline context.");

        if(!context.TryGet(obbInputKey, out YoloObb[] boxes))
            return OpResult.Fail($"Input key '{obbInputKey}' was not found in the pipeline context or is not a YoloObb array.");

        int actualRegionCount = boxes.Length;
        if(actualRegionCount > maxRoiCount)
        {
            if(overflowPolicy == PaddleRecRoiOverflowPolicy.Fail)
                return OpResult.Fail($"Actual ROI count {actualRegionCount} exceeds fixed max ROI count {maxRoiCount}.");

            actualRegionCount = maxRoiCount;
        }

        var imageInfo = image.GetTensorTypeAndShape();
        string? validationError = ValidateImage(imageInfo.Shape, imageInfo.ElementDataType);
        if(validationError is not null)
            return OpResult.Fail(validationError);

        await executionLock.WaitAsync(cancellationToken).ConfigureAwait(false);

        try
        {
            ThrowIfDisposed();
            EnsureRuntime(imageInfo.Shape);

            var transforms = new ICoordinateBackTransform[maxRoiCount];
            using OrtValue matrixHost = CreateMatrixTensor(
                boxes,
                actualRegionCount,
                checked((int)imageInfo.Shape[3]),
                checked((int)imageInfo.Shape[2]),
                transforms);
            using OrtValue? matrixDevice = executionBackend == InferenceBackend.Cpu
                ? null
                : UploadMatricesToDevice(matrixHost);

            OrtValue output = CreateOutputTensor();
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

            if(outputTransformsKey is not null)
                context.Set(outputTransformsKey, transforms);

            if(actualCountOutputKey is not null)
                context.Set(actualCountOutputKey, actualRegionCount);

            return OpResult.Continue;
        }
        finally
        {
            executionLock.Release();
        }
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

    #endregion

    #region Runtime kernels

    void EnsureRuntime(long[] imageShape)
    {
        if(prepareContext is not null && initializedImageShape is not null && initializedImageShape.SequenceEqual(imageShape))
            return;

        DisposeRuntimeState();

        int sourceWidth = checked((int)imageShape[3]);
        int sourceHeight = checked((int)imageShape[2]);
        int sourceChannels = checked((int)imageShape[1]);

        RuntimeOnnxOperatorKernelKey prepareKernelKey = new(
            nameof(Op_Onnx_ExtractObbToLetterboxBatch_FP32_NCHW),
            $"image={FormatShape(imageShape)};target={targetSize.Width}x{targetSize.Height};regions={maxRoiCount};normalization=none;fit={fitMode}",
            executionBackend,
            RuntimeOperatorProviderOptionsKey);

        prepareContext = RuntimeOnnxOperatorKernelCache.CreateContext(
            prepareKernelKey,
            () => PaddleRecRoiPrepareBuilder.BuildBatchedGridSampleFromMatricesFP32Nchw(
                sourceWidth,
                sourceHeight,
                sourceChannels,
                targetSize.Width,
                targetSize.Height,
                maxRoiCount,
                applyPaddleNormalization: false),
            ConfigureRuntimeOperatorProvider,
            "extract-obb-to-letterbox-batch-fp32-nchw.onnx");

        initializedImageShape = [.. imageShape];

        if(executionBackend is InferenceBackend.Cuda or InferenceBackend.TensorRt)
        {
            outputCudaMemoryInfo = new OrtMemoryInfo(
                OrtMemoryInfo.allocatorCUDA,
                OrtAllocatorType.DeviceAllocator,
                0,
                OrtMemType.Default);
            outputCudaAllocator = new OrtAllocator(prepareContext.Model.Session, outputCudaMemoryInfo);

            RuntimeOnnxOperatorKernelKey matrixUploadKernelKey = new(
                "ExtractObbToLetterboxBatch.MatrixUpload",
                $"shape={maxRoiCount}x9;type=Float",
                executionBackend,
                RuntimeOperatorProviderOptionsKey);

            matrixUploadContext = RuntimeOnnxOperatorKernelCache.CreateContext(
                matrixUploadKernelKey,
                () => IdentityBuilder.Build(OnnxDataType.Float, maxRoiCount, 9),
                ConfigureRuntimeOperatorProvider,
                "extract-obb-to-letterbox-batch-matrix-upload.onnx");

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
                trtConfig.EnableFp16 = false;
                trtConfig.EnableBf16 = false;
                trtConfig.BuilderOptimizationLevel = 2;
                break;
            case CudaConfig cudaConfig:
                cudaConfig.EnableCudaGraph = false;
                break;
        }
    }

    const string RuntimeOperatorProviderOptionsKey =
        "runtime-operator;trtEngineCache=keyed;trtFp16=false;trtBf16=false;trtBuilderOptimizationLevel=2;cudaGraph=false";

    static string FormatShape(long[] shape) => string.Join('x', shape);

    #endregion

    #region Matrices

    /// <summary>
    /// Target-to-source matrix per slot: the padded region's axis-aligned bounds are scaled uniformly into the target
    /// canvas and centered, so the matrix is a pure scale + translation (a homography with zero perspective terms).
    /// </summary>
    OrtValue CreateMatrixTensor(
        ReadOnlySpan<YoloObb> boxes,
        int actualRegionCount,
        int sourceWidth,
        int sourceHeight,
        ICoordinateBackTransform[] transforms)
    {
        OrtValue matrixTensor = OrtValue.CreateAllocatedTensorValue(
            OrtAllocator.DefaultInstance,
            TensorElementType.Float,
            [maxRoiCount, 9]);

        Span<float> matrixData = matrixTensor.GetTensorMutableDataAsSpan<float>();
        Span<Point2f> points = stackalloc Point2f[4];
        YoloObb dummyBox = CreateDummyBox(sourceWidth, sourceHeight);

        for(int slotIndex = 0; slotIndex < maxRoiCount; slotIndex++)
        {
            YoloObb box = slotIndex < actualRegionCount
                ? boxes[slotIndex]
                : actualRegionCount > 0
                    ? boxes[actualRegionCount - 1]
                    : dummyBox;

            WritePaddedObbPoints(box, points);
            (float left, float top, float regionWidth, float regionHeight) = GetBounds(points);

            // Letterbox keeps the aspect ratio and centers; Stretch maps the region onto the whole canvas (the plain
            // Resize the CGP bag detector was trained with).
            float scaleX;
            float scaleY;
            float padLeft = 0f;
            float padTop = 0f;
            if(fitMode == RoiBatchFitMode.Stretch)
            {
                scaleX = targetSize.Width / regionWidth;
                scaleY = targetSize.Height / regionHeight;
            }
            else
            {
                scaleX = scaleY = Math.Min(targetSize.Width / regionWidth, targetSize.Height / regionHeight);
                padLeft = (targetSize.Width - (regionWidth * scaleX)) * 0.5f;
                padTop = (targetSize.Height - (regionHeight * scaleY)) * 0.5f;
            }

            float inverseScaleX = 1f / scaleX;
            float inverseScaleY = 1f / scaleY;

            Span<float> matrix = matrixData.Slice(slotIndex * 9, 9);
            matrix[0] = inverseScaleX;
            matrix[1] = 0f;
            matrix[2] = left - (padLeft * inverseScaleX);
            matrix[3] = 0f;
            matrix[4] = inverseScaleY;
            matrix[5] = top - (padTop * inverseScaleY);
            matrix[6] = 0f;
            matrix[7] = 0f;
            matrix[8] = 1f;

            transforms[slotIndex] = new PerspectiveCoordinateBackTransform(
                matrix[0], matrix[1], matrix[2],
                matrix[3], matrix[4], matrix[5],
                matrix[6], matrix[7], matrix[8]);
        }

        return matrixTensor;
    }

    static (float Left, float Top, float Width, float Height) GetBounds(ReadOnlySpan<Point2f> points)
    {
        float minX = float.MaxValue, minY = float.MaxValue, maxX = float.MinValue, maxY = float.MinValue;
        foreach(Point2f point in points)
        {
            minX = Math.Min(minX, point.X);
            minY = Math.Min(minY, point.Y);
            maxX = Math.Max(maxX, point.X);
            maxY = Math.Max(maxY, point.Y);
        }

        return (minX, minY, Math.Max(2f, maxX - minX), Math.Max(2f, maxY - minY));
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
            [maxRoiCount, 9]);

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

    #endregion

    #region Binding

    OrtValue CreateOutputTensor()
    {
        long[] outputShape = [maxRoiCount, 3, targetSize.Height, targetSize.Width];

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

    static OpDescriptor CreateDescriptor(
        string imageInputKey,
        string obbInputKey,
        string outputKey,
        string? outputTransformsKey,
        string? actualCountOutputKey)
    {
        var writes = new List<VarRequirement>(3) { VarRequirement.Write<OrtValue>(outputKey) };
        if(outputTransformsKey is not null)
            writes.Add(VarRequirement.Write<ICoordinateBackTransform[]>(outputTransformsKey));

        if(actualCountOutputKey is not null)
            writes.Add(VarRequirement.Write<int>(actualCountOutputKey));

        return OpDescriptor.Create(
            "Op_Onnx_ExtractObbToLetterboxBatch_FP32_NCHW",
            "op.onnx.extractObbToLetterboxBatch.fp32.nchw",
            [VarRequirement.Read<OrtValue>(imageInputKey), VarRequirement.Read<YoloObb[]>(obbInputKey)],
            writes);
    }

    #endregion

    #region Lifetime

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
        if(disposed)
            return;

        disposed = true;
        DisposeRuntimeState();
        executionLock.Dispose();
        initializedImageShape = null;
    }

    void ThrowIfDisposed()
    {
        if(disposed)
            throw new ObjectDisposedException(nameof(Op_Onnx_ExtractObbToLetterboxBatch_FP32_NCHW));
    }

    #endregion
}
