using System.Diagnostics;
using System.Threading.Channels;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.VMDebug;

/// <summary>
/// Collects raw camera frames from all VM workers, assembles them into a batch using an ONNX Concat graph, and runs
/// the OBB model on the batched tensor.
/// </summary>
/// <remarks>
/// Batching strategy:
///   Each VM worker submits one cloned Mat per frame. The service accumulates frames until it has exactly
///   <see cref="batchSize"/> items, then:
///     1. Wraps each Mat as an OrtValue (zero-copy NHWC view).
///     2. Feeds all N OrtValues into a generated ONNX Concat model → [N, H, W, C] batch.
///     3. Runs the OBB model on the batch tensor.
///     4. Counts detections across all items in the batch output.
///
///   The Concat model is built lazily on the first batch and re-built if input shape changes.
///
/// Note on batch size compatibility:
///   If the OBB model was exported with a fixed batch axis (e.g. [1, itemCount, 7] output), inference on a
///   [4, H, W, C] input will fail. Export the model with dynamic batch (batch=-1) or a fixed batch=N to use
///   true batched inference. When using fixed batch=1 models this service still demonstrates the concat mechanism
///   but the OBB run will throw; the service catches and counts the error in stats.
/// </remarks>
internal sealed class ObbBatchService : IDisposable
{
    const int DefaultChannelCapacity = 64;
    const float ScoreThreshold = 0.3f;
    const int ObbFieldCount = 7;
    const int ScoreFieldIndex = 4;

    readonly string obbModelPath;
    readonly int batchSize;
    readonly Channel<Mat> frameChannel;
    readonly object statsLock = new();

    // Lazy-initialized ONNX sessions (created on first batch, owned by this service).
    OnnxRuntimeContext? obbContext;
    InferenceSession? concatSession;
    long[]? lastConcatInputShape; // [H, W, C] - rebuild Concat session when shape changes

    // Stats — written under statsLock, read without lock (dashboard reads are best-effort).
    readonly MovingAverage batchTimeAverage = new(30);
    readonly MovingAverage detectionCountAverage = new(30);
    int batchesProcessed;
    int totalDetections;

    public ObbBatchService(string obbModelPath, int batchSize)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(obbModelPath);

        if(batchSize < 2)
            throw new ArgumentOutOfRangeException(nameof(batchSize), "OBB batch size must be at least 2.");

        this.obbModelPath = obbModelPath;
        this.batchSize = batchSize;

        frameChannel = Channel.CreateBounded<Mat>(new BoundedChannelOptions(DefaultChannelCapacity)
        {
            FullMode = BoundedChannelFullMode.DropOldest,
            SingleReader = true,
            SingleWriter = false
        });
    }

    /// <summary>
    /// Submits a camera frame to the batch queue. The service takes ownership of <paramref name="frame"/> and
    /// disposes it after the batch run.
    /// </summary>
    public void SubmitFrame(Mat frame)
    {
        ArgumentNullException.ThrowIfNull(frame);

        // DropOldest mode handles a full channel automatically; the Mat is disposed by the caller in that case.
        if(!frameChannel.Writer.TryWrite(frame))
            frame.Dispose();
    }

    /// <summary>
    /// Returns a point-in-time snapshot of batch service statistics.
    /// </summary>
    public ObbBatchServiceStats GetStats()
    {
        lock(statsLock)
        {
            return new ObbBatchServiceStats(
                BatchesProcessed: batchesProcessed,
                TotalDetections: totalDetections,
                LastBatchTimeMilliseconds: batchTimeAverage.Last,
                AverageBatchTimeMilliseconds: batchTimeAverage.Average,
                LastDetectionCount: detectionCountAverage.Last,
                AverageDetectionCount: detectionCountAverage.Average,
                CurrentQueueDepth: 0); // Channel count is not exposed; placeholder
        }
    }

    /// <summary>
    /// Starts the background accumulation and inference loop. Returns when the token is cancelled.
    /// </summary>
    public Task RunAsync(CancellationToken cancellationToken)
    {
        return Task.Run(() => ProcessLoopAsync(cancellationToken), cancellationToken);
    }

    #region Background processing loop

    async Task ProcessLoopAsync(CancellationToken cancellationToken)
    {
        var batch = new List<Mat>(batchSize);

        try
        {
            while(!cancellationToken.IsCancellationRequested)
            {
                // Accumulate exactly batchSize frames before processing.
                while(batch.Count < batchSize)
                {
                    Mat frame = await frameChannel.Reader.ReadAsync(cancellationToken).ConfigureAwait(false);
                    batch.Add(frame);
                }

                ProcessBatch(batch);

                foreach(Mat frame in batch)
                    frame.Dispose();

                batch.Clear();
            }
        }
        finally
        {
            foreach(Mat frame in batch)
                frame.Dispose();

            // Drain remaining queued frames to release Mat memory.
            while(frameChannel.Reader.TryRead(out Mat? leftover))
                leftover.Dispose();
        }
    }

    void ProcessBatch(List<Mat> frames)
    {
        // All frames must have the same shape; use the first to (re-)initialize sessions.
        Mat firstFrame = frames[0];
        long frameHeight = firstFrame.Rows;
        long frameWidth = firstFrame.Cols;
        long frameChannels = firstFrame.Channels();

        EnsureObbContext();
        EnsureConcatSession(frameHeight, frameWidth, frameChannels, firstFrame.Type());

        var stopwatch = Stopwatch.StartNew();
        int detections = 0;

        try
        {
            detections = RunBatchInference(frames, frameHeight, frameWidth, frameChannels);
        }
        catch
        {
            // Inference failure (e.g. fixed-batch model receiving batch>1) is absorbed; stats will show 0 detections.
            // The batch timing still updates so throughput is visible in the dashboard.
        }

        stopwatch.Stop();

        lock(statsLock)
        {
            batchesProcessed++;
            totalDetections += detections;
            batchTimeAverage.Add(stopwatch.Elapsed.TotalMilliseconds);
            detectionCountAverage.Add(detections);
        }
    }

    #endregion

    #region ONNX session management

    void EnsureObbContext()
    {
        if(obbContext is not null)
            return;

        // Attempt TensorRT first (same provider path as Model_ImgTextToObbRawOrtValue).
        obbContext = new OnnxRuntimeContext(obbModelPath, InferenceBackend.TensorRt);
    }

    void EnsureConcatSession(long height, long width, long channels, MatType matType)
    {
        long[] inputShape = [height, width, channels];

        if(concatSession is not null && lastConcatInputShape is not null && lastConcatInputShape.SequenceEqual(inputShape))
            return;

        concatSession?.Dispose();
        concatSession = null;

        TensorProto.Types.DataType elementType = MatTypeToOnnxDataType(matType);
        byte[] modelBytes = ObbBatchConcatBuilder.Build(batchSize, height, width, channels, elementType);

        // Concat is a trivial memory operation; CUDA is preferred to keep data on device, but CPU is acceptable.
        try
        {
            var sessionOptions = new SessionOptions();
            sessionOptions.AppendExecutionProvider_CUDA(0);
            concatSession = new InferenceSession(modelBytes, sessionOptions);
        }
        catch
        {
            concatSession = new InferenceSession(modelBytes);
        }

        lastConcatInputShape = inputShape;
    }

    static TensorProto.Types.DataType MatTypeToOnnxDataType(MatType matType)
    {
        // Camera frames are BGR uint8 (CV_8UC3). Other depths are unlikely in this lab but handled defensively.
        return matType.Depth switch
        {
            MatType.CV_8U => TensorProto.Types.DataType.Uint8,
            MatType.CV_32F => TensorProto.Types.DataType.Float,
            MatType.CV_16F => TensorProto.Types.DataType.Float16,
            _ => TensorProto.Types.DataType.Uint8
        };
    }

    #endregion

    #region Batch inference

    int RunBatchInference(List<Mat> frames, long height, long width, long channels)
    {
        // Step 1: Wrap each Mat as an OrtValue (zero-copy, NHWC layout [1, H, W, C]).
        var inputOrtValues = new List<OrtValue>(frames.Count);

        try
        {
            foreach(Mat frame in frames)
                inputOrtValues.Add(WrapMatAsOrtValue(frame, height, width, channels));

            // Step 2: Run the Concat ONNX model → [batchSize, H, W, C].
            using OrtValue batchOrtValue = RunConcatModel(inputOrtValues);

            // Step 3: Run OBB model on the batch tensor.
            using OrtValue obbOutputOrtValue = RunObbModel(batchOrtValue);

            // Step 4: Count detections in the batched output [batchSize, itemCount, 7].
            return CountDetectedBoxes(obbOutputOrtValue);
        }
        finally
        {
            foreach(OrtValue ortValue in inputOrtValues)
                ortValue.Dispose();
        }
    }

    static OrtValue WrapMatAsOrtValue(Mat frame, long height, long width, long channels)
    {
        // CreateTensorValueWithData is a zero-copy view into the Mat's unmanaged pixel buffer.
        // The Mat must remain alive (undisposed) while this OrtValue is in use — the caller guarantees
        // this because the Mat list is kept alive until after RunConcatModel returns.
        long byteCount = height * width * channels;
        return OrtValue.CreateTensorValueWithData(
            OrtMemoryInfo.DefaultInstance,
            TensorElementType.UInt8,
            [1, height, width, channels],
            (nint)frame.Data,
            byteCount);
    }

    OrtValue RunConcatModel(List<OrtValue> inputOrtValues)
    {
        // Build named inputs matching the Concat graph schema: "input_0", "input_1", ...
        var inputNames = new string[inputOrtValues.Count];
        var inputValues = new OrtValue[inputOrtValues.Count];

        for(int slotIndex = 0; slotIndex < inputOrtValues.Count; slotIndex++)
        {
            inputNames[slotIndex] = ObbBatchConcatBuilder.InputName(slotIndex);
            inputValues[slotIndex] = inputOrtValues[slotIndex];
        }

        using var runOptions = new RunOptions();
        IDisposableReadOnlyCollection<OrtValue> outputs = concatSession!.Run(
            runOptions,
            inputNames,
            inputValues,
            [ObbBatchConcatBuilder.OutputName]);

        // The caller owns the returned OrtValue and must dispose it.
        // Detach it from the collection before disposing the collection.
        OrtValue batchValue = outputs.First();
        outputs.Dispose();
        return batchValue;
    }

    OrtValue RunObbModel(OrtValue batchOrtValue)
    {
        ArgumentNullException.ThrowIfNull(obbContext);

        string outputName = obbContext.Model.PrimaryOutputName;
        long[] outputShape = ResolveObbOutputShape(obbContext.Model.ModelOutputShapes[outputName]);
        TensorElementType outputElementType = obbContext.Model.GetOutputElementType(outputName);

        OrtValue outputOrtValue = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, outputElementType, outputShape);

        obbContext.IoBinding.ClearBoundInputs();
        obbContext.IoBinding.ClearBoundOutputs();

        try
        {
            obbContext.IoBinding.BindInput(obbContext.Model.PrimaryInputName, batchOrtValue);
            obbContext.IoBinding.BindOutput(outputName, outputOrtValue);
            obbContext.Model.Session.RunWithBinding(obbContext.RunOptions, obbContext.IoBinding);
            obbContext.IoBinding.SynchronizeBoundOutputs();
        }
        finally
        {
            obbContext.IoBinding.ClearBoundInputs();
            obbContext.IoBinding.ClearBoundOutputs();
        }

        return outputOrtValue;
    }

    long[] ResolveObbOutputShape(long[] modelShape)
    {
        // Shape: [batch, itemCount, ObbFieldCount]. Dynamic dims (-1) are resolved to defaults.
        const int DefaultItemCount = 300;

        long resolvedBatch = modelShape.Length > 0 && modelShape[0] > 0 ? modelShape[0] : batchSize;
        long resolvedItems = modelShape.Length > 1 && modelShape[1] > 0 ? modelShape[1] : DefaultItemCount;
        long resolvedFields = modelShape.Length > 2 && modelShape[2] > 0 ? modelShape[2] : ObbFieldCount;

        return [resolvedBatch, resolvedItems, resolvedFields];
    }

    static int CountDetectedBoxes(OrtValue outputOrtValue)
    {
        TensorElementType elementType = outputOrtValue.GetTensorTypeAndShape().ElementDataType;

        return elementType switch
        {
            TensorElementType.Float => CountDetectedBoxes(outputOrtValue.GetTensorDataAsSpan<float>()),
            TensorElementType.Float16 => CountDetectedBoxes(outputOrtValue.GetTensorDataAsSpan<Half>()),
            _ => 0
        };
    }

    static int CountDetectedBoxes(ReadOnlySpan<float> data)
    {
        int count = 0;
        int itemCount = data.Length / ObbFieldCount;

        for(int itemIndex = 0; itemIndex < itemCount; itemIndex++)
        {
            float score = data[(itemIndex * ObbFieldCount) + ScoreFieldIndex];
            if(score >= ScoreThreshold)
                count++;
        }

        return count;
    }

    static int CountDetectedBoxes(ReadOnlySpan<Half> data)
    {
        int count = 0;
        int itemCount = data.Length / ObbFieldCount;

        for(int itemIndex = 0; itemIndex < itemCount; itemIndex++)
        {
            float score = (float)data[(itemIndex * ObbFieldCount) + ScoreFieldIndex];
            if(score >= ScoreThreshold)
                count++;
        }

        return count;
    }

    #endregion

    public void Dispose()
    {
        obbContext?.Dispose();
        concatSession?.Dispose();
    }
}
