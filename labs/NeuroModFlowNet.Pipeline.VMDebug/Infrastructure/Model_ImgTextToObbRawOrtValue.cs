using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;

namespace NeuroModFlowNet.Pipeline.VMDebug;

/// <summary>
/// EN: Lab-only command that runs the img-text-to-obb ONNX model directly from an existing OrtValue.
/// RU: Лабораторная команда для запуска ONNX-модели img-text-to-obb напрямую из существующего OrtValue.
/// </summary>
/// <remarks>
/// EN:
/// Essence: Bypasses standard CV conversion pipelines by running inference directly on an already allocated ONNX Runtime OrtValue.
/// Reasons: The camera lab is testing device-placement boundaries. Tensors are wrapped and placed using Copy_OrtTensor_To_ModelDevice,
/// and subsequent models must consume the exact same OrtValue without round-tripping through OpenCV/CPU memory.
/// 
/// RU:
/// Суть: Обход стандартных конвейеров конвертации изображений и запуск инференса напрямую на уже размещенном в памяти ONNX Runtime OrtValue.
/// Причины: Используется в экспериментах для тестирования переноса памяти между устройствами. Тензоры оборачиваются и размещаются с помощью
/// Copy_OrtTensor_To_ModelDevice, и последующие модели должны потреблять тот же самый OrtValue без накладных расходов на обратную конвертацию в OpenCV/CPU.
/// </remarks>
internal sealed class Model_ImgTextToObbRawOrtValue : OpBase, IDisposable
{
    const float ScoreThreshold = 0.3f;
    const int DefaultBatchCount = 1;
    const int DefaultItemCount = 300;
    const int ObbFieldCount = 7;
    const int ScoreFieldIndex = 4;

    readonly string inputKey;
    readonly string outputCountKey;
    readonly string modelPath;
    OnnxRuntimeContext? onnxContext;

    /// <summary>
    /// EN: Initializes a new instance of the <see cref="Model_ImgTextToObbRawOrtValue"/> class with specified keys and model path.
    /// RU: Инициализирует новый экземпляр класса <see cref="Model_ImgTextToObbRawOrtValue"/> с указанными ключами и путем к модели.
    /// </summary>
    /// <param name="inputKey">
    /// EN: The key of the input <see cref="OrtValue"/> in the context.
    /// RU: Ключ входного объекта <see cref="OrtValue"/> в контексте выполнения.
    /// </param>
    /// <param name="outputCountKey">
    /// EN: The key under which the detected box count (<see cref="int"/>) will be written in the context.
    /// RU: Ключ, под которым количество обнаруженных боксов (<see cref="int"/>) будет записано в контекст.
    /// </param>
    /// <param name="modelPath">
    /// EN: Path to the ONNX model file.
    /// RU: Путь к файлу ONNX-модели.
    /// </param>
    /// <remarks>
    /// EN:
    /// Essence of requirements:
    /// - <c>reads: [VarRequirement.Read&lt;OrtValue&gt;(inputKey)]</c>: Declares that the instruction requires an existing, fully populated <see cref="OrtValue"/> tensor at <paramref name="inputKey"/>, avoiding CPU-GPU copy or OpenCV transformation.
    /// - <c>writes: [VarRequirement.Write&lt;int&gt;(outputCountKey)]</c>: Declares that the instruction writes a primitive <see cref="int"/> representing the count of detected OBB boxes to <paramref name="outputCountKey"/>.
    /// 
    /// RU:
    /// Суть требований к переменным:
    /// - <c>reads: [VarRequirement.Read&lt;OrtValue&gt;(inputKey)]</c>: Декларирует необходимость наличия в контексте готового тензора <see cref="OrtValue"/> под ключом <paramref name="inputKey"/>, исключая повторное копирование CPU-GPU или конвертацию OpenCV.
    /// - <c>writes: [VarRequirement.Write&lt;int&gt;(outputCountKey)]</c>: Декларирует запись примитивного значения <see cref="int"/> (количество распознанных боксов) под ключом <paramref name="outputCountKey"/>.
    /// </remarks>
    public Model_ImgTextToObbRawOrtValue(string inputKey, string outputCountKey, string modelPath)
        : base(OpDescriptor.Create(
            "Model_ImgTextToObbRawOrtValue",
            "model.imgTextToObb.rawOrtValue",
            reads: [VarRequirement.Read<OrtValue>(inputKey)],
            writes: [VarRequirement.Write<int>(outputCountKey)]))
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(inputKey);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputCountKey);
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);

        this.inputKey = inputKey;
        this.outputCountKey = outputCountKey;
        this.modelPath = modelPath;
    }

    public override ValueTask<OpResult> ExecuteAsync(
        VmRunContext context,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(context);
        cancellationToken.ThrowIfCancellationRequested();

        if(!context.TryGet(inputKey, out OrtValue inputOrtValue))
            return ValueTask.FromResult(OpResult.Fail($"Input key '{inputKey}' was not found in the pipeline context."));

        if(onnxContext is null)
            InitializeContext();

        using OrtValue outputOrtValue = CreateOutputOrtTensor();

        RunWithFreshBinding(inputOrtValue, outputOrtValue);

        int boxCount = CountDetectedBoxes(outputOrtValue);
        context.Set(outputCountKey, boxCount);

        return ValueTask.FromResult(OpResult.Continue);
    }

    void InitializeContext()
    {
        // This lab intentionally uses the FP16 demo model through TensorRT. A CPU/CUDA fallback would make timings look
        // valid while the instruction is no longer testing the provider path we are studying.
        onnxContext = new OnnxRuntimeContext(modelPath, InferenceBackend.TensorRt);
    }

    OrtValue CreateOutputOrtTensor()
    {
        ArgumentNullException.ThrowIfNull(onnxContext);

        string outputName = onnxContext.Model.PrimaryOutputName;
        long[] outputShape = ResolveOutputShape(onnxContext.Model.ModelOutputShapes[outputName]);
        TensorElementType outputElementType = onnxContext.Model.GetOutputElementType(outputName);

        if(outputElementType is not (TensorElementType.Float or TensorElementType.Float16))
            throw new NotSupportedException($"img-text-to-obb output element type is not supported by the lab counter: {outputElementType}.");

        return OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, outputElementType, outputShape);
    }

    static long[] ResolveOutputShape(long[] modelOutputShape)
    {
        if(modelOutputShape.Length != 3)
            throw new InvalidOperationException($"img-text-to-obb output must be rank 3, actual rank: {modelOutputShape.Length}.");

        long batchCount = modelOutputShape[0] > 0 ? modelOutputShape[0] : DefaultBatchCount;
        long itemCount = modelOutputShape[1] > 0 ? modelOutputShape[1] : DefaultItemCount;
        long fieldCount = modelOutputShape[2] > 0 ? modelOutputShape[2] : ObbFieldCount;

        if(fieldCount != ObbFieldCount)
            throw new InvalidOperationException($"img-text-to-obb output must contain {ObbFieldCount} OBB fields, actual: {fieldCount}.");

        return [batchCount, itemCount, fieldCount];
    }

    void RunWithFreshBinding(OrtValue inputOrtValue, OrtValue outputOrtValue)
    {
        ArgumentNullException.ThrowIfNull(onnxContext);

        // The session is cached across frames, but IoBinding is native mutable state. Clearing before and after the run
        // keeps the binding from retaining OrtValue handles that belong to a disposed warmup or previous frame context.
        onnxContext.IoBinding.ClearBoundInputs();
        onnxContext.IoBinding.ClearBoundOutputs();

        try
        {
            onnxContext.IoBinding.BindInput(onnxContext.Model.PrimaryInputName, inputOrtValue);
            onnxContext.IoBinding.BindOutput(onnxContext.Model.PrimaryOutputName, outputOrtValue);
            onnxContext.Model.Session.RunWithBinding(onnxContext.RunOptions, onnxContext.IoBinding);
            onnxContext.IoBinding.SynchronizeBoundOutputs();
        }
        finally
        {
            onnxContext.IoBinding.ClearBoundInputs();
            onnxContext.IoBinding.ClearBoundOutputs();
        }
    }

    static int CountDetectedBoxes(OrtValue outputOrtValue)
    {
        TensorElementType outputElementType = outputOrtValue.GetTensorTypeAndShape().ElementDataType;
        return outputElementType switch
        {
            TensorElementType.Float => CountDetectedBoxes(outputOrtValue.GetTensorDataAsSpan<float>()),
            TensorElementType.Float16 => CountDetectedBoxes(outputOrtValue.GetTensorDataAsSpan<Half>()),
            _ => throw new NotSupportedException($"img-text-to-obb output element type is not supported by the lab counter: {outputElementType}.")
        };
    }

    static int CountDetectedBoxes(ReadOnlySpan<float> outputData)
    {
        int itemCount = outputData.Length / ObbFieldCount;
        int detectedCount = 0;

        // YOLO OBB NMS output is [X, Y, W, H, Score, ClassId, Angle]. The lab needs only a cheap sanity signal that the
        // model actually produced detections, so counting by Score avoids allocating domain objects or invoking mappers.
        for(int itemIndex = 0; itemIndex < itemCount; itemIndex++)
        {
            float score = outputData[(itemIndex * ObbFieldCount) + ScoreFieldIndex];
            if(score >= ScoreThreshold)
                detectedCount++;
        }

        return detectedCount;
    }

    static int CountDetectedBoxes(ReadOnlySpan<Half> outputData)
    {
        int itemCount = outputData.Length / ObbFieldCount;
        int detectedCount = 0;

        // TensorRT FP16 output keeps the same YOLO OBB NMS layout as FP32. Only the scalar representation changes, so
        // the lab casts just the score field to float and keeps the rest of the output untouched.
        for(int itemIndex = 0; itemIndex < itemCount; itemIndex++)
        {
            float score = (float)outputData[(itemIndex * ObbFieldCount) + ScoreFieldIndex];
            if(score >= ScoreThreshold)
                detectedCount++;
        }

        return detectedCount;
    }

    public void Dispose()
    {
        onnxContext?.Dispose();
    }
}
