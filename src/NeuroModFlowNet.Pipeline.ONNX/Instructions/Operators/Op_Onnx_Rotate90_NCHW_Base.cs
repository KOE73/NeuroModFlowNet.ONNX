using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.ONNX.Graph.Builders;
using NeuroModFlowNet.Pipeline;
using Onnx;

namespace NeuroModFlowNet.Pipeline.ONNX;

public abstract class Op_Onnx_Rotate90_NCHW_Base : Op_Onnx_TensorTransformBase
{
    readonly Rotate90Mode mode;
    readonly string? outputTransformKey;
    readonly TensorElementType expectedElementType;
    readonly TensorProto.Types.DataType graphElementType;

    protected Op_Onnx_Rotate90_NCHW_Base(
        OpDescriptor descriptor,
        string inputKey,
        string outputKey,
        Rotate90Mode mode,
        TensorElementType expectedElementType,
        TensorProto.Types.DataType graphElementType,
        string? outputTransformKey,
        bool isFinal,
        InferenceBackend? executionBackend)
        : base(descriptor, inputKey, outputKey, isFinal, executionBackend)
    {
        this.mode = mode;
        this.expectedElementType = expectedElementType;
        this.graphElementType = graphElementType;
        this.outputTransformKey = outputTransformKey;
    }

    protected override string GraphInputName => Rotate90NchwBuilder.InputName;

    protected override string GraphOutputName => Rotate90NchwBuilder.OutputName;

    protected override string GetCacheSemanticKey(long[] inputShape, TensorElementType inputElementType) =>
        $"mode={mode};graphType={graphElementType}";

    protected override string? ValidateInput(long[] inputShape, TensorElementType inputElementType)
    {
        if(inputElementType != expectedElementType)
            return $"Input tensor must be {expectedElementType} NCHW image data, actual element type: {inputElementType}.";

        if(inputShape.Length != 4)
            return $"Input tensor must be 4D NCHW, actual rank: {inputShape.Length}.";

        if(inputShape[0] != 1)
            return $"Input tensor batch must be 1, actual batch: {inputShape[0]}.";

        if(mode is not (Rotate90Mode.Clockwise90 or Rotate90Mode.Rotate180 or Rotate90Mode.CounterClockwise90))
            return $"Unsupported rotate mode: {mode}.";

        return null;
    }

    protected override byte[] BuildModel(long[] inputShape, TensorElementType inputElementType)
    {
        int channels = checked((int)inputShape[1]);
        int sourceHeight = checked((int)inputShape[2]);
        int sourceWidth = checked((int)inputShape[3]);
        return Rotate90NchwBuilder.Build(sourceWidth, sourceHeight, channels, GetClockwiseQuarterTurns(mode), graphElementType);
    }

    protected override long[] CreateOutputShape(long[] inputShape, TensorElementType inputElementType)
    {
        if(mode == Rotate90Mode.Rotate180)
            return [1, inputShape[1], inputShape[2], inputShape[3]];

        return [1, inputShape[1], inputShape[3], inputShape[2]];
    }

    protected override void WriteAdditionalOutputs(VmRunContext context, long[] inputShape, long[] outputShape)
    {
        if(outputTransformKey is null)
            return;

        int sourceHeight = checked((int)inputShape[2]);
        int sourceWidth = checked((int)inputShape[3]);
        context.Set(outputTransformKey, new Rotate90CoordinateBackTransform(sourceWidth, sourceHeight, mode));
    }

    static int GetClockwiseQuarterTurns(Rotate90Mode mode) =>
        mode switch
        {
            Rotate90Mode.Clockwise90 => 1,
            Rotate90Mode.Rotate180 => 2,
            Rotate90Mode.CounterClockwise90 => 3,
            _ => throw new ArgumentOutOfRangeException(nameof(mode), mode, "Unsupported rotate mode.")
        };

    protected static OpDescriptor CreateDescriptor(
        string descriptorName,
        string opcode,
        string inputKey,
        string outputKey,
        string? outputTransformKey)
    {
        VarRequirement[] writes = outputTransformKey is null
            ? [VarRequirement.Write<OrtValue>(outputKey)]
            : [VarRequirement.Write<OrtValue>(outputKey), VarRequirement.Write<ICoordinateBackTransform>(outputTransformKey)];

        return OpDescriptor.Create(
            descriptorName,
            opcode,
            reads: [VarRequirement.Read<OrtValue>(inputKey)],
            writes: writes);
    }
}
