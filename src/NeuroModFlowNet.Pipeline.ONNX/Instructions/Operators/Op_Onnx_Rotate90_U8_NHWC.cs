using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.ONNX.Graph.Builders;
using NeuroModFlowNet.Pipeline;

namespace NeuroModFlowNet.Pipeline.ONNX;

public sealed class Op_Onnx_Rotate90_U8_NHWC : Op_Onnx_TensorTransformBase
{
    readonly Rotate90Mode mode;
    readonly string? outputTransformKey;

    public Op_Onnx_Rotate90_U8_NHWC(
        string inputKey,
        string outputKey,
        Rotate90Mode mode,
        string? outputTransformKey = null,
        bool isFinal = false,
        InferenceBackend? executionBackend = null)
        : base(CreateDescriptor(inputKey, outputKey, outputTransformKey), inputKey, outputKey, isFinal, executionBackend)
    {
        this.mode = mode;
        this.outputTransformKey = outputTransformKey;
    }

    protected override string GraphInputName => Rotate90Builder.InputName;

    protected override string GraphOutputName => Rotate90Builder.OutputName;

    protected override string DisplayName => "rotate90-u8-nhwc.onnx";

    protected override string? ValidateInput(long[] inputShape, TensorElementType inputElementType)
    {
        if(inputElementType != TensorElementType.UInt8)
            return $"Input tensor must be UInt8 NHWC image data, actual element type: {inputElementType}.";

        if(inputShape.Length != 4)
            return $"Input tensor must be 4D NHWC, actual rank: {inputShape.Length}.";

        if(mode is not (Rotate90Mode.Clockwise90 or Rotate90Mode.Rotate180 or Rotate90Mode.CounterClockwise90))
            return $"Unsupported rotate mode: {mode}.";

        return null;
    }

    protected override byte[] BuildModel(long[] inputShape, TensorElementType inputElementType)
    {
        int sourceHeight = checked((int)inputShape[1]);
        int sourceWidth = checked((int)inputShape[2]);
        int channels = checked((int)inputShape[3]);
        return Rotate90Builder.Build(sourceWidth, sourceHeight, channels, GetClockwiseQuarterTurns(mode));
    }

    protected override long[] CreateOutputShape(long[] inputShape, TensorElementType inputElementType) =>
        mode == Rotate90Mode.Rotate180
            ? [1, inputShape[1], inputShape[2], inputShape[3]]
            : [1, inputShape[2], inputShape[1], inputShape[3]];

    protected override void WriteAdditionalOutputs(VmRunContext context, long[] inputShape, long[] outputShape)
    {
        if(outputTransformKey is null)
            return;

        int sourceHeight = checked((int)inputShape[1]);
        int sourceWidth = checked((int)inputShape[2]);
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

    static OpDescriptor CreateDescriptor(string inputKey, string outputKey, string? outputTransformKey)
    {
        VarRequirement[] writes = outputTransformKey is null
            ? [VarRequirement.Write<OrtValue>(outputKey)]
            : [VarRequirement.Write<OrtValue>(outputKey), VarRequirement.Write<ICoordinateBackTransform>(outputTransformKey)];

        return OpDescriptor.Create("Op_Onnx_Rotate90_U8_NHWC", "op.onnx.rotate90.u8.nhwc", [VarRequirement.Read<OrtValue>(inputKey)], writes);
    }
}
