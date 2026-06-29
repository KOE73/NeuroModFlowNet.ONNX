using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.ONNX.Graph.Builders;
using NeuroModFlowNet.Pipeline;
using CvSize = OpenCvSharp.Size;

namespace NeuroModFlowNet.Pipeline.ONNX;

public sealed class Op_Onnx_Resize_U8_NHWC : Op_Onnx_TensorTransformBase
{
    readonly CvSize targetSize;
    readonly string? outputTransformKey;

    public Op_Onnx_Resize_U8_NHWC(
        string inputKey,
        string outputKey,
        CvSize targetSize,
        string? outputTransformKey = null,
        bool isFinal = false,
        InferenceBackend? executionBackend = null)
        : base(CreateDescriptor(inputKey, outputKey, outputTransformKey), inputKey, outputKey, isFinal, executionBackend)
    {
        if(targetSize.Width <= 0 || targetSize.Height <= 0)
            throw new ArgumentOutOfRangeException(nameof(targetSize), "Width and height must be positive.");

        this.targetSize = targetSize;
        this.outputTransformKey = outputTransformKey;
    }

    protected override string GraphInputName => ResizeBuilder.InputName;

    protected override string GraphOutputName => ResizeBuilder.OutputName;

    protected override string DisplayName => "resize-u8-nhwc.onnx";

    protected override string? ValidateInput(long[] inputShape, TensorElementType inputElementType)
    {
        if(inputElementType != TensorElementType.UInt8)
            return $"Input tensor must be UInt8 NHWC image data, actual element type: {inputElementType}.";

        return inputShape.Length == 4 ? null : $"Input tensor must be 4D NHWC, actual rank: {inputShape.Length}.";
    }

    protected override byte[] BuildModel(long[] inputShape, TensorElementType inputElementType)
    {
        int sourceHeight = checked((int)inputShape[1]);
        int sourceWidth = checked((int)inputShape[2]);
        int channels = checked((int)inputShape[3]);
        return ResizeBuilder.Build(sourceWidth, sourceHeight, targetSize.Width, targetSize.Height, channels);
    }

    protected override long[] CreateOutputShape(long[] inputShape, TensorElementType inputElementType) =>
        [1, targetSize.Height, targetSize.Width, inputShape[3]];

    protected override void WriteAdditionalOutputs(VmRunContext context, long[] inputShape, long[] outputShape)
    {
        if(outputTransformKey is null)
            return;

        int sourceHeight = checked((int)inputShape[1]);
        int sourceWidth = checked((int)inputShape[2]);
        context.Set(outputTransformKey, new ResizeCoordinateBackTransform(sourceWidth, sourceHeight, targetSize.Width, targetSize.Height));
    }

    static OpDescriptor CreateDescriptor(string inputKey, string outputKey, string? outputTransformKey)
    {
        VarRequirement[] writes = outputTransformKey is null
            ? [VarRequirement.Write<OrtValue>(outputKey)]
            : [VarRequirement.Write<OrtValue>(outputKey), VarRequirement.Write<ICoordinateBackTransform>(outputTransformKey)];

        return OpDescriptor.Create("Op_Onnx_Resize_U8_NHWC", "op.onnx.resize.u8.nhwc", [VarRequirement.Read<OrtValue>(inputKey)], writes);
    }
}
