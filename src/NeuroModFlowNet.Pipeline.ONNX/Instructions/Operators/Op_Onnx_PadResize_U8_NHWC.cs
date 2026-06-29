using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.ONNX.Graph.Builders;
using NeuroModFlowNet.Pipeline;
using CvSize = OpenCvSharp.Size;

namespace NeuroModFlowNet.Pipeline.ONNX;

public sealed class Op_Onnx_PadResize_U8_NHWC : Op_Onnx_TensorTransformBase
{
    readonly CvSize targetSize;
    readonly int stride;
    readonly PadResizeMode mode;
    readonly byte padValue;
    readonly string? outputTransformKey;

    public Op_Onnx_PadResize_U8_NHWC(
        string inputKey,
        string outputKey,
        CvSize targetSize,
        string? outputTransformKey = null,
        int stride = 32,
        PadResizeMode mode = PadResizeMode.FixedCanvas,
        byte padValue = 114,
        bool isFinal = false,
        InferenceBackend? executionBackend = null)
        : base(CreateDescriptor(inputKey, outputKey, outputTransformKey), inputKey, outputKey, isFinal, executionBackend)
    {
        if(targetSize.Width <= 0 || targetSize.Height <= 0)
            throw new ArgumentOutOfRangeException(nameof(targetSize), "Width and height must be positive.");

        if(stride <= 0)
            throw new ArgumentOutOfRangeException(nameof(stride), "Stride must be positive.");

        this.targetSize = targetSize;
        this.outputTransformKey = outputTransformKey;
        this.stride = stride;
        this.mode = mode;
        this.padValue = padValue;
    }

    protected override string GraphInputName => PadResizeBuilder.InputName;

    protected override string GraphOutputName => PadResizeBuilder.OutputName;

    protected override string DisplayName => "pad-resize-u8-nhwc.onnx";

    protected override string? ValidateInput(long[] inputShape, TensorElementType inputElementType)
    {
        if(inputElementType != TensorElementType.UInt8)
            return $"Input tensor must be UInt8 NHWC image data, actual element type: {inputElementType}.";

        if(inputShape.Length != 4)
            return $"Input tensor must be 4D NHWC, actual rank: {inputShape.Length}.";

        if(mode is not (PadResizeMode.FixedCanvas or PadResizeMode.AutoStrideCanvas))
            return $"Unsupported pad resize mode: {mode}.";

        return null;
    }

    protected override byte[] BuildModel(long[] inputShape, TensorElementType inputElementType)
    {
        PadResizeLayout layout = CreateLayout(inputShape);
        int channels = checked((int)inputShape[3]);
        return PadResizeBuilder.Build(
            layout.SourceWidth,
            layout.SourceHeight,
            layout.ResizedWidth,
            layout.ResizedHeight,
            layout.OutputWidth,
            layout.OutputHeight,
            layout.PadLeft,
            layout.PadTop,
            layout.PadRight,
            layout.PadBottom,
            channels,
            padValue);
    }

    protected override long[] CreateOutputShape(long[] inputShape, TensorElementType inputElementType)
    {
        PadResizeLayout layout = CreateLayout(inputShape);
        return [1, layout.OutputHeight, layout.OutputWidth, inputShape[3]];
    }

    protected override void WriteAdditionalOutputs(VmRunContext context, long[] inputShape, long[] outputShape)
    {
        if(outputTransformKey is null)
            return;

        PadResizeLayout layout = CreateLayout(inputShape);
        context.Set(outputTransformKey, new PadResizeCoordinateBackTransform(
            layout.SourceWidth, layout.SourceHeight, layout.ResizedWidth, layout.ResizedHeight,
            layout.OutputWidth, layout.OutputHeight, layout.Scale, layout.PadLeft, layout.PadTop,
            layout.PadRight, layout.PadBottom, layout.Stride, layout.Mode));
    }

    PadResizeLayout CreateLayout(long[] inputShape)
    {
        int sourceHeight = checked((int)inputShape[1]);
        int sourceWidth = checked((int)inputShape[2]);
        return PadResizeLayout.Create(sourceWidth, sourceHeight, targetSize.Width, targetSize.Height, stride, mode);
    }

    static OpDescriptor CreateDescriptor(string inputKey, string outputKey, string? outputTransformKey)
    {
        VarRequirement[] writes = outputTransformKey is null
            ? [VarRequirement.Write<OrtValue>(outputKey)]
            : [VarRequirement.Write<OrtValue>(outputKey), VarRequirement.Write<ICoordinateBackTransform>(outputTransformKey)];

        return OpDescriptor.Create("Op_Onnx_PadResize_U8_NHWC", "op.onnx.padResize.u8.nhwc", [VarRequirement.Read<OrtValue>(inputKey)], writes);
    }
}
