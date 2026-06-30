using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.ONNX.Graph.Builders;
using NeuroModFlowNet.Pipeline;
using Onnx;
using CvSize = OpenCvSharp.Size;

namespace NeuroModFlowNet.Pipeline.ONNX;

public abstract class Op_Onnx_PadResize_NCHW_Base : Op_Onnx_TensorTransformBase
{
    readonly CvSize targetSize;
    readonly int stride;
    readonly PadResizeMode mode;
    readonly float padValue;
    readonly string? outputTransformKey;
    readonly TensorElementType expectedElementType;
    readonly TensorProto.Types.DataType graphElementType;

    protected Op_Onnx_PadResize_NCHW_Base(
        OpDescriptor descriptor,
        string inputKey,
        string outputKey,
        CvSize targetSize,
        TensorElementType expectedElementType,
        TensorProto.Types.DataType graphElementType,
        string? outputTransformKey,
        int stride,
        PadResizeMode mode,
        float padValue,
        bool isFinal,
        InferenceBackend? executionBackend)
        : base(descriptor, inputKey, outputKey, isFinal, executionBackend)
    {
        if(targetSize.Width <= 0 || targetSize.Height <= 0)
            throw new ArgumentOutOfRangeException(nameof(targetSize), "Width and height must be positive.");

        if(stride <= 0)
            throw new ArgumentOutOfRangeException(nameof(stride), "Stride must be positive.");

        this.targetSize = targetSize;
        this.expectedElementType = expectedElementType;
        this.graphElementType = graphElementType;
        this.outputTransformKey = outputTransformKey;
        this.stride = stride;
        this.mode = mode;
        this.padValue = padValue;
    }

    protected override string GraphInputName => PadResizeNchwBuilder.InputName;

    protected override string GraphOutputName => PadResizeNchwBuilder.OutputName;

    protected override string GetCacheSemanticKey(long[] inputShape, TensorElementType inputElementType) =>
        $"target={targetSize.Width}x{targetSize.Height};stride={stride};mode={mode};pad={padValue};graphType={graphElementType}";

    protected override string? ValidateInput(long[] inputShape, TensorElementType inputElementType)
    {
        if(inputElementType != expectedElementType)
            return $"Input tensor must be {expectedElementType} NCHW image data, actual element type: {inputElementType}.";

        if(inputShape.Length != 4)
            return $"Input tensor must be 4D NCHW, actual rank: {inputShape.Length}.";

        if(inputShape[0] != 1)
            return $"Input tensor batch must be 1, actual batch: {inputShape[0]}.";

        if(mode is not (PadResizeMode.FixedCanvas or PadResizeMode.AutoStrideCanvas))
            return $"Unsupported pad resize mode: {mode}.";

        return null;
    }

    protected override byte[] BuildModel(long[] inputShape, TensorElementType inputElementType)
    {
        PadResizeLayout layout = CreateLayout(inputShape);
        int channels = checked((int)inputShape[1]);

        return PadResizeNchwBuilder.Build(
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
            graphElementType,
            padValue);
    }

    protected override long[] CreateOutputShape(long[] inputShape, TensorElementType inputElementType)
    {
        PadResizeLayout layout = CreateLayout(inputShape);
        return [1, inputShape[1], layout.OutputHeight, layout.OutputWidth];
    }

    protected override void WriteAdditionalOutputs(VmRunContext context, long[] inputShape, long[] outputShape)
    {
        if(outputTransformKey is null)
            return;

        PadResizeLayout layout = CreateLayout(inputShape);
        context.Set(outputTransformKey, new PadResizeCoordinateBackTransform(
            layout.SourceWidth,
            layout.SourceHeight,
            layout.ResizedWidth,
            layout.ResizedHeight,
            layout.OutputWidth,
            layout.OutputHeight,
            layout.Scale,
            layout.PadLeft,
            layout.PadTop,
            layout.PadRight,
            layout.PadBottom,
            layout.Stride,
            layout.Mode));
    }

    PadResizeLayout CreateLayout(long[] inputShape)
    {
        int sourceHeight = checked((int)inputShape[2]);
        int sourceWidth = checked((int)inputShape[3]);
        return PadResizeLayout.Create(sourceWidth, sourceHeight, targetSize.Width, targetSize.Height, stride, mode);
    }

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
