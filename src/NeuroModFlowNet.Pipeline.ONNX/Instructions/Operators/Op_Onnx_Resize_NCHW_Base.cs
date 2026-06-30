using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.ONNX.Graph.Builders;
using NeuroModFlowNet.Pipeline;
using Onnx;
using CvSize = OpenCvSharp.Size;

namespace NeuroModFlowNet.Pipeline.ONNX;

public abstract class Op_Onnx_Resize_NCHW_Base : Op_Onnx_TensorTransformBase
{
    readonly CvSize targetSize;
    readonly string? outputTransformKey;
    readonly TensorElementType expectedElementType;
    readonly TensorProto.Types.DataType graphElementType;

    protected Op_Onnx_Resize_NCHW_Base(
        OpDescriptor descriptor,
        string inputKey,
        string outputKey,
        CvSize targetSize,
        TensorElementType expectedElementType,
        TensorProto.Types.DataType graphElementType,
        string? outputTransformKey,
        bool isFinal,
        InferenceBackend? executionBackend)
        : base(descriptor, inputKey, outputKey, isFinal, executionBackend)
    {
        if(targetSize.Width <= 0 || targetSize.Height <= 0)
            throw new ArgumentOutOfRangeException(nameof(targetSize), "Width and height must be positive.");

        this.targetSize = targetSize;
        this.outputTransformKey = outputTransformKey;
        this.expectedElementType = expectedElementType;
        this.graphElementType = graphElementType;
    }

    protected override string GraphInputName => ResizeNchwBuilder.InputName;

    protected override string GraphOutputName => ResizeNchwBuilder.OutputName;

    protected override string GetCacheSemanticKey(long[] inputShape, TensorElementType inputElementType) =>
        $"target={targetSize.Width}x{targetSize.Height};graphType={graphElementType}";

    protected override string? ValidateInput(long[] inputShape, TensorElementType inputElementType)
    {
        if(inputElementType != expectedElementType)
            return $"Input tensor must be {expectedElementType} NCHW image data, actual element type: {inputElementType}.";

        if(inputShape.Length != 4)
            return $"Input tensor must be 4D NCHW, actual rank: {inputShape.Length}.";

        if(inputShape[0] != 1)
            return $"Input tensor batch must be 1, actual batch: {inputShape[0]}.";

        return null;
    }

    protected override byte[] BuildModel(long[] inputShape, TensorElementType inputElementType)
    {
        int channels = checked((int)inputShape[1]);
        int sourceHeight = checked((int)inputShape[2]);
        int sourceWidth = checked((int)inputShape[3]);

        return ResizeNchwBuilder.Build(
            sourceWidth,
            sourceHeight,
            targetSize.Width,
            targetSize.Height,
            channels,
            graphElementType);
    }

    protected override long[] CreateOutputShape(long[] inputShape, TensorElementType inputElementType) =>
        [1, inputShape[1], targetSize.Height, targetSize.Width];

    protected override void WriteAdditionalOutputs(VmRunContext context, long[] inputShape, long[] outputShape)
    {
        if(outputTransformKey is null)
            return;

        int sourceHeight = checked((int)inputShape[2]);
        int sourceWidth = checked((int)inputShape[3]);
        context.Set(outputTransformKey, new ResizeCoordinateBackTransform(
            sourceWidth,
            sourceHeight,
            targetSize.Width,
            targetSize.Height));
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
