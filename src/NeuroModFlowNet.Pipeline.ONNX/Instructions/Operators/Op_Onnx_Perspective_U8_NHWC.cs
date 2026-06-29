using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.ONNX.Graph.Builders;
using NeuroModFlowNet.Pipeline;
using OpenCvSharp;
using CvSize = OpenCvSharp.Size;

namespace NeuroModFlowNet.Pipeline.ONNX;

public sealed class Op_Onnx_Perspective_U8_NHWC : Op_Onnx_TensorTransformBase
{
    readonly Point2f[] sourcePoints;
    readonly CvSize outputSize;
    readonly string? outputTransformKey;

    public Op_Onnx_Perspective_U8_NHWC(
        string inputKey,
        string outputKey,
        ReadOnlySpan<Point2f> sourcePoints,
        CvSize outputSize,
        string? outputTransformKey = null,
        bool isFinal = false,
        InferenceBackend? executionBackend = null)
        : base(CreateDescriptor(inputKey, outputKey, outputTransformKey), inputKey, outputKey, isFinal, executionBackend)
    {
        if(outputSize.Width <= 0 || outputSize.Height <= 0)
            throw new ArgumentOutOfRangeException(nameof(outputSize), "Output width and height must be positive.");

        if(sourcePoints.Length < 4)
            throw new ArgumentException("At least four source points are required.", nameof(sourcePoints));

        this.sourcePoints = sourcePoints[..4].ToArray();
        this.outputSize = outputSize;
        this.outputTransformKey = outputTransformKey;
    }

    protected override string GraphInputName => PerspectiveGridBuilder.InputName;

    protected override string GraphOutputName => PerspectiveGridBuilder.OutputName;

    protected override string DisplayName => "perspective-u8-nhwc.onnx";

    protected override string? ValidateInput(long[] inputShape, TensorElementType inputElementType)
    {
        if(inputElementType != TensorElementType.UInt8)
            return $"Input tensor must be UInt8 NHWC image data, actual element type: {inputElementType}.";

        return inputShape.Length == 4 ? null : $"Input tensor must be 4D NHWC, actual rank: {inputShape.Length}.";
    }

    protected override byte[] BuildModel(long[] inputShape, TensorElementType inputElementType)
    {
        Span<float> matrix = stackalloc float[9];
        PerspectiveHomography.WriteTargetToSourceMatrix(sourcePoints, outputSize.Width, outputSize.Height, matrix);

        return PerspectiveGridBuilder.BuildU8Nhwc(
            checked((int)inputShape[2]),
            checked((int)inputShape[1]),
            outputSize.Width,
            outputSize.Height,
            checked((int)inputShape[3]),
            matrix);
    }

    protected override long[] CreateOutputShape(long[] inputShape, TensorElementType inputElementType) =>
        [1, outputSize.Height, outputSize.Width, inputShape[3]];

    protected override void WriteAdditionalOutputs(VmRunContext context, long[] inputShape, long[] outputShape)
    {
        if(outputTransformKey is null)
            return;

        context.Set(outputTransformKey, PerspectiveHomography.CreateTargetToSourceTransform(sourcePoints, outputSize.Width, outputSize.Height));
    }

    static OpDescriptor CreateDescriptor(string inputKey, string outputKey, string? outputTransformKey)
    {
        VarRequirement[] writes = outputTransformKey is null
            ? [VarRequirement.Write<OrtValue>(outputKey)]
            : [VarRequirement.Write<OrtValue>(outputKey), VarRequirement.Write<ICoordinateBackTransform>(outputTransformKey)];

        return OpDescriptor.Create("Op_Onnx_Perspective_U8_NHWC", "op.onnx.perspective.u8.nhwc", [VarRequirement.Read<OrtValue>(inputKey)], writes);
    }
}
