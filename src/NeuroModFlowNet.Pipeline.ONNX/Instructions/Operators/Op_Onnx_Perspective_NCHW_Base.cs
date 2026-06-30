using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.ONNX.Graph.Builders;
using NeuroModFlowNet.Pipeline;
using Onnx;
using OpenCvSharp;
using CvSize = OpenCvSharp.Size;

namespace NeuroModFlowNet.Pipeline.ONNX;

public abstract class Op_Onnx_Perspective_NCHW_Base : Op_Onnx_TensorTransformBase
{
    readonly Point2f[] sourcePoints;
    readonly CvSize outputSize;
    readonly string? outputTransformKey;
    readonly TensorElementType expectedElementType;
    readonly TensorProto.Types.DataType graphElementType;

    protected Op_Onnx_Perspective_NCHW_Base(
        OpDescriptor descriptor,
        string inputKey,
        string outputKey,
        ReadOnlySpan<Point2f> sourcePoints,
        CvSize outputSize,
        TensorElementType expectedElementType,
        TensorProto.Types.DataType graphElementType,
        string? outputTransformKey,
        bool isFinal,
        InferenceBackend? executionBackend)
        : base(descriptor, inputKey, outputKey, isFinal, executionBackend)
    {
        if(outputSize.Width <= 0 || outputSize.Height <= 0)
            throw new ArgumentOutOfRangeException(nameof(outputSize), "Output width and height must be positive.");

        if(sourcePoints.Length < 4)
            throw new ArgumentException("At least four source points are required.", nameof(sourcePoints));

        this.sourcePoints = sourcePoints[..4].ToArray();
        this.outputSize = outputSize;
        this.expectedElementType = expectedElementType;
        this.graphElementType = graphElementType;
        this.outputTransformKey = outputTransformKey;
    }

    protected override string GraphInputName => PerspectiveGridBuilder.InputName;

    protected override string GraphOutputName => PerspectiveGridBuilder.OutputName;

    protected override string GetCacheSemanticKey(long[] inputShape, TensorElementType inputElementType) =>
        $"output={outputSize.Width}x{outputSize.Height};points={FormatPoints(sourcePoints)};graphType={graphElementType}";

    protected override string? ValidateInput(long[] inputShape, TensorElementType inputElementType)
    {
        if(inputElementType != expectedElementType)
            return $"Input tensor must be {expectedElementType} NCHW image data, actual element type: {inputElementType}.";

        return inputShape.Length == 4 ? null : $"Input tensor must be 4D NCHW, actual rank: {inputShape.Length}.";
    }

    protected override byte[] BuildModel(long[] inputShape, TensorElementType inputElementType)
    {
        Span<float> matrix = stackalloc float[9];
        PerspectiveHomography.WriteTargetToSourceMatrix(sourcePoints, outputSize.Width, outputSize.Height, matrix);

        return PerspectiveGridBuilder.BuildNchw(
            checked((int)inputShape[3]),
            checked((int)inputShape[2]),
            outputSize.Width,
            outputSize.Height,
            checked((int)inputShape[1]),
            graphElementType,
            matrix);
    }

    protected override long[] CreateOutputShape(long[] inputShape, TensorElementType inputElementType) =>
        [1, inputShape[1], outputSize.Height, outputSize.Width];

    protected override void WriteAdditionalOutputs(VmRunContext context, long[] inputShape, long[] outputShape)
    {
        if(outputTransformKey is null)
            return;

        context.Set(outputTransformKey, PerspectiveHomography.CreateTargetToSourceTransform(sourcePoints, outputSize.Width, outputSize.Height));
    }

    protected static OpDescriptor CreateDescriptor(string descriptorName, string opcode, string inputKey, string outputKey, string? outputTransformKey)
    {
        VarRequirement[] writes = outputTransformKey is null
            ? [VarRequirement.Write<OrtValue>(outputKey)]
            : [VarRequirement.Write<OrtValue>(outputKey), VarRequirement.Write<ICoordinateBackTransform>(outputTransformKey)];

        return OpDescriptor.Create(descriptorName, opcode, [VarRequirement.Read<OrtValue>(inputKey)], writes);
    }

    static string FormatPoints(ReadOnlySpan<Point2f> points) =>
        string.Join(';', points.ToArray().Select(static point => $"{point.X},{point.Y}"));
}
