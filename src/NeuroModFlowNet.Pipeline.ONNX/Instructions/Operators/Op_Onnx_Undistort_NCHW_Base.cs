using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.ONNX.Graph.Builders;
using NeuroModFlowNet.Pipeline;
using Onnx;
using OpenCvSharp;
using CvSize = OpenCvSharp.Size;

namespace NeuroModFlowNet.Pipeline.ONNX;

public abstract class Op_Onnx_Undistort_NCHW_Base : Op_Onnx_TensorTransformBase
{
    readonly RadialTangentialDistortionParameters distortion;
    readonly CvSize? outputSize;
    readonly string? outputTransformKey;
    readonly TensorElementType expectedElementType;
    readonly TensorProto.Types.DataType graphElementType;

    protected Op_Onnx_Undistort_NCHW_Base(
        OpDescriptor descriptor,
        string inputKey,
        string outputKey,
        RadialTangentialDistortionParameters distortion,
        CvSize? outputSize,
        TensorElementType expectedElementType,
        TensorProto.Types.DataType graphElementType,
        string? outputTransformKey,
        bool isFinal,
        InferenceBackend? executionBackend)
        : base(descriptor, inputKey, outputKey, isFinal, executionBackend)
    {
        ValidateDistortion(distortion);
        if(outputSize is { Width: <= 0 } or { Height: <= 0 })
            throw new ArgumentOutOfRangeException(nameof(outputSize), "Output width and height must be positive.");

        this.distortion = distortion;
        this.outputSize = outputSize;
        this.expectedElementType = expectedElementType;
        this.graphElementType = graphElementType;
        this.outputTransformKey = outputTransformKey;
    }

    protected override string GraphInputName => UndistortGridBuilder.InputName;

    protected override string GraphOutputName => UndistortGridBuilder.OutputName;

    protected override string? ValidateInput(long[] inputShape, TensorElementType inputElementType)
    {
        if(inputElementType != expectedElementType)
            return $"Input tensor must be {expectedElementType} NCHW image data, actual element type: {inputElementType}.";

        return inputShape.Length == 4 ? null : $"Input tensor must be 4D NCHW, actual rank: {inputShape.Length}.";
    }

    protected override byte[] BuildModel(long[] inputShape, TensorElementType inputElementType)
    {
        (int outputWidth, int outputHeight) = ResolveOutputSize(inputShape);
        Span<float> distortionValues = stackalloc float[9];
        WriteDistortionValues(distortion, distortionValues);

        return UndistortGridBuilder.BuildNchw(
            checked((int)inputShape[3]),
            checked((int)inputShape[2]),
            outputWidth,
            outputHeight,
            checked((int)inputShape[1]),
            graphElementType,
            distortionValues);
    }

    protected override long[] CreateOutputShape(long[] inputShape, TensorElementType inputElementType)
    {
        (int outputWidth, int outputHeight) = ResolveOutputSize(inputShape);
        return [1, inputShape[1], outputHeight, outputWidth];
    }

    protected override void WriteAdditionalOutputs(VmRunContext context, long[] inputShape, long[] outputShape)
    {
        if(outputTransformKey is null)
            return;

        context.Set(outputTransformKey, new UndistortCoordinateBackTransform(distortion));
    }

    protected static OpDescriptor CreateDescriptor(string descriptorName, string opcode, string inputKey, string outputKey, string? outputTransformKey)
    {
        VarRequirement[] writes = outputTransformKey is null
            ? [VarRequirement.Write<OrtValue>(outputKey)]
            : [VarRequirement.Write<OrtValue>(outputKey), VarRequirement.Write<ICoordinateBackTransform>(outputTransformKey)];

        return OpDescriptor.Create(descriptorName, opcode, [VarRequirement.Read<OrtValue>(inputKey)], writes);
    }

    (int OutputWidth, int OutputHeight) ResolveOutputSize(long[] inputShape) =>
        outputSize is { } size
            ? (size.Width, size.Height)
            : (checked((int)inputShape[3]), checked((int)inputShape[2]));

    static void ValidateDistortion(RadialTangentialDistortionParameters distortion)
    {
        if(distortion.Fx == 0f)
            throw new ArgumentOutOfRangeException(nameof(distortion), "Fx must be non-zero.");

        if(distortion.Fy == 0f)
            throw new ArgumentOutOfRangeException(nameof(distortion), "Fy must be non-zero.");
    }

    static void WriteDistortionValues(RadialTangentialDistortionParameters distortion, Span<float> destination)
    {
        destination[0] = distortion.Fx;
        destination[1] = distortion.Fy;
        destination[2] = distortion.Cx;
        destination[3] = distortion.Cy;
        destination[4] = distortion.K1;
        destination[5] = distortion.K2;
        destination[6] = distortion.P1;
        destination[7] = distortion.P2;
        destination[8] = distortion.K3;
    }
}
