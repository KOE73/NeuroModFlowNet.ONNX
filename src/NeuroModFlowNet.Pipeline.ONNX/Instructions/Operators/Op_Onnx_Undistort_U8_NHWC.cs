using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.ONNX.Graph.Builders;
using NeuroModFlowNet.Pipeline;
using OpenCvSharp;
using CvSize = OpenCvSharp.Size;

namespace NeuroModFlowNet.Pipeline.ONNX;

public sealed class Op_Onnx_Undistort_U8_NHWC : Op_Onnx_TensorTransformBase
{
    readonly RadialTangentialDistortionParameters distortion;
    readonly CvSize? outputSize;
    readonly string? outputTransformKey;

    public Op_Onnx_Undistort_U8_NHWC(
        string inputKey,
        string outputKey,
        RadialTangentialDistortionParameters distortion,
        CvSize? outputSize = null,
        string? outputTransformKey = null,
        bool isFinal = false,
        InferenceBackend? executionBackend = null)
        : base(CreateDescriptor(inputKey, outputKey, outputTransformKey), inputKey, outputKey, isFinal, executionBackend)
    {
        ValidateDistortion(distortion);
        if(outputSize is { Width: <= 0 } or { Height: <= 0 })
            throw new ArgumentOutOfRangeException(nameof(outputSize), "Output width and height must be positive.");

        this.distortion = distortion;
        this.outputSize = outputSize;
        this.outputTransformKey = outputTransformKey;
    }

    protected override string GraphInputName => UndistortGridBuilder.InputName;

    protected override string GraphOutputName => UndistortGridBuilder.OutputName;

    protected override string DisplayName => "undistort-u8-nhwc.onnx";

    protected override string GetCacheSemanticKey(long[] inputShape, TensorElementType inputElementType)
    {
        (int outputWidth, int outputHeight) = ResolveOutputSize(inputShape);
        return $"output={outputWidth}x{outputHeight};distortion={FormatDistortion(distortion)}";
    }

    protected override string? ValidateInput(long[] inputShape, TensorElementType inputElementType)
    {
        if(inputElementType != TensorElementType.UInt8)
            return $"Input tensor must be UInt8 NHWC image data, actual element type: {inputElementType}.";

        return inputShape.Length == 4 ? null : $"Input tensor must be 4D NHWC, actual rank: {inputShape.Length}.";
    }

    protected override byte[] BuildModel(long[] inputShape, TensorElementType inputElementType)
    {
        (int outputWidth, int outputHeight) = ResolveOutputSize(inputShape);
        Span<float> distortionValues = stackalloc float[13];
        WriteDistortionValues(distortion, distortionValues);

        return UndistortGridBuilder.BuildU8Nhwc(
            checked((int)inputShape[2]),
            checked((int)inputShape[1]),
            outputWidth,
            outputHeight,
            checked((int)inputShape[3]),
            distortionValues);
    }

    protected override long[] CreateOutputShape(long[] inputShape, TensorElementType inputElementType)
    {
        (int outputWidth, int outputHeight) = ResolveOutputSize(inputShape);
        return [1, outputHeight, outputWidth, inputShape[3]];
    }

    protected override void WriteAdditionalOutputs(VmRunContext context, long[] inputShape, long[] outputShape)
    {
        if(outputTransformKey is null)
            return;

        context.Set(outputTransformKey, new UndistortCoordinateBackTransform(distortion));
    }

    (int OutputWidth, int OutputHeight) ResolveOutputSize(long[] inputShape) =>
        outputSize is { } size
            ? (size.Width, size.Height)
            : (checked((int)inputShape[2]), checked((int)inputShape[1]));

    static void ValidateDistortion(RadialTangentialDistortionParameters distortion)
    {
        if(distortion.Fx == 0f)
            throw new ArgumentOutOfRangeException(nameof(distortion), "Fx must be non-zero.");

        if(distortion.Fy == 0f)
            throw new ArgumentOutOfRangeException(nameof(distortion), "Fy must be non-zero.");

        if(distortion.EffectiveOutputFx == 0f || distortion.EffectiveOutputFy == 0f)
            throw new ArgumentOutOfRangeException(nameof(distortion), "Output Fx and Fy must be non-zero.");
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
        destination[9] = distortion.EffectiveOutputFx;
        destination[10] = distortion.EffectiveOutputFy;
        destination[11] = distortion.EffectiveOutputCx;
        destination[12] = distortion.EffectiveOutputCy;
    }

    static OpDescriptor CreateDescriptor(string inputKey, string outputKey, string? outputTransformKey)
    {
        VarRequirement[] writes = outputTransformKey is null
            ? [VarRequirement.Write<OrtValue>(outputKey)]
            : [VarRequirement.Write<OrtValue>(outputKey), VarRequirement.Write<ICoordinateBackTransform>(outputTransformKey)];

        return OpDescriptor.Create("Op_Onnx_Undistort_U8_NHWC", "op.onnx.undistort.u8.nhwc", [VarRequirement.Read<OrtValue>(inputKey)], writes);
    }

    static string FormatDistortion(RadialTangentialDistortionParameters value) =>
        $"{value.Fx},{value.Fy},{value.Cx},{value.Cy},{value.K1},{value.K2},{value.P1},{value.P2},{value.K3};output={value.EffectiveOutputFx},{value.EffectiveOutputFy},{value.EffectiveOutputCx},{value.EffectiveOutputCy}";
}
