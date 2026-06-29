using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.ONNX.Graph.Builders;
using NeuroModFlowNet.Pipeline;
using CvRect = OpenCvSharp.Rect;

namespace NeuroModFlowNet.Pipeline.ONNX;

public sealed class Op_Onnx_Crop_U8_NHWC : Op_Onnx_TensorTransformBase
{
    readonly CvRect cropRect;
    readonly string? outputTransformKey;

    public Op_Onnx_Crop_U8_NHWC(
        string inputKey,
        string outputKey,
        CvRect cropRect,
        string? outputTransformKey = null,
        bool isFinal = false,
        InferenceBackend? executionBackend = null)
        : base(CreateDescriptor(inputKey, outputKey, outputTransformKey), inputKey, outputKey, isFinal, executionBackend)
    {
        if(cropRect.Width <= 0 || cropRect.Height <= 0)
            throw new ArgumentOutOfRangeException(nameof(cropRect), "Crop width and height must be positive.");

        this.cropRect = cropRect;
        this.outputTransformKey = outputTransformKey;
    }

    protected override string GraphInputName => CropBuilder.InputName;

    protected override string GraphOutputName => CropBuilder.OutputName;

    protected override string DisplayName => "crop-u8-nhwc.onnx";

    protected override string? ValidateInput(long[] inputShape, TensorElementType inputElementType)
    {
        if(inputElementType != TensorElementType.UInt8)
            return $"Input tensor must be UInt8 NHWC image data, actual element type: {inputElementType}.";

        if(inputShape.Length != 4)
            return $"Input tensor must be 4D NHWC, actual rank: {inputShape.Length}.";

        int sourceHeight = checked((int)inputShape[1]);
        int sourceWidth = checked((int)inputShape[2]);

        if(cropRect.X < 0 || cropRect.Y < 0)
            return $"Crop origin must be inside the image, actual origin: {cropRect.X},{cropRect.Y}.";

        if(cropRect.Right > sourceWidth || cropRect.Bottom > sourceHeight)
            return $"Crop rectangle {cropRect} exceeds input image bounds {sourceWidth}x{sourceHeight}.";

        return null;
    }

    protected override byte[] BuildModel(long[] inputShape, TensorElementType inputElementType)
    {
        int sourceHeight = checked((int)inputShape[1]);
        int sourceWidth = checked((int)inputShape[2]);
        int channels = checked((int)inputShape[3]);
        return CropBuilder.Build(sourceWidth, sourceHeight, cropRect.X, cropRect.Y, cropRect.Right, cropRect.Bottom, channels);
    }

    protected override long[] CreateOutputShape(long[] inputShape, TensorElementType inputElementType) =>
        [1, cropRect.Height, cropRect.Width, inputShape[3]];

    protected override void WriteAdditionalOutputs(VmRunContext context, long[] inputShape, long[] outputShape)
    {
        if(outputTransformKey is null)
            return;

        context.Set(outputTransformKey, new CropCoordinateBackTransform(cropRect.X, cropRect.Y));
    }

    static OpDescriptor CreateDescriptor(string inputKey, string outputKey, string? outputTransformKey)
    {
        VarRequirement[] writes = outputTransformKey is null
            ? [VarRequirement.Write<OrtValue>(outputKey)]
            : [VarRequirement.Write<OrtValue>(outputKey), VarRequirement.Write<ICoordinateBackTransform>(outputTransformKey)];

        return OpDescriptor.Create("Op_Onnx_Crop_U8_NHWC", "op.onnx.crop.u8.nhwc", [VarRequirement.Read<OrtValue>(inputKey)], writes);
    }
}
