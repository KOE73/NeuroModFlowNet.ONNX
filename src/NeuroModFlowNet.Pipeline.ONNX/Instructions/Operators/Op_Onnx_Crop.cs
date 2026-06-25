using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.ONNX.Graph.Builders;
using NeuroModFlowNet.Pipeline;
using CvRect = OpenCvSharp.Rect;

namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// Applies a generated ONNX Slice crop operator to an ONNX Runtime tensor.
/// </summary>
/// <remarks>
/// The command is an ONNX operator step, not an OpenCV crop. When CUDA is available it can keep intermediate tensors in
/// provider memory and therefore preserve the "explicit placement, no implicit copy" pipeline contract.
/// </remarks>
public sealed class Op_Onnx_Crop : Op_Onnx_TensorTransformBase
{
    readonly CvRect cropRect;
    readonly string? outputTransformKey;

    public Op_Onnx_Crop(
        string inputKey,
        string outputKey,
        CvRect cropRect,
        bool isFinal = false,
        InferenceBackend? executionBackend = null)
        : this(inputKey, outputKey, cropRect, outputTransformKey: null, isFinal, executionBackend)
    {
    }

    public Op_Onnx_Crop(
        string inputKey,
        string outputKey,
        CvRect cropRect,
        string? outputTransformKey,
        bool isFinal = false,
        InferenceBackend? executionBackend = null)
        : base(
            CreateDescriptor(inputKey, outputKey, outputTransformKey),
            inputKey,
            outputKey,
            isFinal,
            executionBackend)
    {
        if(cropRect.Width <= 0 || cropRect.Height <= 0)
            throw new ArgumentOutOfRangeException(nameof(cropRect), "Crop width and height must be positive.");

        this.cropRect = cropRect;
        this.outputTransformKey = outputTransformKey;
    }

    protected override string GraphInputName => CropBuilder.InputName;

    protected override string GraphOutputName => CropBuilder.OutputName;

    protected override string DisplayName => "crop.onnx";

    protected override string? ValidateInput(long[] inputShape, TensorElementType inputElementType)
    {
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

        return CropBuilder.Build(
            sourceWidth,
            sourceHeight,
            cropRect.X,
            cropRect.Y,
            cropRect.X + cropRect.Width,
            cropRect.Y + cropRect.Height,
            channels);
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

        return OpDescriptor.Create(
            "Op_Onnx_Crop",
            "op.onnx.crop",
            reads: [VarRequirement.Read<OrtValue>(inputKey)],
            writes: writes);
    }
}
