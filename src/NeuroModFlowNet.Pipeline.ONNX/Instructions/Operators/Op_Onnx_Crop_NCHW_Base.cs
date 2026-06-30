using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.ONNX.Graph.Builders;
using NeuroModFlowNet.Pipeline;
using Onnx;
using CvRect = OpenCvSharp.Rect;

namespace NeuroModFlowNet.Pipeline.ONNX;

public abstract class Op_Onnx_Crop_NCHW_Base : Op_Onnx_TensorTransformBase
{
    readonly CvRect cropRect;
    readonly string? outputTransformKey;
    readonly TensorElementType expectedElementType;
    readonly TensorProto.Types.DataType graphElementType;

    protected Op_Onnx_Crop_NCHW_Base(
        OpDescriptor descriptor,
        string inputKey,
        string outputKey,
        CvRect cropRect,
        TensorElementType expectedElementType,
        TensorProto.Types.DataType graphElementType,
        string? outputTransformKey,
        bool isFinal,
        InferenceBackend? executionBackend)
        : base(descriptor, inputKey, outputKey, isFinal, executionBackend)
    {
        if(cropRect.Width <= 0 || cropRect.Height <= 0)
            throw new ArgumentOutOfRangeException(nameof(cropRect), "Crop width and height must be positive.");

        this.cropRect = cropRect;
        this.outputTransformKey = outputTransformKey;
        this.expectedElementType = expectedElementType;
        this.graphElementType = graphElementType;
    }

    protected override string GraphInputName => CropNchwBuilder.InputName;

    protected override string GraphOutputName => CropNchwBuilder.OutputName;

    protected override string GetCacheSemanticKey(long[] inputShape, TensorElementType inputElementType) =>
        $"crop={cropRect.X},{cropRect.Y},{cropRect.Width},{cropRect.Height};graphType={graphElementType}";

    protected override string? ValidateInput(long[] inputShape, TensorElementType inputElementType)
    {
        if(inputElementType != expectedElementType)
            return $"Input tensor must be {expectedElementType} NCHW image data, actual element type: {inputElementType}.";

        if(inputShape.Length != 4)
            return $"Input tensor must be 4D NCHW, actual rank: {inputShape.Length}.";

        if(inputShape[0] != 1)
            return $"Input tensor batch must be 1, actual batch: {inputShape[0]}.";

        int sourceHeight = checked((int)inputShape[2]);
        int sourceWidth = checked((int)inputShape[3]);

        if(cropRect.X < 0 || cropRect.Y < 0)
            return $"Crop origin must be inside the image, actual origin: {cropRect.X},{cropRect.Y}.";

        if(cropRect.Right > sourceWidth || cropRect.Bottom > sourceHeight)
            return $"Crop rectangle {cropRect} exceeds input image bounds {sourceWidth}x{sourceHeight}.";

        return null;
    }

    protected override byte[] BuildModel(long[] inputShape, TensorElementType inputElementType)
    {
        int channels = checked((int)inputShape[1]);
        int sourceHeight = checked((int)inputShape[2]);
        int sourceWidth = checked((int)inputShape[3]);

        return CropNchwBuilder.Build(
            sourceWidth,
            sourceHeight,
            cropRect.X,
            cropRect.Y,
            cropRect.X + cropRect.Width,
            cropRect.Y + cropRect.Height,
            channels,
            graphElementType);
    }

    protected override long[] CreateOutputShape(long[] inputShape, TensorElementType inputElementType) =>
        [1, inputShape[1], cropRect.Height, cropRect.Width];

    protected override void WriteAdditionalOutputs(VmRunContext context, long[] inputShape, long[] outputShape)
    {
        if(outputTransformKey is null)
            return;

        context.Set(outputTransformKey, new CropCoordinateBackTransform(cropRect.X, cropRect.Y));
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
