using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX.Graph.Builders;
using NeuroModFlowNet.Pipeline;
using CvSize = OpenCvSharp.Size;

namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// Applies a generated ONNX Resize operator to an ONNX Runtime tensor.
/// </summary>
/// <remarks>
/// The command is an ONNX operator step, not an OpenCV resize. This matters for GPU pipelines: intermediate data can
/// remain in the same execution-provider placement as model inputs instead of silently crossing back to host memory.
/// </remarks>
public sealed class Op_Onnx_Resize : Op_Onnx_TensorTransformBase
{
    readonly CvSize targetSize;

    public Op_Onnx_Resize(string inputKey, string outputKey, CvSize targetSize, bool isFinal = false)
        : base(
            OpDescriptor.Create(
                "Op_Onnx_Resize",
                "op.onnx.resize",
                reads: [VarRequirement.Read<OrtValue>(inputKey)],
                writes: [VarRequirement.Write<OrtValue>(outputKey)]),
            inputKey,
            outputKey,
            isFinal)
    {
        if(targetSize.Width <= 0 || targetSize.Height <= 0)
            throw new ArgumentOutOfRangeException(nameof(targetSize), "Width and height must be positive.");

        this.targetSize = targetSize;
    }

    protected override string GraphInputName => ResizeBuilder.InputName;

    protected override string GraphOutputName => ResizeBuilder.OutputName;

    protected override string DisplayName => "resize.onnx";

    protected override string? ValidateInput(long[] inputShape, TensorElementType inputElementType)
    {
        if(inputShape.Length != 4)
            return $"Input tensor must be 4D NHWC, actual rank: {inputShape.Length}.";

        return null;
    }

    protected override byte[] BuildModel(long[] inputShape, TensorElementType inputElementType)
    {
        int sourceHeight = checked((int)inputShape[1]);
        int sourceWidth = checked((int)inputShape[2]);
        int channels = checked((int)inputShape[3]);

        return ResizeBuilder.Build(
            sourceWidth,
            sourceHeight,
            targetSize.Width,
            targetSize.Height,
            channels);
    }

    protected override long[] CreateOutputShape(long[] inputShape, TensorElementType inputElementType) =>
        [1, targetSize.Height, targetSize.Width, inputShape[3]];
}
