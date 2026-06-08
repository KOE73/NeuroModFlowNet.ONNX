using Microsoft.ML.OnnxRuntime;
using NeuroModFlowNet.Pipeline;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// Copies an ONNX Runtime NHWC byte tensor into a CPU OpenCV Mat image.
/// </summary>
/// <remarks>
/// This is a representation conversion command, not a generic tensor download abstraction. The input contract is
/// deliberately narrow: byte tensor, rank 4, one image, channels that OpenCV can represent as a standard Mat type.
/// </remarks>
public sealed class Copy_OrtValue_To_MatImage : OpBase
{
    readonly string inputKey;
    readonly string outputKey;

    public Copy_OrtValue_To_MatImage(string inputKey, string outputKey)
        : base(OpDescriptor.Create(
            "Copy_OrtValue_To_MatImage",
            "copy.ortValue.toMatImage",
            reads: [VarRequirement.Read<OrtValue>(inputKey)],
            writes: [VarRequirement.Write<Mat>(outputKey)]))
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(inputKey);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputKey);

        this.inputKey = inputKey;
        this.outputKey = outputKey;
    }

    public override ValueTask<OpResult> ExecuteAsync(
        VmRunContext context,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(context);
        cancellationToken.ThrowIfCancellationRequested();

        if(!context.TryGet(inputKey, out OrtValue inputOrtValue))
            return ValueTask.FromResult(OpResult.Fail($"Input key '{inputKey}' was not found in the pipeline context."));

        long[] shape = inputOrtValue.GetTensorTypeAndShape().Shape;

        if(shape.Length != 4)
            return ValueTask.FromResult(OpResult.Fail($"Input tensor must be 4D, actual rank: {shape.Length}."));

        int height = checked((int)shape[1]);
        int width = checked((int)shape[2]);
        int channels = checked((int)shape[3]);

        MatType type = channels switch
        {
            1 => MatType.CV_8UC1,
            3 => MatType.CV_8UC3,
            4 => MatType.CV_8UC4,
            _ => throw new NotSupportedException($"Unsupported channel count: {channels}")
        };

        Mat outputMat = new(height, width, type);
        ReadOnlySpan<byte> sourceSpan = inputOrtValue.GetTensorDataAsSpan<byte>();

        unsafe
        {
            // OpenCV exposes Mat storage through a native pointer. Span gives us a checked managed copy loop while still
            // avoiding per-row marshaling and temporary arrays.
            var destSpan = new Span<byte>((byte*)outputMat.Data, checked((int)(outputMat.Total() * outputMat.ElemSize())));
            sourceSpan.CopyTo(destSpan);
        }

        // The Mat is a normal managed wrapper over native OpenCV memory. Current pipeline consumers own/dispose frame
        // images explicitly, so this command does not attach the Mat to automatic transaction disposal.
        context.Set(outputKey, outputMat, disposeWithContext: false);

        return ValueTask.FromResult(OpResult.Continue);
    }
}
