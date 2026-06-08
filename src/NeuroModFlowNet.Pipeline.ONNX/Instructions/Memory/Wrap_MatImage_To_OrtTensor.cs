using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.Pipeline;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// Wraps CPU-accessible OpenCV Mat image memory as an NHWC ONNX Runtime tensor without copying pixel data.
/// </summary>
/// <remarks>
/// The pipeline treats <c>Mat</c> here as an image buffer, not as an arbitrary OpenCV matrix. The produced shape is
/// <c>[1, height, width, channels]</c>, matching the runtime graph helpers used by the crop/resize commands.
/// </remarks>
public sealed class Wrap_MatImage_To_OrtTensor : Wrap_UnmanagedMem_To_OrtTensorBase<Mat>
{
    public Wrap_MatImage_To_OrtTensor(string inputKey, string outputKey)
        : base(
            "Wrap_MatImage_To_OrtTensor",
            "wrap.matImage.toOrtTensor",
            inputKey,
            outputKey)
    {
    }

    protected override unsafe UnmanagedTensorMemoryView CreateSourceMemoryView(Mat input, VmRunContext context)
    {
        ArgumentNullException.ThrowIfNull(input);

        // ONNX Runtime receives one linear pointer and cannot honor OpenCV row strides in this wrapper path. A
        // non-contiguous ROI therefore has to be cloned; the cloned Mat is attached to the VM context by the base class.
        Mat sourceMat = input.IsContinuous() ? input : input.Clone();
        int channels = sourceMat.Channels();
        long[] shape = [1, sourceMat.Height, sourceMat.Width, channels];
        int byteLength = checked((int)(sourceMat.Total() * sourceMat.ElemSize()));

        return new UnmanagedTensorMemoryView(
            (nint)sourceMat.DataPointer,
            byteLength,
            TensorElementType.UInt8,
            shape,
            ReferenceEquals(sourceMat, input) ? null : sourceMat);
    }
}
