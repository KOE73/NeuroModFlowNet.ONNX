using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline;
using Onnx;
using CvSize = OpenCvSharp.Size;

namespace NeuroModFlowNet.Pipeline.ONNX;

public sealed class Op_Onnx_PadResize_FP16_NCHW : Op_Onnx_PadResize_NCHW_Base
{
    public Op_Onnx_PadResize_FP16_NCHW(
        string inputKey,
        string outputKey,
        CvSize targetSize,
        string? outputTransformKey = null,
        int stride = 32,
        PadResizeMode mode = PadResizeMode.FixedCanvas,
        float padValue = 114f / 255f,
        bool isFinal = false,
        InferenceBackend? executionBackend = null)
        : base(
            CreateDescriptor("Op_Onnx_PadResize_FP16_NCHW", "op.onnx.padResize.fp16.nchw", inputKey, outputKey, outputTransformKey),
            inputKey,
            outputKey,
            targetSize,
            TensorElementType.Float16,
            TensorProto.Types.DataType.Float16,
            outputTransformKey,
            stride,
            mode,
            padValue,
            isFinal,
            executionBackend)
    {
    }

    protected override string DisplayName => "pad-resize-fp16-nchw.onnx";
}
