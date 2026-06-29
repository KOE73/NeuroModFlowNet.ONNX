using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using Onnx;
using CvSize = OpenCvSharp.Size;

namespace NeuroModFlowNet.Pipeline.ONNX;

public sealed class Op_Onnx_Resize_FP32_NCHW : Op_Onnx_Resize_NCHW_Base
{
    public Op_Onnx_Resize_FP32_NCHW(
        string inputKey,
        string outputKey,
        CvSize targetSize,
        string? outputTransformKey = null,
        bool isFinal = false,
        InferenceBackend? executionBackend = null)
        : base(
            CreateDescriptor("Op_Onnx_Resize_FP32_NCHW", "op.onnx.resize.fp32.nchw", inputKey, outputKey, outputTransformKey),
            inputKey,
            outputKey,
            targetSize,
            TensorElementType.Float,
            TensorProto.Types.DataType.Float,
            outputTransformKey,
            isFinal,
            executionBackend)
    {
    }

    protected override string DisplayName => "resize-fp32-nchw.onnx";
}
