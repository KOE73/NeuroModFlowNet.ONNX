using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline;
using Onnx;

namespace NeuroModFlowNet.Pipeline.ONNX;

public sealed class Op_Onnx_Rotate90_FP32_NCHW : Op_Onnx_Rotate90_NCHW_Base
{
    public Op_Onnx_Rotate90_FP32_NCHW(
        string inputKey,
        string outputKey,
        Rotate90Mode mode,
        string? outputTransformKey = null,
        bool isFinal = false,
        InferenceBackend? executionBackend = null)
        : base(
            CreateDescriptor("Op_Onnx_Rotate90_FP32_NCHW", "op.onnx.rotate90.fp32.nchw", inputKey, outputKey, outputTransformKey),
            inputKey,
            outputKey,
            mode,
            TensorElementType.Float,
            TensorProto.Types.DataType.Float,
            outputTransformKey,
            isFinal,
            executionBackend)
    {
    }

    protected override string DisplayName => "rotate90-fp32-nchw.onnx";
}
