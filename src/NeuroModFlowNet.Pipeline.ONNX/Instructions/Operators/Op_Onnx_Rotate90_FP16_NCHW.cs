using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline;
using Onnx;

namespace NeuroModFlowNet.Pipeline.ONNX;

public sealed class Op_Onnx_Rotate90_FP16_NCHW : Op_Onnx_Rotate90_NCHW_Base
{
    public Op_Onnx_Rotate90_FP16_NCHW(
        string inputKey,
        string outputKey,
        Rotate90Mode mode,
        string? outputTransformKey = null,
        bool isFinal = false,
        InferenceBackend? executionBackend = null)
        : base(
            CreateDescriptor("Op_Onnx_Rotate90_FP16_NCHW", "op.onnx.rotate90.fp16.nchw", inputKey, outputKey, outputTransformKey),
            inputKey,
            outputKey,
            mode,
            TensorElementType.Float16,
            TensorProto.Types.DataType.Float16,
            outputTransformKey,
            isFinal,
            executionBackend)
    {
    }

    protected override string DisplayName => "rotate90-fp16-nchw.onnx";
}
