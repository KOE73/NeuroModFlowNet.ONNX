using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using Onnx;
using CvRect = OpenCvSharp.Rect;

namespace NeuroModFlowNet.Pipeline.ONNX;

public sealed class Op_Onnx_Crop_FP16_NCHW : Op_Onnx_Crop_NCHW_Base
{
    public Op_Onnx_Crop_FP16_NCHW(
        string inputKey,
        string outputKey,
        CvRect cropRect,
        string? outputTransformKey = null,
        bool isFinal = false,
        InferenceBackend? executionBackend = null)
        : base(
            CreateDescriptor(
                "Op_Onnx_Crop_FP16_NCHW",
                "op.onnx.crop.fp16.nchw",
                inputKey,
                outputKey,
                outputTransformKey),
            inputKey,
            outputKey,
            cropRect,
            TensorElementType.Float16,
            TensorProto.Types.DataType.Float16,
            outputTransformKey,
            isFinal,
            executionBackend)
    {
    }

    protected override string DisplayName => "crop-fp16-nchw.onnx";
}
