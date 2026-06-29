using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using Onnx;
using OpenCvSharp;
using CvSize = OpenCvSharp.Size;

namespace NeuroModFlowNet.Pipeline.ONNX;

public sealed class Op_Onnx_Perspective_FP16_NCHW : Op_Onnx_Perspective_NCHW_Base
{
    public Op_Onnx_Perspective_FP16_NCHW(
        string inputKey,
        string outputKey,
        ReadOnlySpan<Point2f> sourcePoints,
        CvSize outputSize,
        string? outputTransformKey = null,
        bool isFinal = false,
        InferenceBackend? executionBackend = null)
        : base(
            CreateDescriptor("Op_Onnx_Perspective_FP16_NCHW", "op.onnx.perspective.fp16.nchw", inputKey, outputKey, outputTransformKey),
            inputKey,
            outputKey,
            sourcePoints,
            outputSize,
            TensorElementType.Float16,
            TensorProto.Types.DataType.Float16,
            outputTransformKey,
            isFinal,
            executionBackend)
    {
    }

    protected override string DisplayName => "perspective-fp16-nchw.onnx";
}
