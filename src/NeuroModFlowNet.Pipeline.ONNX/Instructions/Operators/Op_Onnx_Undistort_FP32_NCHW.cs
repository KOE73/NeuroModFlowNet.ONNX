using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline;
using Onnx;
using OpenCvSharp;
using CvSize = OpenCvSharp.Size;

namespace NeuroModFlowNet.Pipeline.ONNX;

public sealed class Op_Onnx_Undistort_FP32_NCHW : Op_Onnx_Undistort_NCHW_Base
{
    public Op_Onnx_Undistort_FP32_NCHW(
        string inputKey,
        string outputKey,
        RadialTangentialDistortionParameters distortion,
        CvSize? outputSize = null,
        string? outputTransformKey = null,
        bool isFinal = false,
        InferenceBackend? executionBackend = null)
        : base(
            CreateDescriptor("Op_Onnx_Undistort_FP32_NCHW", "op.onnx.undistort.fp32.nchw", inputKey, outputKey, outputTransformKey),
            inputKey,
            outputKey,
            distortion,
            outputSize,
            TensorElementType.Float,
            TensorProto.Types.DataType.Float,
            outputTransformKey,
            isFinal,
            executionBackend)
    {
    }

    protected override string DisplayName => "undistort-fp32-nchw.onnx";
}
