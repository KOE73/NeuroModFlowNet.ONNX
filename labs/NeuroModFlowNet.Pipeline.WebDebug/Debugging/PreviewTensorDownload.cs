using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.ONNX.Graph.Builders;
using NeuroModFlowNet.ONNX.Visualizer;
using NeuroModFlowNet.Pipeline.ONNX;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.WebDebug;

/// <summary>
/// EN: Explicit same-size CPU output through ORT Identity on the selected backend; no resize or direct GPU-pointer access.
///
/// RU: Явный CPU-выход исходного размера через ORT Identity на выбранном backend, без resize и чтения GPU-указателя.
/// </summary>
internal static class PreviewTensorDownload
{
    public static Mat Copy(OrtValue input, InferenceBackend backend)
    {
        var information = input.GetTensorTypeAndShape();
        var type = information.ElementDataType switch
        {
            TensorElementType.UInt8 => Onnx.TensorProto.Types.DataType.Uint8,
            TensorElementType.Float => Onnx.TensorProto.Types.DataType.Float,
            TensorElementType.Float16 => Onnx.TensorProto.Types.DataType.Float16,
            _ => throw new NotSupportedException("Original-size preview supports U8, FP16 and FP32 images.")
        };
        var key = new RuntimeOnnxOperatorKernelKey("DebugPreviewDownload", $"type={type};shape={string.Join('x', information.Shape)}", backend, "identity;cpu-output");
        using OnnxExecutionContext execution = RuntimeOnnxOperatorKernelCache.CreateContext(key,
            () => IdentityBuilder.Build(type, information.Shape), configure: null, displayName: "debug-preview-download.onnx");
        using OrtValue output = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, information.ElementDataType, information.Shape);
        try
        {
            execution.IoBinding.BindInput(IdentityBuilder.InputName, input);
            execution.IoBinding.BindOutput(IdentityBuilder.OutputName, output);
            execution.Model.Session.RunWithBinding(execution.RunOptions, execution.IoBinding);
            execution.IoBinding.SynchronizeBoundOutputs();
            return TensorImagePreviewRenderer.CreatePreview(output);
        }
        finally { execution.IoBinding.ClearBoundInputs(); execution.IoBinding.ClearBoundOutputs(); }
    }
}
