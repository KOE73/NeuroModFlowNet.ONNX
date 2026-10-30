using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline.ONNX;

namespace NeuroModFlowNet.Pipeline.MultiCamera;

/// <summary>
/// EN: Optional laboratory-only decorators. The ordinary host supplies none and keeps its original components.
///
/// RU: Необязательные обёртки только для лабораторного хоста. Обычный хост использует исходные компоненты.
/// </summary>
internal interface ISharedInferenceEndpointDiagnostics
{
    IOrtValueBatchInputAssembler ObserveInput(string name, string modelPath, InferenceBackend backend,
        OnnxBatchedResourceOptions options, IOrtValueBatchInputAssembler assembler);

    IOrtValueBatchOutputDecoder<TOutput> ObserveOutput<TOutput>(string name, IOrtValueBatchOutputDecoder<TOutput> decoder);
}
