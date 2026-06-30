using Microsoft.ML.OnnxRuntime;
using NeuroModFlowNet.ONNX;

namespace NeuroModFlowNet.Pipeline.ONNX;

public interface IOrtValueBatchOutputDecoder<TOutput>
{
    IReadOnlyList<TOutput[]> Decode(
        OrtValue output,
        OnnxModel model,
        string outputName,
        int requestCount,
        IReadOnlyList<int>? requestItemCounts);
}
