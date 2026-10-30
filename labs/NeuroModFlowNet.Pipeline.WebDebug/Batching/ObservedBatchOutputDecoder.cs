using Microsoft.ML.OnnxRuntime;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline.ONNX;

namespace NeuroModFlowNet.Pipeline.WebDebug;

internal sealed class ObservedBatchOutputDecoder<TOutput>(IOrtValueBatchOutputDecoder<TOutput> inner,
    BatchEndpointStatistics statistics) : IOrtValueBatchOutputDecoder<TOutput>
{
    public IReadOnlyList<TOutput[]> Decode(OrtValue output, OnnxModel model, string outputName, int requestCount, IReadOnlyList<int>? requestItemCounts)
    {
        if(!statistics.HasPending) return inner.Decode(output, model, outputName, requestCount, requestItemCounts);
        bool success = false;
        try
        {
            IReadOnlyList<TOutput[]> results = inner.Decode(output, model, outputName, requestCount, requestItemCounts);
            success = true;
            return results;
        }
        finally { statistics.Decoded(success); }
    }
}
