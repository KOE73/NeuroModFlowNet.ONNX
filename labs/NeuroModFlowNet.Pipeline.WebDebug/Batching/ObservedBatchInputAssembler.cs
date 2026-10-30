using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline.ONNX;

namespace NeuroModFlowNet.Pipeline.WebDebug;

internal sealed class ObservedBatchInputAssembler(IOrtValueBatchInputAssembler inner, BatchDiagnostics diagnostics,
    BatchEndpointStatistics statistics) : IOrtValueBatchInputAssembler
{
    public OrtValueBatchInput Assemble(IReadOnlyList<OrtValue> inputs, InferenceBackend executionBackend, CancellationToken cancellationToken)
    {
        // The ordinary MultiCamera host has no wrapper. In the debugger an expired lease bypasses all recording.
        if(statistics.HasPending) statistics.InterruptPending();
        if(!diagnostics.IsEnabled) return inner.Assemble(inputs, executionBackend, cancellationToken);
        long timestamp = Stopwatch.GetTimestamp();
        OrtValueBatchInput batch = inner.Assemble(inputs, executionBackend, cancellationToken);
        statistics.Prepared(inputs.Count, batch.BatchSize, batch.RequestItemCounts, timestamp);
        return batch;
    }

    public void Dispose()
    {
        statistics.InterruptPending();
        inner.Dispose();
    }
}
