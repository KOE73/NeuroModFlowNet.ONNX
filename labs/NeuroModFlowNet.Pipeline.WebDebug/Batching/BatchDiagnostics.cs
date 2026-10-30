using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline.MultiCamera;
using NeuroModFlowNet.Pipeline.ONNX;

namespace NeuroModFlowNet.Pipeline.WebDebug;

/// <summary>
/// EN: Bounded, opt-in metadata observation through existing endpoint components; no tensor download or scheduler changes.
///
/// RU: Ограниченное наблюдение метаданных через существующие компоненты endpoint-а, без выгрузки тензоров и изменений планировщика.
/// </summary>
internal sealed class BatchDiagnostics(TimeProvider? timeProvider = null) : ISharedInferenceEndpointDiagnostics
{
    readonly TimeProvider clock = timeProvider ?? TimeProvider.System;
    readonly object syncRoot = new();
    readonly Dictionary<string, BatchEndpointStatistics> endpoints = new(StringComparer.Ordinal);
    long leaseUntil;
    long generation;

    public bool IsEnabled => clock.GetTimestamp() < Volatile.Read(ref leaseUntil);

    public void Reset()
    {
        lock(syncRoot) { endpoints.Clear(); generation++; }
    }

    public BatchDiagnosticsSnapshot Snapshot(bool renew = false)
    {
        if(renew) Volatile.Write(ref leaseUntil, clock.GetTimestamp() + 5 * clock.TimestampFrequency);
        lock(syncRoot)
            return new(generation, IsEnabled, BatchEndpointStatistics.Capacity,
                endpoints.Values.Select(endpoint => endpoint.Snapshot()).ToArray());
    }

    public IOrtValueBatchInputAssembler ObserveInput(string name, string modelPath, InferenceBackend backend,
        OnnxBatchedResourceOptions options, IOrtValueBatchInputAssembler assembler)
    {
        var statistics = new BatchEndpointStatistics(name, Path.GetFileName(modelPath), backend.ToString(), options);
        lock(syncRoot) endpoints.Add(name, statistics);
        return new ObservedBatchInputAssembler(assembler, this, statistics);
    }

    public IOrtValueBatchOutputDecoder<TOutput> ObserveOutput<TOutput>(string name, IOrtValueBatchOutputDecoder<TOutput> decoder)
    {
        lock(syncRoot) return new ObservedBatchOutputDecoder<TOutput>(decoder, endpoints[name]);
    }
}

internal sealed record BatchDiagnosticsSnapshot(long Generation, bool Collecting, int Capacity, BatchEndpointSnapshot[] Endpoints);
