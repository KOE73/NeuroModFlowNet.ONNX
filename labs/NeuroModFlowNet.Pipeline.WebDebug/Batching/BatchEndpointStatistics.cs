using System.Diagnostics;
using NeuroModFlowNet.Pipeline.ONNX;

namespace NeuroModFlowNet.Pipeline.WebDebug;

internal sealed class BatchEndpointStatistics(string name, string model, string backend, OnnxBatchedResourceOptions options)
{
    public const int Capacity = 256;
    readonly object syncRoot = new();
    readonly BatchExecutionSample[] history = new BatchExecutionSample[Capacity];
    long sequence;
    long completed;
    long decodeFailures;
    long pending;
    long startedAt;

    public bool HasPending => Volatile.Read(ref pending) != 0;

    // One endpoint worker executes batches serially. Snapshots never see or retain live OrtValues.
    public void Prepared(int requestCount, int tensorRows, IReadOnlyList<int>? requestRows, long timestamp)
    {
        lock(syncRoot)
        {
            InterruptPending();
            long number = ++sequence;
            history[(number - 1) % Capacity] = new(number, DateTimeOffset.UtcNow, requestCount, tensorRows,
                requestRows?.ToArray() ?? [], Stopwatch.GetElapsedTime(timestamp).TotalMilliseconds, null, "prepared");
            pending = number;
            startedAt = timestamp;
        }
    }

    public void Decoded(bool success)
    {
        lock(syncRoot)
        {
            if(pending == 0) return;
            int index = (int)((pending - 1) % Capacity);
            history[index] = history[index] with
            {
                ProcessingMilliseconds = Stopwatch.GetElapsedTime(startedAt).TotalMilliseconds,
                State = success ? "completed" : "decode-failed"
            };
            if(success) completed++; else decodeFailures++;
            pending = 0;
        }
    }

    public void InterruptPending()
    {
        lock(syncRoot)
        {
            if(pending == 0) return;
            int index = (int)((pending - 1) % Capacity);
            history[index] = history[index] with { State = "interrupted" };
            pending = 0;
        }
    }

    public BatchEndpointSnapshot Snapshot()
    {
        lock(syncRoot)
        {
            int count = (int)Math.Min(sequence, Capacity);
            var samples = new BatchExecutionSample[count];
            for(int index = 0; index < count; index++) samples[index] = history[(sequence - count + index) % Capacity];
            return new(name, model, backend, options.MaxBatchSize, options.EffectiveMaxWaitTime.TotalMilliseconds,
                options.MaxPendingRequests, sequence, completed, decodeFailures, samples);
        }
    }
}

internal readonly record struct BatchExecutionSample(long Sequence, DateTimeOffset Timestamp, int RequestCount, int TensorRows,
    int[] RequestRows, double AssemblyMilliseconds, double? ProcessingMilliseconds, string State);

internal sealed record BatchEndpointSnapshot(string Name, string Model, string Backend, int MaxRequests, double MaxWaitMilliseconds,
    int PendingCapacity, long Prepared, long Completed, long DecodeFailures, BatchExecutionSample[] Samples);
