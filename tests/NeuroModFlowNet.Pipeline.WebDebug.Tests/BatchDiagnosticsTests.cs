using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline.ONNX;
using NeuroModFlowNet.Pipeline.WebDebug;

namespace NeuroModFlowNet.Pipeline.WebDebug.Tests;

public sealed class BatchDiagnosticsTests
{
    [Fact]
    public void ExpiredLeaseBypassesRecordingAndKeepsAssemblerOwnership()
    {
        var clock = new ManualTimeProvider();
        var diagnostics = new BatchDiagnostics(clock);
        var original = new StubAssembler();
        using IOrtValueBatchInputAssembler observer = diagnostics.ObserveInput("test", "test.onnx", InferenceBackend.Cpu,
            new OnnxBatchedResourceOptions(), original);
        observer.Assemble([null!], InferenceBackend.Cpu, TestContext.Current.CancellationToken);
        Assert.Empty(diagnostics.Snapshot().Endpoints.Single().Samples);
        diagnostics.Snapshot(renew: true);
        OrtValueBatchInput batch = observer.Assemble([null!], InferenceBackend.Cpu, TestContext.Current.CancellationToken);
        Assert.False(batch.OwnsValue);
        Assert.Equal(8, batch.BatchSize);
        Assert.Equal([8], batch.RequestItemCounts);
        clock.Timestamp += 6000;
        observer.Assemble([null!], InferenceBackend.Cpu, TestContext.Current.CancellationToken);
        Assert.False(diagnostics.IsEnabled);
        Assert.Single(diagnostics.Snapshot().Endpoints.Single().Samples);
        Assert.Equal(3, original.Calls);
        observer.Dispose();
        Assert.True(original.Disposed);
    }

    [Fact]
    public void HistoryIsBoundedAndRetainsRequestVersusTensorCounts()
    {
        var statistics = new BatchEndpointStatistics("test", "test.onnx", "CPU", new OnnxBatchedResourceOptions(MaxBatchSize: 4));
        int[] rows = [3, 5];
        for(int index = 0; index < 300; index++)
        {
            statistics.Prepared(2, 8, rows, Stopwatch.GetTimestamp());
            statistics.Decoded(success: true);
        }
        rows[0] = 99;
        BatchEndpointSnapshot snapshot = statistics.Snapshot();
        Assert.Equal(300, snapshot.Prepared);
        Assert.Equal(300, snapshot.Completed);
        Assert.Equal(256, snapshot.Samples.Length);
        Assert.Equal(45, snapshot.Samples[0].Sequence);
        Assert.Equal(300, snapshot.Samples[^1].Sequence);
        Assert.All(snapshot.Samples, sample =>
        {
            Assert.Equal(2, sample.RequestCount);
            Assert.Equal(8, sample.TensorRows);
            Assert.Equal([3, 5], sample.RequestRows);
            Assert.Equal("completed", sample.State);
        });
    }

    [Fact]
    public void AdmittedBatchFinishesAfterLeaseExpiryWithoutChangingDecoderResults()
    {
        var clock = new ManualTimeProvider();
        var diagnostics = new BatchDiagnostics(clock);
        using IOrtValueBatchInputAssembler assembler = diagnostics.ObserveInput("test", "test.onnx", InferenceBackend.Cpu,
            new OnnxBatchedResourceOptions(), new StubAssembler());
        int[][] results = [[42]];
        IOrtValueBatchOutputDecoder<int> decoder = diagnostics.ObserveOutput<int>("test", new StubDecoder(results));
        diagnostics.Snapshot(renew: true);
        assembler.Assemble([null!], InferenceBackend.Cpu, TestContext.Current.CancellationToken);
        clock.Timestamp += 6000;
        Assert.Same(results, decoder.Decode(null!, null!, "output", 1, [8]));
        Assert.False(diagnostics.IsEnabled);
        Assert.Equal(1, diagnostics.Snapshot().Endpoints.Single().Completed);
    }

    [Fact]
    public void DecoderExceptionAndInterruptedRunAreNeverCountedAsCompletion()
    {
        var diagnostics = new BatchDiagnostics();
        using IOrtValueBatchInputAssembler assembler = diagnostics.ObserveInput("test", "test.onnx", InferenceBackend.Cpu,
            new OnnxBatchedResourceOptions(), new StubAssembler());
        IOrtValueBatchOutputDecoder<int> decoder = diagnostics.ObserveOutput<int>("test", new ThrowingDecoder());
        diagnostics.Snapshot(renew: true);
        assembler.Assemble([null!], InferenceBackend.Cpu, TestContext.Current.CancellationToken);
        Assert.Throws<InvalidOperationException>(() => decoder.Decode(null!, null!, "output", 1, [8]));
        assembler.Assemble([null!], InferenceBackend.Cpu, TestContext.Current.CancellationToken);
        assembler.Dispose();
        BatchEndpointSnapshot snapshot = diagnostics.Snapshot().Endpoints.Single();
        Assert.Equal(0, snapshot.Completed);
        Assert.Equal(1, snapshot.DecodeFailures);
        Assert.Equal("decode-failed", snapshot.Samples[0].State);
        Assert.Equal("interrupted", snapshot.Samples[1].State);
    }

    sealed class ManualTimeProvider : TimeProvider
    {
        public long Timestamp { get; set; }
        public override long TimestampFrequency => 1000;
        public override long GetTimestamp() => Timestamp;
    }

    sealed class StubAssembler : IOrtValueBatchInputAssembler
    {
        public int Calls { get; private set; }
        public bool Disposed { get; private set; }
        public OrtValueBatchInput Assemble(IReadOnlyList<OrtValue> inputs, InferenceBackend executionBackend, CancellationToken cancellationToken)
        {
            Calls++;
            return new(null!, 8, OwnsValue: false, [8]);
        }
        public void Dispose() => Disposed = true;
    }

    sealed class ThrowingDecoder : IOrtValueBatchOutputDecoder<int>
    {
        public IReadOnlyList<int[]> Decode(OrtValue output, OnnxModel model, string outputName, int requestCount, IReadOnlyList<int>? requestItemCounts) =>
            throw new InvalidOperationException("decoder failed");
    }

    sealed class StubDecoder(IReadOnlyList<int[]> results) : IOrtValueBatchOutputDecoder<int>
    {
        public IReadOnlyList<int[]> Decode(OrtValue output, OnnxModel model, string outputName, int requestCount, IReadOnlyList<int>? requestItemCounts) => results;
    }
}
