using NeuroModFlowNet.Pipeline;
using NeuroModFlowNet.Pipeline.ONNX;

namespace NeuroModFlowNet.ONNX.Tests;

public sealed class OnnxBatchedResourceTests
{
    [Fact]
    public async Task OnnxBatchedResource_RoutesOutputsToOriginalRequests()
    {
        await using var resource = new FakeBatchedResource(
            "fake",
            new OnnxBatchedResourceOptions(MaxBatchSize: 4, MaxWaitTime: TimeSpan.FromMilliseconds(80)));

        Task<string>[] tasks =
        [
            resource.ExecuteAsync(CreateIdentity(0), 10, CancellationToken.None).AsTask(),
            resource.ExecuteAsync(CreateIdentity(1), 20, CancellationToken.None).AsTask(),
            resource.ExecuteAsync(CreateIdentity(2), 30, CancellationToken.None).AsTask(),
            resource.ExecuteAsync(CreateIdentity(3), 40, CancellationToken.None).AsTask()
        ];

        string[] outputs = await Task.WhenAll(tasks);

        Assert.Equal(["value:10", "value:20", "value:30", "value:40"], outputs);
        Assert.Contains(resource.Batches, batch => batch.SequenceEqual([10, 20, 30, 40]));
    }

    static VmRunIdentity CreateIdentity(long runId) =>
        new("source:0", runId, DateTimeOffset.UtcNow);

    sealed class FakeBatchedResource : OnnxBatchedResourceBase<int, string>
    {
        readonly List<int[]> batches = [];

        public FakeBatchedResource(string name, OnnxBatchedResourceOptions options)
            : base(name, options)
        {
        }

        public IReadOnlyList<int[]> Batches => batches;

        protected override ValueTask<IReadOnlyList<string>> ExecuteBatchAsync(
            IReadOnlyList<int> inputs,
            CancellationToken cancellationToken)
        {
            lock(batches)
                batches.Add(inputs.ToArray());

            IReadOnlyList<string> outputs = inputs.Select(input => $"value:{input}").ToArray();
            return ValueTask.FromResult(outputs);
        }
    }
}

