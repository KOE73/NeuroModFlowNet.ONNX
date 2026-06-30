using BenchmarkDotNet.Columns;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Jobs;
using System.Runtime.InteropServices;

namespace NeuroModFlowNet.ONNX.Bench;

public sealed class OcrRoiPrepareConfig : ManualConfig
{
    public OcrRoiPrepareConfig()
    {
        AddColumn(StatisticColumn.OperationsPerSecond);

        Job job = Job.Default
            .WithLaunchCount(1)
            .WithWarmupCount(4)
            .WithIterationCount(20)
            .WithInvocationCount(16)
            .WithUnrollFactor(16);

        IntPtr affinity = GetPreferredAffinity();
        if(affinity != IntPtr.Zero)
            job = job.WithAffinity(affinity);

        AddJob(job);
    }

    static IntPtr GetPreferredAffinity()
    {
        string cpuName = RuntimeInformation.OSDescription.Contains("Windows", StringComparison.OrdinalIgnoreCase)
            ? Environment.GetEnvironmentVariable("PROCESSOR_IDENTIFIER") ?? string.Empty
            : string.Empty;

        if(!cpuName.Contains("Intel", StringComparison.OrdinalIgnoreCase))
            return IntPtr.Zero;

        int totalThreads = Environment.ProcessorCount;
        long pCoreMask = (1L << Math.Min(12, totalThreads)) - 1;
        return (IntPtr)pCoreMask;
    }
}
