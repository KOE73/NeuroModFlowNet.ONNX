using System.Diagnostics;
using System.Text.Json;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline.Video.Nvdec;

namespace NeuroModFlowNet.Pipeline.WebDebug;

/// <summary>
/// EN: Reproducible CPU-only comparison of the incremental catalog cost; never starts cameras or a web listener.
///
/// RU: Воспроизводимое CPU-сравнение добавленной стоимости каталога, без запуска камер и HTTP-сервера.
/// </summary>
internal static class RegisterCatalogBenchmark
{
    public static void Run(string outputPath)
    {
        const int instructionCount = 40, registerCount = 32, framesPerSample = 10000, sampleCount = 7;
        var inspector = new PausedRegisterInspector(InferenceBackend.Cpu, Nv12ColorMatrix.Bt601Full);
        string[] keys = Enumerable.Range(0, registerCount).Select(index => $"register.{index}").ToArray();
        object[] values = Enumerable.Range(0, registerCount).Select(index => (object)index).ToArray();
        var global = new VmGlobalMemory();
        var gates = new VmSyncGateRegistry();
        var inactive = CreateCatalog();
        var active = CreateCatalog();
        RegisterCatalogBoundary[] inactivePoints = Points(inactive), activePoints = Points(active);
        long run = 0;

        // Warm the execution/metadata paths before comparing steady-state samples; cold discovery is reported separately.
        Frames(null, [], 1000); Frames(inactive, inactivePoints, 1000);
        var warmup = CreateCatalog(); warmup.Renew(); Frames(warmup, Points(warmup), 1000);
        active.Renew();
        var cold = Measure(() => Frames(active, activePoints, 1), 1);
        Frames(active, activePoints, 1000);
        var baseline = new List<Measurement>();
        var off = new List<Measurement>();
        var on = new List<Measurement>();
        for(int sample = 0; sample < sampleCount; sample++)
        {
            baseline.Add(Measure(() => Frames(null, [], framesPerSample), framesPerSample));
            off.Add(Measure(() => Frames(inactive, inactivePoints, framesPerSample), framesPerSample));
            active.Renew();
            on.Add(Measure(() => Frames(active, activePoints, framesPerSample), framesPerSample));
        }
        active.Renew();
        Thread.Sleep(1050);
        var refresh = Measure(() => Frames(active, activePoints, 1), 1);
        double baselineMedian = Median(baseline.Select(sample => sample.MicrosecondsPerFrame));
        var result = new
        {
            runtime = System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription,
            configuration = "Release recommended; CPU scalar registers; 40 instructions / 80 boundaries / 32 names",
            framesPerSample, sampleCount, coldDiscovery = cold, metadataRefreshFrame = refresh,
            baselineMedianMicroseconds = baselineMedian,
            inactiveMedianMicroseconds = Median(off.Select(sample => sample.MicrosecondsPerFrame)),
            activeMedianMicroseconds = Median(on.Select(sample => sample.MicrosecondsPerFrame)),
            inactiveAddedMicroseconds = Median(off.Select(sample => sample.MicrosecondsPerFrame)) - baselineMedian,
            activeAddedMicroseconds = Median(on.Select(sample => sample.MicrosecondsPerFrame)) - baselineMedian,
            inactiveAddedBytesPerFrame = Median(off.Select(sample => sample.BytesPerFrame)) - Median(baseline.Select(sample => sample.BytesPerFrame)),
            activeAddedBytesPerFrame = Median(on.Select(sample => sample.BytesPerFrame)) - Median(baseline.Select(sample => sample.BytesPerFrame)),
            baseline, inactive = off, active = on,
            limits = "Measures catalog visits, CPU metadata refresh and end-of-run tracking. Excludes HTTP snapshots/serialization, GPU metadata calls and image/value capture. Ordinary MultiCamera has no catalog calls."
        };
        string json = JsonSerializer.Serialize(result, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(Path.GetFullPath(outputPath), json);
        Console.WriteLine(json);

        RegisterCatalog CreateCatalog()
        {
            var catalog = new RegisterCatalog((context, key) => inspector.Describe(context, key, includeValue: false));
            for(int index = 0; index < instructionCount; index++)
                catalog.Register(new("benchmark", index, "write", "benchmark", [keys[index % registerCount]], [keys[index % registerCount]]));
            return catalog;
        }

        RegisterCatalogBoundary[] Points(RegisterCatalog catalog) => Enumerable.Range(0, instructionCount)
            .SelectMany(index => new[] { catalog.Boundary("benchmark", index, "before"), catalog.Boundary("benchmark", index, "after") }).ToArray();

        void Frames(RegisterCatalog? catalog, RegisterCatalogBoundary[] points, int count)
        {
            for(int frame = 0; frame < count; frame++)
            {
                using var context = new VmRunContext(new("benchmark", ++run, DateTimeOffset.UnixEpoch), global, gates);
                for(int key = 0; key < keys.Length; key++) context.Set(keys[key], values[key]);
                for(int instruction = 0; instruction < instructionCount; instruction++)
                {
                    catalog?.Visit(context, points[instruction * 2]);
                    context.Set(keys[instruction % registerCount], values[(instruction + 1) % registerCount]);
                    catalog?.Visit(context, points[instruction * 2 + 1]);
                }
                catalog?.RunCompleted(context.Identity.RunId);
            }
        }
    }

    static Measurement Measure(Action action, int frames)
    {
        long allocation = GC.GetAllocatedBytesForCurrentThread(), timestamp = Stopwatch.GetTimestamp();
        action();
        double microseconds = Stopwatch.GetElapsedTime(timestamp).TotalMicroseconds;
        return new(microseconds / frames, (GC.GetAllocatedBytesForCurrentThread() - allocation) / (double)frames);
    }

    static double Median(IEnumerable<double> samples) => samples.Order().ElementAt(samples.Count() / 2);
    internal sealed record Measurement(double MicrosecondsPerFrame, double BytesPerFrame);
}
