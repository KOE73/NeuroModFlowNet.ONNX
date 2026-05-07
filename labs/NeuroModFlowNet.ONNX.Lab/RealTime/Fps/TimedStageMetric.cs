using System.Diagnostics;

namespace OnnxTestLoader;

public sealed class TimedStageMetric
{
    private readonly Queue<TimedStageMetricSample> samples = [];
    private readonly TimeSpan window;

    public TimedStageMetric(TimeSpan window)
    {
        this.window = window;
    }

    public long Start() => Stopwatch.GetTimestamp();

    public void AddElapsed(long startTicks, int itemCount = 1)
    {
        long elapsedTicks = Stopwatch.GetTimestamp() - startTicks;
        double milliseconds = elapsedTicks * 1000.0 / Stopwatch.Frequency;
        Add(milliseconds, itemCount);
    }

    public T Measure<T>(Func<T> action, int itemCount = 1)
    {
        long startTicks = Start();
        try
        {
            return action();
        }
        finally
        {
            AddElapsed(startTicks, itemCount);
        }
    }

    public TimedStageMetricSnapshot Snapshot()
    {
        if(samples.Count == 0)
            return TimedStageMetricSnapshot.Empty;

        TimedStageMetricSample current = samples.Last();
        double totalMilliseconds = samples.Sum(sample => sample.Milliseconds);
        int totalItems = samples.Sum(sample => sample.ItemCount);
        int totalCalls = samples.Count;

        return new TimedStageMetricSnapshot(
            current.Milliseconds,
            totalMilliseconds / totalCalls,
            GetRate(totalCalls, totalMilliseconds),
            GetRate(totalItems, totalMilliseconds),
            GetRate(1, current.Milliseconds),
            GetRate(current.ItemCount, current.Milliseconds),
            totalItems / (double)totalCalls,
            current.ItemCount);
    }

    private void Add(double milliseconds, int itemCount)
    {
        long currentTicks = Stopwatch.GetTimestamp();
        long cutoffTicks = currentTicks - (long)(window.TotalSeconds * Stopwatch.Frequency);

        samples.Enqueue(new TimedStageMetricSample(currentTicks, milliseconds, Math.Max(0, itemCount)));

        while(samples.Count > 0 && samples.Peek().Ticks < cutoffTicks)
            samples.Dequeue();
    }

    private static double GetRate(int count, double milliseconds) =>
        count <= 0 || milliseconds <= double.Epsilon ? 0 : count * 1000.0 / milliseconds;

    private readonly record struct TimedStageMetricSample(long Ticks, double Milliseconds, int ItemCount);
}

public readonly record struct TimedStageMetricSnapshot(
    double CurrentMilliseconds,
    double AverageMilliseconds,
    double AverageCallFps,
    double AverageItemFps,
    double CurrentCallFps,
    double CurrentItemFps,
    double AverageItemsPerCall,
    int CurrentItemsPerCall)
{
    public static TimedStageMetricSnapshot Empty { get; } = new(0, 0, 0, 0, 0, 0, 0, 0);
}
