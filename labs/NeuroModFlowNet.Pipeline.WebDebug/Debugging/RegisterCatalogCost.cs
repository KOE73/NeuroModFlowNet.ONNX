using System.Diagnostics;

namespace NeuroModFlowNet.Pipeline.WebDebug;

/// <summary>
/// EN: Explicit twenty-second diagnostic measurement. No counters or allocation probes run outside its window.
///
/// RU: Явный двадцатисекундный замер. Вне его окна счётчики и пробы аллокаций не выполняются.
/// </summary>
internal sealed class RegisterCatalogCost
{
    readonly object syncRoot = new();
    long until, visits, ticks, bytes, maximum;
    public bool Requested => Volatile.Read(ref until) != 0;

    public object Start()
    {
        lock(syncRoot)
        {
            visits = ticks = bytes = maximum = 0;
            Volatile.Write(ref until, Stopwatch.GetTimestamp() + 20 * Stopwatch.Frequency);
            return Snapshot();
        }
    }

    public long Begin()
    {
        long now = Stopwatch.GetTimestamp(), deadline = Volatile.Read(ref until);
        if(now < deadline) return now;
        Interlocked.CompareExchange(ref until, 0, deadline);
        return 0;
    }

    public void Add(long started, long allocatedBefore)
    {
        long elapsed = Stopwatch.GetTimestamp() - started;
        long allocated = GC.GetAllocatedBytesForCurrentThread() - allocatedBefore;
        lock(syncRoot)
        {
            visits++; ticks += elapsed; bytes += allocated; maximum = Math.Max(maximum, elapsed);
        }
    }

    public object Snapshot()
    {
        lock(syncRoot) return new
        {
            measuring = Stopwatch.GetTimestamp() < until, visits,
            totalMilliseconds = ticks * 1000d / Stopwatch.Frequency,
            averageMicroseconds = visits == 0 ? 0 : ticks * 1_000_000d / Stopwatch.Frequency / visits,
            maximumMicroseconds = maximum * 1_000_000d / Stopwatch.Frequency, allocatedBytes = bytes
        };
    }
}
