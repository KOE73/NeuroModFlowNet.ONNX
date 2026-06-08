namespace NeuroModFlowNet.Pipeline.VMDebug;

/// <summary>
/// Fixed-size moving average for live timing samples.
/// </summary>
internal sealed class MovingAverage
{
    readonly Queue<double> samples = new();
    readonly int capacity;
    double sum;

    public MovingAverage(int capacity)
    {
        if(capacity <= 0)
            throw new ArgumentOutOfRangeException(nameof(capacity), "Capacity must be positive.");

        this.capacity = capacity;
    }

    public int Count => samples.Count;

    public double Last { get; private set; }

    public double Average => samples.Count == 0 ? 0 : sum / samples.Count;

    public void Add(double value)
    {
        Last = value;
        samples.Enqueue(value);
        sum += value;

        if(samples.Count <= capacity)
            return;

        sum -= samples.Dequeue();
    }
}
