using OpenCvSharp;

namespace NeuroModFlowNet.ONNX;

public sealed class OcrRoiProcessingPipeline : IOcrRoiProcessingStage, IDisposable
{
    readonly List<IOcrRoiProcessingStage> stages;
    readonly object syncRoot = new();

    public string Name { get; } = "Pipeline";

    public OcrRoiProcessingPipeline(params IOcrRoiProcessingStage[] stages)
    {
        this.stages = [.. stages];
    }

    public int Count
    {
        get
        {
            lock(syncRoot)
                return stages.Count;
        }
    }

    public IReadOnlyList<IOcrRoiProcessingStage> GetStages()
    {
        lock(syncRoot)
            return stages.ToArray();
    }

    public void Add(IOcrRoiProcessingStage stage)
    {
        ArgumentNullException.ThrowIfNull(stage);

        lock(syncRoot)
            stages.Add(stage);
    }

    public void Insert(int index, IOcrRoiProcessingStage stage)
    {
        ArgumentNullException.ThrowIfNull(stage);

        lock(syncRoot)
            stages.Insert(index, stage);
    }

    public IOcrRoiProcessingStage RemoveAt(int index)
    {
        lock(syncRoot)
        {
            IOcrRoiProcessingStage stage = stages[index];
            stages.RemoveAt(index);
            return stage;
        }
    }

    public bool Remove(IOcrRoiProcessingStage stage)
    {
        ArgumentNullException.ThrowIfNull(stage);

        lock(syncRoot)
            return stages.Remove(stage);
    }

    public void Move(int sourceIndex, int targetIndex)
    {
        lock(syncRoot)
        {
            IOcrRoiProcessingStage stage = stages[sourceIndex];
            stages.RemoveAt(sourceIndex);
            stages.Insert(targetIndex, stage);
        }
    }

    public void Clear()
    {
        lock(syncRoot)
            stages.Clear();
    }

    public Mat Process(Mat source)
    {
        ArgumentNullException.ThrowIfNull(source);

        IOcrRoiProcessingStage[] stagesSnapshot;
        lock(syncRoot)
            stagesSnapshot = [.. stages];

        Mat current = source.Clone();

        foreach(IOcrRoiProcessingStage stage in stagesSnapshot)
        {
            Mat next = stage.Process(current);
            current.Dispose();
            current = next;
        }

        return current;
    }

    public void Dispose()
    {
        IOcrRoiProcessingStage[] stagesSnapshot;
        lock(syncRoot)
            stagesSnapshot = [.. stages];

        foreach(IOcrRoiProcessingStage stage in stagesSnapshot)
        {
            if(stage is IDisposable disposableStage)
                disposableStage.Dispose();
        }
    }
}
