using OpenCvSharp;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Ordered post-processing chain for OCR text-region crops.
/// RU: Упорядоченная цепочка постобработки OCR-кропов текстовых областей.
/// </summary>
/// <remarks>
/// EN: The pipeline is mutable for lab/demo controls, but each <see cref="Process"/> call works on a snapshot.
/// That keeps UI reconfiguration isolated from the current frame and avoids holding locks while OpenCV work runs.
/// RU: Pipeline изменяемый для lab/demo-настроек, но каждый вызов <see cref="Process"/> работает со snapshot.
/// Так изменение настроек из UI не вмешивается в текущий кадр, а lock не удерживается во время работы OpenCV.
/// </remarks>
public sealed class TextRegionProcessingPipeline : ITextRegionProcessingStage, IDisposable
{
    readonly List<ITextRegionProcessingStage> stages;
    readonly object syncRoot = new();

    public string Name { get; } = "Pipeline";

    public TextRegionProcessingPipeline(params ITextRegionProcessingStage[] stages)
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

    public IReadOnlyList<ITextRegionProcessingStage> GetStages()
    {
        lock(syncRoot)
            return stages.ToArray();
    }

    public void Add(ITextRegionProcessingStage stage)
    {
        ArgumentNullException.ThrowIfNull(stage);

        lock(syncRoot)
            stages.Add(stage);
    }

    public void Insert(int index, ITextRegionProcessingStage stage)
    {
        ArgumentNullException.ThrowIfNull(stage);

        lock(syncRoot)
            stages.Insert(index, stage);
    }

    public ITextRegionProcessingStage RemoveAt(int index)
    {
        lock(syncRoot)
        {
            ITextRegionProcessingStage stage = stages[index];
            stages.RemoveAt(index);
            return stage;
        }
    }

    public bool Remove(ITextRegionProcessingStage stage)
    {
        ArgumentNullException.ThrowIfNull(stage);

        lock(syncRoot)
            return stages.Remove(stage);
    }

    public void Move(int sourceIndex, int targetIndex)
    {
        lock(syncRoot)
        {
            ITextRegionProcessingStage stage = stages[sourceIndex];
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

        ITextRegionProcessingStage[] stagesSnapshot;
        lock(syncRoot)
            stagesSnapshot = [.. stages];

        Mat current = source.Clone();

        foreach(ITextRegionProcessingStage stage in stagesSnapshot)
        {
            Mat next = stage.Process(current);
            current.Dispose();
            current = next;
        }

        return current;
    }

    public void Dispose()
    {
        ITextRegionProcessingStage[] stagesSnapshot;
        lock(syncRoot)
            stagesSnapshot = [.. stages];

        foreach(ITextRegionProcessingStage stage in stagesSnapshot)
        {
            if(stage is IDisposable disposableStage)
                disposableStage.Dispose();
        }
    }
}
