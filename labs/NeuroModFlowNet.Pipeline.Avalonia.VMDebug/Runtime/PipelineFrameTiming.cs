using System.Diagnostics;

namespace NeuroModFlowNet.Pipeline.Avalonia.VMDebug.Runtime;

internal sealed class PipelineFrameTiming
{
    readonly Stopwatch stopwatch = Stopwatch.StartNew();

    long detectionStartTicks;
    long roiStartTicks;
    long recognitionStartTicks;

    public double DetectionMilliseconds { get; private set; }

    public double RoiMilliseconds { get; private set; }

    public double RecognitionMilliseconds { get; private set; }

    public int RecognitionItemCount { get; set; }

    public void StartDetection() => detectionStartTicks = stopwatch.ElapsedTicks;

    public void StopDetection() => DetectionMilliseconds = ElapsedMilliseconds(detectionStartTicks);

    public void StartRoi() => roiStartTicks = stopwatch.ElapsedTicks;

    public void StopRoi() => RoiMilliseconds = ElapsedMilliseconds(roiStartTicks);

    public void StartRecognition() => recognitionStartTicks = stopwatch.ElapsedTicks;

    public void StopRecognition() => RecognitionMilliseconds = ElapsedMilliseconds(recognitionStartTicks);

    double ElapsedMilliseconds(long startTicks) =>
        (stopwatch.ElapsedTicks - startTicks) * 1000.0 / Stopwatch.Frequency;
}
