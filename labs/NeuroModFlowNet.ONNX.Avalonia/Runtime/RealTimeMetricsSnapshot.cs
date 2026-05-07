namespace NeuroModFlowNet.ONNX.Avalonia.Runtime;

/// <summary>
/// EN: Stores timing and throughput values for one displayed realtime frame.
/// RU: Хранит timing и throughput-значения для одного отображаемого realtime-кадра.
/// </summary>
/// <remarks>
/// EN: The metrics panel reads this value object directly. It contains already calculated values and does not own any
/// unmanaged resources.
/// RU: Панель метрик читает этот value object напрямую. В нем уже рассчитанные значения, unmanaged resources отсутствуют.
/// </remarks>
public readonly struct RealTimeMetricsSnapshot
{
    public RealTimeMetricsSnapshot(
        double fps,
        double detectionMilliseconds,
        double roiMilliseconds,
        double recognitionMilliseconds,
        int recognitionItemCount)
    {
        Fps = fps;
        DetectionMilliseconds = detectionMilliseconds;
        RoiMilliseconds = roiMilliseconds;
        RecognitionMilliseconds = recognitionMilliseconds;
        RecognitionItemCount = recognitionItemCount;
    }

    public double Fps { get; }
    public double DetectionMilliseconds { get; }
    public double RoiMilliseconds { get; }
    public double RecognitionMilliseconds { get; }
    public int RecognitionItemCount { get; }
}
