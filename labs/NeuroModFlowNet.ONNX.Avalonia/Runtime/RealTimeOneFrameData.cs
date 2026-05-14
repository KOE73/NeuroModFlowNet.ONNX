using OpenCvSharp;

namespace NeuroModFlowNet.ONNX.Avalonia.Runtime;

/// <summary>
/// EN: Represents all UI-ready data produced for one realtime frame.
/// RU: Представляет все готовые для UI данные, полученные для одного realtime-кадра.
/// </summary>
/// <remarks>
/// EN: The object owns the frame and recognition ROI <see cref="Mat"/> instances and must be disposed after the UI has
/// copied them into controls. Overlay and metrics are lightweight snapshots.
/// RU: Объект владеет <see cref="Mat"/> кадра и ROI-изображениями recognition, поэтому его нужно освобождать после
/// копирования данных в controls. Overlay и метрики являются легкими snapshot-данными.
/// </remarks>
public sealed class RealTimeOneFrameData : IDisposable
{
    public RealTimeOneFrameData(
        Mat frame,
        Mat detFrame,
        FrameOverlaySnapshot overlay,
        IReadOnlyList<RealTimeRecognitionItemData> recognitionItems,
        RealTimeMetricsSnapshot metrics)
    {
        Frame = frame;
        DetFrame = detFrame;
        Overlay = overlay;
        RecognitionItems = recognitionItems;
        Metrics = metrics;
    }

    public Mat Frame { get; }
    public Mat DetFrame { get; }
    public FrameOverlaySnapshot Overlay { get; }
    public IReadOnlyList<RealTimeRecognitionItemData> RecognitionItems { get; }
    public RealTimeMetricsSnapshot Metrics { get; }

    public void Dispose()
    {
        Frame.Dispose();
        DetFrame.Dispose();
        foreach(RealTimeRecognitionItemData recognitionItem in RecognitionItems)
            recognitionItem.Dispose();
    }
}
