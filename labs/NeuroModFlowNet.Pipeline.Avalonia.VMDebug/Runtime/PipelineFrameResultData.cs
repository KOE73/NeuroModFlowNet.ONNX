using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.Avalonia.VMDebug.Runtime;

/// <summary>
/// EN: DTO with the selected VM variables projected into frame debug data.
/// RU: DTO с выбранными переменными VM, спроецированными в данные отладки кадра.
/// </summary>
/// <remarks>
/// EN: This is intentionally not an Avalonia control type. It owns cloned <see cref="Mat"/> instances and can be
/// consumed by UI, tests, diagnostics, or an outer event-system module.
/// RU: Это намеренно не Avalonia control type. Объект владеет клонированными <see cref="Mat"/> и может потребляться UI,
/// тестами, диагностикой или внешним модулем событийной системы.
/// </remarks>
public sealed class PipelineFrameResultData : IDisposable
{
    public PipelineFrameResultData(
        string sourceId,
        Mat frame,
        FrameOverlaySnapshot overlay)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourceId);
        SourceId = sourceId;
        Frame = frame;
        Overlay = overlay;
    }

    public string SourceId { get; }
    public Mat Frame { get; }
    public FrameOverlaySnapshot Overlay { get; }
    public VmRunTrace? Trace { get; private set; }

    public void AttachTrace(VmRunTrace? trace) => Trace = trace;

    public void Dispose()
    {
        Frame.Dispose();
    }
}
