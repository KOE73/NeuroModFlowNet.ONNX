using Avalonia.Controls;
using NeuroModFlowNet.ONNX.Avalonia.Runtime;

namespace NeuroModFlowNet.ONNX.Avalonia.Controls;

/// <summary>
/// EN: Represents a compact metrics panel for realtime FPS, inference timing, ROI timing, and OCR item count.
/// RU: Представляет компактную панель метрик для FPS, времени inference, времени ROI и количества OCR-элементов.
/// </summary>
/// <remarks>
/// EN: This is currently a text-based diagnostics strip. It can later be replaced or extended with charts without changing
/// the inference engine contract.
/// RU: Сейчас это текстовая диагностическая полоса. Позже ее можно заменить или расширить графиками без изменения
/// контракта inference engine.
/// </remarks>
public partial class MetricsPanelView : UserControl
{
    public MetricsPanelView()
    {
        InitializeComponent();
    }

    public void UpdateMetrics(RealTimeMetricsSnapshot metrics)
    {
        TextBlock_Fps.Text = $"FPS {metrics.Fps,6:F1}";
        TextBlock_DetectionMilliseconds.Text = $"DET {metrics.DetectionMilliseconds,7:F2} ms";
        TextBlock_RoiMilliseconds.Text = $"ROI {metrics.RoiMilliseconds,7:F2} ms";
        TextBlock_RecognitionMilliseconds.Text = $"REC {metrics.RecognitionMilliseconds,7:F2} ms";
        TextBlock_RecognitionItemCount.Text = $"OCR {metrics.RecognitionItemCount,3}";
    }
}
