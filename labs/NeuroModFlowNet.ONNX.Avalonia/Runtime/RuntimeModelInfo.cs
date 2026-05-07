namespace NeuroModFlowNet.ONNX.Avalonia.Runtime;

/// <summary>
/// EN: Contains compact model metadata shown in the right settings panel.
/// RU: Содержит компактные model metadata, которые отображаются в правой панели настроек.
/// </summary>
/// <remarks>
/// EN: The text is intentionally preformatted because the UI needs a very small readable table, not a large diagnostics
/// view. Values are collected after ONNX contexts and persistent buffers are initialized.
/// RU: Текст намеренно заранее форматируется, потому что UI нужна маленькая читаемая таблица, а не большой diagnostics
/// view. Значения собираются после инициализации ONNX contexts и persistent buffers.
/// </remarks>
public sealed class RuntimeModelInfo
{
    public RuntimeModelInfo(string key, string title, string details)
    {
        Key = key;
        Title = title;
        Details = details;
    }

    public string Key { get; }
    public string Title { get; }
    public string Details { get; }
}
