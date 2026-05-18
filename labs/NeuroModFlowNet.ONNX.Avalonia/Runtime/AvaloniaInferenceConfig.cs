namespace NeuroModFlowNet.ONNX.Avalonia.Runtime;

/// <summary>
/// EN: Inference threshold and behavior settings that live outside the core library Options tree.
/// RU: Пороги инференса и поведенческие настройки, которые не входят в дерево Options core-библиотеки.
/// </summary>
public sealed class AvaloniaInferenceConfig
{
    public float ObbThreshold { get; set; } = 0.3f;
}
