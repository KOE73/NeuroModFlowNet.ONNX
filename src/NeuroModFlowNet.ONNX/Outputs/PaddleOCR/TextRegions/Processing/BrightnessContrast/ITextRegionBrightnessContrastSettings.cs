namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Mutable brightness/contrast settings exposed separately from the processing implementation.
/// RU: Изменяемые настройки яркости и контраста, вынесенные отдельно от реализации обработки.
/// </summary>
/// <remarks>
/// EN: Lab and UI code can bind to this narrow contract without depending on a concrete OpenCV stage class.
/// RU: Lab и UI могут привязываться к этому узкому контракту, не завися от конкретного OpenCV-класса stage.
/// </remarks>
public interface ITextRegionBrightnessContrastSettings
{
    double Brightness { get; set; }

    double ContrastPercent { get; set; }
}
