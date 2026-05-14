using OpenCvSharp;

namespace NeuroModFlowNet.ONNX.Avalonia.Runtime;

/// <summary>
/// EN: Represents one OCR recognition row prepared for display in the Avalonia UI.
/// RU: Представляет одну OCR-строку распознавания, подготовленную для отображения в Avalonia UI.
/// </summary>
/// <remarks>
/// EN: The object owns the ROI <see cref="Mat"/> used for the preview image and must be disposed after the UI has copied
/// the image into its own control. The text is the recognition result selected by the current display mode.
/// RU: Объект владеет ROI <see cref="Mat"/>, который используется для preview-картинки, и должен быть освобожден после
/// копирования картинки в UI control. Текст соответствует текущему режиму отображения recognition.
/// </remarks>
public sealed class RealTimeRecognitionItemData : IDisposable
{
    public RealTimeRecognitionItemData(Mat roi, string text, double displayScale, RoiHeightDebugData roiHeightDebug)
    {
        Roi = roi;
        Text = text;
        DisplayScale = displayScale;
        RoiHeightDebug = roiHeightDebug;
    }

    public Mat Roi { get; }
    public string Text { get; }
    public double DisplayScale { get; }
    public RoiHeightDebugData RoiHeightDebug { get; }

    public void Dispose()
    {
        Roi.Dispose();
    }
}
