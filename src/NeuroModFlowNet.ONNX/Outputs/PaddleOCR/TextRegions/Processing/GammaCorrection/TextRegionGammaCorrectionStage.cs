using OpenCvSharp;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Gamma-correction normalization stage for OCR text-region crops.
/// RU: Шаг gamma-коррекции для нормализации OCR-кропов текстовых областей.
/// </summary>
/// <remarks>
/// EN: The lookup table is rebuilt only when gamma changes, so the per-frame path stays as a single OpenCV LUT call.
/// RU: Lookup table перестраивается только при изменении gamma, поэтому на кадр остается один вызов OpenCV LUT.
/// </remarks>
public sealed class TextRegionGammaCorrectionStage : ITextRegionProcessingStage, ITextRegionGammaCorrectionSettings, IDisposable
{
    readonly Mat lookupTable = new(1, 256, MatType.CV_8UC1);
    double gamma;

    public string Name { get; } = "Gamma";

    public TextRegionGammaCorrectionStage()
    {
        Gamma = 1.0;
    }

    public TextRegionGammaCorrectionStage(TextRegionGammaCorrectionOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);

        Gamma = options.Gamma;
    }

    public double Gamma
    {
        get => gamma;
        set
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(value);

            gamma = value;
            RebuildLookupTable();
        }
    }

    void RebuildLookupTable()
    {
        for(int value = 0; value < 256; value++)
        {
            double corrected = Math.Pow(value / 255.0, 1.0 / Gamma) * 255.0;
            lookupTable.Set(0, value, (byte)Math.Clamp((int)Math.Round(corrected), 0, 255));
        }
    }

    public Mat Process(Mat source)
    {
        ArgumentNullException.ThrowIfNull(source);

        var result = new Mat();
        Cv2.LUT(source, lookupTable, result);
        return result;
    }

    public void Dispose()
    {
        lookupTable.Dispose();
    }
}
