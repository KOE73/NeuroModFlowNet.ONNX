using OpenCvSharp;

namespace NeuroModFlowNet.ONNX;

public sealed class OcrRoiGammaCorrectionStage : IOcrRoiProcessingStage, IOcrRoiGammaCorrectionSettings, IDisposable
{
    readonly Mat lookupTable = new(1, 256, MatType.CV_8UC1);
    double gamma;

    public string Name { get; } = "Gamma";

    public OcrRoiGammaCorrectionStage(double gamma)
    {
        Gamma = gamma;
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
