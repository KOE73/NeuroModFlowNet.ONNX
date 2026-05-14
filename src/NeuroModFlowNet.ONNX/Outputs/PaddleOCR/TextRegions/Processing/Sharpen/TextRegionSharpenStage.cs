using OpenCvSharp;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Sharpens OCR text-region crops with unsharp masking.
/// RU: Повышает резкость OCR-кропов через unsharp masking.
/// </summary>
/// <remarks>
/// EN: This is the practical opposite of Gaussian blur for realtime tuning: one blur pass produces a low-frequency
/// image, then AddWeighted subtracts part of it from the source. It is cheaper and more predictable than custom kernels.
/// RU: Это практическая противоположность Gaussian blur для realtime-настройки: один blur-проход дает низкочастотное
/// изображение, затем AddWeighted вычитает его часть из исходника. Так дешевле и предсказуемее, чем вручную подбирать kernel.
/// </remarks>
public sealed class TextRegionSharpenStage : ITextRegionProcessingStage, ITextRegionSharpenSettings
{
    int kernelSize = 3;
    double sigma;
    double amount = 1.0;

    public string Name { get; } = "Sharpen";

    public int KernelSize
    {
        get => kernelSize;
        set => kernelSize = NormalizeKernelSize(value);
    }

    public double Sigma
    {
        get => sigma;
        set => sigma = Math.Max(0, value);
    }

    public double Amount
    {
        get => amount;
        set => amount = Math.Clamp(value, 0, 5);
    }

    public TextRegionSharpenStage()
    {
    }

    public TextRegionSharpenStage(TextRegionSharpenOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);

        KernelSize = options.KernelSize;
        Sigma = options.Sigma;
        Amount = options.Amount;
    }

    public Mat Process(Mat source)
    {
        ArgumentNullException.ThrowIfNull(source);

        using var blurred = new Mat();
        var result = new Mat();

        Cv2.GaussianBlur(source, blurred, new Size(KernelSize, KernelSize), Sigma);
        Cv2.AddWeighted(source, 1.0 + Amount, blurred, -Amount, 0, result);
        return result;
    }

    private static int NormalizeKernelSize(int value)
    {
        int kernel = Math.Clamp(value, 1, 31);
        return kernel % 2 == 0 ? kernel + 1 : kernel;
    }
}
