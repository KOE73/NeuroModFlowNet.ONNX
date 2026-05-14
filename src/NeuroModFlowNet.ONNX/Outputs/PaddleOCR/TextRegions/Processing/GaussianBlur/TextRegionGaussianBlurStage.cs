using OpenCvSharp;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Applies Gaussian blur to OCR text-region crops.
/// RU: Применяет Gaussian blur к OCR-кропам текстовых областей.
/// </summary>
public sealed class TextRegionGaussianBlurStage : ITextRegionProcessingStage, ITextRegionGaussianBlurSettings
{
    int kernelSize = 3;
    double sigma;

    public string Name { get; } = "GaussianBlur";

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

    public TextRegionGaussianBlurStage()
    {
    }

    public TextRegionGaussianBlurStage(TextRegionGaussianBlurOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);

        KernelSize = options.KernelSize;
        Sigma = options.Sigma;
    }

    public Mat Process(Mat source)
    {
        ArgumentNullException.ThrowIfNull(source);

        var result = new Mat();
        Cv2.GaussianBlur(source, result, new Size(KernelSize, KernelSize), Sigma);
        return result;
    }

    private static int NormalizeKernelSize(int value)
    {
        int kernel = Math.Clamp(value, 1, 31);
        return kernel % 2 == 0 ? kernel + 1 : kernel;
    }
}
