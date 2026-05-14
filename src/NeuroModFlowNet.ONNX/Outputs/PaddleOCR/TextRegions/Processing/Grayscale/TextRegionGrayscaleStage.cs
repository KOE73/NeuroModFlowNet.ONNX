using OpenCvSharp;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Converts OCR text-region crops to grayscale while keeping BGR output for the recognition converter.
/// RU: Переводит OCR-кропы в grayscale, но возвращает BGR-изображение для recognition converter.
/// </summary>
public sealed class TextRegionGrayscaleStage : ITextRegionProcessingStage, ITextRegionGrayscaleSettings
{
    public string Name { get; } = "Grayscale";

    public bool UseRed { get; set; } = true;

    public bool UseGreen { get; set; } = true;

    public bool UseBlue { get; set; } = true;

    public TextRegionGrayscaleStage()
    {
    }

    public TextRegionGrayscaleStage(TextRegionGrayscaleOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);

        UseRed = options.UseRed;
        UseGreen = options.UseGreen;
        UseBlue = options.UseBlue;
    }

    public Mat Process(Mat source)
    {
        ArgumentNullException.ThrowIfNull(source);

        using Mat gray = CreateGray(source);
        var result = new Mat();
        Cv2.CvtColor(gray, result, ColorConversionCodes.GRAY2BGR);
        return result;
    }

    private Mat CreateGray(Mat source)
    {
        if(source.Channels() == 1)
            return source.Clone();

        bool useRed = UseRed;
        bool useGreen = UseGreen;
        bool useBlue = UseBlue;

        if(!useRed && !useGreen && !useBlue)
        {
            useRed = true;
            useGreen = true;
            useBlue = true;
        }

        if(useRed && useGreen && useBlue)
        {
            var gray = new Mat();
            Cv2.CvtColor(source, gray, ColorConversionCodes.BGR2GRAY);
            return gray;
        }

        Mat[] channels = Cv2.Split(source);
        try
        {
            var gray32 = new Mat();
            int channelCount = (useBlue ? 1 : 0) + (useGreen ? 1 : 0) + (useRed ? 1 : 0);
            double weight = 1.0 / channelCount;

            if(useBlue)
                AccumulateChannel(channels[0], gray32, weight);
            if(useGreen)
                AccumulateChannel(channels[1], gray32, weight);
            if(useRed)
                AccumulateChannel(channels[2], gray32, weight);

            var gray = new Mat();
            gray32.ConvertTo(gray, MatType.CV_8UC1);
            gray32.Dispose();
            return gray;
        }
        finally
        {
            foreach(Mat channel in channels)
                channel.Dispose();
        }
    }

    private static void AccumulateChannel(Mat channel, Mat accumulator, double weight)
    {
        using var channel32 = new Mat();
        channel.ConvertTo(channel32, MatType.CV_32FC1, weight);

        if(accumulator.Empty())
        {
            channel32.CopyTo(accumulator);
            return;
        }

        Cv2.Add(accumulator, channel32, accumulator);
    }
}
