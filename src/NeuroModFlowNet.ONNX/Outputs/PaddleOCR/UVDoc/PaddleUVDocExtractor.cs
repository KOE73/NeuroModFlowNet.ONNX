namespace NeuroModFlowNet.ONNX;

using System;
using OpenCvSharp;

public class PaddleUVDocExtractor : ResultExtractorBase<Mat>
{
    public int OutputImageHeight { get; private set; }
    public int OutputImageWidth { get; private set; }

    protected override void Init()
    {
        OutputImageHeight = (int)Model.ModelOutputShapes[Model.PrimaryOutputName][2];
        OutputImageWidth = (int)Model.ModelOutputShapes[Model.PrimaryOutputName][3];
    }

    public ReadOnlySpan<float> GetOutputAsSpan()
    {
        return Model.GetTensorDataAsSpan<float>(Model.PrimaryOutputName);
    }

    public override Mat Extract()
    {
        return GetOutputAsMat32FC1();
    }

    public unsafe Mat GetOutputAsMat32FC1()
    {
        var output = GetOutputAsSpan();
        int channelSize = OutputImageWidth * OutputImageHeight;

        fixed(float* p = output)
        {
            using Mat r = Mat.FromPixelData(OutputImageHeight, OutputImageWidth, MatType.CV_32FC1, (nint)p);
            using Mat g = Mat.FromPixelData(OutputImageHeight, OutputImageWidth, MatType.CV_32FC1, (nint)(p + channelSize));
            using Mat b = Mat.FromPixelData(OutputImageHeight, OutputImageWidth, MatType.CV_32FC1, (nint)(p + 2 * channelSize));

            Mat merged = new Mat();
            Cv2.Merge(new[] { b, g, r }, merged);

            return merged;
        }
    }

    public unsafe Mat GetOutputAsMat8UC3()
    {
        using Mat merged32 = GetOutputAsMat32FC1();
        Mat result8U = new Mat();
        merged32.ConvertTo(result8U, MatType.CV_8UC3, 255.0);
        return result8U;
    }
}
