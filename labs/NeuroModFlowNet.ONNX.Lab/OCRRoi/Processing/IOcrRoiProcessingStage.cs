using OpenCvSharp;

namespace NeuroModFlowNet.ONNX;

public interface IOcrRoiProcessingStage
{
    string Name { get; }

    Mat Process(Mat source);
}
