using OpenCvSharp;

namespace NeuroModFlowNet.ONNX;

public readonly record struct ImageResizeTransform(float Scale, float OffsetX, float OffsetY)
{
    public Point2f MapPointToSource(float x, float y) =>
        new((x - OffsetX) / Scale, (y - OffsetY) / Scale);

    public float MapLengthToSource(float value) => value / Scale;
}
