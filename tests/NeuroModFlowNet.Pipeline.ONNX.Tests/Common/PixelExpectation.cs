using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.ONNX.Tests.Common;

public readonly record struct PixelExpectation(
    int X,
    int Y,
    Vec3b ExpectedBgr,
    string Name);
