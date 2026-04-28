using OpenCvSharp;

namespace NeuroModFlowNet.ONNX.Demo.Dashboard;

internal readonly record struct DashboardView(
    string Title,
    Action<Mat> DrawAction);
