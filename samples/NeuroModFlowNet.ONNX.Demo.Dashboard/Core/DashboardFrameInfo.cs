using NeuroModFlowNet.ONNX.Visualizer;

namespace NeuroModFlowNet.ONNX.Demo.Dashboard;

internal readonly record struct DashboardFrameInfo(
    LetterboxInfo LetterboxInfo,
    float SourceWidth,
    float SourceHeight);
