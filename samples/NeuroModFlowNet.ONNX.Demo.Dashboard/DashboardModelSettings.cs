using NeuroModFlowNet.ONNX;

namespace NeuroModFlowNet.ONNX.Demo.Dashboard;

internal sealed record DashboardModelSettings(
    InferenceBackend Backend,
    int InputSize,
    string Precision,
    bool IsByteBgr);
