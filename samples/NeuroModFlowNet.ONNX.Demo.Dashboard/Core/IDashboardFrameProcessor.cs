using OpenCvSharp;

namespace NeuroModFlowNet.ONNX.Demo.Dashboard;

internal interface IDashboardFrameProcessor : IDisposable
{
    ModelSlot Slot { get; }
    string Title { get; }
    bool IsEnabled { get; }

    Task InitializeAsync();
    Task ReloadAsync(DashboardModelSettings settings);
    void Process(Mat modelInput, in DashboardFrameInfo frameInfo);
    void Draw(Mat target, in DashboardFrameInfo frameInfo);
}
