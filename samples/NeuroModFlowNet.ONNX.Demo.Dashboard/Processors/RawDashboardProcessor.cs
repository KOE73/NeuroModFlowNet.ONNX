using OpenCvSharp;

namespace NeuroModFlowNet.ONNX.Demo.Dashboard;

internal sealed class RawDashboardProcessor : IDashboardFrameProcessor
{
    public ModelSlot Slot => ModelSlot.Raw;
    public string Title => "RAW CAMERA";
    public bool IsEnabled => true;

    public Task InitializeAsync() => Task.CompletedTask;
    public Task ReloadAsync(DashboardModelSettings settings) => Task.CompletedTask;
    public void Process(Mat modelInput, in DashboardFrameInfo frameInfo) { }
    public void Draw(Mat target, in DashboardFrameInfo frameInfo) { }
    public void Dispose() { }
}
