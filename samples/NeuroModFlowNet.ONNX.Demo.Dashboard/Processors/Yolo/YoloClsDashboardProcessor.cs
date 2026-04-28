using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.ONNX.Visualizer;
using OpenCvSharp;

namespace NeuroModFlowNet.ONNX.Demo.Dashboard;

internal sealed class YoloClsDashboardProcessor :
    DashboardFrameProcessorBase<IBatchedResult>
{
    public YoloClsDashboardProcessor(DashboardModelSettings settings)
        : base(ModelSlot.Cls, "YOLO CLS", settings.ClsModelName, settings)
    {
    }

    protected override IRunner<Mat, IBatchedResult> CreateRunner(OnnxRuntimeContext context)
        => YoloClsFactory.CreateRunner(context);

    protected override void DrawResult(
        Mat target,
        in DashboardFrameInfo frameInfo,
        IBatchedResult result)
    {
        YoloCls clsResult = result switch
        {
            YoloCls item => item,
            _ => throw new NotSupportedException($"Unsupported CLS result type: {result.GetType().Name}")
        };

        ClsPainter.DrawCls(target, clsResult, GetClassName);
    }
}
