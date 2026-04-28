using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.ONNX.Visualizer;
using OpenCvSharp;

namespace NeuroModFlowNet.ONNX.Demo.Dashboard;

internal sealed class YoloBoxDashboardProcessor :
    DashboardFrameProcessorBase<IDetectionResult<YoloBox>>
{
    public YoloBoxDashboardProcessor(DashboardModelSettings settings)
        : base(ModelSlot.Box, "YOLO BOX DETECT", "yolo26s", settings)
    {
    }

    protected override IRunner<Mat, IDetectionResult<YoloBox>> CreateRunner(OnnxRuntimeContext context)
        => YoloBoxFactory.CreateRunner<IDetectionResult<YoloBox>>(context);

    protected override void DrawResult(
        Mat target,
        in DashboardFrameInfo frameInfo,
        IDetectionResult<YoloBox> result)
    {
        BoxPainter.DrawBox(
            target,
            result.GetBatch(0).ToArray(),
            frameInfo.LetterboxInfo,
            GetScaleX(target, frameInfo),
            GetScaleY(target, frameInfo),
            GetClassName);
    }
}
