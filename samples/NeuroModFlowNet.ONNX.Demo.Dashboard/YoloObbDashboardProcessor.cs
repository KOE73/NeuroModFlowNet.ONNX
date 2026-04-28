using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.ONNX.Visualizer;
using OpenCvSharp;

namespace NeuroModFlowNet.ONNX.Demo.Dashboard;

internal sealed class YoloObbDashboardProcessor :
    DashboardFrameProcessorBase<IDetectionResult<YoloObb>>
{
    public YoloObbDashboardProcessor(DashboardModelSettings settings)
        : base(ModelSlot.Obb, "YOLO OBB DETECT", "yolo26s-obb", settings)
    {
    }

    protected override IRunner<Mat, IDetectionResult<YoloObb>> CreateRunner(OnnxRuntimeContext context)
        => YoloObbFactory.CreateRunner<IDetectionResult<YoloObb>>(context);

    protected override void DrawResult(
        Mat target,
        in DashboardFrameInfo frameInfo,
        IDetectionResult<YoloObb> result)
    {
        ObbPainter.DrawObb(
            target,
            result.GetBatch(0).ToArray(),
            frameInfo.LetterboxInfo,
            GetScaleX(target, frameInfo),
            GetScaleY(target, frameInfo),
            Scalar.Lime,
            GetClassName);
    }
}
