using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.ONNX.Visualizer;
using OpenCvSharp;

namespace NeuroModFlowNet.ONNX.Demo.Dashboard;

internal sealed class YoloPoseDashboardProcessor :
    DashboardFrameProcessorBase<IDetectionResult<YoloPose>>
{
    public YoloPoseDashboardProcessor(DashboardModelSettings settings)
        : base(ModelSlot.Pose, "YOLO POSE", settings.PoseModelName, settings)
    {
    }

    protected override IRunner<Mat, IDetectionResult<YoloPose>> CreateRunner(OnnxRuntimeContext context)
        => YoloPoseFactory.CreateRunner(context);

    protected override void DrawResult(
        Mat target,
        in DashboardFrameInfo frameInfo,
        IDetectionResult<YoloPose> result)
    {
        YoloPosePainter.DrawPose(
            target,
            result.GetBatch(0).ToArray(),
            frameInfo.LetterboxInfo,
            GetScaleX(target, frameInfo),
            GetScaleY(target, frameInfo),
            GetClassName);
    }
}
