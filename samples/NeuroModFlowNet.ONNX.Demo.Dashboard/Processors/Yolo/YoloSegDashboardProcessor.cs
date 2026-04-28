using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.ONNX.Visualizer;
using OpenCvSharp;

namespace NeuroModFlowNet.ONNX.Demo.Dashboard;

internal sealed class YoloSegDashboardProcessor :
    DashboardFrameProcessorBase<IBatchedResult>
{
    public YoloSegDashboardProcessor(DashboardModelSettings settings)
        : base(ModelSlot.Seg, "YOLO SEG", settings.SegModelName, settings)
    {
    }

    protected override IRunner<Mat, IBatchedResult> CreateRunner(OnnxRuntimeContext context)
        => YoloSegFactory.CreateRunner(context);

    protected override void DrawResult(
        Mat target,
        in DashboardFrameInfo frameInfo,
        IBatchedResult result)
    {
        YoloSegResult_FP32_Mask32 fp32Result = result switch
        {
            YoloSegResult_FP32_Mask32 item => item,
            YoloSegResult_FP16_Mask32 item => ConvertToFP32(item),
            _ => throw new NotSupportedException($"Unsupported SEG result type: {result.GetType().Name}")
        };

        SegPainter.DrawSeg(
            target,
            fp32Result.Values,
            fp32Result.Masks,
            frameInfo.LetterboxInfo,
            GetScaleX(target, frameInfo),
            GetScaleY(target, frameInfo),
            GetClassName);
    }

    private static YoloSegResult_FP32_Mask32 ConvertToFP32(YoloSegResult_FP16_Mask32 result)
    {
        var values = new YoloSeg_FP32_XYWHSC_Mask32[result.Values.Length];
        for(int index = 0; index < result.Values.Length; index++)
        {
            YoloSeg_FP16_XYWHSC_Mask32 item = result.Values[index];
            var maskCoefficients = new InlineArray_FP32_Mask32();
            for(int maskIndex = 0; maskIndex < InlineArray_FP16_Mask_Count32.Length; maskIndex++)
                maskCoefficients[maskIndex] = (float)item.MaskCoefficients[maskIndex];

            values[index] = new YoloSeg_FP32_XYWHSC_Mask32(
                (float)item.X,
                (float)item.Y,
                (float)item.W,
                (float)item.H,
                (float)item.Score,
                (float)item.ClassId,
                maskCoefficients);
        }

        var masks = new float[result.Masks.Length][];
        for(int index = 0; index < result.Masks.Length; index++)
        {
            Half[] sourceMask = result.Masks[index];
            float[] targetMask = new float[sourceMask.Length];
            for(int maskIndex = 0; maskIndex < sourceMask.Length; maskIndex++)
                targetMask[maskIndex] = (float)sourceMask[maskIndex];

            masks[index] = targetMask;
        }

        return new YoloSegResult_FP32_Mask32
        {
            Values = values,
            Masks = masks,
            PrototypeShape = result.PrototypeShape,
        };
    }
}
