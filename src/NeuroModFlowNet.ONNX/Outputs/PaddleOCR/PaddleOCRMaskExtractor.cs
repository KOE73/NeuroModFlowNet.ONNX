namespace NeuroModFlowNet.ONNX;

public class PaddleOCRMaskExtractor : ResultExtractorBase<Mat>
{
    public override unsafe Mat Extract()
    {
        return default!;
        // TODO Востановить!
        //var data = Model.GetTensorDataAsSpan<float>();
        //var shape = Model.ModelOutputShapes[Model.PrimaryOutputName];
        //int h = (int)shape[2], w = (int)shape[3];
        //fixed (float* p = data)
        //{
        //    using Mat mask32F = Cv2.MatFromPixelData(h, w, MatType.CV_32FC1, (nint)p);
        //    Mat mask8U = new Mat();
        //    mask32F.ConvertTo(mask8U, MatType.CV_8UC1, 255.0);
        //    return mask8U;
        //}
    }
}
