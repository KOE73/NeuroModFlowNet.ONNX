namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Base class for PaddleOCR Detection extractors.
/// <br/>
/// RU: Базовый класс для экстракторов PaddleOCR Detection.
/// </summary>
public abstract class PaddleOCRDetExtractorBase<TOut> : ResultExtractorBase<TOut>
{

    protected override void Init()
    {
        var shape = Model.ModelOutputShapes[Model.PrimaryOutputName];
        ModelOutputBatchCount = (int)shape[0];
        ModelOutputImageHeight = (int)shape[2];
        ModelOutputImageWidth = (int)shape[3];
        ModelOneImageSize = ModelOutputImageHeight * ModelOutputImageWidth;
    }

    public int ModelOutputBatchCount { get; private set; }
    public int ModelOutputImageHeight { get; private set; }
    public int ModelOutputImageWidth { get; private set; }
    public int ModelOneImageSize { get; private set; }


    public int BatchCount => ModelOutputBatchCount < 0 ? (int)Model.GetRealOutputShape(Model.PrimaryOutputName)[0] : ModelOutputBatchCount;
    public int OutputImageHeight => ModelOutputImageHeight < 0 ? (int)Model.GetRealOutputShape(Model.PrimaryOutputName)[2] : ModelOutputImageHeight;
    public int OutputImageWidth => ModelOutputImageWidth < 0 ? (int)Model.GetRealOutputShape(Model.PrimaryOutputName)[3] : ModelOutputImageWidth;
    public int OneImageSize=> OutputImageHeight * OutputImageWidth;



    public abstract Mat GetOutputAsMat_8UC1(int batchIndex = 0);

    public Mat GetOutputAsMat_8UC3_RGB(int batchIndex = 0)
    {
        using var mask8UC1 = GetOutputAsMat_8UC1(batchIndex);
        var mask8UC3 = new Mat();
        Cv2.CvtColor(mask8UC1, mask8UC3, ColorConversionCodes.GRAY2RGB);
        return mask8UC3;
    }

}
