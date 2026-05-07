using Microsoft.ML.OnnxRuntime.Tensors;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: YOLO Segmentation extractor for models with FP32 detections and FP16 mask prototypes.
/// RU: Экстрактор YOLO Segmentation для моделей, где detections в FP32, а прототипы масок в FP16.
/// </summary>
public class YoloSegFP32MixedSingleExtractor : YoloSegFP32ExtractorBase<YoloSegResult_FP32_Mask32>
{
    protected override void Check()
    {
        base.Check();

        if(BatchCount != 1)
            throw new InvalidOperationException($"Invalid BatchCount for {nameof(YoloSegFP32MixedSingleExtractor)}: BatchCount={BatchCount}. Expected 1.");

        var prototypeMeta = Model.Session.OutputMetadata[MaskPrototypesOutputName];
        if(prototypeMeta.ElementDataType != TensorElementType.Float16)
            throw new InvalidOperationException($"Model mask prototypes produce {prototypeMeta.ElementDataType}, but mixed SEG extractor requires Float16.");
    }

    public override YoloSegResult_FP32_Mask32 Extract()
    {
        return GetOutput_Mixed();
    }
}
