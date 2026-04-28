using Microsoft.ML.OnnxRuntime.Tensors;
using Float16 = Microsoft.ML.OnnxRuntime.Float16;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Base class for YOLO Segmentation extractors for Float16.
/// RU: Базовый класс для экстракторов YOLO Segmentation для Float16.
/// </summary>
/// <remarks>
///  OUT : output0 | Detection | Float    | [1x300x38]
///  OUT : output1 | Prototype | Float16  | [1x32x160x160]
/// </remarks>
public abstract class YoloSegFP16ExtractorBase<TOut> : YoloSegExtractorBase<TOut>
    where TOut : IBatchedResult
{
    protected override void Check()
    {
        base.Check();
        var meta = Model.Session.OutputMetadata[MaskPrototypesOutputName];
        if(meta.ElementDataType != TensorElementType.Float16)
            throw new InvalidOperationException($"Model produces {meta.ElementDataType}, but FP16 extractor requires Float16 for mask prototypes.");
    }

    public unsafe YoloSegResult_FP16_Mask32 GetOutput()
    {
        var detectionData = Model.GetTensorDataAsSpan<float>(DetectionWithMaskCoefficientsOutputName);
        var maskData = Model.GetTensorDataAsSpan<Float16>(MaskPrototypesOutputName);

        var lineSeg = MemoryMarshal.Cast<float, YoloSeg_FP16_XYWHSC_Mask32>(detectionData);
        YoloSeg_FP16_XYWHSC_Mask32* tmp = stackalloc YoloSeg_FP16_XYWHSC_Mask32[ItemCount];

        int addIndex = 0;
        var batchDetections = lineSeg.Slice(0, ItemCount);

        for(int itemIndex = 0; itemIndex < ItemCount; itemIndex++)
            if((float)batchDetections[itemIndex].Score >= Threshold)
                tmp[addIndex++] = batchDetections[itemIndex];

        var result = new YoloSegResult_FP16_Mask32();
        result.PrototypeShape = MaskPrototypesShape;
        result.Values = new Span<YoloSeg_FP16_XYWHSC_Mask32>(tmp, addIndex).ToArray();

        var protoBatch = maskData.Slice(0, MaskPrototypesCount * MaskPixels);
        var batchMasks = new Half[addIndex][];

        for(int detectionIndex = 0; detectionIndex < addIndex; detectionIndex++)
        {
            ReadOnlySpan<Half> coeffs = result.Values[detectionIndex].MaskCoefficients;

            var mask = new Half[MaskPixels];
            Span<Half> maskSpan = mask;

            for(int prototypeIndex = 0; prototypeIndex < MaskPrototypesCount; prototypeIndex++)
            {
                float coeff = (float)coeffs[prototypeIndex];
                if(coeff == 0f) continue;

                var proto = protoBatch.Slice(prototypeIndex * MaskPixels, MaskPixels);
                for(int pixelIndex = 0; pixelIndex < MaskPixels; pixelIndex++)
                    maskSpan[pixelIndex] = (Half)((float)maskSpan[pixelIndex] + coeff * (float)proto[pixelIndex]);
            }

            // Sigmoid
            for(int pixelIndex = 0; pixelIndex < MaskPixels; pixelIndex++)
            {
                float val = (float)maskSpan[pixelIndex];
                maskSpan[pixelIndex] = (Half)(1.0f / (1.0f + MathF.Exp(-val)));
            }

            batchMasks[detectionIndex] = mask;
        }
        result.Masks = batchMasks;
        return result;
    }
}
