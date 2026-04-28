using Microsoft.ML.OnnxRuntime.Tensors;
using Float16 = Microsoft.ML.OnnxRuntime.Float16;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Base class for YOLO Segmentation extractors for FP32.
/// RU: Базовый класс для экстракторов YOLO Segmentation для FP32.
/// </summary>
/// <remarks>
///  OUT : output0 | Detection | Float  | [1x300x38]
///  OUT : output1 | Prototype | Float  | [1x32x160x160]
/// </remarks>
public abstract class YoloSegFP32ExtractorBase<TOut> : YoloSegExtractorBase<TOut>
    where TOut : IBatchedResult
{
    protected override void Check()
    {
        base.Check();
        var meta = Model.Session.OutputMetadata[DetectionWithMaskCoefficientsOutputName];
        if(meta.ElementDataType != TensorElementType.Float)
            throw new InvalidOperationException($"Model produces {meta.ElementDataType}, but FP32 extractor requires Float.");
    }

    public unsafe YoloSegResult_FP32_Mask32 GetOutput()
    {
        var detectionData = Model.GetTensorDataAsSpan<float>(DetectionWithMaskCoefficientsOutputName);
        var prototypeData = Model.GetTensorDataAsSpan<float>(MaskPrototypesOutputName);

        var lineSeg = MemoryMarshal.Cast<float, YoloSeg_FP32_XYWHSC_Mask32>(detectionData);
        var tensorSeg = new ReadOnlyTensorSpan<YoloSeg_FP32_XYWHSC_Mask32>(lineSeg, [BatchCount, ItemCount]);

        YoloSeg_FP32_XYWHSC_Mask32* tmp = stackalloc YoloSeg_FP32_XYWHSC_Mask32[ItemCount];

        int addIndex = 0;
        for(int itemIndex = 0; itemIndex < ItemCount; itemIndex++)
            if(tensorSeg[0, itemIndex].Score >= Threshold)
                tmp[addIndex++] = tensorSeg[0, itemIndex];

        var result = new YoloSegResult_FP32_Mask32();
        result.PrototypeShape = MaskPrototypesShape;
        result.Values = new Span<YoloSeg_FP32_XYWHSC_Mask32>(tmp, addIndex).ToArray();

        var protoBatch = prototypeData.Slice(0, MaskPrototypesCount * MaskPixels);
        var batchMasks = new float[addIndex][];

        for(int detectionIndex = 0; detectionIndex < addIndex; detectionIndex++)
        {
            ReadOnlySpan<float> coeffs = result.Values[detectionIndex].MaskCoefficients;
            var mask = new float[MaskPixels];
            Span<float> maskSpan = mask;

            for(int prototypeIndex = 0; prototypeIndex < MaskPrototypesCount; prototypeIndex++)
            {
                float coeff = coeffs[prototypeIndex];
                if(coeff == 0f) continue;

                var proto = protoBatch.Slice(prototypeIndex * MaskPixels, MaskPixels);
                TensorPrimitives.MultiplyAdd(proto, coeff, (ReadOnlySpan<float>)maskSpan, maskSpan);
            }

            TensorPrimitives.Sigmoid(maskSpan, maskSpan);
            batchMasks[detectionIndex] = mask;
        }

        result.Masks = batchMasks;
        return result;
    }

    public unsafe YoloSegResult_FP32_Mask32 GetOutput_Mixed()
    {
        var data = Model.GetTensorDataAsSpan<float>(DetectionWithMaskCoefficientsOutputName);
        var data1 = Model.GetTensorDataAsSpan<Float16>(MaskPrototypesOutputName);

        var lineSeg = MemoryMarshal.Cast<float, YoloSeg_FP32_XYWHSC_Mask32>(data);
        var tensorSeg = new ReadOnlyTensorSpan<YoloSeg_FP32_XYWHSC_Mask32>(lineSeg, [BatchCount, ItemCount]);

        YoloSeg_FP32_XYWHSC_Mask32* tmp = stackalloc YoloSeg_FP32_XYWHSC_Mask32[ItemCount];

        int addIndex = 0;
        for(int itemIndex = 0; itemIndex < ItemCount; itemIndex++)
            if(tensorSeg[0, itemIndex].Score >= Threshold)
                tmp[addIndex++] = tensorSeg[0, itemIndex];

        var result = new YoloSegResult_FP32_Mask32();
        result.PrototypeShape = MaskPrototypesShape;
        result.Values = new Span<YoloSeg_FP32_XYWHSC_Mask32>(tmp, addIndex).ToArray();

        var protoBatch = data1.Slice(0, MaskPrototypesCount * MaskPixels);
        var batchMasks = new float[addIndex][];

        for(int detectionIndex = 0; detectionIndex < addIndex; detectionIndex++)
        {
            ReadOnlySpan<float> coeffs = result.Values[detectionIndex].MaskCoefficients;
            var mask = new float[MaskPixels];
            Span<float> maskSpan = mask;

            for(int prototypeIndex = 0; prototypeIndex < MaskPrototypesCount; prototypeIndex++)
            {
                float coeff = coeffs[prototypeIndex];
                if(coeff == 0f) continue;

                var proto = protoBatch.Slice(prototypeIndex * MaskPixels, MaskPixels);
                for(int pixelIndex = 0; pixelIndex < MaskPixels; pixelIndex++)
                    maskSpan[pixelIndex] += coeff * (float)proto[pixelIndex];
            }

            TensorPrimitives.Sigmoid(maskSpan, maskSpan);
            batchMasks[detectionIndex] = mask;
        }

        result.Masks = batchMasks;
        return result;
    }
}
