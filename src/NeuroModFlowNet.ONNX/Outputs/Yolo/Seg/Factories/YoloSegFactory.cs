using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX.Converters.Algorithms;
using NeuroModFlowNet.ONNX.Converters.Images;
using NeuroModFlowNet.ONNX.Converters.Nchw;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Factory for YOLO Segmentation runners.
/// RU: Фабрика раннеров YOLO сегментации.
/// </summary>
public static class YoloSegFactory
{
    // ──────────────────────────────── Single Mat, FP32 ────────────────────────────────

    public static ImageRunner<Mat, YoloSegResult_FP32_Mask32, ConverterMatSingleNchw<float, PosCvdnnFP32>, YoloSegFP32SingleExtractor> Single_PosCvdnn_FP32(OnnxRuntimeContext context) => new(context);
    public static ImageRunner<Mat, YoloSegResult_FP32_Mask32, ConverterMatSingleNchw<float, SymCvdnnFP32>, YoloSegFP32SingleExtractor> Single_SymCvdnn_FP32(OnnxRuntimeContext context) => new(context);
    public static ImageRunner<Mat, YoloSegResult_FP32_Mask32, ConverterMatSingleBgrDirectU8, YoloSegFP32SingleExtractor> Single_BgrDirect_FP32(OnnxRuntimeContext context) => new(context);

    // ──────────────────────────────── Single Mat, FP16 (mixed output) ─────────────────

    public static ImageRunner<Mat, YoloSegResult_FP16_Mask32, ConverterMatSingleNchw<Float16, PosCvdnnFP16>, YoloSegFP16SingleExtractor> Single_PosCvdnn_FP16(OnnxRuntimeContext context) => new(context);
    public static ImageRunner<Mat, YoloSegResult_FP16_Mask32, ConverterMatSingleNchw<Float16, SymCvdnnFP16>, YoloSegFP16SingleExtractor> Single_SymCvdnn_FP16(OnnxRuntimeContext context) => new(context);
    public static ImageRunner<Mat, YoloSegResult_FP16_Mask32, ConverterMatSingleBgrDirectU8, YoloSegFP16SingleExtractor> Single_BgrDirect_FP16(OnnxRuntimeContext context) => new(context);

    // ─────────────────────────── Auto-detect from metadata ───────────────────────────

    /// <summary>
    /// EN: Automatically selects converter and extractor based on model metadata (Single Mat).
    ///     Seg output type is determined by the prototype tensor (output1):
    ///     FP32 prototype → FP32 result, FP16 prototype → FP16 (mixed) result.
    /// RU: Автоматически выбирает конвертер и экстрактор по метаданным модели.
    ///     Тип результата определяется по тензору прототипов (output1):
    ///     FP32 прототипы → FP32 результат, FP16 → смешанный результат.
    /// </summary>
    public static IRunner<Mat, IBatchedResult> CreateRunner(OnnxRuntimeContext context)
    {
        var inputMeta  = context.Session.InputMetadata.Values.First();

        // Detection tensor may stay FP32 while prototype tensor is FP16.
        // In that mixed case detections must still be decoded as FP32.
        var outputMetas = context.Session.OutputMetadata.Values.ToArray();
        var detectionMeta = outputMetas[0];
        var prototypeMeta = outputMetas.Length > 1 ? outputMetas[1] : outputMetas[0];
        bool detectionIsFP16 = detectionMeta.ElementDataType == TensorElementType.Float16;
        bool prototypeIsFP16 = prototypeMeta.ElementDataType == TensorElementType.Float16;

        Type converterType = inputMeta.ElementDataType switch
        {
            TensorElementType.UInt8   => typeof(ConverterMatSingleBgrDirectU8),
            TensorElementType.Float   => typeof(ConverterMatSingleNchw<float,   PosCvdnnFP32>),
            TensorElementType.Float16 => typeof(ConverterMatSingleNchw<Float16, PosCvdnnFP16>),
            _ => throw new NotSupportedException($"Unsupported input data type: {inputMeta.ElementDataType}")
        };

        (Type resultType, Type extractorType) = detectionIsFP16 && prototypeIsFP16
            ? (typeof(YoloSegResult_FP16_Mask32), typeof(YoloSegFP16SingleExtractor))
            : prototypeIsFP16
                ? (typeof(YoloSegResult_FP32_Mask32), typeof(YoloSegFP32MixedSingleExtractor))
            : (typeof(YoloSegResult_FP32_Mask32), typeof(YoloSegFP32SingleExtractor));

        Type closedType = typeof(ImageRunner<,,,>).MakeGenericType(
            typeof(Mat),
            resultType,
            converterType,
            extractorType);

        return (IRunner<Mat, IBatchedResult>)Activator.CreateInstance(closedType, context)!;
    }
}
