using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX.Converters.Algorithms;
using NeuroModFlowNet.ONNX.Converters.Images;
using NeuroModFlowNet.ONNX.Converters.Nchw;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Factory for YOLO Classification runners.
/// RU: Фабрика раннеров YOLO классификации.
/// </summary>
public static class YoloClsFactory
{
    // ──────────────────────────────── Single Mat, FP32 ────────────────────────────────

    public static ImageRunner<Mat, YoloCls, ConverterMatSingleNchw<float, PosCvdnnFP32>, YoloClsFP32SingleExtractor> Single_PosCvdnn_FP32(OnnxRuntimeContext context) => new(context);
    public static ImageRunner<Mat, YoloCls, ConverterMatSingleNchw<float, SymCvdnnFP32>, YoloClsFP32SingleExtractor> Single_SymCvdnn_FP32(OnnxRuntimeContext context) => new(context);
    public static ImageRunner<Mat, YoloCls, ConverterMatSingleBgrDirectU8, YoloClsFP32SingleExtractor> Single_BgrDirect_FP32(OnnxRuntimeContext context) => new(context);

    // ──────────────────────────────── Single Mat, FP16 ────────────────────────────────

    public static ImageRunner<Mat, YoloCls, ConverterMatSingleNchw<Float16, PosCvdnnFP16>, YoloClsFP16SingleExtractor> Single_PosCvdnn_FP16(OnnxRuntimeContext context) => new(context);
    public static ImageRunner<Mat, YoloCls, ConverterMatSingleNchw<Float16, SymCvdnnFP16>, YoloClsFP16SingleExtractor> Single_SymCvdnn_FP16(OnnxRuntimeContext context) => new(context);
    public static ImageRunner<Mat, YoloCls, ConverterMatSingleBgrDirectU8, YoloClsFP16SingleExtractor> Single_BgrDirect_FP16(OnnxRuntimeContext context) => new(context);

    // ──────────────────────────── List<Mat> Batch, FP32 ──────────────────────────────

    public static ImageRunner<List<Mat>, YoloCls, ConverterMatListNchw<float, PosCvdnnFP32>, YoloClsFP32SingleExtractor> List_PosCvdnn_FP32(OnnxRuntimeContext context) => new(context);
    public static ImageRunner<List<Mat>, YoloCls, ConverterMatListNchw<float, SymCvdnnFP32>, YoloClsFP32SingleExtractor> List_SymCvdnn_FP32(OnnxRuntimeContext context) => new(context);

    // ──────────────────────────── List<Mat> Batch, FP16 ──────────────────────────────

    public static ImageRunner<List<Mat>, YoloCls, ConverterMatListNchw<Float16, PosCvdnnFP16>, YoloClsFP16SingleExtractor> List_PosCvdnn_FP16(OnnxRuntimeContext context) => new(context);
    public static ImageRunner<List<Mat>, YoloCls, ConverterMatListNchw<Float16, SymCvdnnFP16>, YoloClsFP16SingleExtractor> List_SymCvdnn_FP16(OnnxRuntimeContext context) => new(context);

    // ─────────────────────────── Auto-detect from metadata ───────────────────────────

    /// <summary>
    /// EN: Automatically selects converter and extractor based on model metadata (Single Mat).
    ///     For Cls models, output precision matches input precision:
    ///     FP32 → FP32, FP16 → FP16, BGR-U8 → determined by output tensor type.
    /// RU: Автоматически выбирает конвертер и экстрактор по метаданным модели.
    ///     Для Cls-моделей точность выхода соответствует входу.
    /// </summary>
    public static IRunner<Mat, IBatchedResult> CreateRunner(OnnxRuntimeContext context)
    {
        var inputMeta  = context.Session.InputMetadata.Values.First();
        var outputMeta = context.Session.OutputMetadata.Values.First();

        bool outputIsFP16 = outputMeta.ElementDataType == TensorElementType.Float16;

        Type converterType = inputMeta.ElementDataType switch
        {
            TensorElementType.UInt8   => typeof(ConverterMatSingleBgrDirectU8),
            TensorElementType.Float   => typeof(ConverterMatSingleNchw<float,   PosCvdnnFP32>),
            TensorElementType.Float16 => typeof(ConverterMatSingleNchw<Float16, PosCvdnnFP16>),
            _ => throw new NotSupportedException($"Unsupported input data type: {inputMeta.ElementDataType}")
        };

        (Type resultType, Type extractorType) = outputIsFP16
            ? (typeof(YoloCls), typeof(YoloClsFP16SingleExtractor))
            : (typeof(YoloCls), typeof(YoloClsFP32SingleExtractor));

        Type closedType = typeof(ImageRunner<,,,>).MakeGenericType(
            typeof(Mat),
            resultType,
            converterType,
            extractorType);

        return (IRunner<Mat, IBatchedResult>)Activator.CreateInstance(closedType, context)!;
    }
}
