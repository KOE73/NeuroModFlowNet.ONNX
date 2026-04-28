using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX.Converters.Algorithms;
using NeuroModFlowNet.ONNX.Converters.Images;
using NeuroModFlowNet.ONNX.Converters.Nchw;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Factory for YOLO OBB (Oriented Bounding Box) detection runners.
/// RU: Фабрика раннеров YOLO OBB (ориентированные прямоугольники) детекции.
/// </summary>
public static class YoloObbFactory
{
    // ──────────────────────────────── Single Mat, FP32 ────────────────────────────────

    public static ImageRunner<Mat, IDetectionResult<YoloObb>, ConverterMatSingleNchw<float, PosCvdnnFP32>, YoloObbNmsFP32StdExtractor> Single_PosCvdnn_FP32(OnnxRuntimeContext context) => new(context);
    public static ImageRunner<Mat, IDetectionResult<YoloObb>, ConverterMatSingleNchw<float, SymCvdnnFP32>, YoloObbNmsFP32StdExtractor> Single_SymCvdnn_FP32(OnnxRuntimeContext context) => new(context);
    public static ImageRunner<Mat, IDetectionResult<YoloObb>, ConverterMatSingleBgrDirectU8, YoloObbNmsFP32StdExtractor> Single_BgrDirect_FP32(OnnxRuntimeContext context) => new(context);

    // ──────────────────────────────── Single Mat, Internal FP32 ──────────────────────

    public static ImageRunner<Mat, IDetectionResult<YoloObb_FP32_XYWHSCA>, ConverterMatSingleNchw<float, PosCvdnnFP32>, YoloObbNmsFP32Extractor> SingleInternal_PosCvdnn_FP32(OnnxRuntimeContext context) => new(context);

    // ──────────────────────────── List<Mat> Batch, FP32 ──────────────────────────────

    public static ImageRunner<List<Mat>, IDetectionResult<YoloObb>, ConverterMatListNchw<float, PosCvdnnFP32>, YoloObbNmsFP32StdExtractor> List_PosCvdnn_FP32(OnnxRuntimeContext context) => new(context);
    public static ImageRunner<List<Mat>, IDetectionResult<YoloObb>, ConverterMatListNchw<float, SymCvdnnFP32>, YoloObbNmsFP32StdExtractor> List_SymCvdnn_FP32(OnnxRuntimeContext context) => new(context);

    // ────────────────────────────── Generic Creation ────────────────────────────────

    /// <summary>
    /// EN: Automatically selects converter and extractor based on model metadata (Single Mat).
    /// RU: Автоматически выбирает конвертер и экстрактор по метаданным модели (Single Mat).
    /// </summary>
    public static IRunner<Mat, TOut> CreateRunner<TOut>(OnnxRuntimeContext context)
         where TOut : IBatchedResult
    {
        var inputMeta = context.Session.InputMetadata[context.PrimaryInputName];
        var outputMeta = context.Session.OutputMetadata[context.PrimaryOutputName];

        // OBB output is always FP32 according to specs
        bool isFp16 = outputMeta.ElementDataType == TensorElementType.Float16;

        Type converterType = inputMeta.ElementDataType switch
        {
            TensorElementType.UInt8 => typeof(ConverterMatSingleBgrDirectU8),
            TensorElementType.Float => typeof(ConverterMatSingleNchw<float, PosCvdnnFP32>),
            TensorElementType.Float16 => typeof(ConverterMatSingleNchw<Float16, PosCvdnnFP16>),
            _ => throw new NotSupportedException($"Unsupported input data type: {inputMeta.ElementDataType}")
        };

        Type extractorType = GetExtractorType(typeof(TOut), isFp16);

        Type closedType = typeof(ImageRunner<,,,>).MakeGenericType(typeof(Mat), typeof(TOut), converterType, extractorType);

        var runner = (IRunner<Mat, TOut>)Activator.CreateInstance(closedType, context)!;
        return runner;
    }

    private static Type GetExtractorType(Type typeOut, bool isFp16)
    {
        if(typeOut == typeof(IDetectionResult<YoloObb>))
            return typeof(YoloObbNmsFP32StdExtractor);

        if(typeOut == typeof(IDetectionResult<YoloObb_FP32_XYWHSCA>))
            return typeof(YoloObbNmsFP32Extractor);

        throw new NotSupportedException($"Unsupported output type for OBB factory: {typeOut.Name}");
    }
}
