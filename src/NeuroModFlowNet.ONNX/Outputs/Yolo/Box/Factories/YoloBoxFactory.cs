using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX.Converters.Algorithms;
using NeuroModFlowNet.ONNX.Converters.Images;
using NeuroModFlowNet.ONNX.Converters.Nchw;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Factory for YOLO Box detection runners.
/// RU: Фабрика раннеров YOLO Box детекции.
/// </summary>
public static class YoloBoxFactory
{
    // ──────────────────────────────── Single Mat, FP32 ────────────────────────────────

    public static ImageRunner<Mat, IDetectionResult<YoloBox>, ConverterMatSingleNchw<float, PosCvdnnFP32>, YoloBoxNmsFP32StdExtractor> Single_PosCvdnn_FP32(OnnxRuntimeContext context) => new(context);

    public static ImageRunner<Mat, IDetectionResult<YoloBox>,
        ConverterMatSingleNchw<float, SymCvdnnFP32>,
        YoloBoxNmsFP32StdExtractor>
        Single_SymCvdnn_FP32(OnnxRuntimeContext context) => new(context);

    public static ImageRunner<Mat, IDetectionResult<YoloBox>,
        ConverterMatSingleBgrDirectU8,
        YoloBoxNmsFP32StdExtractor>
        Single_BgrDirect_FP32(OnnxRuntimeContext context) => new(context);

    // ──────────────────────────────── Single Mat, FP16 ────────────────────────────────

    public static ImageRunner<Mat, IDetectionResult<YoloBox>,
        ConverterMatSingleNchw<Float16, PosCvdnnFP16>,
        YoloBoxNmsFP16StdExtractor>
        Single_PosCvdnn_FP16(OnnxRuntimeContext context) => new(context);

    public static ImageRunner<Mat, IDetectionResult<YoloBox>,
        ConverterMatSingleNchw<Float16, SymCvdnnFP16>,
        YoloBoxNmsFP16StdExtractor>
        Single_SymCvdnn_FP16(OnnxRuntimeContext context) => new(context);

    // ──────────────────────────── List<Mat> Batch, FP32 ──────────────────────────────

    public static ImageRunner<List<Mat>, IDetectionResult<YoloBox>,
        ConverterMatListNchw<float, PosCvdnnFP32>,
        YoloBoxNmsFP32StdExtractor>
        List_PosCvdnn_FP32(OnnxRuntimeContext context) => new(context);

    public static ImageRunner<List<Mat>, IDetectionResult<YoloBox>,
        ConverterMatListNchw<float, SymCvdnnFP32>,
        YoloBoxNmsFP32StdExtractor>
        List_SymCvdnn_FP32(OnnxRuntimeContext context) => new(context);

    // ──────────────────────────── List<Mat> Batch, FP16 ──────────────────────────────

    public static ImageRunner<List<Mat>, IDetectionResult<YoloBox>,
        ConverterMatListNchw<Float16, PosCvdnnFP16>,
        YoloBoxNmsFP16StdExtractor>
        List_PosCvdnn_FP16(OnnxRuntimeContext context) => new(context);

    public static ImageRunner<List<Mat>, IDetectionResult<YoloBox>,
        ConverterMatListNchw<Float16, SymCvdnnFP16>,
        YoloBoxNmsFP16StdExtractor>
        List_SymCvdnn_FP16(OnnxRuntimeContext context) => new(context);




    // Новый обобщенный метод, который заменит все старые
    public static IRunner<Mat, TOut> CreateRunner<TOut>(OnnxRuntimeContext context, bool isByteBgr)
         where TOut : IBatchedResult
    {
        // 1. Определяем типы, которые подставим в генерик
        Type typeIn = typeof(Mat);
        Type typeOut = typeof(TOut);
        Type typeAdapter = isByteBgr ?
            typeof(ConverterMatSingleBgrDirectU8) :
            typeof(ConverterMatSingleNchw<float, PosCvdnnFP32>);
        Type typeExtractor = GetExtractorType(typeOut, isFp16: false);

        Type openType = typeof(ImageRunner<,,,>);

        Type closedType = openType.MakeGenericType(typeIn, typeOut, typeAdapter, typeExtractor);

        var runner = (IRunner<Mat, TOut>)Activator.CreateInstance(closedType, context)!;
        return runner;


    }



    /// <summary>
    /// EN: Automatically selects converter and extractor based on model metadata (Single Mat).
    /// RU: Автоматически выбирает конвертер и экстрактор по метаданным модели (Single Mat).
    /// </summary>
    public static IRunner<Mat, TOut> CreateRunner<TOut>(OnnxRuntimeContext context)
         where TOut : IBatchedResult
    {
        var inputMeta = context.Session.InputMetadata[context.PrimaryInputName];
        var outputMeta = context.Session.OutputMetadata[context.PrimaryOutputName];

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
        if(typeOut == typeof(IDetectionResult<YoloBox>))
            return isFp16 ? typeof(YoloBoxNmsFP16StdExtractor) : typeof(YoloBoxNmsFP32StdExtractor);

        if(typeOut == typeof(IDetectionResult<YoloBox_FP32_XYWHSC>))
            return typeof(YoloBoxNmsFP32Extractor);

        if(typeOut == typeof(IDetectionResult<YoloBox_FP16_XYWHSC>))
            return typeof(YoloBoxNmsFP16Extractor);

        throw new NotSupportedException($"Unsupported output type for factory: {typeOut.Name}");
    }


}
