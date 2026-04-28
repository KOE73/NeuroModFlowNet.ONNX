using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using NeuroModFlowNet.ONNX.Converters.Algorithms;
using NeuroModFlowNet.ONNX.Converters.Nchw;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Factory for PaddleOCR Detection runners.
/// <br/>
/// RU: Фабрика раннеров PaddleOCR Detection.
/// </summary>
public static class PaddleOCRDetFactory
{
    #region Static Methods

    // Single Mat
    public static ImageRunner<Mat, Mat, ConverterMatSingleNchw<float, SymCvdnnFP32>, PaddleOCRDetFP32_32FC1_SafeExtractor> Single_FP32_32FC1_Safe(OnnxRuntimeContext context) => new(context);
    public static ImageRunner<Mat, Mat, ConverterMatSingleNchw<float, SymCvdnnFP32>, PaddleOCRDetFP32_32FC1_UnsafeExtractor> Single_FP32_32FC1_Unsafe(OnnxRuntimeContext context) => new(context);
    public static ImageRunner<Mat, Mat, ConverterMatSingleNchw<float, SymCvdnnFP32>, PaddleOCRDetFP32_8UC1_Extractor> Single_FP32_8UC1(OnnxRuntimeContext context) => new(context);
    public static ImageRunner<Mat, Mat, ConverterMatSingleNchw<float, SymCvdnnFP32>, PaddleOCRDetFP32_8UC3_RGBExtractor> Single_FP32_8UC3_RGB(OnnxRuntimeContext context) => new(context);

    public static ImageRunner<Mat, Mat, ConverterMatSingleNchw<Float16, SymCvdnnFP16>, PaddleOCRDetFP16_16FC1_SafeExtractor> Single_FP16_16FC1_Safe(OnnxRuntimeContext context) => new(context);
    public static ImageRunner<Mat, Mat, ConverterMatSingleNchw<Float16, SymCvdnnFP16>, PaddleOCRDetFP16_16FC1_UnsafeExtractor> Single_FP16_16FC1_Unsafe(OnnxRuntimeContext context) => new(context);
    public static ImageRunner<Mat, Mat, ConverterMatSingleNchw<Float16, SymCvdnnFP16>, PaddleOCRDetFP16_8UC1_Extractor> Single_FP16_8UC1(OnnxRuntimeContext context) => new(context);
    public static ImageRunner<Mat, Mat, ConverterMatSingleNchw<Float16, SymCvdnnFP16>, PaddleOCRDetFP16_8UC3_RGBExtractor> Single_FP16_8UC3_RGB(OnnxRuntimeContext context) => new(context);

    // List<Mat>
    public static ImageRunner<List<Mat>, List<Mat>, ConverterMatListNchw<float, SymCvdnnFP32>, PaddleOCRDetFP32_32FC1_SafeListExtractor> List_FP32_32FC1_Safe(OnnxRuntimeContext context) => new(context);
    public static ImageRunner<List<Mat>, List<Mat>, ConverterMatListNchw<float, SymCvdnnFP32>, PaddleOCRDetFP32_8UC1_SafeListExtractor> List_FP32_8UC1_Safe(OnnxRuntimeContext context) => new(context);
    public static ImageRunner<List<Mat>, List<Mat>, ConverterMatListNchw<float, SymCvdnnFP32>, PaddleOCRDetFP32_8UC3_RGB_SafeListExtractor> List_FP32_8UC3_RGB_Safe(OnnxRuntimeContext context) => new(context);
    public static ImageRunner<List<Mat>, List<Mat>, ConverterMatListNchw<Float16, SymCvdnnFP16>, PaddleOCRDetFP16_16FC1_SafeListExtractor> List_FP16_16FC1_Safe(OnnxRuntimeContext context) => new(context);
    public static ImageRunner<List<Mat>, List<Mat>, ConverterMatListNchw<Float16, SymCvdnnFP16>, PaddleOCRDetFP16_8UC1_SafeListExtractor> List_FP16_8UC1_Safe(OnnxRuntimeContext context) => new(context);
    public static ImageRunner<List<Mat>, List<Mat>, ConverterMatListNchw<Float16, SymCvdnnFP16>, PaddleOCRDetFP16_8UC3_RGB_SafeListExtractor> List_FP16_8UC3_RGB_Safe(OnnxRuntimeContext context) => new(context);

    /// <summary>
    /// EN: Automatically selects converter and extractor based on model metadata, requested output MatType and safety mode.
    /// RU: Автоматически выбирает конвертер и экстрактор по метаданным модели, запрошенному MatType выхода и режиму безопасного доступа.
    /// </summary>
    public static IRunner<TIn, TOut> CreateRunner<TIn, TOut>(OnnxRuntimeContext context, MatType outputMatType, bool safe = true)
    {
        var inputMeta = context.Session.InputMetadata[context.PrimaryInputName];
        var outputMeta = context.Session.OutputMetadata[context.PrimaryOutputName];

        bool isInputFp16 = inputMeta.ElementDataType == TensorElementType.Float16;
        bool isOutputFp16 = outputMeta.ElementDataType == TensorElementType.Float16;

        Type converterType = GetConverterType(typeof(TIn), isInputFp16);
        Type extractorType = GetExtractorType(typeof(TOut), isOutputFp16, outputMatType, safe);

        Type closedType = typeof(ImageRunner<,,,>).MakeGenericType(typeof(TIn), typeof(TOut), converterType, extractorType);

        return (IRunner<TIn, TOut>)Activator.CreateInstance(closedType, context)!;
    }

    private static Type GetConverterType(Type typeIn, bool isFp16)
    {
        if(typeIn == typeof(Mat))
            return isFp16 ? typeof(ConverterMatSingleNchw<Float16, SymCvdnnFP16>) : typeof(ConverterMatSingleNchw<float, SymCvdnnFP32>);

        if(typeIn == typeof(List<Mat>))
            return isFp16 ? typeof(ConverterMatListNchw<Float16, SymCvdnnFP16>) : typeof(ConverterMatListNchw<float, SymCvdnnFP32>);

        throw new NotSupportedException($"Unsupported input type for PaddleOCRDetFactory: {typeIn.Name}");
    }

    private static Type GetExtractorType(Type typeOut, bool isFp16, MatType outputMatType, bool safe)
    {
        if(typeOut == typeof(Mat))
            return GetSingleExtractorType(isFp16, outputMatType, safe);

        if(typeOut == typeof(List<Mat>))
            return GetListExtractorType(isFp16, outputMatType);

        throw new NotSupportedException($"Unsupported output type for PaddleOCRDetFactory: {typeOut.Name}");
    }

    private static Type GetSingleExtractorType(bool isFp16, MatType outputMatType, bool safe)
    {
        if(isFp16)
        {
            if(outputMatType == MatType.CV_16FC1)
                return safe ? typeof(PaddleOCRDetFP16_16FC1_SafeExtractor) : typeof(PaddleOCRDetFP16_16FC1_UnsafeExtractor);

            if(outputMatType == MatType.CV_8UC1)
                return typeof(PaddleOCRDetFP16_8UC1_Extractor);

            if(outputMatType == MatType.CV_8UC3)
                return typeof(PaddleOCRDetFP16_8UC3_RGBExtractor);
        }
        else
        {
            if(outputMatType == MatType.CV_32FC1)
                return safe ? typeof(PaddleOCRDetFP32_32FC1_SafeExtractor) : typeof(PaddleOCRDetFP32_32FC1_UnsafeExtractor);

            if(outputMatType == MatType.CV_8UC1)
                return typeof(PaddleOCRDetFP32_8UC1_Extractor);

            if(outputMatType == MatType.CV_8UC3)
                return typeof(PaddleOCRDetFP32_8UC3_RGBExtractor);
        }

        throw new NotSupportedException($"Unsupported PaddleOCRDet output combination for single Mat: tensor={GetTensorFormatName(isFp16)}, matType={outputMatType}, safe={safe}.");
    }

    private static Type GetListExtractorType(bool isFp16, MatType outputMatType)
    {
        if(isFp16 && outputMatType == MatType.CV_16FC1)
            return typeof(PaddleOCRDetFP16_16FC1_SafeListExtractor);

        if(isFp16 && outputMatType == MatType.CV_8UC1)
            return typeof(PaddleOCRDetFP16_8UC1_SafeListExtractor);

        if(isFp16 && outputMatType == MatType.CV_8UC3)
            return typeof(PaddleOCRDetFP16_8UC3_RGB_SafeListExtractor);

        if(!isFp16 && outputMatType == MatType.CV_32FC1)
            return typeof(PaddleOCRDetFP32_32FC1_SafeListExtractor);

        if(!isFp16 && outputMatType == MatType.CV_8UC1)
            return typeof(PaddleOCRDetFP32_8UC1_SafeListExtractor);

        if(!isFp16 && outputMatType == MatType.CV_8UC3)
            return typeof(PaddleOCRDetFP32_8UC3_RGB_SafeListExtractor);

        throw new NotSupportedException($"Unsupported PaddleOCRDet output combination for List<Mat>: tensor={GetTensorFormatName(isFp16)}, matType={outputMatType}.");
    }

    private static string GetTensorFormatName(bool isFp16) => isFp16 ? "FP16" : "FP32";

    #endregion
}
