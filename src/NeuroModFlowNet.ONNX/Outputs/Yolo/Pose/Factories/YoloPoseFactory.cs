using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX.Converters.Algorithms;
using NeuroModFlowNet.ONNX.Converters.Images;
using NeuroModFlowNet.ONNX.Converters.Nchw;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Factory for YOLO Pose estimation runners.
/// RU: Фабрика раннеров YOLO Pose оценки поз.
/// </summary>
public static class YoloPoseFactory
{
    // ──────────────────────────────── Single Mat, FP32 ────────────────────────────────

    public static ImageRunner<Mat, IDetectionResult<YoloPose_FP32_Size57_Keypoint17>, ConverterMatSingleNchw<float, PosCvdnnFP32>, YoloPoseFP32Keypoint17Extractor> Single_PosCvdnn_FP32(OnnxRuntimeContext context) => new(context);
    public static ImageRunner<Mat, IDetectionResult<YoloPose_FP32_Size57_Keypoint17>, ConverterMatSingleNchw<float, SymCvdnnFP32>, YoloPoseFP32Keypoint17Extractor> Single_SymCvdnn_FP32(OnnxRuntimeContext context) => new(context);
    public static ImageRunner<Mat, IDetectionResult<YoloPose_FP32_Size57_Keypoint17>, ConverterMatSingleBgrDirectU8, YoloPoseFP32Keypoint17Extractor> Single_BgrDirect_FP32(OnnxRuntimeContext context) => new(context);

    // ──────────────────────────────── Single Mat, FP16 ────────────────────────────────

    public static ImageRunner<Mat, IDetectionResult<YoloPose_FP32_Size57_Keypoint17>, ConverterMatSingleNchw<Float16, PosCvdnnFP16>, YoloPoseFP32Keypoint17Extractor> Single_PosCvdnn_FP16(OnnxRuntimeContext context) => new(context);
    public static ImageRunner<Mat, IDetectionResult<YoloPose_FP32_Size57_Keypoint17>, ConverterMatSingleNchw<Float16, SymCvdnnFP16>, YoloPoseFP32Keypoint17Extractor> Single_SymCvdnn_FP16(OnnxRuntimeContext context) => new(context);
    public static ImageRunner<Mat, IDetectionResult<YoloPose_FP32_Size57_Keypoint17>, ConverterMatSingleBgrDirectU8, YoloPoseFP32Keypoint17Extractor> Single_BgrDirect_FP16(OnnxRuntimeContext context) => new(context);

    // ──────────────────────────── List<Mat> Batch, FP32 ──────────────────────────────

    public static ImageRunner<List<Mat>, IDetectionResult<YoloPose_FP32_Size57_Keypoint17>, ConverterMatListNchw<float, PosCvdnnFP32>, YoloPoseFP32Keypoint17Extractor> List_PosCvdnn_FP32(OnnxRuntimeContext context) => new(context);
    public static ImageRunner<List<Mat>, IDetectionResult<YoloPose_FP32_Size57_Keypoint17>, ConverterMatListNchw<float, SymCvdnnFP32>, YoloPoseFP32Keypoint17Extractor> List_SymCvdnn_FP32(OnnxRuntimeContext context) => new(context);

    // ──────────────────────────── List<Mat> Batch, FP16 ──────────────────────────────

    public static ImageRunner<List<Mat>, IDetectionResult<YoloPose_FP32_Size57_Keypoint17>, ConverterMatListNchw<Float16, PosCvdnnFP16>, YoloPoseFP32Keypoint17Extractor> List_PosCvdnn_FP16(OnnxRuntimeContext context) => new(context);
    public static ImageRunner<List<Mat>, IDetectionResult<YoloPose_FP32_Size57_Keypoint17>, ConverterMatListNchw<Float16, SymCvdnnFP16>, YoloPoseFP32Keypoint17Extractor> List_SymCvdnn_FP16(OnnxRuntimeContext context) => new(context);

    // ─────────────────────────── Auto-detect from metadata ───────────────────────────

    /// <summary>
    /// EN: Automatically selects converter based on model input metadata (Single Mat).
    ///     Output is always FP32 for Pose models.
    /// RU: Автоматически выбирает конвертер по метаданным входного тензора.
    ///     Выход всегда FP32 для Pose-моделей.
    /// </summary>
    public static IRunner<Mat, IDetectionResult<YoloPose_FP32_Size57_Keypoint17>> CreateRunner17(OnnxRuntimeContext context)
    {
        var inputMeta = context.Session.InputMetadata.Values.First();

        Type converterType = inputMeta.ElementDataType switch
        {
            TensorElementType.UInt8    => typeof(ConverterMatSingleBgrDirectU8),
            TensorElementType.Float    => typeof(ConverterMatSingleNchw<float,   PosCvdnnFP32>),
            TensorElementType.Float16  => typeof(ConverterMatSingleNchw<Float16, PosCvdnnFP16>),
            _ => throw new NotSupportedException($"Unsupported input data type: {inputMeta.ElementDataType}")
        };

        Type closedType = typeof(ImageRunner<,,,>).MakeGenericType(
            typeof(Mat),
            typeof(IDetectionResult<YoloPose_FP32_Size57_Keypoint17>),
            converterType,
            typeof(YoloPoseFP32Keypoint17Extractor));

        return (IRunner<Mat, IDetectionResult<YoloPose_FP32_Size57_Keypoint17>>)Activator.CreateInstance(closedType, context)!;
    }


    public static IRunner<Mat, IDetectionResult<YoloPose>> CreateRunner(OnnxRuntimeContext context)
    {
        var inputMeta = context.Session.InputMetadata.Values.First();

        Type converterType = inputMeta.ElementDataType switch
        {
            TensorElementType.UInt8    => typeof(ConverterMatSingleBgrDirectU8),
            TensorElementType.Float    => typeof(ConverterMatSingleNchw<float,   PosCvdnnFP32>),
            TensorElementType.Float16  => typeof(ConverterMatSingleNchw<Float16, PosCvdnnFP16>),
            _ => throw new NotSupportedException($"Unsupported input data type: {inputMeta.ElementDataType}")
        };

        Type closedType = typeof(ImageRunner<,,,>).MakeGenericType(
            typeof(Mat),
            typeof(IDetectionResult<YoloPose>),
            converterType,
            typeof(YoloPoseFP32UniversalExtractor));

        return (IRunner<Mat, IDetectionResult<YoloPose>>)Activator.CreateInstance(closedType, context)!;
    }
}
