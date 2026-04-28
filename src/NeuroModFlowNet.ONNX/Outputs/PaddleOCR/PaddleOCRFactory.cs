using NeuroModFlowNet.ONNX.Converters.Algorithms;
using NeuroModFlowNet.ONNX.Converters.Nchw;

namespace NeuroModFlowNet.ONNX;

public static class PaddleOCRFactory
{
    public static ImageRunner<Mat, Mat, ConverterMatSingleNchw<float, SymCvdnnFP32>, PaddleOCRDetFP32_32FC1_SafeExtractor>
        CreateDet(string modelPath, InferenceBackend inferenceBackend = InferenceBackend.Cuda, Action<ExecutionProviderConfig>? configure = null)
    {
        OnnxRuntimeContext context = new OnnxRuntimeContext(modelPath, inferenceBackend, configure);
        return PaddleOCRDetFactory.Single_FP32_32FC1_Safe(context);
    }

    public static ImageRunner<Mat, Mat, PaddleUVDocConverter, PaddleUVDocExtractor>
        CreateUVDoc(string modelPath, InferenceBackend inferenceBackend = InferenceBackend.Cuda, Action<ExecutionProviderConfig>? configure = null)
    {
        OnnxRuntimeContext context = new OnnxRuntimeContext(modelPath, inferenceBackend, configure);
        return new ImageRunner<Mat, Mat, PaddleUVDocConverter, PaddleUVDocExtractor>(context);
    }

    public static ImageRunner<Mat, List<PaddleOCRRecExtractor.OcrResult>, PaddleOCRRecSingleConverter, PaddleOCRRecExtractor>
        CreateRecSingle(string modelPath, InferenceBackend inferenceBackend = InferenceBackend.Cuda, Action<ExecutionProviderConfig>? configure = null)
    {
        OnnxRuntimeContext context = new OnnxRuntimeContext(modelPath, inferenceBackend, configure);
        return new ImageRunner<Mat, List<PaddleOCRRecExtractor.OcrResult>, PaddleOCRRecSingleConverter, PaddleOCRRecExtractor>(context);
    }

    public static ImageRunner<List<Mat>, List<PaddleOCRRecExtractor.OcrResult>, PaddleOCRRecListConverter, PaddleOCRRecExtractor>
        CreateRecList(string modelPath, InferenceBackend inferenceBackend = InferenceBackend.Cuda, Action<ExecutionProviderConfig>? configure = null)
    {
        OnnxRuntimeContext context = new OnnxRuntimeContext(modelPath, inferenceBackend, configure);
        return new ImageRunner<List<Mat>, List<PaddleOCRRecExtractor.OcrResult>, PaddleOCRRecListConverter, PaddleOCRRecExtractor>(context);
    }
}
