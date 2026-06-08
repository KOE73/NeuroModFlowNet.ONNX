using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// Runs the PaddleOCR text-recognition model for a batch of cropped text line images.
/// </summary>
/// <remarks>
/// Recognition width and batch shape can be changed by the debug/runtime UI. The instruction therefore uses a reloadable
/// resource: the VM command name and register contract stay stable while the concrete runner is swapped under a lock.
/// </remarks>
public sealed class Model_PaddleRec : ReloadableOnnxRunner<List<Mat>, List<PaddleOCRRecExtractor.OcrResult>>
{
    public Model_PaddleRec(string name, string inputKey, string outputKey, ReloadableOnnxRunnerResource<List<Mat>, List<PaddleOCRRecExtractor.OcrResult>> resource)
        : base(
            OpDescriptor.Create(
                name,
                "onnx.paddleRec",
                [VarRequirement.Read<List<Mat>>(inputKey)],
                [VarRequirement.Write<List<PaddleOCRRecExtractor.OcrResult>>(outputKey)],
                hasSideEffects: true),
            inputKey,
            outputKey,
            resource)
    {
    }
}
