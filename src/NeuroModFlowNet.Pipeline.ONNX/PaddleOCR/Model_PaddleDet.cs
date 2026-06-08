using NeuroModFlowNet.Pipeline;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// Runs the PaddleOCR text-detection model and writes the CPU score map used by region extraction.
/// </summary>
/// <remarks>
/// The command is named as a model instruction because the pipeline step is semantic: it is not just "run ONNX", it
/// produces the Paddle detection map that the following OCR preparation step understands.
/// </remarks>
public sealed class Model_PaddleDet : OnnxRunner<Mat, Mat>
{
    public Model_PaddleDet(string name, string inputKey, string outputKey, OnnxRunnerResource<Mat, Mat> resource)
        : base(
            OpDescriptor.Create(
                name,
                "onnx.paddleDet",
                [VarRequirement.Read<Mat>(inputKey)],
                [VarRequirement.Write<Mat>(outputKey)],
                hasSideEffects: true),
            inputKey,
            outputKey,
            resource,
            disposeOutputWithContext: true)
    {
    }
}
