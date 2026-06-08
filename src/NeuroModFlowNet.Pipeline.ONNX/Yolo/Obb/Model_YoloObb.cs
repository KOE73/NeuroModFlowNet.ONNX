using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// Runs a YOLO oriented-bounding-box model and writes one detection list for the current frame.
/// </summary>
public sealed class Model_YoloObb :
    OnnxBatchedInference<Mat, YoloDetectionBatchResult<YoloObb>>
{
    public Model_YoloObb(string name, string inputKey, string outputKey, YoloObbBatchedResource resource)
        : base(
            OpDescriptor.Create(
                name,
                "onnx.yoloObb",
                [VarRequirement.Read<Mat>(inputKey)],
                [VarRequirement.Write<YoloDetectionBatchResult<YoloObb>>(outputKey)],
                hasSideEffects: true),
            inputKey,
            outputKey,
            resource)
    {
    }
}
