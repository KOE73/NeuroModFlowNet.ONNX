using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// Runs a YOLO axis-aligned bounding-box model and writes one detection list for the current frame.
/// </summary>
public sealed class Model_YoloBox :
    OnnxBatchedInference<Mat, YoloDetectionBatchResult<YoloBox>>
{
    public Model_YoloBox(string name, string inputKey, string outputKey, YoloBoxBatchedResource resource)
        : base(
            OpDescriptor.Create(
                name,
                "onnx.yoloBox",
                [VarRequirement.Read<Mat>(inputKey)],
                [VarRequirement.Write<YoloDetectionBatchResult<YoloBox>>(outputKey)],
                hasSideEffects: true),
            inputKey,
            outputKey,
            resource)
    {
    }
}
