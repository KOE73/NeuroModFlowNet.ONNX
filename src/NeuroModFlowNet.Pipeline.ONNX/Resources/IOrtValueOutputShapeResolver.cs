using NeuroModFlowNet.ONNX;

namespace NeuroModFlowNet.Pipeline.ONNX;

public interface IOrtValueOutputShapeResolver
{
    long[] ResolveOutputShape(OnnxModel model, string outputName, int requestCount);
}
