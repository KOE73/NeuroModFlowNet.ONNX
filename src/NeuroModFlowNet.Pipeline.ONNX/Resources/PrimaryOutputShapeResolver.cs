using NeuroModFlowNet.ONNX;

namespace NeuroModFlowNet.Pipeline.ONNX;

public sealed class PrimaryOutputShapeResolver : IOrtValueOutputShapeResolver
{
    public long[] ResolveOutputShape(OnnxModel model, string outputName, int requestCount)
    {
        ArgumentNullException.ThrowIfNull(model);

        long[] modelShape = model.ModelOutputShapes[outputName];
        long[] resolvedShape = [.. modelShape];

        if(resolvedShape.Length == 0)
            throw new InvalidOperationException($"Model output '{outputName}' is scalar. Tensor output is required.");

        if(resolvedShape[0] <= 0)
            resolvedShape[0] = requestCount;

        for(int index = 0; index < resolvedShape.Length; index++)
        {
            if(resolvedShape[index] <= 0)
                throw new InvalidOperationException($"Model output '{outputName}' has unresolved dynamic dimension at index {index}: [{string.Join(", ", modelShape)}].");
        }

        return resolvedShape;
    }
}
