using NeuroModFlowNet.ONNX;

namespace NeuroModFlowNet.Pipeline.ONNX;

public sealed class PaddleOCRRecOutputShapeResolver : IOrtValueOutputShapeResolver
{
    readonly int targetWidth;
    readonly int outputWidthStride;

    public PaddleOCRRecOutputShapeResolver(
        int targetWidth,
        int outputWidthStride = PaddleOCRRecExtractor.OutputWidthStride)
    {
        if(targetWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(targetWidth), "Target width must be positive.");

        if(outputWidthStride <= 0)
            throw new ArgumentOutOfRangeException(nameof(outputWidthStride), "Output width stride must be positive.");

        if(targetWidth % outputWidthStride != 0)
            throw new ArgumentException($"Target width {targetWidth} must be divisible by output width stride {outputWidthStride}.", nameof(targetWidth));

        this.targetWidth = targetWidth;
        this.outputWidthStride = outputWidthStride;
    }

    public long[] ResolveOutputShape(OnnxModel model, string outputName, int requestCount)
    {
        ArgumentNullException.ThrowIfNull(model);

        long[] modelShape = model.ModelOutputShapes[outputName];
        if(modelShape.Length != 3)
            throw new InvalidOperationException($"PaddleOCR Rec output '{outputName}' must be rank 3, actual: [{string.Join(", ", modelShape)}].");

        long symbolCount = modelShape[2];
        if(symbolCount <= 0)
            throw new InvalidOperationException($"PaddleOCR Rec output '{outputName}' has unresolved symbol dimension: [{string.Join(", ", modelShape)}].");

        return [requestCount, targetWidth / outputWidthStride, symbolCount];
    }
}
