using System.Diagnostics;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// Single input or output row shown by the debugger proxy for <see cref="OnnxRuntimeContext"/>.
/// </summary>
[DebuggerDisplay("{DebuggerDisplay,nq}")]
internal sealed class OnnxRuntimeContextDebugNode
{
    public OnnxRuntimeContextDebugNode(
        string direction,
        string name,
        string elementType,
        long[] modelShape,
        long[]? realShape)
    {
        Direction = direction;
        Name = name;
        ElementType = elementType;
        ModelShape = modelShape;
        RealShape = realShape;
    }

    public string Direction { get; }
    public string Name { get; }
    public string ElementType { get; }
    public long[] ModelShape { get; }
    public long[]? RealShape { get; }

    string DebuggerDisplay =>
        $"{Direction}: {Name} {ElementType} {FormatShape(ModelShape)}{FormatRealShape(RealShape)}";

    static string FormatRealShape(long[]? realShape) =>
        realShape is null ? string.Empty : $" real={FormatShape(realShape)}";

    static string FormatShape(IReadOnlyList<long> shape) =>
        "[" + string.Join(" x ", shape.Select(dimension => dimension < 0 ? "?" : dimension.ToString())) + "]";
}
