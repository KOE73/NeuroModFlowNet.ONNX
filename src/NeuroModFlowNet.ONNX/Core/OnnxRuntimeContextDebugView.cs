using Microsoft.ML.OnnxRuntime;
using System.Diagnostics;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// Debugger proxy for <see cref="OnnxRuntimeContext"/>.
/// </summary>
internal sealed class OnnxRuntimeContextDebugView
{
    readonly OnnxRuntimeContext context;

    public OnnxRuntimeContextDebugView(OnnxRuntimeContext context)
    {
        this.context = context;
    }

    public InferenceBackend InferenceBackend => context.InferenceBackend;
    public string ModelName => Path.GetFileName(context.ModelPath);
    public string ModelPath => Path.GetFullPath(context.ModelPath);

    [DebuggerBrowsable(DebuggerBrowsableState.RootHidden)]
    public OnnxRuntimeContextDebugNode[] Nodes =>
    [
        .. context.Session.InputMetadata.Select(item => CreateNode("IN", item.Key, item.Value, context.IsInputPersistentValueInitialized, context.GetRealInputShape)),
        .. context.Session.OutputMetadata.Select(item => CreateNode("OUT", item.Key, item.Value, context.IsOutputPersistentValueInitialized, context.GetRealOutputShape)),
    ];

    public IReadOnlyDictionary<string, string> CustomMetadata => context.Session.ModelMetadata.CustomMetadataMap;

    static OnnxRuntimeContextDebugNode CreateNode(
        string direction,
        string name,
        NodeMetadata nodeMetadata,
        Func<string, bool> isPersistentValueInitialized,
        Func<string, long[]> getRealShape)
    {
        long[]? realShape = nodeMetadata.IsTensor && isPersistentValueInitialized(name)
            ? getRealShape(name)
            : null;

        return new OnnxRuntimeContextDebugNode(
            direction,
            name,
            nodeMetadata.IsTensor ? nodeMetadata.ElementDataType.ToString() : "non-tensor",
            nodeMetadata.IsTensor ? [.. nodeMetadata.Dimensions.Select(dimension => (long)dimension)] : [],
            realShape);
    }
}
