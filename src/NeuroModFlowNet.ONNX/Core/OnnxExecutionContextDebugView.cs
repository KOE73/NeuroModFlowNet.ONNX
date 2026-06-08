using Microsoft.ML.OnnxRuntime;
using System.Diagnostics;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// Debugger proxy for <see cref="OnnxExecutionContext"/>.
/// </summary>
internal sealed class OnnxExecutionContextDebugView
{
    readonly OnnxExecutionContext context;

    public OnnxExecutionContextDebugView(OnnxExecutionContext context)
    {
        this.context = context;
    }

    public InferenceBackend InferenceBackend => context.Model.InferenceBackend;
    public OnnxRuntimeModelSourceKind ModelSourceKind => context.Model.ModelSource.Kind;
    public string ModelName => context.Model.ModelSource.DisplayName;
    public string? ModelPath => context.Model.ModelSource.Path is null ? null : Path.GetFullPath(context.Model.ModelSource.Path);

    [DebuggerBrowsable(DebuggerBrowsableState.RootHidden)]
    public OnnxExecutionContextDebugNode[] Nodes =>
    [
        .. context.Model.Session.InputMetadata.Select(item => CreateNode("IN", item.Key, item.Value, context.IsInputPersistentValueInitialized, context.GetRealInputShape)),
        .. context.Model.Session.OutputMetadata.Select(item => CreateNode("OUT", item.Key, item.Value, context.IsOutputPersistentValueInitialized, context.GetRealOutputShape)),
    ];

    public IReadOnlyDictionary<string, string> CustomMetadata => context.Model.Session.ModelMetadata.CustomMetadataMap;

    static OnnxExecutionContextDebugNode CreateNode(
        string direction,
        string name,
        NodeMetadata nodeMetadata,
        Func<string, bool> isPersistentValueInitialized,
        Func<string, long[]> getRealShape)
    {
        long[]? realShape = nodeMetadata.IsTensor && isPersistentValueInitialized(name)
            ? getRealShape(name)
            : null;

        return new OnnxExecutionContextDebugNode(
            direction,
            name,
            nodeMetadata.IsTensor ? nodeMetadata.ElementDataType.ToString() : "non-tensor",
            nodeMetadata.IsTensor ? [.. nodeMetadata.Dimensions.Select(dimension => (long)dimension)] : [],
            realShape);
    }
}
