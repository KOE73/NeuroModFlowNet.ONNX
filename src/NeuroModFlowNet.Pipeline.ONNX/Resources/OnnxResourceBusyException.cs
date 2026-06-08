namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// Raised when a bounded ONNX batching resource cannot accept another request.
/// </summary>
public sealed class OnnxResourceBusyException : Exception
{
    public OnnxResourceBusyException(string resourceName)
        : base($"ONNX resource '{resourceName}' is full and cannot accept another request.")
    {
        ResourceName = resourceName;
    }

    public string ResourceName { get; }
}

