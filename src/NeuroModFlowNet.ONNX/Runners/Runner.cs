namespace NeuroModFlowNet.ONNX;

public abstract class Runner : IDisposable
{
    protected Runner(OnnxRuntimeContext context) => Context = context;

    protected OnnxRuntimeContext Context { get; }

    public virtual void Dispose() => Context?.Dispose();
}
