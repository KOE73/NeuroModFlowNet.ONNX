namespace NeuroModFlowNet.ONNX;

public abstract class Runner : IDisposable
{
    protected Runner(OnnxExecutionContext context) => Context = context;

    protected OnnxExecutionContext Context { get; }

    public virtual void Dispose() => Context?.Dispose();
}
