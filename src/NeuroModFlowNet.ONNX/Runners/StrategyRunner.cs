namespace NeuroModFlowNet.ONNX;


public abstract class StrategyRunner<TIn, TOut, TAdapter, TExtractor> : Runner, 
    IRunner<TIn, TOut>
    where TAdapter : IInputConverter<TIn>, new()
    where TExtractor : IResultExtractor<TOut>, new()
{
    protected StrategyRunner(OnnxRuntimeContext context) : base(context)
    {
        Adapter.SetModel(context);
        Extractor.SetModel(context);
    }



    protected TAdapter Adapter { get; } = new();
    protected TExtractor Extractor { get; } = new();


    #region " IRunner Core "

    public virtual TOut Predict(TIn input)
    {
        try
        {
            Adapter.Prepare(input);
            Context.Run();
            return Extractor.Extract();

        }
        finally
        {
            Context.Cleanup();
        }
    }

    #endregion

    #region " Capability Discovery (Extension Interfaces) "
    /// <summary>
    /// EN: Searches for capability T in input (adapter), then output (extractor), then this runner itself.
    /// RU: Ищет возможность T на входе (адаптер), затем на выходе (экстрактор), закончив самим раннером.
    /// </summary>
    public virtual T? As<T>() where T : class
    {
        return InAs<T>() ?? OutAs<T>() ?? this as T;
    }

    public virtual T? InAs<T>() where T : class => Adapter as T;
    public virtual T? OutAs<T>() where T : class => Extractor as T;
    #endregion
}
