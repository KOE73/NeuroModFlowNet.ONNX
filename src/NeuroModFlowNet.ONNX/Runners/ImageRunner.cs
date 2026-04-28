namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: General runner for image-related model inference. Ensures IImageAdapter usage at compile time.
/// RU: Общий раннер для инференса изображений. Гарантирует использование IImageAdapter на этапе компиляции.
/// </summary>
public class ImageRunner<TIn, TOut, TAdapter, TExtractor> : 
    StrategyRunner<TIn, TOut, TAdapter, TExtractor>, 
    IImageInfo
    where TAdapter : IImageConverter<TIn>, new()
    where TExtractor : IResultExtractor<TOut>, new()
{
    public ImageRunner(OnnxRuntimeContext context) : base(context)
    {
        // Init is handled by StrategyRunner base constructor
    }


    #region IImageInfo / IAdapterInfo Delegation
    public int Batch => Adapter.Batch;
    public int Channels => Adapter.Channels;
    public int Width => Adapter.Width;
    public int Height => Adapter.Height;
    #endregion
}
