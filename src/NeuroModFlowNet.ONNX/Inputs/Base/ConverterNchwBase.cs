namespace NeuroModFlowNet.ONNX.Converters;

/// <summary>
/// EN: Base class for input adapters with NCHW shape (Batch, Channels, Height, Width).
/// RU: Базовый класс для адаптеров ввода с формой NCHW (Batch, Channels, Height, Width).
/// </summary>
public abstract class ConverterNchwBase<TIn> :
    ConverterBase<TIn>,
    IImageConverter<TIn>
{
    public int Width { get; protected set; }
    public int Height { get; protected set; }
    public int Channels { get; protected set; }
    public int Batch { get; protected set; }

    /// <summary> W × H </summary>
    public int PixelsCount => Width * Height;
    /// <summary> W × H × C (one image size in elements) </summary>
    public int SizeOne => Width * Height * Channels;

    protected override void Init()
    {
        var shape = Model.ModelInputShapes[Model.PrimaryInputName];
        if(shape.Length == 4)
        {
            Batch = (int)shape[0];    // N
            Channels = (int)shape[1]; // C
            Height = (int)shape[2];   // H
            Width = (int)shape[3];    // W
        }
    }
}
