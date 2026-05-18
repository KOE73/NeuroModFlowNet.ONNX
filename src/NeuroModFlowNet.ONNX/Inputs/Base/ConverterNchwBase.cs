namespace NeuroModFlowNet.ONNX.Converters;

/// <summary>
/// EN: Base class for input adapters with NCHW shape (Batch, Channels, Height, Width).
/// Note: While using OpenCvSharp's <c>CvDnn.BlobFromImage</c> is considered the gold standard (reference implementation) for accuracy and correctness,
/// it is not the most optimal in terms of performance and execution speed due to extra allocations and data copying.
/// <para/>
/// RU: Базовый класс для адаптеров ввода с формой NCHW (Batch, Channels, Height, Width).
/// Примечание: Использование метода <c>CvDnn.BlobFromImage</c> библиотеки OpenCvSharp считается образцовым (эталонным) с точки зрения корректности,
/// однако не является лучшим по быстродействию из-за дополнительных аллокаций и копирования данных.
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
