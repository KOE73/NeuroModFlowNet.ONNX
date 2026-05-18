using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace NeuroModFlowNet.ONNX.Converters.Images;

/// <summary>
/// EN: Input adapter for models expecting BGR-bytes (uint8) with direct zero-copy usage of <see cref="Mat"/> memory.
/// This adapter leverages a key project feature: support for a modified model head that accepts native <see cref="Mat"/> pointer data directly.
/// This achieves zero allocation and zero copying by pinning the Mat memory buffer.
/// Expects a 4D input tensor with shape [1, H, W, 3] and uint8 data type.
/// <para/>
/// RU: Адаптер ввода для моделей, ожидающих BGR-байты (uint8), с прямым (zero-copy) использованием памяти <see cref="Mat"/> без копирования.
/// Данный адаптер реализует ключевую фичу проекта — работу с модифицированной головой модели, которая принимает данные <see cref="Mat"/> напрямую.
/// Это позволяет исключить аллокации и копирование данных благодаря передаче указателя на буфер памяти Mat.
/// Ожидается четырехмерный входной тензор с формой [1, H, W, 3] и типом данных uint8.
/// </summary>
public class ConverterMatSingleBgrDirectU8 : ConverterBase<Mat>,
    IImageConverter<Mat>
{

    #region " IImageInfo Properties "
    public int Width { get; private set; }
    public int Height { get; private set; }
    public int Channels { get; private set; }
    public int Batch { get; private set; }
    #endregion

    #region " IAdapterInfo Properties "
    public override string ConverterName => "BgrDirectAdapter (ZeroCopy UInt8)";
    public Type InputType => typeof(Mat);
    #endregion

    protected override void Init()
    {
        var shape = Model.ModelInputShapes[Model.PrimaryInputName];
        if(shape.Length == 4)
        {
            Batch = (int)shape[0];    // N
            Height = (int)shape[1];   // H
            Width = (int)shape[2];    // W 
            Channels = (int)shape[3]; // C
        }
    }

    protected override void Check()
    {
        base.Check();
        Debug.Assert(Batch == 1, $"Expected batch size of 1 for single image input, but got {Batch}.");
        Debug.Assert(Channels == 3, $"Expected 3 channels for BGR format, but got {Channels}.");
    }

    public override unsafe void Prepare(Mat image)
    {
        // Прямое обращение к памяти Mat без копирования
        var value = OrtValue.CreateTensorValueWithData(
            OrtMemoryInfo.DefaultInstance,
            TensorElementType.UInt8,
            Model.ModelInputShapes[Model.PrimaryInputName],
            (nint)image.DataPointer,
            (int)(image.Total() * image.Channels()));

        Model.SetInput(Model.InputNames[0], value);
    }
}
