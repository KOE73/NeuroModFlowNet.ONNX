using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace NeuroModFlowNet.ONNX.Converters.Images;

/// <summary>
/// Конвертер для моделей, ожидающих вход в виде BGR-байтов (uint8), с прямым использованием данных из Mat без копирования.
/// Ожидается, что модель принимает входной тензор с формой [1, H, W, 3] и типом данных uint8.
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
