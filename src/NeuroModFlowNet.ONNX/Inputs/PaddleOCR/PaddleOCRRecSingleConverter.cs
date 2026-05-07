using NeuroModFlowNet.ONNX.Converters.Algorithms;

namespace NeuroModFlowNet.ONNX;

public class PaddleOCRRecSingleConverter : IImageConverter<Mat>
{
    public OnnxRuntimeContext Model { get; private set; } = default!;
    public string ConverterName => "PaddleOCRRecSingleConverter";

    public int Width { get; private set; }
    public int Height { get; private set; }
    public int Channels { get; private set; }
    public int Batch { get; private set; }

    public void SetModel(OnnxRuntimeContext context)
    {
        Model = context;
        long[] shape = Model.IsInputPersistentValueInitialized(Model.PrimaryInputName)
            ? Model.GetRealInputShape(Model.PrimaryInputName)
            : Model.ModelInputShapes[Model.PrimaryInputName];

        Batch = (int)shape[0];
        Channels = (int)shape[1];
        Height = (int)shape[2];
        Width = (int)shape[3];
    }

    public unsafe void Prepare(Mat input)
    {
        var buffer = Model.GetInputBuffer<float>(Model.PrimaryInputName);
        if(input.Empty())
        {
            buffer.Clear();
            return;
        }
        List<Mat> mats = [input];
        SymReorderNormPtrFP32.Fill(mats, buffer, 1, Width * Height * Channels, Width * Height);
    }
}
