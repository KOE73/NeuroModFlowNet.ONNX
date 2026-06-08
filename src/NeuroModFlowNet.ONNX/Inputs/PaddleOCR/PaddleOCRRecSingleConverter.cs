using NeuroModFlowNet.ONNX.Converters.Algorithms;

namespace NeuroModFlowNet.ONNX;

public class PaddleOCRRecSingleConverter : IImageConverter<Mat>
{
    public OnnxExecutionContext Context { get; private set; } = default!;
    OnnxModel Model => Context.Model;

    public string ConverterName => "PaddleOCRRecSingleConverter";

    public int Width { get; private set; }
    public int Height { get; private set; }
    public int Channels { get; private set; }
    public int Batch { get; private set; }

    public void SetModel(OnnxExecutionContext context)
    {
        Context = context;
        long[] shape = Context.IsInputPersistentValueInitialized(Model.PrimaryInputName)
            ? Context.GetRealInputShape(Model.PrimaryInputName)
            : Model.ModelInputShapes[Model.PrimaryInputName];

        Batch = (int)shape[0];
        Channels = (int)shape[1];
        Height = (int)shape[2];
        Width = (int)shape[3];
    }

    public unsafe void Prepare(Mat input)
    {
        var buffer = Context.GetInputBuffer<float>(Model.PrimaryInputName);
        if(input.Empty())
        {
            buffer.Clear();
            return;
        }
        List<Mat> mats = [input];
        SymReorderNormPtrFP32.Fill(mats, buffer, 1, Width * Height * Channels, Width * Height);
    }
}
