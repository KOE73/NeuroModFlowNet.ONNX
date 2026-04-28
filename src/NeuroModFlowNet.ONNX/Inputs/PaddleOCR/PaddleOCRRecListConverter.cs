using NeuroModFlowNet.ONNX.Converters.Algorithms;

namespace NeuroModFlowNet.ONNX;

public class PaddleOCRRecListConverter : IImageConverter<List<Mat>>
{
    public OnnxRuntimeContext Model { get; private set; } = default!;
    public string ConverterName => "PaddleOCRRecListConverter";

    public int Width { get; private set; }
    public int Height { get; private set; }
    public int Channels { get; private set; }
    public int Batch { get; private set; }

    public void SetModel(OnnxRuntimeContext context)
    {
        Model = context;
        Batch = (int)Model.ModelInputShapes[Model.PrimaryInputName][0];
        Channels = (int)Model.ModelInputShapes[Model.PrimaryInputName][1];
        Height = (int)Model.ModelInputShapes[Model.PrimaryInputName][2];
        Width = (int)Model.ModelInputShapes[Model.PrimaryInputName][3];
    }

    public unsafe void Prepare(List<Mat> input)
    {
        var buffer = Model.GetInputBuffer<float>(Model.PrimaryInputName);
        if(input == null || input.Count == 0)
        {
            buffer.Clear();
            return;
        }

        List<Mat> mats = input.Take(Batch).ToList();
        SymReorderNormPtrFP32.Fill(mats, buffer, mats.Count, Width * Height * Channels, Width * Height);
    }
}
