namespace NeuroModFlowNet.ONNX;

public class PaddleUVDocConverter : IImageConverter<Mat>
{
    public OnnxRuntimeContext Model { get; private set; } = default!;
    public string ConverterName => "PaddleUVDocConverter";

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

    public unsafe void Prepare(Mat input)
    {
        var buffer = Model.GetInputBuffer<float>(Model.PrimaryInputName);
        if(input.Empty())
        {
            buffer.Clear();
            return;
        }

        double scale = 1.0 / 255.0;
        Scalar mean = new Scalar(0, 0, 0);

        using var blob = CvDnn.BlobFromImage(
            input,
            scale,
            new Size(Width, Height),
            mean,
            swapRB: true,
            crop: false);

        var sourceData = new ReadOnlySpan<float>((float*)blob.DataPointer, (int)blob.Total());

        sourceData.CopyTo(buffer);
    }
}
