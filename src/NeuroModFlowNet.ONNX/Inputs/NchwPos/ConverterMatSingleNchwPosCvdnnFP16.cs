using Microsoft.ML.OnnxRuntime;

namespace NeuroModFlowNet.ONNX.Converters.Images;

public class ConverterMatSingleNchwPosCvdnnFP16 : ConverterNchwBase<Mat>
    {

    public override string ConverterName => "CvDnnBlobAdapter (Float16 Normalized)";
    public Type InputType => typeof(Mat);

    public override unsafe void Prepare(Mat image)
    {
        var buffer = Model.GetInputBuffer<Float16>(Model.PrimaryInputName);
        if(image.Empty()) { buffer.Clear(); return; }

        double scale = 1.0 / 255.0;
        using var blob = CvDnn.BlobFromImage(image, scale, swapRB: true, crop: false);

        var sourceData = new ReadOnlySpan<float>((float*)blob.DataPointer, (int)blob.Total());
        
        Span<Half> viewFloat16AsHalf = MemoryMarshal.Cast<Float16, Half>(buffer);
        TensorPrimitives.ConvertToHalf(sourceData, viewFloat16AsHalf);
    }
}
