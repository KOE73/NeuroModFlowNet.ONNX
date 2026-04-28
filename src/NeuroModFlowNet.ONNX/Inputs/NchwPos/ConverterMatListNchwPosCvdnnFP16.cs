using Microsoft.ML.OnnxRuntime;

namespace NeuroModFlowNet.ONNX.Converters.Images;

public class ConverterMatListNchwPosCvdnnFP16 : ConverterNchwBase<List<Mat>>
{

    public override string ConverterName => "CvDnnBlobBatchAdapter (Float16 Normalized)";
    public Type InputType => typeof(List<Mat>);

    public override unsafe void Prepare(List<Mat> images)
    {
        var buffer = Model.GetInputBuffer<Float16>(Model.PrimaryInputName);
        if(images == null || images.Count == 0) { buffer.Clear(); return; }

        double scale = 1.0 / 255.0;
        using var blob = CvDnn.BlobFromImages(images, scale, swapRB: true, crop: false);

        var sourceData = new ReadOnlySpan<float>((float*)blob.DataPointer, (int)blob.Total());

        Span<Half> viewFloat16AsHalf = MemoryMarshal.Cast<Float16, Half>(buffer);
        TensorPrimitives.ConvertToHalf(sourceData, viewFloat16AsHalf.Slice(0, (int)blob.Total()));
    }
}
