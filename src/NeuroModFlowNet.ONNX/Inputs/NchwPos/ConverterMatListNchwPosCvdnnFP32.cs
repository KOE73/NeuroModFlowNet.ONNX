namespace NeuroModFlowNet.ONNX.Converters.Images;

public class ConverterMatListNchwPosCvdnnFP32 : ConverterNchwBase<List<Mat>>
    {

    public override string ConverterName => "CvDnnBlobBatchAdapter (Float32 Normalized)";
    public Type InputType => typeof(List<Mat>);

    public override unsafe void Prepare(List<Mat> images)
    {
        var buffer = Model.GetInputBuffer<float>(Model.PrimaryInputName);
        if(images == null || images.Count == 0) { buffer.Clear(); return; }

        double scale = 1.0 / 255.0;
        using var blob = CvDnn.BlobFromImages(images, scale, swapRB: true, crop: false);

        var sourceData = new ReadOnlySpan<float>((float*)blob.DataPointer, (int)blob.Total());
        sourceData.CopyTo(buffer.Slice(0, (int)blob.Total()));
    }
}
