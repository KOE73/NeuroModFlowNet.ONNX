using Microsoft.ML.OnnxRuntime;

namespace NeuroModFlowNet.ONNX.Converters.Images;

/// <summary>
/// EN: NCHW converter for a batch of Mat inputs utilizing <c>CvDnn.BlobFromImages</c> (Float16).
/// This implementation represents the reference standard for correctness, but is not the most performant due to allocation and copy overhead.
/// <para/>
/// RU: NCHW-конвертер для батча изображений Mat с использованием <c>CvDnn.BlobFromImages</c> (Float16).
/// Использование данного подхода считается образцовым по правильности, но не лучшим по быстродействию из-за аллокаций и копирования данных.
/// </summary>
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
