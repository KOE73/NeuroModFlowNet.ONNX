using Microsoft.ML.OnnxRuntime;

namespace NeuroModFlowNet.ONNX.Converters.Images;

/// <summary>
/// EN: NCHW converter for a single Mat input utilizing <c>CvDnn.BlobFromImage</c> (Float16).
/// This implementation represents the reference standard for correctness, but is not the most performant due to allocation and copy overhead.
/// <para/>
/// RU: NCHW-конвертер для одного изображения Mat с использованием <c>CvDnn.BlobFromImage</c> (Float16).
/// Использование данного подхода считается образцовым по правильности, но не лучшим по быстродействию из-за аллокаций и копирования данных.
/// </summary>
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
