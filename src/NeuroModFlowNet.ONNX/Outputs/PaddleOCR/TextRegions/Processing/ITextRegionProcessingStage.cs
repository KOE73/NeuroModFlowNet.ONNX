using OpenCvSharp;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Optional image normalization step applied after geometry extraction and before OCR recognition.
/// RU: Необязательный шаг нормализации изображения после геометрического вырезания и до OCR-распознавания.
/// </summary>
/// <remarks>
/// EN: Processing is separated from extraction because brightness, contrast, gamma, thresholding, and similar
/// operations are model/data experiments, while perspective crop geometry is a stable pipeline primitive.
/// RU: Processing отделен от extraction, потому что яркость, контраст, гамма, thresholding и похожие операции
/// являются экспериментами под модель и данные, а геометрия perspective crop остается стабильным примитивом pipeline.
/// </remarks>
public interface ITextRegionProcessingStage
{
    string Name { get; }

    Mat Process(Mat source);
}
