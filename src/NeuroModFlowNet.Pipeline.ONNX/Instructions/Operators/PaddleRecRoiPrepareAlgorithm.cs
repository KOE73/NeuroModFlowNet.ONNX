namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// EN: ONNX graph layout used to prepare OCR recognition ROIs.
///
/// RU: Схема ONNX-графа для подготовки ROI распознавания OCR.
/// </summary>
/// <remarks>
/// EN: Initializer-based variants are useful for benchmarks and fixed ROI experiments, but dynamic OBB payloads must use
/// <see cref="BatchedGridSampleFromMatrices"/> so the cached graph stays valid while ROI matrices change per frame.
///
/// RU: Варианты через initializer полезны для benchmark и экспериментов с фиксированными ROI, но динамические OBB payload
/// должны использовать <see cref="BatchedGridSampleFromMatrices"/>, чтобы кэшированный граф оставался корректным при
/// изменении ROI-матриц на каждом кадре.
/// </remarks>
public enum PaddleRecRoiPrepareAlgorithm
{
    PerRegionGridSampleConcat,
    BatchedGridSample,
    BatchedGridSampleFromMatrices
}
