using NeuroModFlowNet.ONNX;

namespace NeuroModFlowNet.Pipeline.MultiCamera;

/// <summary>
/// EN: Text OBB found on one crop of the letterbox batch: <see cref="CropIndex"/> is the batch slot (crop) the box
/// belongs to, <see cref="Box"/> is in crop canvas pixels. The index is what lets text be attributed to the track
/// whose crop it came from after the batch is flattened by the endpoint.
///
/// RU: Текстовый OBB, найденный на одной вырезке letterbox-батча: <see cref="CropIndex"/> это слот батча (вырезка),
/// <see cref="Box"/> в пикселях холста вырезки. Индекс позволяет отнести текст к треку после того, как endpoint
/// вернул результат батча плоским массивом.
/// </summary>
internal readonly record struct CropTextRegion(int CropIndex, YoloObb Box);
