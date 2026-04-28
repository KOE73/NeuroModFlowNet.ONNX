namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: YOLO Segmentation result for a single image (FP16).
/// RU: Результат YOLO Segmentation для одного изображения (FP16).
/// </summary>
public class YoloSegResult_FP16_Mask32 : IBatchedResult
{
    int IBatchedResult.BatchCount => 1;
    /// <summary> EN: Detections (boxes and mask coefficients). RU: Детекции (боксы и коэффициенты масок). </summary>
    public YoloSeg_FP16_XYWHSC_Mask32[] Values { get; set; } = [];

    /// <summary> EN: Shape of the prototype masks tensor. RU: Размер тензора прототипов. </summary>
    public long[] PrototypeShape { get; set; } = [];

    /// <summary>
    /// EN: Computed masks per detection. Masks[detectionIdx] = Half[H*W].
    /// RU: Вычисленные маски.
    /// </summary>
    public Half[][] Masks { get; set; } = [];
}
