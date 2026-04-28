namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: YOLO Segmentation result for a single image (FP32).
/// RU: Результат YOLO Segmentation для одного изображения (FP32).
/// </summary>
public class YoloSegResult_FP32_Mask32 : IBatchedResult
{
    int IBatchedResult.BatchCount => 1;
    /// <summary> EN: Detections (boxes and mask coefficients). RU: Детекции (боксы и коэффициенты масок). </summary>
    public YoloSeg_FP32_XYWHSC_Mask32[] Values { get; set; } = [];

    /// <summary> EN: Shape of the prototype masks tensor. RU: Текущая форма тензора прототипов масок. </summary>
    public long[] PrototypeShape { get; set; } = [];

    /// <summary>
    /// EN: Computed masks per detection. Masks[detectionIdx] = float[H*W].
    /// RU: Вычисленные маски для каждой детекции.
    /// </summary>
    public float[][] Masks { get; set; } = [];
}
