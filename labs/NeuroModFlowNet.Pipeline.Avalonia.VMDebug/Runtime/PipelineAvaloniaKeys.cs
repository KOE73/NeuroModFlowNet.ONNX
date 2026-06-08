namespace NeuroModFlowNet.Pipeline.Avalonia.VMDebug.Runtime;

internal static class PipelineAvaloniaKeys
{
    public const string SourceFrame = "source.frame";
    public const string SourceId = "source.id";
    public const string ResizedFrame = "frame.resized";
    public const string ModelInputFrame = "model.input.frame";
    public const string LetterboxInfo = "model.input.letterbox";
    public const string SourceLetterboxInfo = "source.letterbox";
    public const string FrameTiming = "frame.timing";
    public const string YoloBoxResult = "yolo.box";
    public const string YoloObbResult = "yolo.obb";
    public const string YoloSegResult = "yolo.seg";
    public const string YoloClsResult = "yolo.cls";
    public const string YoloPoseResult = "yolo.pose";
    public const string PaddleDetScoreMap = "paddle.det.scoreMap";
    public const string PaddleDetRegions = "paddle.det.regions";
    public const string RecognitionRois = "ocr.recognition.rois";
    public const string RecognitionRows = "ocr.recognition.rows";
    public const string FrameResult = "vm.frame.result";
}
