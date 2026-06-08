using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.Avalonia.VMDebug.Runtime;

internal sealed record PreparedRecognitionRoi(
    Mat Roi,
    Point LabelPoint,
    RoiHeightDebugData RoiHeightDebug);

