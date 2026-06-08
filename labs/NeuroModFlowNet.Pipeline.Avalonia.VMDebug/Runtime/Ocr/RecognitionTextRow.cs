using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.Avalonia.VMDebug.Runtime;

internal sealed record RecognitionTextRow(
    string Text,
    Point LabelPoint,
    Mat Roi,
    RoiHeightDebugData RoiHeightDebug);

