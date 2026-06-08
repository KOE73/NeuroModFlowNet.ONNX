using Avalonia.Controls;
using NeuroModFlowNet.Pipeline.Avalonia.VMDebug.Runtime;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.Avalonia.VMDebug.Controls;

/// <summary>
/// Displays the single current VM frame output and overlay.
/// </summary>
public partial class VideoSceneView : UserControl
{
    public VideoSceneView()
    {
        InitializeComponent();
    }

    public void UpdateFrame(Mat frame, FrameOverlaySnapshot overlay)
    {
        SkiaFrameView_MainFrame.UpdateFrame(frame);
        SkiaOverlayView_MainOverlay.UpdateOverlay(overlay);
    }
}
