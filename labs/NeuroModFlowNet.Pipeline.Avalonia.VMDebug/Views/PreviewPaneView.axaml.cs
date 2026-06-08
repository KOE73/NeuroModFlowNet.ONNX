using Avalonia.Controls;
using NeuroModFlowNet.Pipeline.Avalonia.VMDebug.Runtime;
using NeuroModFlowNet.Pipeline.Avalonia.VMDebug.ViewModels;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.Avalonia.VMDebug.Views;

/// <summary>
/// Displays the current VM frame. The view subscribes to frame events because OpenCV/Skia rendering is visual state,
/// while the VM and trace state remain in the workspace view model.
/// </summary>
public partial class PreviewPaneView : UserControl
{
    VmDebugWorkspaceViewModel? viewModel;

    public PreviewPaneView()
    {
        InitializeComponent();
        DataContextChanged += OnDataContextChanged;
        ShowInitialFrame();
    }

    void OnDataContextChanged(object? sender, EventArgs e)
    {
        if(viewModel is not null)
            viewModel.FrameUpdated -= OnFrameUpdated;

        viewModel = DataContext as VmDebugWorkspaceViewModel;
        if(viewModel is null)
            return;

        viewModel.FrameUpdated += OnFrameUpdated;
        if(viewModel.TryGetLatestFrame(out Mat? frame, out FrameOverlaySnapshot? overlay) && frame is not null && overlay is not null)
            VideoSceneView_Main.UpdateFrame(frame, overlay);
    }

    void OnFrameUpdated(Mat frame, FrameOverlaySnapshot overlay)
    {
        VideoSceneView_Main.UpdateFrame(frame, overlay);
    }

    void ShowInitialFrame()
    {
        using var placeholder = new Mat(720, 1280, MatType.CV_8UC3, Scalar.Black);
        Cv2.PutText(
            placeholder,
            "VM Debug",
            new OpenCvSharp.Point(500, 360),
            HersheyFonts.HersheySimplex,
            1.5,
            new Scalar(80, 80, 80),
            2,
            LineTypes.AntiAlias);

        using var bgra = new Mat();
        Cv2.CvtColor(placeholder, bgra, ColorConversionCodes.BGR2BGRA);
        VideoSceneView_Main.UpdateFrame(bgra, new FrameOverlaySnapshot(bgra.Width, bgra.Height, [], []));
    }
}
