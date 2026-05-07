using Avalonia;
using Avalonia.Controls;
using Avalonia.Layout;
using Avalonia.Media;
using NeuroModFlowNet.ONNX.Avalonia.Runtime;
using OpenCvSharp;

namespace NeuroModFlowNet.ONNX.Avalonia.Controls;


/// <summary>
/// Represents a user control for displaying the main video frame, neural network overlay, and OCR recognition rows.
/// </summary>
/// <remarks>Use this control as the center scene of the realtime UI. The upper area shows the camera frame and overlay,
/// while the lower resizable area shows a list of OCR results with ROI image previews on the left and recognized text on
/// the right. Thread safety is not guaranteed; updates should be performed on the UI thread.</remarks>
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

    public void UpdateRecognition(IReadOnlyList<RealTimeRecognitionItemData> recognitionItems)
    {
        ListBox_RecognitionResults.ItemsSource = recognitionItems
            .Select(CreateRecognitionResultRow)
            .ToArray();
    }

    private static Control CreateRecognitionResultRow(RealTimeRecognitionItemData recognitionItem)
    {
        double displayScale = recognitionItem.DisplayScale;

        var skiaFrameView = new Rendering.SkiaFrameView
        {
            Width = Math.Max(1, recognitionItem.Roi.Width * displayScale),
            Height = Math.Max(1, recognitionItem.Roi.Height * displayScale),
            HorizontalAlignment = HorizontalAlignment.Left,
            VerticalAlignment = VerticalAlignment.Top,
        };
        skiaFrameView.UpdateFrame(recognitionItem.Roi);

        var textBlock = new TextBlock
        {
            Text = recognitionItem.Text,
            TextWrapping = TextWrapping.Wrap,
            VerticalAlignment = VerticalAlignment.Center,
            FontFamily = new FontFamily("Consolas"),
            FontSize = 13,
        };

        var grid = new Grid
        {
            ColumnDefinitions = new ColumnDefinitions("Auto,*"),
            MinHeight = Math.Max(1, recognitionItem.Roi.Height * displayScale) + 2,
            ColumnSpacing = 8,
            Margin = new Thickness(0, 0, 0, 1),
        };

        grid.Children.Add(new Border
        {
            Background = Brushes.Black,
            BorderBrush = Brushes.DimGray,
            BorderThickness = new Thickness(1),
            ClipToBounds = true,
            Child = skiaFrameView,
        });

        Grid.SetColumn(textBlock, 1);
        grid.Children.Add(textBlock);

        return grid;
    }
}
