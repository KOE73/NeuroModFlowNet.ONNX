using Avalonia.Controls;
using Avalonia.Media;
using NeuroModFlowNet.ONNX.Avalonia.Runtime;

namespace NeuroModFlowNet.ONNX.Avalonia.Controls;

/// <summary>
/// EN: Represents the left inference selector for choosing which model pipelines run on the shared scene.
/// RU: Представляет левый inference selector для выбора model pipeline-ов, которые запускаются на общей сцене.
/// </summary>
/// <remarks>
/// EN: Checked slots are executed by the background engine and drawn on the overlay. Disabled future slots are shown as
/// placeholders so the operator can see the intended surface without invoking missing runners.
/// RU: Отмеченные слоты выполняются background engine и рисуются поверх кадра. Выключенные будущие слоты показаны как
/// placeholders, чтобы было видно будущую поверхность без вызова отсутствующих runners.
/// </remarks>
public partial class ModeNavigationView : UserControl
{
    public event EventHandler? StartRequested;
    public event EventHandler? StopRequested;
    public event EventHandler? OptionsChanged;

    RecognitionOptions? options;
    bool isRefreshing;

    public ModeNavigationView()
    {
        InitializeComponent();

        Button_RuntimeStart.Click += (_, _) => StartRequested?.Invoke(this, EventArgs.Empty);
        Button_RuntimeStop.Click += (_, _) => StopRequested?.Invoke(this, EventArgs.Empty);
        Button_FrameWidthMinus.Click += (_, _) => ChangeOptions(item => item.AdjustFrameWidth(-40));
        Button_FrameWidthPlus.Click += (_, _) => ChangeOptions(item => item.AdjustFrameWidth(40));
        Button_RecognitionBatchMinus.Click += (_, _) => ChangeOptions(item => item.AdjustBatchSize(-1));
        Button_RecognitionBatchPlus.Click += (_, _) => ChangeOptions(item => item.AdjustBatchSize(1));
        CheckBox_InferenceOcr.IsCheckedChanged += (_, _) => ChangeOptions(item => item.InferenceSelection.OcrEnabled = CheckBox_InferenceOcr.IsChecked == true);
        CheckBox_InferenceBox.IsCheckedChanged += (_, _) => ChangeOptions(item => item.InferenceSelection.BoxDetectionEnabled = CheckBox_InferenceBox.IsChecked == true);
        CheckBox_InferenceObb.IsCheckedChanged += (_, _) => ChangeOptions(item => item.InferenceSelection.ObbDetectionEnabled = CheckBox_InferenceObb.IsChecked == true);
        CheckBox_InferenceSegmentation.IsCheckedChanged += (_, _) => ChangeOptions(item => item.InferenceSelection.SegmentationEnabled = CheckBox_InferenceSegmentation.IsChecked == true);
        CheckBox_InferenceClassification.IsCheckedChanged += (_, _) => ChangeOptions(item => item.InferenceSelection.ClassificationEnabled = CheckBox_InferenceClassification.IsChecked == true);
        CheckBox_InferencePose.IsCheckedChanged += (_, _) => ChangeOptions(item => item.InferenceSelection.PoseEnabled = CheckBox_InferencePose.IsChecked == true);
    }

    private void ChangeOptions(Action<RecognitionOptions> change)
    {
        if(options is null || isRefreshing) return;

        change(options);
        Refresh();
        OptionsChanged?.Invoke(this, EventArgs.Empty);
    }

    public void BindOptions(RecognitionOptions recognitionOptions)
    {
        options = recognitionOptions;
        Refresh();
    }

    public void Refresh()
    {
        if(options is null) return;

        isRefreshing = true;
        try
        {
            CheckBox_InferenceOcr.IsChecked = options.InferenceSelection.OcrEnabled;
            CheckBox_InferenceBox.IsChecked = options.InferenceSelection.BoxDetectionEnabled;
            CheckBox_InferenceObb.IsChecked = options.InferenceSelection.ObbDetectionEnabled;
            CheckBox_InferenceSegmentation.IsChecked = options.InferenceSelection.SegmentationEnabled;
            CheckBox_InferenceClassification.IsChecked = options.InferenceSelection.ClassificationEnabled;
            CheckBox_InferencePose.IsChecked = options.InferenceSelection.PoseEnabled;
            TextBlock_FrameWidth.Text = $"Display width: {options.FrameWidth}";
            TextBlock_RecognitionBatch.Text = $"Batch: {options.BatchSize}";
        }
        finally
        {
            isRefreshing = false;
        }
    }

    public void SetStatus(string status)
    {
        TextBlock_RuntimeStatus.Text = status;
        ApplyStatusStyle(status);
    }

    public void UpdateModelInfo(IReadOnlyList<RuntimeModelInfo> modelInfos)
    {
        TextBlock_InferenceOcrInfo.Text = FindModelDetails("ocr");
        TextBlock_InferenceBoxInfo.Text = FindModelDetails("box");
        TextBlock_InferenceObbInfo.Text = FindModelDetails("obb");
        TextBlock_InferenceSegmentationInfo.Text = FindModelDetails("seg");
        TextBlock_InferenceClassificationInfo.Text = FindModelDetails("cls");
        TextBlock_InferencePoseInfo.Text = FindModelDetails("pose");

        string FindModelDetails(string key) => modelInfos.FirstOrDefault(item => string.Equals(item.Key, key, StringComparison.OrdinalIgnoreCase))?.Details ?? "not loaded";
    }

    private void ApplyStatusStyle(string status)
    {
        (Color foreground, Color background, Color border) = ResolveStatusColors(status);
        TextBlock_RuntimeStatus.Foreground = new SolidColorBrush(foreground);
        TextBlock_RuntimeStatusIndicator.Foreground = new SolidColorBrush(foreground);
        Border_RuntimeStatus.Background = new SolidColorBrush(background);
        Border_RuntimeStatus.BorderBrush = new SolidColorBrush(border);
    }

    private static (Color Foreground, Color Background, Color Border) ResolveStatusColors(string status)
    {
        if(status.Contains("Running", StringComparison.OrdinalIgnoreCase))
            return (Color.FromRgb(38, 186, 92), Color.FromRgb(20, 48, 32), Color.FromRgb(46, 120, 72));

        if(status.Contains("Starting", StringComparison.OrdinalIgnoreCase) ||
           status.Contains("Initializing", StringComparison.OrdinalIgnoreCase))
            return (Color.FromRgb(255, 190, 62), Color.FromRgb(58, 45, 18), Color.FromRgb(130, 95, 26));

        if(status.Contains("failed", StringComparison.OrdinalIgnoreCase) ||
           status.Contains("Camera only", StringComparison.OrdinalIgnoreCase))
            return (Color.FromRgb(255, 92, 92), Color.FromRgb(62, 24, 24), Color.FromRgb(145, 45, 45));

        return (Color.FromRgb(84, 96, 112), Color.FromRgb(232, 237, 242), Color.FromRgb(184, 194, 204));
    }
}
