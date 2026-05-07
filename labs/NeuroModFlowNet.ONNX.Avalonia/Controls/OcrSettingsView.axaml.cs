using Avalonia.Controls;
using NeuroModFlowNet.ONNX.Avalonia.Runtime;

namespace NeuroModFlowNet.ONNX.Avalonia.Controls;

/// <summary>
/// EN: Represents the settings panel for realtime OCR, ROI extraction, and ROI image processing parameters.
/// RU: Представляет панель настроек realtime OCR, вырезания ROI и обработки ROI-изображений.
/// </summary>
/// <remarks>
/// EN: The control exposes simple events for start, stop, and option changes. It edits a shared <see cref="RecognitionOptions"/>
/// instance and refreshes labels after each button click. Thread safety is not guaranteed; use it from the UI thread.
/// RU: Control отдает наружу простые события start, stop и option changes. Он редактирует общий объект
/// <see cref="RecognitionOptions"/> и обновляет подписи после каждого нажатия. Потокобезопасность не гарантируется;
/// используйте его из UI thread.
/// </remarks>
public partial class OcrSettingsView : UserControl
{
    public event EventHandler? OptionsChanged;

    RecognitionOptions? options;

    public OcrSettingsView()
    {
        InitializeComponent();

        Button_RecognitionInputWidthMinus.Click += (_, _) => ChangeOptions(item => item.AdjustRecognitionInputWidth(-32));
        Button_RecognitionInputWidthPlus.Click += (_, _) => ChangeOptions(item => item.AdjustRecognitionInputWidth(32));
        Button_RoiHeightScaleMinus.Click += (_, _) => ChangeOptions(item => item.AdjustRoiHeightScale(-0.25f));
        Button_RoiHeightScalePlus.Click += (_, _) => ChangeOptions(item => item.AdjustRoiHeightScale(0.25f));
        Button_RoiDisplayScaleMinus.Click += (_, _) => ChangeOptions(item => item.AdjustRoiDisplayScale(-0.25));
        Button_RoiDisplayScalePlus.Click += (_, _) => ChangeOptions(item => item.AdjustRoiDisplayScale(0.25));
        Button_BrightnessMinus.Click += (_, _) => ChangeOptions(item => item.AdjustBrightness(-5));
        Button_BrightnessPlus.Click += (_, _) => ChangeOptions(item => item.AdjustBrightness(5));
        Button_ContrastMinus.Click += (_, _) => ChangeOptions(item => item.AdjustContrast(-5));
        Button_ContrastPlus.Click += (_, _) => ChangeOptions(item => item.AdjustContrast(5));
        Button_GammaMinus.Click += (_, _) => ChangeOptions(item => item.AdjustGamma(-0.25));
        Button_GammaPlus.Click += (_, _) => ChangeOptions(item => item.AdjustGamma(0.25));
        Button_RotateProcessingStages.Click += (_, _) => ChangeOptions(item => item.RotateProcessingStages());
        CheckBox_RoiProcessingEnabled.IsCheckedChanged += (_, _) => ChangeOptions(item => item.ProcessingEnabled = CheckBox_RoiProcessingEnabled.IsChecked == true);
        ComboBox_RecognitionTextMode.SelectionChanged += (_, _) => ChangeOptions(item => item.RecognitionMode = ComboBox_RecognitionTextMode.SelectedIndex + 1);
    }

    public void BindOptions(RecognitionOptions recognitionOptions)
    {
        options = recognitionOptions;
        Refresh();
    }

    public void Refresh()
    {
        if(options is null) return;

        TextBlock_RecognitionInputWidth.Text = $"Rec width: {options.RecognitionInputWidth}";
        TextBlock_RoiHeightScale.Text = $"ROI height scale: {options.RoiHeightScale:F2}";
        TextBlock_RoiDisplayScale.Text = $"ROI display scale: x{options.RoiDisplayScale:F2}";
        TextBlock_Brightness.Text = $"Brightness: {options.Brightness:F0}";
        TextBlock_Contrast.Text = $"Contrast: {options.ContrastPercent:F0}%";
        TextBlock_Gamma.Text = $"Gamma: {options.Gamma:F2}";
        TextBlock_ProcessingPipeline.Text = $"Pipeline: {options.GetProcessingStageList()}";

        CheckBox_RoiProcessingEnabled.IsChecked = options.ProcessingEnabled;
        ComboBox_RecognitionTextMode.SelectedIndex = Math.Clamp(options.RecognitionMode - 1, 0, 2);
    }

    private void ChangeOptions(Action<RecognitionOptions> change)
    {
        if(options is null) return;

        change(options);
        Refresh();
        OptionsChanged?.Invoke(this, EventArgs.Empty);
    }
}
