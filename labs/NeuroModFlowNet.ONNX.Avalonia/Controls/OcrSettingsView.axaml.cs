using Avalonia;
using Avalonia.Controls;
using Avalonia.Layout;
using Avalonia.Media;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.ONNX.Avalonia.Runtime;

namespace NeuroModFlowNet.ONNX.Avalonia.Controls;

/// <summary>
/// EN: Represents the settings panel for realtime OCR, ROI extraction, and ROI image processing parameters.
/// </summary>
/// <remarks>
/// EN: The control exposes simple events for start, stop, and option changes. It edits a shared <see cref="RecognitionOptions"/>
/// instance and refreshes labels after each button click. Thread safety is not guaranteed; use it from the UI thread.
/// </remarks>
public partial class OcrSettingsView : UserControl
{
    public event EventHandler? OptionsChanged;

    RecognitionOptions? options;
    bool isRefreshing;

    public OcrSettingsView()
    {
        InitializeComponent();

        Button_RecognitionInputWidthMinus.Click += (_, _) => ChangeOptions(item => item.AdjustRecognitionInputWidth(-32));
        Button_RecognitionInputWidthPlus.Click += (_, _) => ChangeOptions(item => item.AdjustRecognitionInputWidth(32));
        Button_RoiHeightScaleMinus.Click += (_, _) => ChangeOptions(item => item.AdjustRoiHeightScale(-0.25f));
        Button_RoiHeightScalePlus.Click += (_, _) => ChangeOptions(item => item.AdjustRoiHeightScale(0.25f));
        CheckBox_AdaptiveRoiHeight.IsCheckedChanged += (_, _) => ChangeOptions(item => item.AdaptiveRoiHeightEnabled = CheckBox_AdaptiveRoiHeight.IsChecked == true);
        Button_RoiAdaptiveReset.Click += (_, _) => ChangeOptions(item => item.ResetAdaptiveRoiHeight());
        Button_RoiAdaptiveBasePadMinus.Click += (_, _) => ChangeOptions(item => item.AdjustRoiAdaptiveBasePad(-1));
        Button_RoiAdaptiveBasePadPlus.Click += (_, _) => ChangeOptions(item => item.AdjustRoiAdaptiveBasePad(1));
        Button_RoiAdaptivePadRatioMinus.Click += (_, _) => ChangeOptions(item => item.AdjustRoiAdaptivePadRatio(-0.05f));
        Button_RoiAdaptivePadRatioPlus.Click += (_, _) => ChangeOptions(item => item.AdjustRoiAdaptivePadRatio(0.05f));
        Button_RoiAdaptiveMaxPadMinus.Click += (_, _) => ChangeOptions(item => item.AdjustRoiAdaptiveMaxPad(-1));
        Button_RoiAdaptiveMaxPadPlus.Click += (_, _) => ChangeOptions(item => item.AdjustRoiAdaptiveMaxPad(1));
        Button_RoiDisplayScaleMinus.Click += (_, _) => ChangeOptions(item => item.AdjustRoiDisplayScale(-0.25));
        Button_RoiDisplayScalePlus.Click += (_, _) => ChangeOptions(item => item.AdjustRoiDisplayScale(0.25));
        Button_AnalyzerMaxRegionsMinus.Click += (_, _) => ChangeOptions(item => item.AdjustMaxRegionsPerFrame(-8));
        Button_AnalyzerMaxRegionsPlus.Click += (_, _) => ChangeOptions(item => item.AdjustMaxRegionsPerFrame(8));
        Button_AnalyzerHeightLimitsReset.Click += (_, _) => ChangeOptions(item => item.ResetRegionHeightLimits());
        Button_AnalyzerMinHeightMinus.Click += (_, _) => ChangeOptions(item => item.AdjustMinRegionHeight(-2));
        Button_AnalyzerMinHeightPlus.Click += (_, _) => ChangeOptions(item => item.AdjustMinRegionHeight(2));
        Button_AnalyzerMaxHeightMinus.Click += (_, _) => ChangeOptions(item => item.AdjustMaxRegionHeight(-2));
        Button_AnalyzerMaxHeightPlus.Click += (_, _) => ChangeOptions(item => item.AdjustMaxRegionHeight(2));
        Button_AnalyzerWidthLimitsReset.Click += (_, _) => ChangeOptions(item => item.ResetRegionWidthLimits());
        Button_AnalyzerMinWidthMinus.Click += (_, _) => ChangeOptions(item => item.AdjustMinRegionWidth(-4));
        Button_AnalyzerMinWidthPlus.Click += (_, _) => ChangeOptions(item => item.AdjustMinRegionWidth(4));
        Button_AnalyzerMaxWidthMinus.Click += (_, _) => ChangeOptions(item => item.AdjustMaxRegionWidth(-16));
        Button_AnalyzerMaxWidthPlus.Click += (_, _) => ChangeOptions(item => item.AdjustMaxRegionWidth(16));
        Button_AnalyzerAspectLimitsReset.Click += (_, _) => ChangeOptions(item => item.ResetRegionAspectRatioLimits());
        Button_AnalyzerMinAspectMinus.Click += (_, _) => ChangeOptions(item => item.AdjustMinRegionAspectRatio(-0.5f));
        Button_AnalyzerMinAspectPlus.Click += (_, _) => ChangeOptions(item => item.AdjustMinRegionAspectRatio(0.5f));
        Button_AnalyzerMaxAspectMinus.Click += (_, _) => ChangeOptions(item => item.AdjustMaxRegionAspectRatio(-0.5f));
        Button_AnalyzerMaxAspectPlus.Click += (_, _) => ChangeOptions(item => item.AdjustMaxRegionAspectRatio(0.5f));
        CheckBox_AnalyzerOverlapSuppression.IsCheckedChanged += (_, _) => ChangeOptions(item => item.OverlapSuppressionEnabled = CheckBox_AnalyzerOverlapSuppression.IsChecked == true);
        Button_AnalyzerOverlapRatioMinus.Click += (_, _) => ChangeOptions(item => item.AdjustOverlapSuppressionRatio(-0.05f));
        Button_AnalyzerOverlapRatioPlus.Click += (_, _) => ChangeOptions(item => item.AdjustOverlapSuppressionRatio(0.05f));
        CheckBox_AnalyzerLineMerge.IsCheckedChanged += (_, _) => ChangeOptions(item => item.LineMergeEnabled = CheckBox_AnalyzerLineMerge.IsChecked == true);
        Button_AnalyzerMergeAngleMinus.Click += (_, _) => ChangeOptions(item => item.AdjustMergeAngleDeltaDegrees(-1));
        Button_AnalyzerMergeAnglePlus.Click += (_, _) => ChangeOptions(item => item.AdjustMergeAngleDeltaDegrees(1));
        Button_AnalyzerMergeNormalMinus.Click += (_, _) => ChangeOptions(item => item.AdjustMergeNormalOffsetInHeights(-0.05f));
        Button_AnalyzerMergeNormalPlus.Click += (_, _) => ChangeOptions(item => item.AdjustMergeNormalOffsetInHeights(0.05f));
        Button_AnalyzerMergeGapMinus.Click += (_, _) => ChangeOptions(item => item.AdjustMergeGapInHeights(-0.25f));
        Button_AnalyzerMergeGapPlus.Click += (_, _) => ChangeOptions(item => item.AdjustMergeGapInHeights(0.25f));
        Button_AnalyzerMergeHeightRatioMinus.Click += (_, _) => ChangeOptions(item => item.AdjustMergeHeightRatio(-0.1f));
        Button_AnalyzerMergeHeightRatioPlus.Click += (_, _) => ChangeOptions(item => item.AdjustMergeHeightRatio(0.1f));
        Button_AnalyzerMergeCoverageMinus.Click += (_, _) => ChangeOptions(item => item.AdjustMinimumMergedCoverageRatio(-0.05f));
        Button_AnalyzerMergeCoveragePlus.Click += (_, _) => ChangeOptions(item => item.AdjustMinimumMergedCoverageRatio(0.05f));
        Button_AnalyzerMaxMergedWidthReset.Click += (_, _) => ChangeOptions(item => item.ResetMaxMergedRegionWidth());
        Button_AnalyzerMaxMergedWidthMinus.Click += (_, _) => ChangeOptions(item => item.AdjustMaxMergedRegionWidth(-16));
        Button_AnalyzerMaxMergedWidthPlus.Click += (_, _) => ChangeOptions(item => item.AdjustMaxMergedRegionWidth(16));
        Button_AnalyzerMaxMergedAspectReset.Click += (_, _) => ChangeOptions(item => item.ResetMaxMergedRegionAspectRatio());
        Button_AnalyzerMaxMergedAspectMinus.Click += (_, _) => ChangeOptions(item => item.AdjustMaxMergedRegionAspectRatio(-0.5f));
        Button_AnalyzerMaxMergedAspectPlus.Click += (_, _) => ChangeOptions(item => item.AdjustMaxMergedRegionAspectRatio(0.5f));
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

        isRefreshing = true;

        TextBlock_RecognitionInputWidth.Text = $"Rec width: {options.RecognitionInputWidth}";
        TextBlock_RoiHeightScale.Text = $"ROI fallback scale: {options.RoiHeightScale:F2}";
        CheckBox_AdaptiveRoiHeight.IsChecked = options.AdaptiveRoiHeightEnabled;
        TextBlock_RoiAdaptiveBasePad.Text = $"adaptive base pad: {options.RoiAdaptiveBasePad:F0}px";
        TextBlock_RoiAdaptivePadRatio.Text = $"adaptive pad ratio: {options.RoiAdaptivePadRatio:F2}";
        TextBlock_RoiAdaptiveMaxPad.Text = $"adaptive max pad: {options.RoiAdaptiveMaxPad:F0}px";
        TextBlock_RoiDisplayScale.Text = $"ROI display scale: x{options.RoiDisplayScale:F2}";
        TextBlock_ProcessingPipeline.Text = $"Pipeline: {options.GetProcessingStageList()}";
        RefreshAnalyzerSettings();

        RebuildAvailableProcessingList();
        RebuildSelectedProcessingCards();
        CheckBox_RoiProcessingEnabled.IsChecked = options.ProcessingEnabled;
        ComboBox_RecognitionTextMode.SelectedIndex = Math.Clamp(options.RecognitionMode - 1, 0, 2);

        isRefreshing = false;
    }

    private void ChangeOptions(Action<RecognitionOptions> change, bool raiseOptionsChanged = true)
    {
        if(isRefreshing) return;
        if(options is null) return;

        change(options);
        Refresh();
        if(raiseOptionsChanged)
            OptionsChanged?.Invoke(this, EventArgs.Empty);
    }

    #region OBB analyzer panel

    private void RefreshAnalyzerSettings()
    {
        if(options is null) return;

        TextBlock_AnalyzerMaxRegions.Text = $"max regions/frame: {options.MaxRegionsPerFrame}";
        TextBlock_AnalyzerHeightLimits.Text = $"height: {options.MinRegionHeight:F0}..{FormatOptionalPixels(options.MaxRegionHeight)}";
        TextBlock_AnalyzerWidthLimits.Text = $"width: {options.MinRegionWidth:F0}..{FormatOptionalPixels(options.MaxRegionWidth)}";
        TextBlock_AnalyzerAspectLimits.Text = $"aspect: {options.MinRegionAspectRatio:F1}..{FormatOptionalRatio(options.MaxRegionAspectRatio)}";
        CheckBox_AnalyzerOverlapSuppression.IsChecked = options.OverlapSuppressionEnabled;
        TextBlock_AnalyzerOverlapRatio.Text = $"overlap ratio: {options.OverlapSuppressionRatio:F2}";
        CheckBox_AnalyzerLineMerge.IsChecked = options.LineMergeEnabled;
        TextBlock_AnalyzerMergeAngle.Text = $"angle delta: {options.MergeAngleDeltaDegrees:F0} deg";
        TextBlock_AnalyzerMergeNormal.Text = $"normal offset: {options.MergeNormalOffsetInHeights:F2} h";
        TextBlock_AnalyzerMergeGap.Text = $"axis gap: {options.MergeGapInHeights:F2} h";
        TextBlock_AnalyzerMergeHeightRatio.Text = $"height ratio: {options.MergeHeightRatio:F1}";
        TextBlock_AnalyzerMergeCoverage.Text = $"coverage: {options.MinimumMergedCoverageRatio:F2}";
        TextBlock_AnalyzerMaxMergedWidth.Text = $"max merged width: {FormatOptionalPixels(options.MaxMergedRegionWidth)}";
        TextBlock_AnalyzerMaxMergedAspect.Text = $"max merged aspect: {FormatOptionalMergedAspect(options)}";
    }

    private static string FormatOptionalPixels(float value) =>
        float.IsPositiveInfinity(value) ? "off" : $"{value:F0}px";

    private static string FormatOptionalRatio(float value) =>
        float.IsPositiveInfinity(value) ? "off" : $"{value:F1}";

    private static string FormatOptionalMergedAspect(RecognitionOptions recognitionOptions) =>
        float.IsPositiveInfinity(recognitionOptions.MaxMergedRegionAspectRatio)
            ? $"rec {recognitionOptions.RecognitionInputWidth / (float)recognitionOptions.RecognitionInputHeight:F1}"
            : $"{recognitionOptions.MaxMergedRegionAspectRatio:F1}";

    #endregion

    private void RebuildAvailableProcessingList()
    {
        Panel_AvailableProcessingList.Children.Clear();

        AddAvailableProcessingRow("Brightness/Contrast", TextRegionProcessingStageKind.BrightnessContrast);
        AddAvailableProcessingRow("Gamma", TextRegionProcessingStageKind.Gamma);
        AddAvailableProcessingRow("Gray RGB", TextRegionProcessingStageKind.Grayscale);
        AddAvailableProcessingRow("Black/White", TextRegionProcessingStageKind.Threshold);
        AddAvailableProcessingRow("GaussianBlur", TextRegionProcessingStageKind.GaussianBlur);
        AddAvailableProcessingRow("Sharpen", TextRegionProcessingStageKind.Sharpen);
        AddAvailableProcessingRow("AutoContrast", TextRegionProcessingStageKind.AutoContrast);
    }

    private void AddAvailableProcessingRow(string name, TextRegionProcessingStageKind stageKind)
    {
        var row = new Grid
        {
            ColumnDefinitions = new ColumnDefinitions("*,Auto"),
            ColumnSpacing = 2,
        };

        row.Children.Add(new TextBlock
        {
            Text = name,
            VerticalAlignment = VerticalAlignment.Center,
            FontSize = 11,
        });

        Button addButton = CreateCompactButton("+");
        addButton.Click += (_, _) => ChangeOptions(item => item.AddProcessingStage(stageKind));
        Grid.SetColumn(addButton, 1);
        row.Children.Add(addButton);

        Panel_AvailableProcessingList.Children.Add(row);
    }

    private void RebuildSelectedProcessingCards()
    {
        Panel_SelectedProcessingCards.Children.Clear();
        if(options is null) return;

        IReadOnlyList<ITextRegionProcessingStage> stages = options.GetProcessingStages();
        if(stages.Count == 0)
        {
            Panel_SelectedProcessingCards.Children.Add(new TextBlock
            {
                Text = "No processing modules selected",
                FontSize = 10,
                Foreground = Brushes.Gray,
            });
            return;
        }

        for(int index = 0; index < stages.Count; index++)
            Panel_SelectedProcessingCards.Children.Add(CreateProcessingCard(index, stages[index]));
    }

    private Control CreateProcessingCard(int index, ITextRegionProcessingStage stage)
    {
        var card = new Border
        {
            BorderBrush = index == options?.SelectedProcessingStageIndex ? Brushes.DeepSkyBlue : Brushes.DimGray,
            BorderThickness = new Thickness(index == options?.SelectedProcessingStageIndex ? 2 : 1),
            CornerRadius = new CornerRadius(3),
            Padding = new Thickness(3),
        };

        var grid = new Grid
        {
            ColumnDefinitions = new ColumnDefinitions("Auto,*"),
            ColumnSpacing = 3,
        };

        grid.Children.Add(CreateMoveButtons(index));

        StackPanel settingsPanel = CreateSettingsPanel(index, stage);
        Grid.SetColumn(settingsPanel, 1);
        grid.Children.Add(settingsPanel);

        card.Child = grid;
        return card;
    }

    private Grid CreateMoveButtons(int index)
    {
        var panel = new Grid
        {
            RowDefinitions = new RowDefinitions("Auto,*,Auto"),
            VerticalAlignment = VerticalAlignment.Stretch,
        };

        Button upButton = CreateCompactButton("↑");
        upButton.Click += (_, _) => ChangeStageOptions(index, item => item.MoveSelectedProcessingStageUp());
        panel.Children.Add(upButton);

        Button downButton = CreateCompactButton("↓");
        downButton.Click += (_, _) => ChangeStageOptions(index, item => item.MoveSelectedProcessingStageDown());
        Grid.SetRow(downButton, 2);
        panel.Children.Add(downButton);

        return panel;
    }

    private StackPanel CreateSettingsPanel(int index, ITextRegionProcessingStage stage)
    {
        var panel = new StackPanel
        {
            Spacing = 2,
        };

        var titleRow = new Grid
        {
            ColumnDefinitions = new ColumnDefinitions("*,Auto"),
            ColumnSpacing = 2,
        };

        titleRow.Children.Add(new TextBlock
        {
            Text = $"Processing {index + 1}: {stage.Name}",
            FontSize = 11,
            FontWeight = FontWeight.SemiBold,
        });

        Button removeButton = CreateIconButton("🗑");
        removeButton.Click += (_, _) => ChangeStageOptions(index, item => item.RemoveSelectedProcessingStage());
        Grid.SetColumn(removeButton, 1);
        titleRow.Children.Add(removeButton);

        panel.Children.Add(titleRow);

        if(stage is ITextRegionBrightnessContrastSettings brightnessContrastSettings)
        {
            panel.Children.Add(CreateValueRow(
                $"brightness {brightnessContrastSettings.Brightness:F0}",
                "0",
                () => ChangeStageOptions(index, item => item.ResetSelectedBrightness()),
                () => ChangeStageOptions(index, item => item.AdjustSelectedBrightness(-5)),
                () => ChangeStageOptions(index, item => item.AdjustSelectedBrightness(5))));

            panel.Children.Add(CreateValueRow(
                $"contrast   {brightnessContrastSettings.ContrastPercent:F0}%",
                "100",
                () => ChangeStageOptions(index, item => item.ResetSelectedContrast()),
                () => ChangeStageOptions(index, item => item.AdjustSelectedContrast(-5)),
                () => ChangeStageOptions(index, item => item.AdjustSelectedContrast(5))));
        }

        if(stage is ITextRegionGammaCorrectionSettings gammaSettings)
        {
            panel.Children.Add(CreateValueRow(
                $"gamma      {gammaSettings.Gamma:F2}",
                "1",
                () => ChangeStageOptions(index, item => item.ResetSelectedGamma()),
                () => ChangeStageOptions(index, item => item.AdjustSelectedGamma(-0.25)),
                () => ChangeStageOptions(index, item => item.AdjustSelectedGamma(0.25))));
        }

        if(stage is ITextRegionGrayscaleSettings grayscaleSettings)
        {
            var row = new StackPanel
            {
                Orientation = Orientation.Horizontal,
                Spacing = 5,
            };

            row.Children.Add(CreateChannelCheckBox("R", grayscaleSettings.UseRed, () => ChangeStageOptions(index, item => item.ToggleSelectedRedChannel())));
            row.Children.Add(CreateChannelCheckBox("G", grayscaleSettings.UseGreen, () => ChangeStageOptions(index, item => item.ToggleSelectedGreenChannel())));
            row.Children.Add(CreateChannelCheckBox("B", grayscaleSettings.UseBlue, () => ChangeStageOptions(index, item => item.ToggleSelectedBlueChannel())));
            panel.Children.Add(row);
        }

        if(stage is ITextRegionThresholdSettings thresholdSettings)
        {
            panel.Children.Add(CreateChannelCheckBox(
                "auto Otsu",
                thresholdSettings.UseOtsu,
                () => ChangeStageOptions(index, item => item.ToggleSelectedOtsu())));

            panel.Children.Add(CreateValueRow(
                thresholdSettings.UseOtsu ? $"threshold  auto ({thresholdSettings.Threshold:F0})" : $"threshold  {thresholdSettings.Threshold:F0}",
                "128",
                () => ChangeStageOptions(index, item => item.ResetSelectedThreshold()),
                () => ChangeStageOptions(index, item => item.AdjustSelectedThreshold(-5)),
                () => ChangeStageOptions(index, item => item.AdjustSelectedThreshold(5))));
        }

        if(stage is ITextRegionGaussianBlurSettings gaussianBlurSettings)
        {
            panel.Children.Add(CreateValueRow(
                $"kernel     {gaussianBlurSettings.KernelSize}",
                "reset",
                () => ChangeStageOptions(index, item => item.ResetSelectedGaussian()),
                () => ChangeStageOptions(index, item => item.AdjustSelectedGaussianKernel(-2)),
                () => ChangeStageOptions(index, item => item.AdjustSelectedGaussianKernel(2))));

            panel.Children.Add(CreateValueRow(
                $"sigma      {gaussianBlurSettings.Sigma:F2}",
                "",
                null,
                () => ChangeStageOptions(index, item => item.AdjustSelectedGaussianSigma(-0.25)),
                () => ChangeStageOptions(index, item => item.AdjustSelectedGaussianSigma(0.25))));
        }

        if(stage is ITextRegionSharpenSettings sharpenSettings)
        {
            panel.Children.Add(CreateValueRow(
                $"kernel     {sharpenSettings.KernelSize}",
                "reset",
                () => ChangeStageOptions(index, item => item.ResetSelectedSharpen()),
                () => ChangeStageOptions(index, item => item.AdjustSelectedSharpenKernel(-2)),
                () => ChangeStageOptions(index, item => item.AdjustSelectedSharpenKernel(2))));

            panel.Children.Add(CreateValueRow(
                $"sigma      {sharpenSettings.Sigma:F2}",
                "",
                null,
                () => ChangeStageOptions(index, item => item.AdjustSelectedSharpenSigma(-0.25)),
                () => ChangeStageOptions(index, item => item.AdjustSelectedSharpenSigma(0.25))));

            panel.Children.Add(CreateValueRow(
                $"amount     {sharpenSettings.Amount:F2}",
                "",
                null,
                () => ChangeStageOptions(index, item => item.AdjustSelectedSharpenAmount(-0.25)),
                () => ChangeStageOptions(index, item => item.AdjustSelectedSharpenAmount(0.25))));
        }

        return panel;
    }

    private Grid CreateValueRow(
        string label,
        string resetText,
        Action? resetAction,
        Action minusAction,
        Action plusAction)
    {
        var row = new Grid
        {
            ColumnDefinitions = new ColumnDefinitions("*,Auto,Auto,Auto"),
            ColumnSpacing = 2,
        };

        row.Children.Add(new TextBlock
        {
            Text = label,
            FontSize = 11,
            VerticalAlignment = VerticalAlignment.Center,
        });

        Button resetButton = CreateCompactButton(resetText);
        resetButton.IsVisible = resetAction is not null;
        resetButton.Click += (_, _) => resetAction?.Invoke();
        Grid.SetColumn(resetButton, 1);
        row.Children.Add(resetButton);

        Button minusButton = CreateCompactButton("-");
        minusButton.Click += (_, _) => minusAction();
        Grid.SetColumn(minusButton, 2);
        row.Children.Add(minusButton);

        Button plusButton = CreateCompactButton("+");
        plusButton.Click += (_, _) => plusAction();
        Grid.SetColumn(plusButton, 3);
        row.Children.Add(plusButton);

        return row;
    }

    private CheckBox CreateChannelCheckBox(string text, bool isChecked, Action change)
    {
        var checkBox = new CheckBox
        {
            Content = text,
            IsChecked = isChecked,
            FontSize = 11,
            Padding = new Thickness(0),
        };

        checkBox.IsCheckedChanged += (_, _) => change();
        return checkBox;
    }

    private Button CreateCompactButton(string text) =>
        new()
        {
            Content = text,
            Padding = new Thickness(2, 0),
            MinHeight = 16,
            MinWidth = 20,
            FontSize = 9,
            FontFamily = new FontFamily("Consolas"),
            HorizontalContentAlignment = HorizontalAlignment.Center,
            VerticalContentAlignment = VerticalAlignment.Center,
        };

    private Button CreateIconButton(string text) =>
        new()
        {
            Content = text,
            Padding = new Thickness(1),
            MinHeight = 16,
            MinWidth = 16,
            FontSize = 12,
            Background = Brushes.Transparent,
            BorderBrush = Brushes.Transparent,
            BorderThickness = new Thickness(0),
            HorizontalContentAlignment = HorizontalAlignment.Center,
            VerticalContentAlignment = VerticalAlignment.Center,
        };

    private void ChangeStageOptions(int index, Action<RecognitionOptions> change)
    {
        ChangeOptions(item =>
        {
            item.SelectProcessingStage(index);
            change(item);
        });
    }
}
