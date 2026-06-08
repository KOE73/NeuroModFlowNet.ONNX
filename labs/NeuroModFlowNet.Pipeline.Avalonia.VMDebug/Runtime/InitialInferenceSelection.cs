using System.Configuration;

namespace NeuroModFlowNet.Pipeline.Avalonia.VMDebug.Runtime;

/// <summary>
/// Applies optional lab-only inference slot presets from App.config/App.local.config.
/// </summary>
/// <remarks>
/// This is intentionally kept in the Avalonia lab. The VM core should not know about UI checkboxes, and the ONNX
/// adapter should not know how an operator chooses demo modes. The preset gives automated runs a stable hook without
/// adding desktop UI automation code to the product path.
/// </remarks>
internal static class InitialInferenceSelection
{
    const string EnabledSlotsKey = "EnabledSlots";

    public static void ApplyFromConfig(InferenceSelectionOptions selection)
    {
        ArgumentNullException.ThrowIfNull(selection);

        string? rawValue = ConfigurationManager.AppSettings[EnabledSlotsKey];
        if(string.IsNullOrWhiteSpace(rawValue))
            return;

        selection.OcrEnabled = false;
        selection.BoxDetectionEnabled = false;
        selection.ObbDetectionEnabled = false;
        selection.SegmentationEnabled = false;
        selection.ClassificationEnabled = false;
        selection.PoseEnabled = false;

        foreach(string slot in rawValue.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries))
        {
            if(slot.Equals("OCR", StringComparison.OrdinalIgnoreCase))
                selection.OcrEnabled = true;
            else if(slot.Equals("Box", StringComparison.OrdinalIgnoreCase))
                selection.BoxDetectionEnabled = true;
            else if(slot.Equals("OBB", StringComparison.OrdinalIgnoreCase) || slot.Equals("Obb", StringComparison.OrdinalIgnoreCase))
                selection.ObbDetectionEnabled = true;
            else if(slot.Equals("Seg", StringComparison.OrdinalIgnoreCase) || slot.Equals("Segmentation", StringComparison.OrdinalIgnoreCase))
                selection.SegmentationEnabled = true;
            else if(slot.Equals("Cls", StringComparison.OrdinalIgnoreCase) || slot.Equals("Classification", StringComparison.OrdinalIgnoreCase))
                selection.ClassificationEnabled = true;
            else if(slot.Equals("Pose", StringComparison.OrdinalIgnoreCase))
                selection.PoseEnabled = true;
            // Raw/Det are display/source slots in this lab, not independent model toggles.
        }
    }
}
