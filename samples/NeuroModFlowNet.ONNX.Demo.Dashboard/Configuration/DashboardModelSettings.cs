using NeuroModFlowNet.ONNX;
using System.Configuration;

namespace NeuroModFlowNet.ONNX.Demo.Dashboard;

internal sealed record DashboardModelSettings(
    InferenceBackend Backend,
    int InputSize,
    string Precision,
    bool IsByteBgr,
    IReadOnlySet<ModelSlot> EnabledSlots,
    string BoxModelName,
    string ObbModelName,
    string SegModelName,
    string PoseModelName,
    string ClsModelName)
{
    public static DashboardModelSettings FromConfig()
    {
        return new DashboardModelSettings(
            ReadEnum("DashboardBackend", InferenceBackend.TensorRt),
            ReadInt("DashboardInputSize", 640),
            ReadString("DashboardPrecision", "fp32"),
            ReadBool("DashboardIsByteBgr", false),
            ReadEnabledSlots(),
            ReadString("DashboardBoxModelName", "yolo26n"),
            ReadString("DashboardObbModelName", "yolo26n-obb"),
            ReadString("DashboardSegModelName", "yolo26n-seg"),
            ReadString("DashboardPoseModelName", "yolo26n-pose"),
            ReadString("DashboardClsModelName", "yolo26n-cls"));
    }

    private static string ReadString(string key, string defaultValue)
    {
        string? value = ConfigurationManager.AppSettings[key];
        return string.IsNullOrWhiteSpace(value) ? defaultValue : value.Trim();
    }

    private static int ReadInt(string key, int defaultValue)
    {
        string? value = ConfigurationManager.AppSettings[key];
        return int.TryParse(value, out int result) ? result : defaultValue;
    }

    private static bool ReadBool(string key, bool defaultValue)
    {
        string? value = ConfigurationManager.AppSettings[key];
        return bool.TryParse(value, out bool result) ? result : defaultValue;
    }

    private static TEnum ReadEnum<TEnum>(string key, TEnum defaultValue)
        where TEnum : struct
    {
        string? value = ConfigurationManager.AppSettings[key];
        return Enum.TryParse(value, ignoreCase: true, out TEnum result) ? result : defaultValue;
    }

    private static IReadOnlySet<ModelSlot> ReadEnabledSlots()
    {
        string value = ReadString("DashboardEnabledSlots", "Raw,Box,Obb,Seg,Pose,Cls");
        var enabledSlots = new HashSet<ModelSlot>();

        foreach(string slotName in value.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            if(Enum.TryParse(slotName, ignoreCase: true, out ModelSlot slot))
                enabledSlots.Add(slot);
        }

        return enabledSlots.Count == 0
            ? new HashSet<ModelSlot> { ModelSlot.Raw, ModelSlot.Box, ModelSlot.Obb, ModelSlot.Seg, ModelSlot.Pose, ModelSlot.Cls }
            : enabledSlots;
    }
}
