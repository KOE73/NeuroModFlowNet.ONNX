using System.Configuration;

namespace NeuroModFlowNet.Pipeline.VMDebug;

/// <summary>
/// Reads all lab settings from App.config / App.local.config.
/// </summary>
internal sealed record LabSettings(
    LabMode Mode,
    int CameraIndex,
    bool ShowOpenCvWindows,
    bool SyncLoopToCameraFps,
    string ObbModelPrecision,
    bool ObbModelUseByteBgr,
    SingleStreamSettings SingleStream,
    MultiStreamSettings MultiStream)
{
    public static LabSettings FromConfig()
    {
        return new LabSettings(
            Mode: ReadEnum("LabMode", LabMode.SingleStream),
            CameraIndex: ReadInt("CameraIndex", 0),
            ShowOpenCvWindows: ReadBool("ShowOpenCvWindows", true),
            SyncLoopToCameraFps: ReadBool("SyncLoopToCameraFps", true),
            ObbModelPrecision: ReadString("ObbModelPrecision", "fp16"),
            ObbModelUseByteBgr: ReadBool("ObbModelUseByteBgr", true),
            SingleStream: SingleStreamSettings.FromConfig(),
            MultiStream: MultiStreamSettings.FromConfig());
    }

    #region Config readers

    static string ReadString(string key, string defaultValue)
    {
        string? value = ConfigurationManager.AppSettings[key];
        return string.IsNullOrWhiteSpace(value) ? defaultValue : value.Trim();
    }

    static int ReadInt(string key, int defaultValue)
    {
        string? value = ConfigurationManager.AppSettings[key];
        return int.TryParse(value, out int result) ? result : defaultValue;
    }

    static bool ReadBool(string key, bool defaultValue)
    {
        string? value = ConfigurationManager.AppSettings[key];
        return bool.TryParse(value, out bool result) ? result : defaultValue;
    }

    static TEnum ReadEnum<TEnum>(string key, TEnum defaultValue)
        where TEnum : struct
    {
        string? value = ConfigurationManager.AppSettings[key];
        return Enum.TryParse(value, ignoreCase: true, out TEnum result) ? result : defaultValue;
    }

    #endregion
}

/// <summary>Single-stream specific settings.</summary>
internal sealed record SingleStreamSettings(int MaxInFlight)
{
    public static SingleStreamSettings FromConfig()
    {
        string? value = ConfigurationManager.AppSettings["SingleStream.MaxInFlight"];
        int maxInFlight = int.TryParse(value, out int parsed) ? parsed : 1;
        return new SingleStreamSettings(MaxInFlight: maxInFlight);
    }
}

/// <summary>Multi-stream specific settings.</summary>
internal sealed record MultiStreamSettings(
    int VmCount,
    int ObbBatchSize,
    int MaxInFlightPerVm)
{
    public static MultiStreamSettings FromConfig()
    {
        return new MultiStreamSettings(
            VmCount: ReadInt("MultiStream.VmCount", 4),
            ObbBatchSize: ReadInt("MultiStream.ObbBatchSize", 4),
            MaxInFlightPerVm: ReadInt("MultiStream.MaxInFlightPerVm", 1));
    }

    static int ReadInt(string key, int defaultValue)
    {
        string? value = ConfigurationManager.AppSettings[key];
        return int.TryParse(value, out int result) ? result : defaultValue;
    }
}
