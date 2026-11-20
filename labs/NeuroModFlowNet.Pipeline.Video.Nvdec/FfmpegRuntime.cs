namespace NeuroModFlowNet.Pipeline.Video.Nvdec;

public static class FfmpegRuntime
{
    public static int ErrorAgain => NeuroModFlowNet.Video.FFmpeg.FfmpegRuntime.ErrorAgain;
    public static string Version => NeuroModFlowNet.Video.FFmpeg.FfmpegRuntime.Version;
    public static void Initialize(string librariesPath) => NeuroModFlowNet.Video.FFmpeg.FfmpegRuntime.Initialize(librariesPath);
    public static int Check(int result, string operation) => NeuroModFlowNet.Video.FFmpeg.FfmpegRuntime.Check(result, operation);
    public static string Describe(int error) => NeuroModFlowNet.Video.FFmpeg.FfmpegRuntime.Describe(error);
}
