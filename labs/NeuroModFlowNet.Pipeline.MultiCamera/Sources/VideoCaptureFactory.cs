using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.MultiCamera;

/// <summary>
/// EN: Opens an FFmpeg <see cref="VideoCapture"/> with an explicit hardware decode mode ("none", "d3d11", "any").
/// A requested hardware mode that the backend did not actually enable is an error, not a silent CPU decode.
///
/// RU: Открывает FFmpeg <see cref="VideoCapture"/> с явным режимом аппаратного декода ("none", "d3d11", "any").
/// Если запрошенный аппаратный режим не включился, это ошибка, а не тихий переход на CPU-декод.
/// </summary>
internal static class VideoCaptureFactory
{
    public static VideoAccelerationType ParseAcceleration(string? value) =>
        value?.ToLowerInvariant() switch
        {
            null or "" or "none" => VideoAccelerationType.None,
            "d3d11" => VideoAccelerationType.D3D11,
            "any" => VideoAccelerationType.Any,
            _ => throw new InvalidDataException($"Unknown hwAcceleration '{value}'. Use none, d3d11 or any.")
        };

    /// <summary>Returns an opened capture or <c>null</c> when the source cannot be opened at all.</summary>
    public static VideoCapture? TryOpen(string source, VideoAccelerationType acceleration)
    {
        var capture = new VideoCapture(
            source,
            VideoCaptureAPIs.FFMPEG,
            [(int)VideoCaptureProperties.HwAcceleration, (int)acceleration]);

        if(!capture.IsOpened())
        {
            capture.Dispose();
            return null;
        }

        if(acceleration != VideoAccelerationType.None && capture.Get(VideoCaptureProperties.HwAcceleration) <= 0)
        {
            capture.Dispose();
            throw new InvalidOperationException($"Hardware decode '{acceleration}' was requested but not enabled for '{source}'.");
        }

        return capture;
    }
}
