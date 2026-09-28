namespace NeuroModFlowNet.Pipeline.MultiCamera;

/// <summary>
/// EN: Creates the frame source of a camera. <c>decoder</c> "cpu" = OpenCV/FFmpeg decode into a <c>Mat</c> (optionally
/// D3D11VA); "nvdec" = FFmpeg 9 NVDEC decode into CUDA memory, frames never touch the host.
///
/// RU: Создаёт источник кадров камеры. <c>decoder</c> "cpu" = декод OpenCV/FFmpeg в <c>Mat</c> (опционально D3D11VA);
/// "nvdec" = декод FFmpeg 9 NVDEC в память CUDA, кадры не попадают в host.
/// </summary>
internal static class FrameSourceFactory
{
    public static IFrameSource Create(SourceConfig source, int extraHardwareFrames)
    {
        if(source.IsNvdec)
        {
            return source.Kind.ToLowerInvariant() switch
            {
                "folder" => NvdecFrameSource.FromFiles(VideoFileLoopFrameSource.EnumerateFolder(source.Path!, source.Pattern), source.Loop, extraHardwareFrames),
                "file" => NvdecFrameSource.FromFiles([source.Path!], source.Loop, extraHardwareFrames),
                "files" => NvdecFrameSource.FromFiles(source.Files!, source.Loop, extraHardwareFrames),
                "rtsp" => NvdecFrameSource.FromRtsp(source.Url!, extraHardwareFrames),
                _ => throw new InvalidDataException($"Source kind '{source.Kind}' is not supported with decoder 'nvdec'.")
            };
        }

        return source.Kind.ToLowerInvariant() switch
        {
            "folder" => VideoFileLoopFrameSource.FromFolder(source.Path!, source.Pattern, source.Loop, VideoCaptureFactory.ParseAcceleration(source.HwAcceleration)),
            "file" => VideoFileLoopFrameSource.FromFile(source.Path!, source.Loop, VideoCaptureFactory.ParseAcceleration(source.HwAcceleration)),
            "files" => VideoFileLoopFrameSource.FromFiles(source.Files!, source.Loop, VideoCaptureFactory.ParseAcceleration(source.HwAcceleration)),
            "rtsp" => LiveCaptureFrameSource.FromRtsp(source.Url!, source.Transport, VideoCaptureFactory.ParseAcceleration(source.HwAcceleration)),
            "camera" => LiveCaptureFrameSource.FromCameraIndex(source.Index!.Value),
            _ => throw new InvalidDataException($"Unknown source kind '{source.Kind}'.")
        };
    }
}
