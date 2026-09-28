namespace NeuroModFlowNet.Pipeline.Video.Nvdec;

/// <summary>
/// EN: YUV→RGB matrix and range of an NV12 stream. Chosen explicitly: streams often report "unknown"; yuvj420p cameras
/// are full-range BT.601 (what FFmpeg swscale and OpenCV assume).
///
/// RU: Матрица YUV→RGB и диапазон NV12-потока. Выбирается явно: потоки часто сообщают "unknown"; камеры yuvj420p это
/// full-range BT.601 (так считают FFmpeg swscale и OpenCV).
/// </summary>
public enum Nv12ColorMatrix
{
    Bt601Full,
    Bt601Limited,
    Bt709Full,
    Bt709Limited
}
