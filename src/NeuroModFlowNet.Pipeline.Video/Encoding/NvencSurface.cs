using FFmpeg.AutoGen.Abstractions;

namespace NeuroModFlowNet.Pipeline.Video;

internal sealed unsafe class NvencSurface(AVFrame* frame) : IDisposable
{
    public AVFrame* Frame => frame;

    public void Dispose()
    {
        AVFrame* owned = frame;
        frame = null;
        if(owned is not null)
            ffmpeg.av_frame_free(&owned);
    }
}
