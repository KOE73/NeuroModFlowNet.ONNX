using NeuroModFlowNet.Pipeline.Video.Nvdec;

namespace NeuroModFlowNet.Pipeline.VideoOrt;

/// <summary>Register names of the video → ORT lab programs.</summary>
internal static class VideoOrtKeys
{
    /// <summary>Owner of the decoded surface (<see cref="NvdecFrame"/>); disposed with the run context.</summary>
    public const string Frame = "video.frame";

    /// <summary>Zero-copy NV12 view over the CUDA surface, [allocHeight * 3 / 2, pitch] U8.</summary>
    public const string Nv12 = "video.nv12";

    /// <summary>BGR U8 NHWC [1, H, W, 3], the same bridge format prepare programs start from.</summary>
    public const string Bgr = "video.bgr";

    public const string Resized = "video.resized";

    public const string Mat = "video.mat";
}
