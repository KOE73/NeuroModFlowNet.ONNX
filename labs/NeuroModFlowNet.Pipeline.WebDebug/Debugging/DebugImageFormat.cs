using NeuroModFlowNet.Pipeline.Video.Nvdec;

namespace NeuroModFlowNet.Pipeline.WebDebug;

internal sealed record DebugImageFormat(string? ActualCountRegister = null, bool SignedUnitRange = false,
    int Nv12Width = 0, int Nv12Height = 0, Nv12ColorMatrix Nv12Matrix = default);
