using System.Runtime.Versioning;
using Microsoft.ML.OnnxRuntime;

namespace NeuroModFlowNet.Pipeline.Video;

[SupportedOSPlatform("windows")]
internal sealed class Op_Video_PrepareSurface(
    string frameKey,
    string surfaceKey,
    D3D11VideoRenderer renderer,
    D3D11Nv12SurfacePool surfaces)
    : OpBase(OpDescriptor.Create("PREPARE SURFACE", "video.prepare-surface",
        reads: [VarRequirement.Read<OrtValue>(frameKey)],
        writes: [VarRequirement.Write<NvencSurface>(surfaceKey)],
        hasSideEffects: true))
{
    public override ValueTask<OpResult> ExecuteAsync(VmRunContext context, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        OrtValue frame = context.Get<OrtValue>(frameKey);
        NvencSurface surface = surfaces.Rent();
        try
        {
            context.Set(surfaceKey, surface, disposeWithContext: true);
        }
        catch
        {
            surface.Dispose();
            throw;
        }

        renderer.Prepare(frame);
        return ValueTask.FromResult(OpResult.Continue);
    }
}
