using System.Runtime.Versioning;

namespace NeuroModFlowNet.Pipeline.Video;

[SupportedOSPlatform("windows")]
internal sealed class Op_Video_DrawDirect2D(
    string surfaceKey,
    string drawerName,
    IDirect2DDrawer drawer,
    D3D11VideoRenderer renderer)
    : OpBase(OpDescriptor.Create($"DRAW DIRECT2D {surfaceKey}, \"{drawerName}\"", "video.draw-direct2d",
        reads: [VarRequirement.Read<NvencSurface>(surfaceKey)],
        writes: [VarRequirement.Write<NvencSurface>(surfaceKey)],
        hasSideEffects: true))
{
    public override ValueTask<OpResult> ExecuteAsync(VmRunContext context, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        NvencSurface surface = context.Get<NvencSurface>(surfaceKey);
        drawer.Draw(new VmRegisterReader(context), renderer.BeginDraw());
        renderer.EndDraw(surface);
        return ValueTask.FromResult(OpResult.Continue);
    }
}
