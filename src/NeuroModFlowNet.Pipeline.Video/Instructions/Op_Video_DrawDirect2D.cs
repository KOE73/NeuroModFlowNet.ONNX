using System.Runtime.Versioning;
using NeuroModFlowNet.ONNX.Visualizer;

namespace NeuroModFlowNet.Pipeline.Video;

[SupportedOSPlatform("windows")]
internal sealed class Op_Video_DrawDirect2D(
    string surfaceKey,
    string rectanglesKey,
    D3D11VideoRenderer renderer)
    : OpBase(OpDescriptor.Create("DRAW DIRECT2D", "video.draw-direct2d",
        reads: [VarRequirement.Read<NvencSurface>(surfaceKey),
            VarRequirement.Read<ReadOnlyMemory<Direct2DRectangle>>(rectanglesKey)],
        writes: [VarRequirement.Write<NvencSurface>(surfaceKey)],
        hasSideEffects: true))
{
    public override ValueTask<OpResult> ExecuteAsync(VmRunContext context, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        NvencSurface surface = context.Get<NvencSurface>(surfaceKey);
        ReadOnlyMemory<Direct2DRectangle> rectangles = context.Get<ReadOnlyMemory<Direct2DRectangle>>(rectanglesKey);
        renderer.Draw(surface, rectangles.Span);
        return ValueTask.FromResult(OpResult.Continue);
    }
}
