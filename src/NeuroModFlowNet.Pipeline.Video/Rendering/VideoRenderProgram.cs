using System.Runtime.Versioning;

namespace NeuroModFlowNet.Pipeline.Video;

[SupportedOSPlatform("windows")]
internal sealed class VideoRenderProgram
{
    const string FrameKey = "frame";
    const string RectanglesKey = "rectangles";
    const string SurfaceKey = "surface";
    const string HandoffKey = "video.handoff";

    readonly D3D11VideoRenderer renderer;
    readonly VmProgram program;
    readonly VmGlobalMemory globalMemory = new();
    readonly VmSyncGateRegistry syncGates = new();
    long runId;

    public VideoRenderProgram(D3D11VideoRenderer renderer, D3D11Nv12SurfacePool surfaces)
    {
        this.renderer = renderer;
        program = new VmProgramBuilder("render")
            .Step(new Op_Video_PrepareSurface(FrameKey, SurfaceKey, renderer, surfaces))
            .Step(new Op_Video_DrawDirect2D(SurfaceKey, RectanglesKey, renderer))
            .Step(new Op_PostOwned<NvencSurface>(SurfaceKey, HandoffKey))
            .Build();
    }

    public async ValueTask<NvencSurface> ExecuteAsync(string sourceId, VideoInputFrame input, CancellationToken cancellationToken)
    {
        using var handoff = new VmOwnedHandoff<NvencSurface>();
        var context = new VmRunContext(
            new VmRunIdentity(sourceId, ++runId, DateTimeOffset.UtcNow, input.Sequence), globalMemory, syncGates);
        await using(context.ConfigureAwait(false))
        {
            context.Set(FrameKey, input.Tensor);
            context.Set(RectanglesKey, input.Rectangles);
            context.Set(HandoffKey, handoff);

            try
            {
                VmProgramExit outcome = await program.ExecuteAsync(context, cancellationToken, captureVariablesInTrace: false)
                    .ConfigureAwait(false);
                if(outcome != VmProgramExit.Completed)
                    throw new InvalidOperationException("The render program stopped before publishing a surface.");
            }
            catch
            {
                // Finish abandoned GPU work before releasing the source tensor or surface.
                renderer.Reset();
                throw;
            }
        }

        cancellationToken.ThrowIfCancellationRequested();
        return handoff.Take()
            ?? throw new InvalidOperationException("The render program completed without posting a surface.");
    }
}
