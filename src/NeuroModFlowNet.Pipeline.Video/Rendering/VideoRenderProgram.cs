using System.Runtime.Versioning;

namespace NeuroModFlowNet.Pipeline.Video;

[SupportedOSPlatform("windows")]
internal sealed class VideoRenderProgram
{
    readonly D3D11VideoRenderer renderer;
    readonly VmProgram program;
    readonly VmGlobalMemory globalMemory = new();
    readonly VmSyncGateRegistry syncGates = new();
    long runId;

    public VideoRenderProgram(D3D11VideoRenderer renderer, D3D11Nv12SurfacePool surfaces, string drawerName, IDirect2DDrawer drawer)
    {
        this.renderer = renderer;
        program = new VmProgramBuilder("render")
            .Step(new Op_Video_PrepareSurface(VideoInputFrame.FrameRegister, VideoInputFrame.SurfaceRegister, renderer, surfaces))
            .Step(new Op_Video_DrawDirect2D(VideoInputFrame.SurfaceRegister, drawerName, drawer, renderer))
            .Step(new Op_PostOwned<NvencSurface>(VideoInputFrame.SurfaceRegister, VideoInputFrame.HandoffRegister))
            .Build();
    }

    public async ValueTask<NvencSurface> ExecuteAsync(string sourceId, VideoInputFrame input, CancellationToken cancellationToken)
    {
        using var handoff = new VmOwnedHandoff<NvencSurface>();
        var context = new VmRunContext(
            new VmRunIdentity(sourceId, ++runId, DateTimeOffset.UtcNow, input.Sequence), globalMemory, syncGates);
        await using(context.ConfigureAwait(false))
        {
            foreach(var (key, value) in input.Registers)
                context.Set(key, value);
            context.Set(VideoInputFrame.FrameRegister, input.Tensor);
            context.Set(VideoInputFrame.HandoffRegister, handoff);

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
