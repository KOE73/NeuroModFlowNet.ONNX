using NeuroModFlowNet.Pipeline;

namespace NeuroModFlowNet.Pipeline.Avalonia.VMDebug.Runtime;

internal sealed class DetectionTimingInstruction : OpBase
{
    readonly bool start;

    public DetectionTimingInstruction(bool start)
        : base(OpDescriptor.Create(
            start ? "timing-detection-start" : "timing-detection-stop",
            start ? "timing.startDetection" : "timing.stopDetection",
            [VarRequirement.Read<PipelineFrameTiming>(PipelineAvaloniaKeys.FrameTiming)]))
    {
        this.start = start;
    }

    public override ValueTask<OpResult> ExecuteAsync(VmRunContext context, CancellationToken cancellationToken)
    {
        PipelineFrameTiming timing = context.Get<PipelineFrameTiming>(PipelineAvaloniaKeys.FrameTiming);

        if(start)
            timing.StartDetection();
        else
            timing.StopDetection();

        return ValueTask.FromResult(OpResult.Continue);
    }
}
