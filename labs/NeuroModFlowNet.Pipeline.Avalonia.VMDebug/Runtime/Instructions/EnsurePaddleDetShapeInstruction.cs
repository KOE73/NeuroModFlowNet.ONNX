using NeuroModFlowNet.Pipeline;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.Avalonia.VMDebug.Runtime;

internal sealed class EnsurePaddleDetShapeInstruction : OpBase
{
    readonly PipelineModelResources resources;

    public EnsurePaddleDetShapeInstruction(PipelineModelResources resources)
        : base(OpDescriptor.Create(
            "ensure-paddle-det-shape",
            "paddleDet.ensureShape",
            [VarRequirement.Read<Mat>(PipelineAvaloniaKeys.ModelInputFrame)],
            hasSideEffects: true))
    {
        this.resources = resources;
    }

    public override ValueTask<OpResult> ExecuteAsync(VmRunContext context, CancellationToken cancellationToken)
    {
        resources.EnsureDetFrameShape(context.Get<Mat>(PipelineAvaloniaKeys.ModelInputFrame));
        return ValueTask.FromResult(OpResult.Continue);
    }
}
