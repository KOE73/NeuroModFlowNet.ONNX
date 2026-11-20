namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// EN: Stages an owned register in a named host handoff. The host publishes it after the successful outcome.
///
/// RU: Переносит принадлежащий контексту регистр в именованную передачу хосту для публикации после успеха.
/// </summary>
public sealed class Op_PostOwned<T>(string valueKey, string handoffKey)
    : OpBase(OpDescriptor.Create("POST MOVE", "post.owned",
        reads: [VarRequirement.Read<T>(valueKey), VarRequirement.Read<VmOwnedHandoff<T>>(handoffKey, required: false)],
        hasSideEffects: true))
    where T : class, IDisposable
{
    public override ValueTask<OpResult> ExecuteAsync(VmRunContext context, CancellationToken cancellationToken)
    {
        if(context.TryGet(handoffKey, out VmOwnedHandoff<T> handoff))
            handoff.MoveFrom(context, valueKey);
        return ValueTask.FromResult(OpResult.Continue);
    }
}
