namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// EN: Holds one value moved out of a VM run until its host accepts the successful outcome.
/// The host owns the handoff and disposes it after the run, including rejected or failed runs.
///
/// RU: Удерживает одно значение, перенесённое из VM до принятия успешного результата хостом.
/// Хост владеет передачей и освобождает её после запуска, включая отказ и ошибку.
/// </summary>
public sealed class VmOwnedHandoff<T> : IDisposable where T : class, IDisposable
{
    T? value;

    internal void MoveFrom(VmRunContext context, string key)
    {
        if(value is not null)
            throw new InvalidOperationException("The handoff already holds a value.");
        value = context.MoveOwned<T>(key);
    }

    public T? Take()
    {
        T? result = value;
        value = null;
        return result;
    }

    public void Dispose() => Take()?.Dispose();
}
