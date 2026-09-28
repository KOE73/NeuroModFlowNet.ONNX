namespace NeuroModFlowNet.Pipeline.MultiCamera;

/// <summary>
/// EN: One decoded frame on its way into a VM run. The frame knows its own representation: a CPU <c>Mat</c> (CPU decode)
/// or an NV12 surface in CUDA memory (NVDEC). <see cref="AddTo"/> puts its registers into the run inputs and hands
/// ownership to the run context (disposed with it). If the run is rejected, the caller disposes the frame.
///
/// RU: Один декодированный кадр на пути в запуск VM. Кадр знает своё представление: <c>Mat</c> в CPU (CPU-декод) или
/// NV12-поверхность в памяти CUDA (NVDEC). <see cref="AddTo"/> кладёт его регистры во входы запуска и передаёт владение
/// контексту запуска (освобождается вместе с ним). Если запуск отклонён, кадр освобождает вызывающий.
/// </summary>
internal interface ISourceFrame : IDisposable
{
    int Width { get; }

    int Height { get; }

    void AddTo(VmRunInputs inputs);
}
