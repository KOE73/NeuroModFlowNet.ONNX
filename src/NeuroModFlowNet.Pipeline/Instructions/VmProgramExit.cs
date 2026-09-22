namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// EN:
/// How a program finished its instruction loop without failing.
///
/// RU:
/// Как программа завершила цикл инструкций без ошибки.
/// </summary>
/// <remarks>
/// EN:
/// A controller may execute several programs for one accepted run with one shared <see cref="VmRunContext"/>.
/// <see cref="Stopped"/> means an instruction returned <see cref="OpResultKind.Stop"/>: the whole run is finished and
/// the following programs in the chain must not execute. <see cref="Completed"/> means the program ran off its last
/// instruction and the chain continues.
///
/// RU:
/// Контроллер может выполнять несколько программ для одного принятого запуска с общим <see cref="VmRunContext"/>.
/// <see cref="Stopped"/> означает, что инструкция вернула <see cref="OpResultKind.Stop"/>: запуск завершен целиком и
/// следующие программы цепочки выполняться не должны. <see cref="Completed"/> означает, что программа дошла до последней
/// инструкции и цепочка продолжается.
/// </remarks>
public enum VmProgramExit
{
    Completed,
    Stopped
}
