namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// EN: Container for a single named input value passed into the VM.
/// RU: Контейнер для одного именованного входного значения, передаваемого в VM.
/// </summary>
/// <remarks>
/// EN:
/// Essence: Wraps an input payload together with its resource lifetime policy (disposal flag).
/// Reasons: Tensors, frames, and buffers cross host/VM boundaries with varying lifetimes. Binding disposal rules directly to the value ensures that resources (like cloned frames) are cleaned up correctly by the execution context without manual tracking.
/// 
/// RU:
/// Суть: Оборачивает входные данные вместе с флагом владения ресурсом (политикой освобождения памяти).
/// Причины: Тензоры, кадры и буферы пересекают границы хоста и VM с разными жизненными циклами. 
///          Привязка правила очистки непосредственно к значению гарантирует своевременное освобождение ресурсов 
///          (например, клонированных кадров) контекстом запуска без ручного отслеживания.
/// </remarks>
public readonly record struct VmInputValue(object? Value, bool DisposeWithContext = false);
