using NeuroModFlowNet.ONNX;

namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// EN: Full identity of a generated ONNX operator session cached across VM programs.
///
/// RU: Полная идентичность сгенерированной ONNX operator session, кэшируемой между VM-программами.
/// </summary>
/// <remarks>
/// EN: The key must contain every value that can change graph semantics, tensor layout, element type, provider behavior
/// or output shape. The cache never compares model bytes and never switches implementation at runtime, so an incomplete
/// key is a contract bug in the caller.
///
/// RU: Ключ должен содержать все значения, которые меняют семантику графа, tensor layout, element type, поведение
/// provider-а или output shape. Кэш не сравнивает model bytes и не переключает реализацию в runtime, поэтому неполный
/// ключ является ошибкой контракта вызывающего кода.
/// </remarks>
public readonly record struct RuntimeOnnxOperatorKernelKey(
    string OperationKind,
    string SemanticKey,
    InferenceBackend Backend,
    string ProviderOptionsKey);
