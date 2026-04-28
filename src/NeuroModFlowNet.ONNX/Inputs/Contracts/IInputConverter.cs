namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: General interface for input adapters.
/// RU: Общий интерфейс для адаптеров данных ввода.
/// </summary>
public interface IInputConverter<in TIn>
{
    string ConverterName { get; }

    /// <summary>
    /// EN: Initializes the adapter with model metadata (e.g. input shapes).
    /// RU: Инициализация адаптера метаданными модели (напр. размерами тензоров входа).
    /// </summary>
    void SetModel(OnnxRuntimeContext context);

    /// <summary>
    /// EN: Prepares the model input tensors from the provided raw object.
    /// RU: Подготавливает тензоры входа модели из предоставленного сырого объекта.
    /// </summary>
    void Prepare(TIn input);

}
