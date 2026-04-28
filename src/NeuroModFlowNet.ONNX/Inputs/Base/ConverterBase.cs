namespace NeuroModFlowNet.ONNX.Converters;

/// <summary>
/// EN: Base class for input adapters with model storage and initialization flow.
/// RU: Базовый класс для адаптеров ввода с хранением модели и этапами инициализации.
/// </summary>
public abstract class ConverterBase<TIn> :
    IInputConverter<TIn>
    {
    public OnnxRuntimeContext Model { get; private set; } = default!;

    public virtual string ConverterName => GetType().Name;

    /// <summary>
    /// EN: Main entry for initialization.
    /// RU: Основной вход для инициализации.
    /// </summary>
    public void SetModel(OnnxRuntimeContext context)
    {
        Model = context;
        Init();
        Check();
    }

    /// <summary>
    /// EN: Stage 1: Resource setup (e.g. input geometry extraction).
    /// RU: Этап 1: Подготовка ресурсов (например, извлечение геометрии входа).
    /// </summary>
    protected virtual void Init() { }

    /// <summary>
    /// EN: Stage 2: Validation against the model.
    /// RU: Этап 2: Валидация соответствия адаптера модели.
    /// </summary>
    protected virtual void Check() { }

    public abstract void Prepare(TIn input);
}
