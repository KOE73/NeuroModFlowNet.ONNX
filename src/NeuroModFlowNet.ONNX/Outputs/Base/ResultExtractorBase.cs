namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Base class for result extractors with tiered initialization flow.
/// RU: Базовый класс для экстракторов результатов с многоэтапной инициализацией.
/// </summary>
public abstract class ResultExtractorBase<TOut> : IResultExtractor<TOut>
{
    public OnnxRuntimeContext Model { get; private set; } = default!;

    /// <summary>
    /// EN: Main entrance for initialization. RU: Основной вход для инициализации.
    /// </summary>
    public void SetModel(OnnxRuntimeContext context)
    {
        Model = context;
        Init();
        Check();
    }

    /// <summary>
    /// EN: Stage 1: Resource setup (e.g. metadata Extraction). 
    /// RU: Этап 1: Подготовка ресурсов (например, извлечение метаданных).
    /// </summary>
    protected virtual void Init() { }

    /// <summary>
    /// EN: Stage 2: Validation against the model.
    /// RU: Этап 2: Валидация соответствия экстрактора модели.
    /// </summary>
    protected virtual void Check() { }

    public abstract TOut Extract();
}
