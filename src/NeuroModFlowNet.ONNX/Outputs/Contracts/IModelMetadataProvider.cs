using Microsoft.ML.OnnxRuntime.Tensors;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Provides a read-only facade for accessing static model metadata (shapes, data types, names).
/// RU: Предоставляет фасад только-для-чтения для доступа к статическим метаданным модели (формы, типы данных, имена).
/// </summary>
/// <remarks>
/// <para>EN: Design Rationale (Causes and Effects):</para>
/// <list type="bullet">
/// <item><term>Decoupling from Session</term><description> Separates metadata access from the concrete InferenceSession and dynamic execution state (OrtValues). This allows extractors to be initialized via lightweight contexts (e.g. in NeuroModFlowNet.Pipeline.ONNX) without allocating a full monolithic runner.</description></item>
/// <item><term>Lazy Initialization (Warm-up)</term><description> Since extractors no longer depend on InferenceSession directly, a pipeline can dynamically construct this provider during a "warm-up" pass by simply extracting TensorElementType and Shape from incoming OrtValues via GetTensorTypeAndShape().</description></item>
/// <item><term>Zero Overhead</term><description> Eliminates the need to pass heavy context objects around on the hot path. Extractors cache necessary static metrics (like BatchCount and ItemCount) during SetModel(IModelMetadataProvider) and only retrieve raw Spans during Extract(IOnnxModelOutputs).</description></item>
/// </list>
/// <br/>
/// <para>RU: Обоснование архитектуры (Причины и следствия):</para>
/// <list type="bullet">
/// <item><term>Отвязка от сессии (Session)</term><description> Отделяет доступ к метаданным от конкретного класса InferenceSession и динамического состояния (OrtValues). Это позволяет инициализировать экстракторы через легковесные контексты (например, в пайплайнах NeuroModFlowNet.Pipeline.ONNX) без создания тяжеловесного раннера.</description></item>
/// <item><term>Ленивая инициализация (Прогрев)</term><description> Поскольку экстракторы больше не зависят напрямую от InferenceSession, пайплайн может динамически создать этот провайдер во время прохода "прогрева" (warm-up), просто извлекая TensorElementType и Shape из входящих OrtValues через GetTensorTypeAndShape().</description></item>
/// <item><term>Нулевой оверхед (Zero Overhead)</term><description> Устраняет необходимость передачи тяжелых контекстов на "горячем пути" (hot path). Экстракторы кэшируют нужные статические метрики (например, BatchCount и ItemCount) во время SetModel(IModelMetadataProvider), а на этапе выполнения Extract(IOnnxModelOutputs) берут только сырые спаны из IOnnxModelOutputs.</description></item>
/// </list>
/// </remarks>
public interface IModelMetadataProvider
{
    string PrimaryInputName => InputNames[0];
    string PrimaryOutputName => OutputNames[0];
    IReadOnlyList<string> InputNames { get; }
    IReadOnlyList<string> OutputNames { get; }

    IReadOnlyDictionary<string, long[]> ModelInputShapes { get; }
    IReadOnlyDictionary<string, long[]> ModelOutputShapes { get; }

    string ModelPath { get; }

    /// <summary>
    /// EN: Retrieves the element data type for an input. Replaces direct access to Session.InputMetadata.
    /// RU: Возвращает тип данных для входа. Заменяет прямое обращение к Session.InputMetadata.
    /// </summary>
    TensorElementType GetInputElementType(string name);

    /// <summary>
    /// EN: Retrieves the element data type for an output. Replaces direct access to Session.OutputMetadata.
    /// RU: Возвращает тип данных для выхода. Заменяет прямое обращение к Session.OutputMetadata.
    /// </summary>
    TensorElementType GetOutputElementType(string name);
}
