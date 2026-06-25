namespace NeuroModFlowNet.Pipeline;

public enum VarRequirementDirection
{
    Read,
    Write,
}

/// <summary>
/// EN: Declares one named transaction variable used by an instruction.
/// RU: Объявление именованной переменной транзакции, используемой инструкцией.
/// </summary>
/// <remarks>
/// EN:
/// Essence: Declaratively describes a pipeline variable dependency (its key, value type, and presence requirement).
/// Reasons: The declaration is descriptive, not a lazy getter. Instructions still execute explicitly so hot-path GPU work does
/// not become hidden inside a dictionary read.
/// 
/// RU:
/// Суть: Декларативно описывает зависимость от переменной в пайплайне (её ключ, тип значения и обязательность присутствия).
/// Причины: Декларативное описание отделено от ленивого геттера. Инструкции выполняются явно, чтобы тяжелые операции в hot-path
/// (например, на GPU) не скрывались внутри неявных чтений словаря.
/// </remarks>
/// <param name="Key">
/// EN: The unique string key of the variable in the pipeline execution context.
/// RU: Уникальный строковый ключ переменной в контексте выполнения пайплайна.
/// </param>
/// <param name="ValueType">
/// EN: The expected runtime type of the variable's value.
/// RU: Ожидаемый runtime-тип значения переменной.
/// </param>
/// <param name="Required">
/// EN: Indicates whether the variable must exist in the context for the instruction to execute.
/// RU: Указывает, обязательно ли переменная должна существовать в контексте для выполнения инструкции.
/// </param>
public sealed record VarRequirement(
    string Key,
    Type ValueType,
    bool Required = true,
    VarRequirementDirection Direction = VarRequirementDirection.Read)
{
    /// <summary>
    /// EN: Creates a requirement for reading a variable of type <typeparamref name="T"/> from the pipeline context.
    /// RU: Создает требование на чтение переменной типа <typeparamref name="T"/> из контекста выполнения пайплайна.
    /// </summary>
    /// <typeparam name="T">
    /// EN: The expected type of the variable to read.
    /// RU: Ожидаемый тип считываемой переменной.
    /// </typeparam>
    /// <param name="key">
    /// EN: The key of the variable in the context.
    /// RU: Ключ переменной в контексте.
    /// </param>
    /// <param name="required">
    /// EN: <c>true</c> if the variable must be present; otherwise, <c>false</c>.
    /// RU: <c>true</c>, если переменная обязательно должна присутствовать; иначе — <c>false</c>.
    /// </param>
    /// <returns>
    /// EN: A new <see cref="VarRequirement"/> instance.
    /// RU: Новый экземпляр <see cref="VarRequirement"/>.
    /// </returns>
    public static VarRequirement Read<T>(string key, bool required = true) =>
        new(key, typeof(T), required, VarRequirementDirection.Read);

    /// <summary>
    /// EN: Creates a requirement for writing a variable of type <typeparamref name="T"/> to the pipeline context.
    /// RU: Создает требование на запись переменной типа <typeparamref name="T"/> в контекст выполнения пайплайна.
    /// </summary>
    /// <typeparam name="T">
    /// EN: The type of the variable to write.
    /// RU: Тип записываемой переменной.
    /// </typeparam>
    /// <param name="key">
    /// EN: The key of the variable in the context.
    /// RU: Ключ переменной в контексте.
    /// </param>
    /// <param name="required">
    /// EN: <c>true</c> if the variable must be present or created; otherwise, <c>false</c>.
    /// RU: <c>true</c>, если переменная обязательно должна быть создана; иначе — <c>false</c>.
    /// </param>
    /// <returns>
    /// EN: A new <see cref="VarRequirement"/> instance.
    /// RU: Новый экземпляр <see cref="VarRequirement"/>.
    /// </returns>
    public static VarRequirement Write<T>(string key, bool required = true) =>
        new(key, typeof(T), required, VarRequirementDirection.Write);
}

