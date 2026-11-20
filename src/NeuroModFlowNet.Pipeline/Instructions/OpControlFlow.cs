namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// EN: Optional static description of an instruction's guards and possible exits.
/// Labels refer to the containing program. Guards are ordered from outermost to innermost.
///
/// RU: Необязательное статическое описание условий инструкции и её возможных выходов.
/// Метки локальны для программы. Условия перечислены от внешнего к внутреннему.
/// </summary>
/// <remarks>
/// This describes existing execution behavior. The VM still follows the returned OpResult.
/// A null guard name describes a condition without a display name.
/// </remarks>
public sealed record OpControlFlow
{
    public IReadOnlyList<string?> Guards { get; init; } = [];
    public bool FallsThrough { get; init; } = true;
    public IReadOnlyList<string> JumpLabels { get; init; } = [];
    public bool CanStop { get; init; }
}
