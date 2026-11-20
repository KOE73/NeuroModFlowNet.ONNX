namespace NeuroModFlowNet.Pipeline.WebDebug;

internal static class DebugInstructionCatalog
{
    public static IEnumerable<DebugInstruction> Describe(VmProgram program, string programName)
    {
        ILookup<int, string> labels = program.Labels.ToLookup(pair => pair.Value, pair => pair.Key);
        int instructionCount = program.Instructions.Count;
        for(int index = 0; index < instructionCount; index++)
        {
            OpDescriptor descriptor = program.Instructions[index].Descriptor;
            DebugInstructionFlow? flow = descriptor.ControlFlow is { } controlFlow
                ? new(controlFlow.Guards, controlFlow.FallsThrough, controlFlow.JumpLabels.Select(label =>
                    new DebugJump(label, program.Labels.TryGetValue(label, out int targetIndex) ? targetIndex : null)).ToArray(),
                    controlFlow.CanStop)
                : null;
            yield return new(programName, index, descriptor.Name, descriptor.Operation,
                descriptor.Reads.Select(item => item.Key).ToArray(), descriptor.Writes.Select(item => item.Key).ToArray(),
                labels[index].ToArray(), flow, index == instructionCount - 1 ? labels[instructionCount].ToArray() : null);
        }
    }
}
