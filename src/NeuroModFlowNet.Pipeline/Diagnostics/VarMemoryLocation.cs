namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// Coarse memory placement visible to VM diagnostics.
/// </summary>
/// <remarks>
/// The core pipeline cannot know every provider-specific device buffer. The value is intentionally conservative:
/// Unknown is better than pretending that an OrtValue, TensorRT binding, or future ROCm object is safely CPU-resident.
/// </remarks>
public enum VarMemoryLocation
{
    Unknown,
    Cpu,
    Gpu,
    ExternalDevice,
}
