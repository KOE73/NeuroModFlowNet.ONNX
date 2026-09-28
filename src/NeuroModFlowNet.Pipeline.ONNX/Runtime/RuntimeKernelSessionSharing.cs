namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// EN: How runtime-generated operator sessions are shared. See <see cref="RuntimeOnnxOperatorKernelCache.SessionSharing"/>.
///
/// RU: Как разделяются сессии runtime-операторов. См. <see cref="RuntimeOnnxOperatorKernelCache.SessionSharing"/>.
/// </summary>
public enum RuntimeKernelSessionSharing
{
    /// <summary>One session per kernel key for the whole process (least memory, runs are serialized).</summary>
    Shared,

    /// <summary>One session per operator instance; engine files on disk are still shared by kernel key.</summary>
    PerCaller
}
