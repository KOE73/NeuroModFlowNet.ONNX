namespace NeuroModFlowNet.ONNX;

/// <summary>
/// Resolves logical ONNX-related assets into concrete file or byte sources.
/// </summary>
/// <remarks>
/// Downloading, local cache layout, version policy, authorization, and deployment-specific storage belong to
/// implementations. Runners and VM instructions should only consume resolved sources.
/// </remarks>
public interface IOnnxAssetResolver
{
    ValueTask<OnnxAssetSource> ResolveAssetAsync(
        OnnxAssetRequest request,
        CancellationToken cancellationToken = default);
}
