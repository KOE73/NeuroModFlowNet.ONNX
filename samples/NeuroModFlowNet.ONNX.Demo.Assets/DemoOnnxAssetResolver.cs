using NeuroModFlowNet.ONNX;

namespace NeuroModFlowNet.ONNX.Demo.Assets;

/// <summary>
/// Demo resolver that maps logical asset ids to files managed by the sample asset downloader.
/// </summary>
/// <remarks>
/// This adapter keeps HuggingFace/demo cache rules out of consumers. A real application can implement the same
/// IOnnxAssetResolver contract with its own file storage, database, byte-array, or deployment-specific source.
/// </remarks>
public sealed class DemoOnnxAssetResolver : IOnnxAssetResolver
{
    public async ValueTask<OnnxAssetSource> ResolveAssetAsync(
        OnnxAssetRequest request,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        request.Validate();
        cancellationToken.ThrowIfCancellationRequested();

        string assetPath = await AssetsManager.GetAssetPathAsync(request.AssetId).ConfigureAwait(false);
        cancellationToken.ThrowIfCancellationRequested();
        return OnnxAssetSource.FromFile(assetPath);
    }
}
