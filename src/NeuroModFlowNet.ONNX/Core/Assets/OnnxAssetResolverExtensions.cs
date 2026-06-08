namespace NeuroModFlowNet.ONNX;

public static class OnnxAssetResolverExtensions
{
    public static async ValueTask<OnnxRuntimeModelSource> ResolveModelSourceAsync(
        this IOnnxAssetResolver resolver,
        string modelAssetId,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(resolver);

        OnnxAssetSource assetSource = await resolver
            .ResolveAssetAsync(new OnnxAssetRequest(modelAssetId), cancellationToken)
            .ConfigureAwait(false);

        return assetSource.ToModelSource();
    }
}
