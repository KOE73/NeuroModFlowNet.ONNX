namespace NeuroModFlowNet.ONNX;

/// <summary>
/// Logical request for an ONNX-related asset.
/// </summary>
/// <remarks>
/// The core library should not know whether an asset comes from a local file, demo downloader, database, encrypted
/// store, or embedded resource. Assets are broader than models: OCR dictionaries, labels, calibration files, YAML VM
/// programs, and model files can all share the same resolution boundary.
/// </remarks>
public sealed record OnnxAssetRequest(string AssetId)
{
    public void Validate() => ArgumentException.ThrowIfNullOrWhiteSpace(AssetId);
}
