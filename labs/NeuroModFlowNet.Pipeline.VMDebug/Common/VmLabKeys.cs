namespace NeuroModFlowNet.Pipeline.VMDebug;

/// <summary>
/// Register names used by the lab VM program.
/// </summary>
internal static class VmLabKeys
{
    public const string InputImage = "input_image";
    public const string InputTensor = "input_tensor";
    public const string ModelDeviceTensor = "model_device_tensor";
    public const string CroppedTensor = "cropped_tensor";
    public const string ResizedTensor = "resized_tensor";
    public const string TextObbCount = "text_obb_count";
    public const string OutputImage = "output_image";
}
