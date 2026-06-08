namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// Converts arbitrary transaction values into conservative debug metadata.
/// </summary>
internal static class VarDebugInspector
{
    public static VarDebugInfo Create(string key, object? value)
    {
        Type? valueType = value?.GetType();
        string typeName = valueType?.FullName ?? "<null>";
        return new VarDebugInfo(
            key,
            typeName,
            DetectMemoryLocation(value, valueType),
            value is IDisposable,
            value is IAsyncDisposable,
            TryDescribeShape(value));
    }

    internal static VarMemoryLocation DetectMemoryLocation(object? value, Type? valueType)
    {
        if(value is null || valueType is null)
            return VarMemoryLocation.Cpu;

        string fullName = valueType.FullName ?? valueType.Name;
        if(fullName == "OpenCvSharp.Mat" || valueType.IsPrimitive || value is string)
            return VarMemoryLocation.Cpu;

        if(fullName == "Microsoft.ML.OnnxRuntime.OrtValue")
        {
            try
            {
                dynamic ortVal = value;
                dynamic memInfo = ortVal.GetTensorMemoryInfo();
                string deviceTypeStr = memInfo.DeviceType.ToString();
                
                if (deviceTypeStr.Contains("Cuda", StringComparison.OrdinalIgnoreCase) ||
                    deviceTypeStr.Contains("Gpu", StringComparison.OrdinalIgnoreCase) ||
                    deviceTypeStr.Contains("DirectML", StringComparison.OrdinalIgnoreCase) ||
                    deviceTypeStr.Contains("Dml", StringComparison.OrdinalIgnoreCase))
                {
                    return VarMemoryLocation.Gpu;
                }
                
                return VarMemoryLocation.Cpu;
            }
            catch
            {
                return VarMemoryLocation.Unknown;
            }
        }

        if(fullName.Contains("Gpu", StringComparison.OrdinalIgnoreCase) ||
           fullName.Contains("Cuda", StringComparison.OrdinalIgnoreCase) ||
           fullName.Contains("Device", StringComparison.OrdinalIgnoreCase))
            return VarMemoryLocation.ExternalDevice;

        return VarMemoryLocation.Unknown;
    }

    static string? TryDescribeShape(object? value)
    {
        if(value is null)
            return null;

        Type valueType = value.GetType();
        string fullName = valueType.FullName ?? valueType.Name;
        if(fullName == "OpenCvSharp.Mat")
        {
            dynamic mat = value;
            return $"{mat.Width}x{mat.Height}x{mat.Channels()}";
        }

        if(fullName == "Microsoft.ML.OnnxRuntime.OrtValue")
        {
            try
            {
                dynamic ortVal = value;
                dynamic typeAndShape = ortVal.GetTensorTypeAndShapeInfo();
                long[] shape = typeAndShape.Shape;
                return string.Join("x", shape);
            }
            catch
            {
                return "OrtValue(NoShape)";
            }
        }

        if(value is Array array)
            return $"Length={array.Length}";

        return null;
    }
}
