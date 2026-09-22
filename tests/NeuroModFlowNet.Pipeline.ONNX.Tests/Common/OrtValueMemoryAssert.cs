using Microsoft.ML.OnnxRuntime;
using NeuroModFlowNet.ONNX;

namespace NeuroModFlowNet.Pipeline.ONNX.Tests.Common;

internal static class OrtValueMemoryAssert
{
    public static void AssertGpuTensor(OrtValue value, InferenceBackend executionBackend)
    {
        object memoryInfo = value.GetTensorMemoryInfo();
        string allocatorName = GetPropertyString(memoryInfo, "Name");
        string deviceMemoryType = InvokeString(memoryInfo, "GetDeviceMemoryType");
        string memoryType = InvokeString(memoryInfo, "GetMemoryType");
        string allocatorType = InvokeString(memoryInfo, "GetAllocatorType");

        Assert.True(
            IsGpuMemoryName(allocatorName) || IsGpuMemoryName(deviceMemoryType),
            $"Backend {executionBackend} was expected to produce a GPU tensor, actual memory: name={allocatorName}, deviceMemoryType={deviceMemoryType}, memoryType={memoryType}, allocatorType={allocatorType}.");
    }

    static string GetPropertyString(object instance, string propertyName) =>
        instance.GetType().GetProperty(propertyName)?.GetValue(instance)?.ToString() ?? string.Empty;

    static string InvokeString(object instance, string methodName) =>
        instance.GetType().GetMethod(methodName, Type.EmptyTypes)?.Invoke(instance, null)?.ToString() ?? string.Empty;

    static bool IsGpuMemoryName(string value) =>
        value.Contains("Cuda", StringComparison.OrdinalIgnoreCase) ||
        value.Contains("Gpu", StringComparison.OrdinalIgnoreCase) ||
        value.Contains("DirectML", StringComparison.OrdinalIgnoreCase) ||
        value.Contains("Dml", StringComparison.OrdinalIgnoreCase);
}
