using Microsoft.ML.OnnxRuntime;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline.Video.Nvdec;
using NeuroModFlowNet.Pipeline.WebDebug;

namespace NeuroModFlowNet.Pipeline.WebDebug.Tests;

public sealed class RegisterInspectionCapabilitiesTests
{
    [Fact]
    public void SnapshotOffersOnlyActionsSupportedByTheInspector()
    {
        using var context = new VmRunContext(new("camera",1,DateTimeOffset.UtcNow),new VmGlobalMemory(),new VmSyncGateRegistry());
        using OrtValue image = OrtValue.CreateTensorValueFromMemory(new float[12],new long[] {1,3,2,2});
        using OrtValue batch = OrtValue.CreateTensorValueFromMemory(new float[24],new long[] {2,3,2,2});
        using OrtValue recognition = OrtValue.CreateTensorValueFromMemory(new float[12],new long[] {1,3,2,2});
        using var resource = new NativeResource();
        context.Set("image",image);
        context.Set("batch",batch);
        context.Set("ocr.recInput",recognition);
        context.Set("count",2);
        context.Set("results",new[] {1,2,3});
        context.Set("resource",resource);
        context.Set("nativeArray",new[] {resource});
        var inspector = new PausedRegisterInspector(InferenceBackend.Cpu,Nv12ColorMatrix.Bt601Full);
        Dictionary<string,DebugVariable> variables = inspector.Snapshot(context).ToDictionary(variable => variable.Key);

        Assert.True(variables["image"].CanPreviewImage);
        Assert.False(variables["image"].CanInspectValue);
        Assert.False(variables["batch"].CanPreviewImage);
        Assert.False(variables["batch"].CanInspectValue);
        Assert.False(variables["ocr.recInput"].CanPreviewImage);
        Assert.True(variables["count"].CanInspectValue);
        Assert.False(variables["count"].CanPreviewImage);
        Assert.True(variables["results"].CanInspectValue);
        Assert.False(variables["resource"].CanInspectValue);
        Assert.False(variables["nativeArray"].CanInspectValue);
    }

    sealed class NativeResource : IDisposable
    {
        public void Dispose() { }
    }
}
