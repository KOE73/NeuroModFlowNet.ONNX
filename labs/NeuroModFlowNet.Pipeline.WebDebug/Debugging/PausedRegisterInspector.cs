using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.ONNX.Visualizer;
using NeuroModFlowNet.Pipeline.ONNX;
using NeuroModFlowNet.Pipeline.Video.Nvdec;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.WebDebug;

internal sealed class PausedRegisterInspector(InferenceBackend backend, Nv12ColorMatrix colorMatrix, int sourceWidth = 0, int sourceHeight = 0)
{
    static readonly JsonSerializerOptions JsonOptions = new() { IncludeFields = true, WriteIndented = true, MaxDepth = 12 };

    public DebugVariable[] Snapshot(VmRunContext context) => context.Keys.Order(StringComparer.Ordinal).Select(key =>
    {
        context.TryGetObject(key, out object? value);
        string? elementType = value is OrtValue tensor ? tensor.GetTensorTypeAndShape().ElementDataType.ToString() : null;
        string memory = value is OrtValue ? "Unknown" : "Managed";
        string? shape = value switch
        {
            OrtValue shapedTensor => string.Join("×", shapedTensor.GetTensorTypeAndShape().Shape),
            Mat image => $"{image.Width}×{image.Height}×{image.Channels()}",
            Array array => $"Length={array.Length}",
            _ => null
        };
        if(value is OrtValue deviceTensor)
        {
            using OrtMemoryInfo memoryInfo = deviceTensor.GetTensorMemoryInfo();
            memory = $"{memoryInfo.Name}:{memoryInfo.Id}";
        }
        return new DebugVariable(key, value?.GetType().FullName ?? "null", memory, shape,
            elementType, value is null or string or int or long or float or double or bool ? Convert.ToString(value, System.Globalization.CultureInfo.InvariantCulture) : null);
    }).ToArray();

    public async ValueTask<DebugInspectionResult> InspectAsync(VmRunContext context, DebugInspectionRequest request, CancellationToken cancellationToken)
    {
        if(!context.TryGetObject(request.Key, out object? value))
            throw new KeyNotFoundException("The register is not present at this stop.");
        if(request.Format == "png")
            return new("image/png", await PreviewAsync(value, request, cancellationToken).ConfigureAwait(false));
        if(request.Format != "json") throw new ArgumentException("Inspection format must be json or png.");
        if(value is OrtValue or Mat)
            throw new NotSupportedException("Tensor data is available through the explicit PNG preview; metadata is in the stop snapshot.");
        // Native wrappers and arbitrary resource graphs must never be reflected into JSON.
        if(value is not null && value is not Array && value is not string && value is not ICoordinateBackTransform && !value.GetType().IsValueType)
            throw new NotSupportedException("This resource is metadata-only.");
        if(value is Array nativeArray && nativeArray.Cast<object?>().Take(256).Any(item => item is IDisposable or IAsyncDisposable))
            throw new NotSupportedException("Arrays of native resources are metadata-only.");
        if(value is Array { Length: > 256 } array)
        {
            object?[] firstItems = array.Cast<object?>().Take(256).ToArray();
            return new("application/json", JsonSerializer.SerializeToUtf8Bytes(new { totalCount = array.Length, truncated = true, items = firstItems }, JsonOptions));
        }
        return new("application/json", JsonSerializer.SerializeToUtf8Bytes(value, JsonOptions));
    }

    async Task<byte[]> PreviewAsync(object? value, DebugInspectionRequest request, CancellationToken cancellationToken)
    {
        if(request.MaxEdge is < 100 or > 2000) throw new ArgumentOutOfRangeException(nameof(request.MaxEdge));
        if(value is Mat image) return TensorImagePreviewRenderer.EncodePng(image, request.MaxEdge);
        if(value is not OrtValue tensor) throw new NotSupportedException("Select an image tensor or Mat.");

        using var temporary = new VmRunContext(new("debug-preview", 0, DateTimeOffset.UtcNow), new VmGlobalMemory(), new VmSyncGateRegistry());
        temporary.Set("input", tensor);
        var information = tensor.GetTensorTypeAndShape();
        long[] shape = information.Shape;
        bool nv12 = request.Key == "image.source.nv12";
        if(!nv12 && (shape.Length != 4 || shape[0] != 1))
            throw new NotSupportedException("Preview currently supports single images. Batch tensors remain available as metadata.");

        bool nhwc = information.ElementDataType == TensorElementType.UInt8;
        int width = nv12 ? sourceWidth : checked((int)shape[nhwc ? 2 : 3]);
        int height = nv12 ? sourceHeight : checked((int)shape[nhwc ? 1 : 2]);
        IDisposable? conversion = null;
        try
        {
            if(nv12)
            {
                var nv12Conversion = new Op_Onnx_Nv12_To_BgrU8Nhwc("input", "converted", width, height, colorMatrix, executionBackend: backend);
                conversion = nv12Conversion;
                EnsureSuccess(await nv12Conversion.ExecuteAsync(temporary, cancellationToken).ConfigureAwait(false));
            }
            double scale = Math.Min(1, request.MaxEdge / (double)Math.Max(width, height));
            var size = new Size(Math.Max(1, (int)(width * scale)), Math.Max(1, (int)(height * scale)));
            string inputKey = nv12 ? "converted" : "input";
            using Op_Onnx_TensorTransformBase resize = information.ElementDataType switch
            {
                TensorElementType.UInt8 => new Op_Onnx_Resize_U8_NHWC(inputKey, "preview", size, isFinal: true, executionBackend: backend),
                TensorElementType.Float => new Op_Onnx_Resize_FP32_NCHW(inputKey, "preview", size, isFinal: true, executionBackend: backend),
                TensorElementType.Float16 => new Op_Onnx_Resize_FP16_NCHW(inputKey, "preview", size, isFinal: true, executionBackend: backend),
                _ => throw new NotSupportedException("Image preview supports U8 NHWC and FP16/FP32 NCHW only.")
            };
            try
            {
                EnsureSuccess(await resize.ExecuteAsync(temporary, cancellationToken).ConfigureAwait(false));
                return TensorImagePreviewRenderer.EncodePng(temporary.Get<OrtValue>("preview"));
            }
            finally
            {
                // Provider-owned intermediate tensors must die before the resize allocator/session.
                temporary.Dispose();
            }
        }
        finally
        {
            temporary.Dispose();
            conversion?.Dispose();
        }
    }

    static void EnsureSuccess(OpResult result)
    {
        if(result.Kind == OpResultKind.Fail) throw new InvalidOperationException(result.Reason, result.Exception);
    }
}
