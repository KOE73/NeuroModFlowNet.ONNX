using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.ONNX.Visualizer;
using NeuroModFlowNet.Pipeline.ONNX;
using NeuroModFlowNet.Pipeline.Video.Nvdec;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.WebDebug;

internal sealed class PausedRegisterInspector(InferenceBackend backend, IReadOnlyDictionary<string, DebugImageFormat>? imageFormats = null)
{
    static readonly JsonSerializerOptions JsonOptions = new() { IncludeFields = true, WriteIndented = true, MaxDepth = 12 };

    public DebugVariable[] Snapshot(VmRunContext context) => context.Keys.Order(StringComparer.Ordinal).Select(key => Describe(context, key)).ToArray();

    public DebugVariable Describe(VmRunContext context, string key, bool includeValue = true)
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
            elementType, includeValue && (value is null or string or int or long or float or double or bool) ? Convert.ToString(value, System.Globalization.CultureInfo.InvariantCulture) : null,
            CanInspectValue(value), CanPreviewImage(key, value));
    }

    static bool CanInspectValue(object? value) => value switch
    {
        OrtValue or Mat or IntPtr or UIntPtr => false,
        null or string or ICoordinateBackTransform => true,
        Array array => !array.Cast<object?>().Take(256).Any(item => item is IDisposable or IAsyncDisposable),
        _ => value.GetType().IsValueType
    };

    bool CanPreviewImage(string key, object? value)
    {
        if(value is Mat image) return !image.Empty();
        if(value is not OrtValue tensor) return false;
        var information = tensor.GetTensorTypeAndShape();
        long[] shape = information.Shape;
        DebugImageFormat? format = imageFormats?.GetValueOrDefault(key);
        if(format is { Nv12Width: > 0, Nv12Height: > 0 })
            return shape.Length == 2 && information.ElementDataType == TensorElementType.UInt8;
        if(shape.Length != 4 || shape[0] is < 1 or > 64) return false;
        return information.ElementDataType switch
        {
            TensorElementType.UInt8 => shape[3] == 3,
            TensorElementType.Float or TensorElementType.Float16 => shape[1] == 3,
            _ => false
        };
    }

    public async ValueTask<DebugInspectionResult> InspectAsync(VmRunContext context, DebugInspectionRequest request, CancellationToken cancellationToken)
    {
        using CapturedRegister captured = await CaptureAsync(context, request, cancellationToken).ConfigureAwait(false);
        return captured.Encode();
    }

    public async ValueTask<CapturedRegister> CaptureAsync(VmRunContext context, DebugInspectionRequest request, CancellationToken cancellationToken)
    {
        if(!context.TryGetObject(request.Key, out object? value))
            throw new KeyNotFoundException("The register is not present at this instruction boundary.");
        if(request.Format == "png")
            return new(await PreviewAsync(context, value, request, cancellationToken).ConfigureAwait(false), null);
        if(request.Format != "json") throw new ArgumentException("Inspection format must be json or png.");
        if(value is OrtValue or Mat)
            throw new NotSupportedException("Tensor data is available through the explicit PNG preview; metadata is in the stop snapshot.");
        // Native wrappers and arbitrary resource graphs must never be reflected into JSON.
        if(!CanInspectValue(value)) throw new NotSupportedException("This resource is metadata-only.");
        if(value is Array { Length: > 256 } array)
        {
            object?[] firstItems = array.Cast<object?>().Take(256).ToArray();
            return new(null, new("application/json", JsonSerializer.SerializeToUtf8Bytes(new { totalCount = array.Length, truncated = true, items = firstItems }, JsonOptions)));
        }
        return new(null, new("application/json", JsonSerializer.SerializeToUtf8Bytes(value, JsonOptions)));
    }

    async Task<Mat> PreviewAsync(VmRunContext context, object? value, DebugInspectionRequest request, CancellationToken cancellationToken)
    {
        if(!PreviewImageSize.IsValid(request.MaxEdge)) throw new ArgumentOutOfRangeException(nameof(request.MaxEdge));
        if(!CanPreviewImage(request.Key, value))
            throw new NotSupportedException("Image preview requires supported BGR U8 NHWC or RGB FP16/FP32 NCHW images, up to 64 batch slots.");
        if(value is Mat image) return PreviewImageSize.CopyCpu(image, request.MaxEdge);
        if(value is not OrtValue tensor) throw new NotSupportedException("Select an image tensor or Mat.");
        long[] imageShape = tensor.GetTensorTypeAndShape().Shape;
        DebugImageFormat? format = imageFormats?.GetValueOrDefault(request.Key);
        if(imageShape.Length == 4 && (imageShape[0] > 1 || format?.ActualCountRegister is not null))
            return PreviewBatch(context, tensor, request, format, cancellationToken);
        bool nv12 = format is { Nv12Width: > 0, Nv12Height: > 0 };
        using(OrtMemoryInfo memory = tensor.GetTensorMemoryInfo())
            if(!nv12 && string.Equals(memory.Name, "Cpu", StringComparison.OrdinalIgnoreCase))
            {
                using Mat downloaded = TensorImagePreviewRenderer.CreatePreview(tensor, format?.SignedUnitRange == true);
                return PreviewImageSize.CopyCpu(downloaded, request.MaxEdge);
            }

        using var temporary = new VmRunContext(new("debug-preview", 0, DateTimeOffset.UtcNow), new VmGlobalMemory(), new VmSyncGateRegistry());
        temporary.Set("input", tensor);
        var information = tensor.GetTensorTypeAndShape();
        long[] shape = information.Shape;
        if(!nv12 && (shape.Length != 4 || shape[0] != 1))
            throw new NotSupportedException("Preview currently supports single images. Batch tensors remain available as metadata.");

        bool nhwc = information.ElementDataType == TensorElementType.UInt8;
        int width = nv12 ? format!.Nv12Width : checked((int)shape[nhwc ? 2 : 3]);
        int height = nv12 ? format!.Nv12Height : checked((int)shape[nhwc ? 1 : 2]);
        Size size = PreviewImageSize.Resolve(width, height, request.MaxEdge);
        bool resizeNeeded = size.Width != width || size.Height != height;
        IDisposable? conversion = null;
        try
        {
            if(nv12)
            {
                var nv12Conversion = new Op_Onnx_Nv12_To_BgrU8Nhwc("input", "converted", width, height, format!.Nv12Matrix, isFinal: !resizeNeeded, executionBackend: backend);
                conversion = nv12Conversion;
                EnsureSuccess(await nv12Conversion.ExecuteAsync(temporary, cancellationToken).ConfigureAwait(false));
            }
            string inputKey = nv12 ? "converted" : "input";
            if(!resizeNeeded)
                return nv12 ? TensorImagePreviewRenderer.CreatePreview(temporary.Get<OrtValue>(inputKey)) : PreviewTensorDownload.Copy(tensor, backend, cpu => TensorImagePreviewRenderer.CreatePreview(cpu, format?.SignedUnitRange == true));
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
                return TensorImagePreviewRenderer.CreatePreview(temporary.Get<OrtValue>("preview"), format?.SignedUnitRange == true);
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

    Mat PreviewBatch(VmRunContext context, OrtValue tensor, DebugInspectionRequest request, DebugImageFormat? format, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var information = tensor.GetTensorTypeAndShape();
        long[] shape = information.Shape;
        int capacity = checked((int)shape[0]);
        string? countKey = format?.ActualCountRegister;
        int actualCount = capacity;
        if(countKey is not null && !context.TryGet(countKey, out actualCount))
            throw new InvalidOperationException($"{countKey} is not present at this boundary; real images cannot be distinguished from padding.");
        if(actualCount < 0 || actualCount > capacity) throw new InvalidOperationException("Actual image count exceeds the batch capacity.");
        bool nhwc = information.ElementDataType == TensorElementType.UInt8;
        int width = checked((int)shape[nhwc ? 2 : 3]), height = checked((int)shape[nhwc ? 1 : 2]);
        Size itemSize = PreviewImageSize.Resolve(width, height, request.MaxEdge);
        Size gridSize = TensorBatchPreviewRenderer.GridSize(itemSize, actualCount);
        if((long)gridSize.Width * gridSize.Height > PreviewImageSize.MaximumPixels)
            throw new InvalidOperationException("Batch preview grid exceeds 16 megapixels. Choose a smaller image size.");
        if(actualCount == 0) return TensorBatchPreviewRenderer.CreateEmpty(itemSize, capacity);
        bool paddleNormalized = format?.SignedUnitRange == true;
        Mat Render(OrtValue cpu) => TensorBatchPreviewRenderer.CreatePreview(cpu, actualCount, itemSize, paddleNormalized);
        int bytesPerElement = information.ElementDataType == TensorElementType.UInt8 ? 1 : information.ElementDataType == TensorElementType.Float16 ? 2 : 4;
        if((long)capacity * width * height * 3 * bytesPerElement > 128 * 1024 * 1024)
            throw new InvalidOperationException("Full batch snapshot exceeds the 128 MiB limit.");
        using(OrtMemoryInfo memory = tensor.GetTensorMemoryInfo())
            if(string.Equals(memory.Name, "Cpu", StringComparison.OrdinalIgnoreCase)) return Render(tensor);
        // Existing geometry commands require batch=1. Keep their contracts: download one bounded batch and build the CPU grid.
        return PreviewTensorDownload.Copy(tensor, backend, Render);
    }

    static void EnsureSuccess(OpResult result)
    {
        if(result.Kind == OpResultKind.Fail) throw new InvalidOperationException(result.Reason, result.Exception);
    }
}
