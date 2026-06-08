using Microsoft.ML.OnnxRuntime.Tensors;

namespace NeuroModFlowNet.Pipeline.ONNX;

public readonly record struct UnmanagedTensorMemoryView(
    nint Pointer,
    int ByteLength,
    TensorElementType ElementType,
    long[] Shape,
    IDisposable? OwnedResource)
{
    public void Validate()
    {
        if(Pointer == 0)
            throw new ArgumentException("Unmanaged tensor memory pointer must not be zero.", nameof(Pointer));
        if(ByteLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(ByteLength), "Unmanaged tensor memory length must be positive.");
        if(Shape.Length == 0)
            throw new ArgumentException("Unmanaged tensor memory shape must not be empty.", nameof(Shape));
    }
}
