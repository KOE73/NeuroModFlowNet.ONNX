using Microsoft.ML.OnnxRuntime;
using NeuroModFlowNet.Pipeline;

namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// Base command for exposing an existing unmanaged memory block as an ONNX Runtime tensor value without copying data.
/// </summary>
/// <remarks>
/// The command does not decide where the source memory came from. Derived classes translate a domain object, for example
/// an OpenCV <c>Mat</c>, into pointer, byte length, element type and tensor shape. This keeps the fragile ONNX Runtime
/// wrapping rule in one place: the tensor is only a view, so the original memory must stay alive for the whole VM context
/// lifetime that can observe the produced <see cref="OrtValue"/>.
/// </remarks>
public abstract class Wrap_UnmanagedMem_To_OrtTensorBase<TInput> : OpBase
{
    readonly string inputKey;
    readonly string outputKey;

    protected Wrap_UnmanagedMem_To_OrtTensorBase(
        string descriptorName,
        string operation,
        string inputKey,
        string outputKey)
        : base(OpDescriptor.Create(
            descriptorName,
            operation,
            reads: [VarRequirement.Read<TInput>(inputKey)],
            writes: [VarRequirement.Write<OrtValue>(outputKey)]))
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(inputKey);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputKey);

        this.inputKey = inputKey;
        this.outputKey = outputKey;
    }

    public override ValueTask<OpResult> ExecuteAsync(
        VmRunContext context,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(context);
        cancellationToken.ThrowIfCancellationRequested();

        if(!context.TryGet(inputKey, out TInput input))
            return ValueTask.FromResult(OpResult.Fail($"Input key '{inputKey}' was not found in the pipeline context."));

        UnmanagedTensorMemoryView sourceMemory = CreateSourceMemoryView(input, context);
        sourceMemory.Validate();

        unsafe
        {
            // CreateTensorValueWithData does not allocate tensor storage and does not copy bytes. ONNX Runtime stores
            // the pointer and trusts the caller to keep that native buffer valid until the OrtValue is disposed.
            var ortTensor = OrtValue.CreateTensorValueWithData(
                OrtMemoryInfo.DefaultInstance,
                sourceMemory.ElementType,
                sourceMemory.Shape,
                sourceMemory.Pointer,
                sourceMemory.ByteLength);

            if(sourceMemory.OwnedResource is not null)
                context.AddOwnedResource(sourceMemory.OwnedResource);

            // The OrtValue itself is owned by the pipeline context. The underlying memory may be owned separately by
            // sourceMemory.OwnedResource, which is why both registrations are needed for cloned/non-contiguous inputs.
            context.Set(outputKey, ortTensor, disposeWithContext: true);
        }

        return ValueTask.FromResult(OpResult.Continue);
    }

    protected abstract UnmanagedTensorMemoryView CreateSourceMemoryView(TInput input, VmRunContext context);
}
