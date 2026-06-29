using NeuroModFlowNet.Pipeline;

namespace NeuroModFlowNet.Pipeline.ONNX;

/// <summary>
/// Applies one named back-transform register to one coordinate payload register.
/// </summary>
public sealed class Op_Map_Coordinates : OpBase
{
    readonly string inputKey;
    readonly string transformKey;
    readonly string outputKey;
    readonly CoordinateMappingShapePolicy shapePolicy;
    Type? executorInputType;
    IMapCoordinatesExecutor? executor;

    public Op_Map_Coordinates(string inputKey, string transformKey, string outputKey)
        : this(inputKey, transformKey, outputKey, CoordinateMappingShapePolicy.PreserveShape)
    {
    }

    public Op_Map_Coordinates(
        string inputKey,
        string transformKey,
        string outputKey,
        CoordinateMappingShapePolicy shapePolicy)
        : base(OpDescriptor.Create(
            "Op_Map_Coordinates",
            "op.map.coordinates",
            reads: [VarRequirement.Read<object>(inputKey), VarRequirement.Read<ICoordinateBackTransform>(transformKey)],
            writes: [VarRequirement.Write<object>(outputKey)]))
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(inputKey);
        ArgumentException.ThrowIfNullOrWhiteSpace(transformKey);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputKey);

        this.inputKey = inputKey;
        this.transformKey = transformKey;
        this.outputKey = outputKey;
        this.shapePolicy = shapePolicy;
    }

    public override ValueTask<OpResult> ExecuteAsync(VmRunContext context, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(context);
        cancellationToken.ThrowIfCancellationRequested();

        if(!context.TryGetObject(inputKey, out object? input) || input is null)
            return ValueTask.FromResult(OpResult.Fail($"Input key '{inputKey}' was not found in the pipeline context."));

        if(!context.TryGet(transformKey, out ICoordinateBackTransform transform))
            return ValueTask.FromResult(OpResult.Fail($"Transform key '{transformKey}' was not found in the pipeline context."));

        Type inputType = input.GetType();
        if(executor is null || executorInputType != inputType)
        {
            if(!CoordinatePayloadMapperRegistry.TryCreateExecutor(inputType, out executor))
                return ValueTask.FromResult(OpResult.Fail($"No coordinate mapper is registered for payload type '{inputType.FullName}'."));

            executorInputType = inputType;
        }

        object output;
        try
        {
            output = executor.Map(input, transform, shapePolicy);
        }
        catch(NotSupportedException exception)
        {
            return ValueTask.FromResult(OpResult.Fail(exception.Message, exception));
        }
        catch(InvalidOperationException exception)
        {
            return ValueTask.FromResult(OpResult.Fail(exception.Message, exception));
        }

        context.Set(outputKey, output);
        return ValueTask.FromResult(OpResult.Continue);
    }
}
