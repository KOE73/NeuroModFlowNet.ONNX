using NeuroModFlowNet.ONNX;

namespace NeuroModFlowNet.Pipeline.ONNX;

internal static class CoordinatePayloadMapperRegistry
{
    static readonly Dictionary<Type, Type> MapperTypes = new()
    {
        [typeof(YoloBox)] = typeof(YoloBoxCoordinatePayloadMapper),
        [typeof(YoloObb)] = typeof(YoloObbCoordinatePayloadMapper),
        [typeof(OcrQuadRegion)] = typeof(OcrQuadRegionCoordinatePayloadMapper),
    };

    public static bool TryCreateExecutor(Type payloadType, out IMapCoordinatesExecutor executor)
    {
        ArgumentNullException.ThrowIfNull(payloadType);

        if(MapperTypes.TryGetValue(payloadType, out Type? mapperType))
        {
            executor = CreateExecutor(typeof(MapCoordinatesExecutor<,>), payloadType, mapperType);
            return true;
        }

        if(payloadType.IsArray)
        {
            Type? elementType = payloadType.GetElementType();
            if(elementType is not null && MapperTypes.TryGetValue(elementType, out mapperType))
            {
                executor = CreateExecutor(typeof(ArrayMapCoordinatesExecutor<,>), elementType, mapperType);
                return true;
            }
        }

        if(payloadType.IsGenericType)
        {
            Type genericDefinition = payloadType.GetGenericTypeDefinition();
            Type elementType = payloadType.GetGenericArguments()[0];

            if(genericDefinition == typeof(List<>) && MapperTypes.TryGetValue(elementType, out mapperType))
            {
                executor = CreateExecutor(typeof(ListMapCoordinatesExecutor<,>), elementType, mapperType);
                return true;
            }

            if(genericDefinition == typeof(YoloDetectionBatchResult<>) && MapperTypes.TryGetValue(elementType, out mapperType))
            {
                executor = CreateExecutor(typeof(YoloDetectionBatchResultMapCoordinatesExecutor<,>), elementType, mapperType);
                return true;
            }
        }

        executor = null!;
        return false;
    }

    static IMapCoordinatesExecutor CreateExecutor(Type executorGenericDefinition, Type payloadType, Type mapperType)
    {
        Type executorType = executorGenericDefinition.MakeGenericType(payloadType, mapperType);
        return (IMapCoordinatesExecutor)Activator.CreateInstance(executorType)!;
    }
}
