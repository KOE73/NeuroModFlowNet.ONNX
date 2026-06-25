using NeuroModFlowNet.Pipeline;

namespace NeuroModFlowNet.Pipeline.ONNX;

internal interface IMapCoordinatesExecutor
{
    object Map(object input, ICoordinateBackTransform transform);
}
