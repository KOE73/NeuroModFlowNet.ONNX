
namespace NeuroModFlowNet.ONNX;

public interface IResultExtractor<out TOut> 
{
    void SetModel(OnnxRuntimeContext context);
    TOut Extract();
}
