
namespace NeuroModFlowNet.ONNX;

public interface IResultExtractor<out TOut> 
{
    void SetModel(IModelMetadataProvider metadata);
    TOut Extract(IOnnxModelOutputs outputs);
}
