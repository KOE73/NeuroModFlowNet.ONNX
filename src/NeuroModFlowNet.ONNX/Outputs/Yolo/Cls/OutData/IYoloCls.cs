namespace NeuroModFlowNet.ONNX;

public interface IYoloCls
{
    int ClassId { get; }
    float Score { get; }
    float[] Scores { get; }
}
