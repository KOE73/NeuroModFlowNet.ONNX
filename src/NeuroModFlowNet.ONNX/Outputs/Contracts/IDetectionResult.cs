namespace NeuroModFlowNet.ONNX;

public interface IDetectionResult<T> : IBatchedResult
{
    int GetResultCount(int batchIndex = 0);
    T GetResult(int index, int batchIndex = 0);
    ReadOnlySpan<T> GetBatch(int batchIndex = 0);
}
