namespace NeuroModFlowNet.ONNX;

public struct YoloPose : IOutAsT<YoloPose>
{
    public float X { get; init; }
    public float Y { get; init; }
    public float W { get; init; }
    public float H { get; init; }
    public float Score { get; init; }
    public float ClassId { get; init; }

    // Здесь уже обычный управляемый массив, потому что результат должен жить после выхода из метода.
    public required YoloPoseKeypointXYV[] Keypoints { get; init; }

    public YoloPose AsStd() => this;
}

