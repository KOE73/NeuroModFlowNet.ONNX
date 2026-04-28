namespace NeuroModFlowNet.ONNX;

public readonly struct YoloPoseKeypointXYV
{
    public readonly float X;
    public readonly float Y;
    public readonly float V;

    public YoloPoseKeypointXYV(float x, float y, float v)
    {
        X = x;
        Y = y;
        V = v;
    }
}

