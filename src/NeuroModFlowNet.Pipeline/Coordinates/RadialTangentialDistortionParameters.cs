namespace NeuroModFlowNet.Pipeline;

public readonly record struct RadialTangentialDistortionParameters(
    float Fx,
    float Fy,
    float Cx,
    float Cy,
    float K1,
    float K2 = 0f,
    float P1 = 0f,
    float P2 = 0f,
    float K3 = 0f);
