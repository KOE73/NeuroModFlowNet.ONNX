using System.Numerics;

namespace NeuroModFlowNet.Pipeline;

public readonly record struct UndistortCoordinateBackTransform(
    RadialTangentialDistortionParameters Distortion) : ICoordinateBackTransform
{
    public bool TryMapBackward(Vector2 point, out Vector2 mappedPoint)
    {
        // Output pixels live in the output camera; the distorted source pixel is projected with the source camera.
        float normalizedX = (point.X - Distortion.EffectiveOutputCx) / Distortion.EffectiveOutputFx;
        float normalizedY = (point.Y - Distortion.EffectiveOutputCy) / Distortion.EffectiveOutputFy;
        float radius2 = normalizedX * normalizedX + normalizedY * normalizedY;
        float radius4 = radius2 * radius2;
        float radius6 = radius4 * radius2;
        float radial = 1f + Distortion.K1 * radius2 + Distortion.K2 * radius4 + Distortion.K3 * radius6;

        float distortedX = normalizedX * radial +
            2f * Distortion.P1 * normalizedX * normalizedY +
            Distortion.P2 * (radius2 + 2f * normalizedX * normalizedX);
        float distortedY = normalizedY * radial +
            Distortion.P1 * (radius2 + 2f * normalizedY * normalizedY) +
            2f * Distortion.P2 * normalizedX * normalizedY;

        mappedPoint = new Vector2(
            distortedX * Distortion.Fx + Distortion.Cx,
            distortedY * Distortion.Fy + Distortion.Cy);
        return true;
    }
}
