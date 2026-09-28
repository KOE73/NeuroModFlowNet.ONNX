namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// EN: OpenCV-style camera intrinsics and distortion coefficients for undistort operators and back transforms.
/// <c>Fx..Cy</c> describe the source (distorted) camera. <c>OutputFx..OutputCy</c> optionally describe the undistorted
/// output camera (OpenCV <c>newCameraMatrix</c>, for example from <c>getOptimalNewCameraMatrix</c>); when omitted the
/// output uses the source intrinsics, which matches <c>cv::undistort</c> without a new camera matrix.
///
/// RU: Intrinsics и коэффициенты дисторсии в стиле OpenCV для undistort-операторов и обратных преобразований.
/// <c>Fx..Cy</c> описывают исходную (искажённую) камеру. <c>OutputFx..OutputCy</c> опционально описывают выходную
/// камеру (OpenCV <c>newCameraMatrix</c>, например из <c>getOptimalNewCameraMatrix</c>); если не заданы, выход
/// использует исходные intrinsics, что соответствует <c>cv::undistort</c> без новой матрицы камеры.
/// </summary>
public readonly record struct RadialTangentialDistortionParameters(
    float Fx,
    float Fy,
    float Cx,
    float Cy,
    float K1,
    float K2 = 0f,
    float P1 = 0f,
    float P2 = 0f,
    float K3 = 0f,
    float? OutputFx = null,
    float? OutputFy = null,
    float? OutputCx = null,
    float? OutputCy = null)
{
    public float EffectiveOutputFx => OutputFx ?? Fx;

    public float EffectiveOutputFy => OutputFy ?? Fy;

    public float EffectiveOutputCx => OutputCx ?? Cx;

    public float EffectiveOutputCy => OutputCy ?? Cy;

    public bool HasOutputIntrinsics => OutputFx.HasValue || OutputFy.HasValue || OutputCx.HasValue || OutputCy.HasValue;
}
