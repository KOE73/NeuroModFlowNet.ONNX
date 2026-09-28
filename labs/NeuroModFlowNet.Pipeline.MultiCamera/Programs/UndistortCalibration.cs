using OpenCvSharp;
using CvSize = OpenCvSharp.Size;

namespace NeuroModFlowNet.Pipeline.MultiCamera;

/// <summary>
/// EN: Reproduces CGPCam2026 DistortionCorrector calibration for the ONNX undistort operator. That tool builds K as
/// fx = fy = width * focalLengthFactor, cx = width / 2, cy = height / 2 and undistorts through
/// <c>getOptimalNewCameraMatrix(alpha)</c>. The same new camera matrix is passed as the operator's output intrinsics,
/// so perspective points calibrated on the tool's output apply unchanged.
///
/// RU: Воспроизводит калибровку DistortionCorrector из CGPCam2026 для ONNX undistort. Инструмент строит K как
/// fx = fy = width * focalLengthFactor, cx = width / 2, cy = height / 2 и исправляет через
/// <c>getOptimalNewCameraMatrix(alpha)</c>. Та же новая матрица передаётся оператору как выходные intrinsics, поэтому
/// точки перспективы, откалиброванные на выходе инструмента, применяются без пересчёта.
/// </summary>
internal static class UndistortCalibration
{
    public static RadialTangentialDistortionParameters Create(TransformConfig transform, int width, int height)
    {
        ArgumentNullException.ThrowIfNull(transform);
        DistortionCoefficientsConfig coefficients = transform.DistortionCoefficients
            ?? throw new InvalidDataException("radialUndistort requires distortionCoefficients.");

        double focal = width * transform.FocalLengthFactor;
        double cx = width / 2.0;
        double cy = height / 2.0;

        using var cameraMatrix = new Mat(3, 3, MatType.CV_64FC1, Scalar.All(0));
        cameraMatrix.Set(0, 0, focal);
        cameraMatrix.Set(1, 1, focal);
        cameraMatrix.Set(0, 2, cx);
        cameraMatrix.Set(1, 2, cy);
        cameraMatrix.Set(2, 2, 1.0);

        using var distortionMatrix = new Mat(1, 5, MatType.CV_64FC1);
        distortionMatrix.Set(0, 0, (double)coefficients.K1);
        distortionMatrix.Set(0, 1, (double)coefficients.K2);
        distortionMatrix.Set(0, 2, (double)coefficients.P1);
        distortionMatrix.Set(0, 3, (double)coefficients.P2);
        distortionMatrix.Set(0, 4, (double)coefficients.K3);

        var imageSize = new CvSize(width, height);
        using Mat optimalMatrix = Cv2.GetOptimalNewCameraMatrix(cameraMatrix, distortionMatrix, imageSize, transform.Alpha, imageSize, out _);

        return new RadialTangentialDistortionParameters(
            (float)focal,
            (float)focal,
            (float)cx,
            (float)cy,
            coefficients.K1,
            coefficients.K2,
            coefficients.P1,
            coefficients.P2,
            coefficients.K3,
            OutputFx: (float)optimalMatrix.At<double>(0, 0),
            OutputFy: (float)optimalMatrix.At<double>(1, 1),
            OutputCx: (float)optimalMatrix.At<double>(0, 2),
            OutputCy: (float)optimalMatrix.At<double>(1, 2));
    }
}
