using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline.ONNX;
using NeuroModFlowNet.Pipeline.Video.Nvdec;
using OpenCvSharp;
using CvRect = OpenCvSharp.Rect;
using CvSize = OpenCvSharp.Size;

namespace NeuroModFlowNet.Pipeline.MultiCamera;

/// <summary>
/// EN: Builds the per-camera <c>prepare</c> program: host Mat -> OrtValue -> model device -> configured geometry chain
/// (undistort / perspective / rotate / crop / resize as ONNX operators) -> <see cref="MultiCameraKeys.ImageRect"/>.
/// The transform list mirrors CGPCam2026 <c>cameras.json</c>, so calibration done there can be pasted here.
///
/// RU: Собирает prepare-программу камеры: Mat хоста -> OrtValue -> устройство модели -> настроенная цепочка геометрии
/// (undistort / perspective / rotate / crop / resize как ONNX-операторы) -> <see cref="MultiCameraKeys.ImageRect"/>.
/// Список transforms повторяет CGPCam2026 <c>cameras.json</c>, калибровка оттуда переносится копированием.
/// </summary>
/// <remarks>
/// EN: Every controller gets its own program instance because geometry ops hold mutable native state (IoBinding,
/// RunOptions). Kernels are still shared through the global runtime kernel cache keyed by shape and parameters.
///
/// RU: У каждого контроллера свой экземпляр программы, потому что geometry-ops держат mutable native state
/// (IoBinding, RunOptions). Kernels при этом общие через глобальный кэш по форме и параметрам.
/// </remarks>
internal static class CameraPrepareProgramFactory
{
    public static (VmProgram Program, CameraGeometry Geometry) Create(CameraConfig camera, InferenceBackend backend)
    {
        ArgumentNullException.ThrowIfNull(camera);

        var builder = new VmProgramBuilder($"prepare:{camera.Id}");
        var geometry = new CameraGeometry(camera.Resolution.Width, camera.Resolution.Height);

        // Ingest: the only part that depends on the decoder. Both variants leave the same bridge format in model device
        // memory (BGR U8 NHWC), so every transform below and the whole common program are identical.
        bool hasTransforms = camera.Transforms.Count > 0;
        string ingestOutput = hasTransforms ? MultiCameraKeys.SourceDevice : MultiCameraKeys.ImageRect;
        if(camera.Source.IsNvdec)
        {
            // NVDEC surface is already in CUDA memory: one colour/layout op, no host copy, no upload.
            builder.Step(new Op_Onnx_Nv12_To_BgrU8Nhwc(
                MultiCameraKeys.SourceNv12,
                ingestOutput,
                camera.Resolution.Width,
                camera.Resolution.Height,
                camera.Source.ColorMatrix,
                isFinal: false,
                executionBackend: backend));
        }
        else
        {
            // Zero-copy wrap of the host frame, then one explicit device upload. No implicit CPU->GPU copies later.
            builder.Step(new Wrap_MatImage_To_OrtTensor(MultiCameraKeys.SourceImage, MultiCameraKeys.SourceTensor));
            builder.Step(new Copy_OrtTensor_To_ModelDevice(MultiCameraKeys.SourceTensor, ingestOutput, backend));
        }

        string inputKey = MultiCameraKeys.SourceDevice;

        for(int index = 0; index < camera.Transforms.Count; index++)
        {
            TransformConfig transform = camera.Transforms[index];
            bool isLast = index == camera.Transforms.Count - 1;
            string outputKey = isLast ? MultiCameraKeys.ImageRect : MultiCameraKeys.PrepareStage(index);
            string transformKey = MultiCameraKeys.PrepareTransform(index);

            switch(transform.Type.ToLowerInvariant())
            {
                case "radialundistort":
                {
                    // Output intrinsics come from getOptimalNewCameraMatrix(alpha), exactly as in the CGPCam2026 tool,
                    // so the following transforms can use the points calibrated on that tool's output.
                    builder.Step(new Op_Onnx_Undistort_U8_NHWC(
                        inputKey,
                        outputKey,
                        UndistortCalibration.Create(transform, geometry.Width, geometry.Height),
                        outputSize: null,
                        outputTransformKey: transformKey,
                        isFinal: false,
                        executionBackend: backend));
                    break;
                }
                case "perspective":
                {
                    Point2f[] sourcePoints = transform.SourcePoints!
                        .Select(static point => new Point2f(point.X, point.Y))
                        .ToArray();
                    var outputSize = new CvSize(transform.OutputSize!.Width, transform.OutputSize.Height);

                    builder.Step(new Op_Onnx_Perspective_U8_NHWC(
                        inputKey,
                        outputKey,
                        sourcePoints,
                        outputSize,
                        outputTransformKey: transformKey,
                        isFinal: false,
                        executionBackend: backend));

                    geometry = new CameraGeometry(outputSize.Width, outputSize.Height);
                    break;
                }
                case "rotate":
                {
                    Rotate90Mode mode = transform.Degrees switch
                    {
                        90 => Rotate90Mode.Clockwise90,
                        180 => Rotate90Mode.Rotate180,
                        _ => Rotate90Mode.CounterClockwise90
                    };

                    builder.Step(new Op_Onnx_Rotate90_U8_NHWC(
                        inputKey,
                        outputKey,
                        mode,
                        outputTransformKey: transformKey,
                        isFinal: false,
                        executionBackend: backend));

                    if(mode != Rotate90Mode.Rotate180)
                        geometry = new CameraGeometry(geometry.Height, geometry.Width);

                    break;
                }
                case "crop":
                {
                    var cropRect = new CvRect(transform.X, transform.Y, transform.Width, transform.Height);

                    if(cropRect.X < 0 || cropRect.Y < 0 || cropRect.Right > geometry.Width || cropRect.Bottom > geometry.Height)
                        throw new InvalidDataException($"camera '{camera.Id}': crop {cropRect} exceeds the {geometry.Width}x{geometry.Height} frame.");

                    builder.Step(new Op_Onnx_Crop_U8_NHWC(
                        inputKey,
                        outputKey,
                        cropRect,
                        outputTransformKey: transformKey,
                        isFinal: false,
                        executionBackend: backend));

                    geometry = new CameraGeometry(cropRect.Width, cropRect.Height);
                    break;
                }
                case "resize":
                {
                    var targetSize = new CvSize(transform.Width, transform.Height);
                    builder.Step(new Op_Onnx_Resize_U8_NHWC(
                        inputKey,
                        outputKey,
                        targetSize,
                        outputTransformKey: transformKey,
                        isFinal: false,
                        executionBackend: backend));

                    geometry = new CameraGeometry(targetSize.Width, targetSize.Height);
                    break;
                }
                default:
                    throw new InvalidDataException($"camera '{camera.Id}': unknown transform type '{transform.Type}'.");
            }

            inputKey = outputKey;
        }

        return (builder.Build(), geometry);
    }
}
