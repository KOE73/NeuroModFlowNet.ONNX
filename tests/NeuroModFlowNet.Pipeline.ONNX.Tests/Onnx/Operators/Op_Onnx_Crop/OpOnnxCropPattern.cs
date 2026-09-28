using NeuroModFlowNet.Pipeline.ONNX.Tests.Common;
using NeuroModFlowNet.ONNX;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.ONNX.Tests.Onnx.Operators.Op_Onnx_Crop;

/// <summary>
/// EN: Creates reproducible diagnostic crop cases where the source image, crop rectangle, expected output size, and
/// expected marker pixels are owned by one operation-specific test pattern.
///
/// RU: Создает воспроизводимые диагностические сценарии crop, где исходное изображение, прямоугольник обрезки,
/// ожидаемый размер результата и ожидаемые marker pixels принадлежат одному operation-specific test pattern.
/// </summary>
/// <remarks>
/// EN: Crop bugs are hard to understand from tensor values alone: the most common failures are an off-by-one border,
/// swapped X/Y coordinates, an inclusive/exclusive rectangle mistake, or a layout/channel assumption that still returns
/// a tensor with the right shape. This pattern makes those failures visible and automatically checkable.
///
/// The source image is synthetic on purpose. It uses a stable background grid plus high-contrast marker pixels at
/// positions derived from the requested crop rectangle. After the operator runs, the test does not need to duplicate
/// the crop math: the generated <see cref="OpOnnxCropPatternCase"/> carries the input image, crop rectangle, expected
/// output size, and the exact pixels that must appear in the cropped result.
///
/// Keep crop-specific assertions here instead of moving them to a generic visual helper. Generic helpers can compare
/// pixels and save artifacts, but this class owns the domain decision about which pixels prove that Crop is correct.
/// Add new cases here when a new failure mode appears: center fill catches broad rectangle placement issues, corner
/// markers catch border and coordinate mistakes, and seeded random cases give wider coverage while keeping failures
/// reproducible from the stored seed.
///
/// RU: Ошибки crop трудно понять только по значениям tensor: чаще всего ломаются границы на один пиксель, путаются
/// координаты X/Y, возникает ошибка inclusive/exclusive прямоугольника или неверное предположение о layout/channel,
/// при котором tensor все еще имеет правильную форму. Этот pattern делает такие сбои видимыми и проверяемыми
/// автоматически.
///
/// Исходное изображение намеренно синтетическое. Оно использует стабильную фоновую сетку и контрастные marker pixels
/// в позициях, вычисленных из запрошенного crop rectangle. После выполнения операции тесту не нужно дублировать crop
/// math: созданный <see cref="OpOnnxCropPatternCase"/> несет входное изображение, crop rectangle, ожидаемый размер
/// результата и точные pixels, которые должны оказаться в результате.
///
/// Crop-specific assertions должны оставаться здесь, а не уходить в общий visual helper. Общие helpers могут сравнивать
/// pixels и сохранять artifacts, но этот класс владеет доменным решением о том, какие pixels доказывают корректность
/// Crop. Новые cases нужно добавлять сюда при появлении нового failure mode: center fill ловит грубые ошибки положения
/// прямоугольника, corner markers ловят ошибки границ и координат, а seeded random cases расширяют покрытие и сохраняют
/// воспроизводимость падения по seed.
/// </remarks>
internal static class OpOnnxCropPattern
{
    static readonly Vec3b Background = new(17, 29, 43);
    static readonly Vec3b CenterMarker = new(191, 32, 224);
    static readonly Vec3b TopLeftMarker = new(9, 73, 201);
    static readonly Vec3b TopRightMarker = new(31, 211, 47);
    static readonly Vec3b BottomLeftMarker = new(223, 67, 11);
    static readonly Vec3b BottomRightMarker = new(241, 241, 241);

    public static IEnumerable<object[]> DeterministicCases()
    {
        yield return [CreateCenteredMarkerCase(1024, 768, new Rect(256, 192, 512, 384))];
        yield return [CreateCornerMarkerCase(1000, 750, new Rect(120, 90, 640, 480))];
    }

    public static OpOnnxCropPatternCase CreateCenteredMarkerCase(int width, int height, Rect cropRect)
    {
        Mat source = CreateCoordinateBackground(width, height);
        Cv2.Rectangle(source, cropRect, CenterMarker.ToScalar(), thickness: -1);

        return new OpOnnxCropPatternCase
        {
            Name = $"center_marker_{width}x{height}_crop_{cropRect.X}_{cropRect.Y}_{cropRect.Width}_{cropRect.Height}",
            Source = source,
            CropRect = cropRect,
            ExpectedSize = new Size(cropRect.Width, cropRect.Height),
            ExpectedPixels =
            [
                new PixelExpectation(0, 0, CenterMarker, "top-left crop pixel"),
                new PixelExpectation(cropRect.Width - 1, 0, CenterMarker, "top-right crop pixel"),
                new PixelExpectation(0, cropRect.Height - 1, CenterMarker, "bottom-left crop pixel"),
                new PixelExpectation(cropRect.Width - 1, cropRect.Height - 1, CenterMarker, "bottom-right crop pixel")
            ]
        };
    }

    public static OpOnnxCropPatternCase CreateCornerMarkerCase(int width, int height, Rect cropRect)
    {
        Mat source = CreateCoordinateBackground(width, height);
        source.Set(cropRect.Y, cropRect.X, TopLeftMarker);
        source.Set(cropRect.Y, cropRect.Right - 1, TopRightMarker);
        source.Set(cropRect.Bottom - 1, cropRect.X, BottomLeftMarker);
        source.Set(cropRect.Bottom - 1, cropRect.Right - 1, BottomRightMarker);

        return new OpOnnxCropPatternCase
        {
            Name = $"corner_marker_{width}x{height}_crop_{cropRect.X}_{cropRect.Y}_{cropRect.Width}_{cropRect.Height}",
            Source = source,
            CropRect = cropRect,
            ExpectedSize = new Size(cropRect.Width, cropRect.Height),
            ExpectedPixels =
            [
                new PixelExpectation(0, 0, TopLeftMarker, "top-left crop pixel"),
                new PixelExpectation(cropRect.Width - 1, 0, TopRightMarker, "top-right crop pixel"),
                new PixelExpectation(0, cropRect.Height - 1, BottomLeftMarker, "bottom-left crop pixel"),
                new PixelExpectation(cropRect.Width - 1, cropRect.Height - 1, BottomRightMarker, "bottom-right crop pixel")
            ]
        };
    }

    public static OpOnnxCropPatternCase CreateRandomCase(int seed, int width, int height)
    {
        var random = new Random(seed);
        int cropWidth = random.Next(4, Math.Max(5, width / 2));
        int cropHeight = random.Next(4, Math.Max(5, height / 2));
        int cropX = random.Next(0, width - cropWidth);
        int cropY = random.Next(0, height - cropHeight);

        OpOnnxCropPatternCase testCase = CreateCornerMarkerCase(width, height, new Rect(cropX, cropY, cropWidth, cropHeight));
        return new OpOnnxCropPatternCase
        {
            Name = $"{testCase.Name}_seed_{seed}",
            Source = testCase.Source,
            CropRect = testCase.CropRect,
            ExpectedSize = testCase.ExpectedSize,
            ExpectedPixels = testCase.ExpectedPixels,
            Tolerance = testCase.Tolerance,
            Seed = seed
        };
    }

    public static void AssertResult(Mat actual, OpOnnxCropPatternCase testCase)
    {
        VisualAssert.MatSize(actual, testCase.ExpectedSize.Width, testCase.ExpectedSize.Height);

        foreach(PixelExpectation expectation in testCase.ExpectedPixels)
            VisualAssert.PixelBgrNear(actual, expectation, testCase.Tolerance);
    }

    public static void SaveArtifacts(Mat actual, OpOnnxCropPatternCase testCase, InferenceBackend executionBackend)
    {
        PipelineOnnxTestEnvironment environment = PipelineOnnxTestEnvironment.Current;

        VisualTestOutput.SaveOrShow(
            $"Op_Onnx_Crop {executionBackend} source: {testCase.Name}",
            testCase.Source,
            environment.CreateOperationArtifactPath(
                "Onnx",
                "Op_Onnx_Crop_U8_NHWC",
                executionBackend.ToString(),
                testCase.Name,
                "source"));

        VisualTestOutput.SaveOrShow(
            $"Op_Onnx_Crop {executionBackend} actual: {testCase.Name}",
            actual,
            environment.CreateOperationArtifactPath(
                "Onnx",
                "Op_Onnx_Crop_U8_NHWC",
                executionBackend.ToString(),
                testCase.Name,
                "actual"));
    }

    static Mat CreateCoordinateBackground(int width, int height)
    {
        var source = new Mat(height, width, MatType.CV_8UC3, Background.ToScalar());

        for(int y = 0; y < height; y++)
        {
            for(int x = 0; x < width; x++)
            {
                if(x % 8 == 0 || y % 8 == 0)
                    source.Set(y, x, new Vec3b((byte)x, (byte)y, (byte)((x + y) & 0xFF)));
            }
        }

        return source;
    }

    static Scalar ToScalar(this Vec3b color) => new(color.Item0, color.Item1, color.Item2);
}
