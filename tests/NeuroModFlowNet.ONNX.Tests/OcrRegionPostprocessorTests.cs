using NeuroModFlowNet.ONNX;
using OpenCvSharp;
using Xunit;

namespace NeuroModFlowNet.ONNX.Tests.PaddleOCR;

public class OcrRegionPostprocessorTests
{
    [Fact]
    public void Process_FiltersRegionsByDomainHeight()
    {
        OcrQuadRegion[] regions =
        [
            CreateRegion(0, 0, 40, 5),
            CreateRegion(0, 10, 40, 20),
            CreateRegion(0, 40, 40, 60),
        ];

        var options = new OcrRegionPostprocessorOptions
        {
            MinRegionHeight = 10,
            MaxRegionHeight = 40,
        };

        List<OcrQuadRegion> destination = [];
        int count = OcrRegionPostprocessor.Shared.Process(regions, options, destination);

        Assert.Equal(1, count);
        Assert.Single(destination);
        Assert.Equal(10, destination[0].Y0);
    }

    [Fact]
    public void Process_SuppressesStronglyOverlappingRegions()
    {
        OcrQuadRegion[] regions =
        [
            CreateRegion(0, 0, 100, 20),
            CreateRegion(5, 2, 95, 18),
        ];

        var options = new OcrRegionPostprocessorOptions
        {
            EnableOverlapSuppression = true,
            OverlapSuppressionRatio = 0.5f,
        };

        List<OcrQuadRegion> destination = [];
        int count = OcrRegionPostprocessor.Shared.Process(regions, options, destination);

        Assert.Equal(1, count);
        Assert.Single(destination);
    }

    [Fact]
    public void Process_MergesNeighborLineRegionsWhenEnabled()
    {
        OcrQuadRegion[] regions =
        [
            CreateRegion(0, 0, 40, 12),
            CreateRegion(45, 0, 40, 12),
        ];

        var options = new OcrRegionPostprocessorOptions
        {
            EnableLineMerge = true,
            MergeGapInHeights = 1f,
            MinimumMergedCoverageRatio = 0.5f,
            MaxMergedRegionAspectRatio = 10f,
        };

        List<OcrQuadRegion> destination = [];
        int count = OcrRegionPostprocessor.Shared.Process(regions, options, destination);

        Assert.Equal(1, count);
        Assert.Single(destination);
        Assert.True(destination[0].X1 - destination[0].X0 > 80);
    }

    [Fact]
    public void Extract_DetMaskScoreMap_ReturnsQuadRegion()
    {
        using var scoreMap = new Mat(64, 128, MatType.CV_32FC1, Scalar.Black);
        using Mat region = scoreMap[new Rect(20, 20, 60, 16)];
        region.SetTo(new Scalar(0.95));

        var options = new PaddleOCRDetMaskRegionExtractorOptions
        {
            EnableUnclip = false,
            BitmapThreshold = 0.3f,
            BoxScoreThreshold = 0.7f,
            MinimumBoxSide = 3f,
        };

        List<OcrQuadRegion> destination = [];
        int count = PaddleOCRDetMaskRegionExtractor.Shared.Extract(scoreMap, options, destination);

        Assert.Equal(1, count);
        Assert.Single(destination);
        Assert.True(destination[0].X0 >= 15);
        Assert.True(destination[0].Y0 >= 15);
    }

    private static OcrQuadRegion CreateRegion(float x, float y, float width, float height) =>
        new(
            x,
            y,
            x + width,
            y,
            x + width,
            y + height,
            x,
            y + height);
}
