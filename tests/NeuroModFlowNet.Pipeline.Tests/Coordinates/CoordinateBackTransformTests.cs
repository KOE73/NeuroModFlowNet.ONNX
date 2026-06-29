using System.Numerics;

namespace NeuroModFlowNet.Pipeline.Tests.Coordinates;

public sealed class CoordinateBackTransformTests
{
    [Fact]
    public void CropCoordinateBackTransform_MapsPointBackByAddingCropOrigin()
    {
        var transform = new CropCoordinateBackTransform(12, 25);

        bool mapped = transform.TryMapBackward(new Vector2(3, 4), out Vector2 point);

        Assert.True(mapped);
        Assert.Equal(new Vector2(15, 29), point);
    }

    [Fact]
    public void ResizeCoordinateBackTransform_MapsPointBackBySourceToTargetScale()
    {
        var transform = new ResizeCoordinateBackTransform(
            sourceWidth: 640,
            sourceHeight: 480,
            targetWidth: 320,
            targetHeight: 120);

        bool mapped = transform.TryMapBackward(new Vector2(10, 20), out Vector2 point);

        Assert.True(mapped);
        Assert.Equal(new Vector2(20, 80), point);
    }

    [Fact]
    public void BackTransforms_CanBeAppliedAsExplicitRiscChain()
    {
        ICoordinateBackTransform resizedToCrop = new ResizeCoordinateBackTransform(
            sourceWidth: 200,
            sourceHeight: 100,
            targetWidth: 100,
            targetHeight: 50);
        ICoordinateBackTransform cropToSource = new CropCoordinateBackTransform(5, 7);

        bool mappedToCrop = resizedToCrop.TryMapBackward(new Vector2(10, 20), out Vector2 cropPoint);
        bool mappedToSource = cropToSource.TryMapBackward(cropPoint, out Vector2 sourcePoint);

        Assert.True(mappedToCrop);
        Assert.True(mappedToSource);
        Assert.Equal(new Vector2(20, 40), cropPoint);
        Assert.Equal(new Vector2(25, 47), sourcePoint);
    }

    [Theory]
    [InlineData(Rotate90Mode.Clockwise90, 0, 0, 0, 479)]
    [InlineData(Rotate90Mode.Clockwise90, 100, 25, 25, 379)]
    [InlineData(Rotate90Mode.Rotate180, 0, 0, 639, 479)]
    [InlineData(Rotate90Mode.Rotate180, 100, 25, 539, 454)]
    [InlineData(Rotate90Mode.CounterClockwise90, 0, 0, 639, 0)]
    [InlineData(Rotate90Mode.CounterClockwise90, 100, 25, 614, 100)]
    public void Rotate90CoordinateBackTransform_MapsPointBackToSource(
        Rotate90Mode mode,
        float x,
        float y,
        float expectedX,
        float expectedY)
    {
        var transform = new Rotate90CoordinateBackTransform(
            sourceWidth: 640,
            sourceHeight: 480,
            mode);

        bool mapped = transform.TryMapBackward(new Vector2(x, y), out Vector2 point);

        Assert.True(mapped);
        Assert.Equal(new Vector2(expectedX, expectedY), point);
    }

    [Fact]
    public void PadResizeLayout_FixedCanvas_KeepsPanoramicTargetWithoutPadding()
    {
        PadResizeLayout layout = PadResizeLayout.Create(
            sourceWidth: 2048,
            sourceHeight: 256,
            targetWidth: 2048,
            targetHeight: 256,
            stride: 32,
            PadResizeMode.FixedCanvas);

        Assert.Equal(2048, layout.OutputWidth);
        Assert.Equal(256, layout.OutputHeight);
        Assert.Equal(2048, layout.ResizedWidth);
        Assert.Equal(256, layout.ResizedHeight);
        Assert.Equal(0, layout.PadLeft);
        Assert.Equal(0, layout.PadTop);
        Assert.Equal(0, layout.PadRight);
        Assert.Equal(0, layout.PadBottom);
    }

    [Fact]
    public void PadResizeLayout_FixedCanvas_AddsCenteredPadding()
    {
        PadResizeLayout layout = PadResizeLayout.Create(
            sourceWidth: 1000,
            sourceHeight: 500,
            targetWidth: 640,
            targetHeight: 640,
            stride: 32,
            PadResizeMode.FixedCanvas);

        Assert.Equal(640, layout.OutputWidth);
        Assert.Equal(640, layout.OutputHeight);
        Assert.Equal(640, layout.ResizedWidth);
        Assert.Equal(320, layout.ResizedHeight);
        Assert.Equal(0, layout.PadLeft);
        Assert.Equal(160, layout.PadTop);
        Assert.Equal(0, layout.PadRight);
        Assert.Equal(160, layout.PadBottom);
    }

    [Fact]
    public void PadResizeLayout_AutoStrideCanvas_RoundsOutputToStride()
    {
        PadResizeLayout layout = PadResizeLayout.Create(
            sourceWidth: 1000,
            sourceHeight: 400,
            targetWidth: 641,
            targetHeight: 641,
            stride: 32,
            PadResizeMode.AutoStrideCanvas);

        Assert.Equal(672, layout.OutputWidth);
        Assert.Equal(256, layout.OutputHeight);
        Assert.Equal(641, layout.ResizedWidth);
        Assert.Equal(256, layout.ResizedHeight);
        Assert.Equal(15, layout.PadLeft);
        Assert.Equal(0, layout.PadTop);
        Assert.Equal(16, layout.PadRight);
        Assert.Equal(0, layout.PadBottom);
    }

    [Fact]
    public void PadResizeCoordinateBackTransform_RemovesPaddingAndScale()
    {
        var transform = new PadResizeCoordinateBackTransform(
            SourceWidth: 1000,
            SourceHeight: 500,
            ResizedWidth: 640,
            ResizedHeight: 320,
            OutputWidth: 640,
            OutputHeight: 640,
            Scale: 0.64f,
            PadLeft: 0,
            PadTop: 160,
            PadRight: 0,
            PadBottom: 160,
            Stride: 32,
            Mode: PadResizeMode.FixedCanvas);

        bool mapped = transform.TryMapBackward(new Vector2(64, 224), out Vector2 point);

        Assert.True(mapped);
        Assert.Equal(new Vector2(100, 100), point);
    }

    [Fact]
    public void PerspectiveCoordinateBackTransform_MapsPointByHomography()
    {
        var transform = new PerspectiveCoordinateBackTransform(
            2, 0, 10,
            0, 3, 20,
            0, 0, 1);

        bool mapped = transform.TryMapBackward(new Vector2(5, 7), out Vector2 point);

        Assert.True(mapped);
        Assert.Equal(new Vector2(20, 41), point);
    }

    [Fact]
    public void UndistortCoordinateBackTransform_MapsPointToDistortedSource()
    {
        var transform = new UndistortCoordinateBackTransform(
            new RadialTangentialDistortionParameters(
                Fx: 100,
                Fy: 100,
                Cx: 50,
                Cy: 50,
                K1: 0.1f));

        bool mapped = transform.TryMapBackward(new Vector2(150, 50), out Vector2 point);

        Assert.True(mapped);
        Assert.Equal(new Vector2(160, 50), point);
    }
}
