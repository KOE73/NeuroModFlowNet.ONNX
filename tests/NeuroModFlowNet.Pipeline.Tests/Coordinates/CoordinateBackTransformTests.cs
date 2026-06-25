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
}

