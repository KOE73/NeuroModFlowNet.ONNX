namespace NeuroModFlowNet.Pipeline;

public readonly record struct PadResizeLayout(
    int SourceWidth,
    int SourceHeight,
    int ResizedWidth,
    int ResizedHeight,
    int OutputWidth,
    int OutputHeight,
    float Scale,
    int PadLeft,
    int PadTop,
    int PadRight,
    int PadBottom,
    int Stride,
    PadResizeMode Mode)
{
    public static PadResizeLayout Create(
        int sourceWidth,
        int sourceHeight,
        int targetWidth,
        int targetHeight,
        int stride,
        PadResizeMode mode)
    {
        if(sourceWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(sourceWidth), "Source width must be positive.");

        if(sourceHeight <= 0)
            throw new ArgumentOutOfRangeException(nameof(sourceHeight), "Source height must be positive.");

        if(targetWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(targetWidth), "Target width must be positive.");

        if(targetHeight <= 0)
            throw new ArgumentOutOfRangeException(nameof(targetHeight), "Target height must be positive.");

        if(stride <= 0)
            throw new ArgumentOutOfRangeException(nameof(stride), "Stride must be positive.");

        float scale = Math.Min((float)targetWidth / sourceWidth, (float)targetHeight / sourceHeight);
        int resizedWidth = Math.Max(1, (int)Math.Round(sourceWidth * scale));
        int resizedHeight = Math.Max(1, (int)Math.Round(sourceHeight * scale));

        int outputWidth;
        int outputHeight;
        if(mode == PadResizeMode.FixedCanvas)
        {
            outputWidth = targetWidth;
            outputHeight = targetHeight;
        }
        else if(mode == PadResizeMode.AutoStrideCanvas)
        {
            outputWidth = RoundUpToStride(resizedWidth, stride);
            outputHeight = RoundUpToStride(resizedHeight, stride);
        }
        else
        {
            throw new ArgumentOutOfRangeException(nameof(mode), mode, "Unsupported pad resize mode.");
        }

        int padLeft = (outputWidth - resizedWidth) / 2;
        int padTop = (outputHeight - resizedHeight) / 2;
        int padRight = outputWidth - resizedWidth - padLeft;
        int padBottom = outputHeight - resizedHeight - padTop;

        if(padLeft < 0 || padTop < 0 || padRight < 0 || padBottom < 0)
            throw new InvalidOperationException("Pad resize layout produced negative padding.");

        return new PadResizeLayout(
            sourceWidth,
            sourceHeight,
            resizedWidth,
            resizedHeight,
            outputWidth,
            outputHeight,
            scale,
            padLeft,
            padTop,
            padRight,
            padBottom,
            stride,
            mode);
    }

    static int RoundUpToStride(int value, int stride) =>
        ((value + stride - 1) / stride) * stride;
}
