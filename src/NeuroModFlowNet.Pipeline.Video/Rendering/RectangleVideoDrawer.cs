using NeuroModFlowNet.ONNX.Visualizer;

namespace NeuroModFlowNet.Pipeline.Video;

internal sealed class RectangleVideoDrawer : IDirect2DDrawer
{
    public const string RegisterKey = "rectangles";

    public void Draw(IVmRegisterReader registers, IDirect2DCanvas canvas)
    {
        var rectangles = registers.Get<ReadOnlyMemory<Direct2DRectangle>>(RegisterKey);
        float fontSize = Math.Clamp(canvas.Height / 48f, 14f, 36f);
        foreach(Direct2DRectangle rectangle in rectangles.Span)
        {
            canvas.DrawRectangle(rectangle.CenterX, rectangle.CenterY, rectangle.Width, rectangle.Height,
                rectangle.AngleDegrees, rectangle.Rgba, rectangle.StrokeWidth);
            if(rectangle.Label is { Length: > 0 } label)
            {
                float labelLeft = Math.Max(0, rectangle.CenterX - rectangle.Width * 0.5f);
                float labelTop = Math.Max(0, rectangle.CenterY - rectangle.Height * 0.5f - fontSize - 2);
                if(labelLeft < canvas.Width)
                    canvas.DrawText(label, labelLeft, labelTop, canvas.Width - labelLeft, fontSize + 4, rectangle.Rgba);
            }
        }
    }
}
