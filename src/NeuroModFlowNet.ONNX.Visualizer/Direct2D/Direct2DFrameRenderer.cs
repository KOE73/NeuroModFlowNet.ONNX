using System.Buffers;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;

namespace NeuroModFlowNet.ONNX.Visualizer;

/// <summary>
/// EN: Renders GPU-resident image data into a caller-owned NV12 D3D11 texture on the same device.
/// Serialize access to an instance. Complete the CUDA write before preparing the background or calling Draw.
///
/// RU: Рисует изображение из памяти GPU в предоставленную текстуру D3D11 NV12 на том же устройстве.
/// Доступ к экземпляру последовательный. Запись CUDA должна завершиться до подготовки фона или вызова Draw.
/// </summary>
[SupportedOSPlatform("windows")]
public sealed unsafe class Direct2DFrameRenderer : IDisposable
{
    readonly nint library;
    readonly delegate* unmanaged[Cdecl]<nint, nint> device;
    readonly delegate* unmanaged[Cdecl]<nint, ulong> map;
    readonly delegate* unmanaged[Cdecl]<nint, nint, uint, NativeRectangle*, uint, char*, int> draw;
    readonly delegate* unmanaged[Cdecl]<nint, int> prepareBackground;
    readonly delegate* unmanaged[Cdecl]<nint, nint, uint, NativeRectangle*, uint, char*, int> drawPrepared;
    readonly delegate* unmanaged[Cdecl]<nint, int> reset;
    readonly delegate* unmanaged[Cdecl]<nint, int> beginDraw;
    readonly delegate* unmanaged[Cdecl]<nint, nint, uint, int> endDraw;
    readonly delegate* unmanaged[Cdecl]<nint, float, float, float, float, float, uint, float, int> drawRectangle;
    readonly delegate* unmanaged[Cdecl]<nint, float, float, float, float, uint, float, int> drawLine;
    readonly delegate* unmanaged[Cdecl]<nint, char*, uint, float, float, float, float, uint, float, int> drawText;
    readonly delegate* unmanaged[Cdecl]<nint, void> destroy;
    readonly delegate* unmanaged[Cdecl]<nint> error;
    readonly Direct2DCanvas canvas;
    nint renderer;

    public Direct2DFrameRenderer(string libraryPath, int deviceId, int width, int height, int outputWidth, int outputHeight)
        : this(libraryPath, deviceId, width, height, outputWidth, outputHeight, requireStagedDrawing: false)
    {
    }

    public Direct2DFrameRenderer(string libraryPath, int deviceId, int width, int height, int outputWidth, int outputHeight,
        bool requireStagedDrawing)
        : this(libraryPath, deviceId, width, height, outputWidth, outputHeight, requireStagedDrawing, requireCanvasDrawing: false)
    {
    }

    public Direct2DFrameRenderer(string libraryPath, int deviceId, int width, int height, int outputWidth, int outputHeight,
        bool requireStagedDrawing, bool requireCanvasDrawing)
    {
        canvas = new(this, width, height);
        library = NativeLibrary.Load(Path.GetFullPath(libraryPath));
        try
        {
            var create = (delegate* unmanaged[Cdecl]<int, int, int, int, int, nint>)NativeLibrary.GetExport(library, "NmfD2DCreate");
            device = (delegate* unmanaged[Cdecl]<nint, nint>)NativeLibrary.GetExport(library, "NmfD2DDevice");
            map = (delegate* unmanaged[Cdecl]<nint, ulong>)NativeLibrary.GetExport(library, "NmfD2DMap");
            draw = (delegate* unmanaged[Cdecl]<nint, nint, uint, NativeRectangle*, uint, char*, int>)NativeLibrary.GetExport(library, "NmfD2DDraw");
            destroy = (delegate* unmanaged[Cdecl]<nint, void>)NativeLibrary.GetExport(library, "NmfD2DDestroy");
            error = (delegate* unmanaged[Cdecl]<nint>)NativeLibrary.GetExport(library, "NmfD2DError");
            if(NativeLibrary.TryGetExport(library, "NmfD2DPrepareBackground", out nint prepareBackgroundAddress) &&
                NativeLibrary.TryGetExport(library, "NmfD2DDrawPrepared", out nint drawPreparedAddress) &&
                NativeLibrary.TryGetExport(library, "NmfD2DReset", out nint resetAddress))
            {
                prepareBackground = (delegate* unmanaged[Cdecl]<nint, int>)prepareBackgroundAddress;
                drawPrepared = (delegate* unmanaged[Cdecl]<nint, nint, uint, NativeRectangle*, uint, char*, int>)drawPreparedAddress;
                reset = (delegate* unmanaged[Cdecl]<nint, int>)resetAddress;
            }
            if(NativeLibrary.TryGetExport(library, "NmfD2DBeginDraw", out nint beginDrawAddress) &&
                NativeLibrary.TryGetExport(library, "NmfD2DEndDraw", out nint endDrawAddress) &&
                NativeLibrary.TryGetExport(library, "NmfD2DDrawRectangle", out nint drawRectangleAddress) &&
                NativeLibrary.TryGetExport(library, "NmfD2DDrawLine", out nint drawLineAddress) &&
                NativeLibrary.TryGetExport(library, "NmfD2DDrawText", out nint drawTextAddress))
            {
                beginDraw = (delegate* unmanaged[Cdecl]<nint, int>)beginDrawAddress;
                endDraw = (delegate* unmanaged[Cdecl]<nint, nint, uint, int>)endDrawAddress;
                drawRectangle = (delegate* unmanaged[Cdecl]<nint, float, float, float, float, float, uint, float, int>)drawRectangleAddress;
                drawLine = (delegate* unmanaged[Cdecl]<nint, float, float, float, float, uint, float, int>)drawLineAddress;
                drawText = (delegate* unmanaged[Cdecl]<nint, char*, uint, float, float, float, float, uint, float, int>)drawTextAddress;
            }
            if(requireStagedDrawing || requireCanvasDrawing)
                RequireStagedDrawing();
            if(requireCanvasDrawing)
                RequireCanvasDrawing();
            renderer = create(deviceId, width, height, outputWidth, outputHeight);
            if(renderer == 0)
                throw ReadError();
        }
        catch { NativeLibrary.Free(library); throw; }
    }

    #region GPU frame operations

    public nint Device => device(GetRenderer());

    public nint MapPackedBgra()
    {
        ulong address = map(GetRenderer());
        if(address == 0)
            throw ReadError();
        return (nint)address;
    }

    public void Draw(nint nv12Texture, uint slice, ReadOnlySpan<Direct2DRectangle> rectangles) =>
        DrawCore(nv12Texture, slice, rectangles, draw);

    /// <summary>
    /// EN: Returns the CUDA buffer to D3D11 and submits the background draw on the GPU.
    ///
    /// RU: Возвращает буфер CUDA в D3D11 и ставит рисование фона в очередь GPU.
    /// </summary>
    public void PrepareBackground()
    {
        RequireStagedDrawing();
        if(prepareBackground(GetRenderer()) != 0)
            throw ReadError();
    }

    /// <summary>
    /// EN: Draws the overlay over the prepared background and waits for the GPU to finish writing the NV12 target.
    ///
    /// RU: Рисует поверх подготовленного фона и дожидается записи кадра в целевую текстуру NV12 на GPU.
    /// </summary>
    public void DrawPrepared(nint nv12Texture, uint slice, ReadOnlySpan<Direct2DRectangle> rectangles)
    {
        RequireStagedDrawing();
        DrawCore(nv12Texture, slice, rectangles, drawPrepared);
    }

    /// <summary>
    /// EN: Opens a drawing pass on the prepared background. The canvas is valid until EndDraw or Reset.
    ///
    /// RU: Открывает проход рисования поверх подготовленного фона. Холст действителен до EndDraw или Reset.
    /// </summary>
    public IDirect2DCanvas BeginDraw()
    {
        RequireCanvasDrawing();
        if(beginDraw(GetRenderer()) != 0)
            throw ReadError();
        return canvas;
    }

    /// <summary>
    /// EN: Finishes drawing, writes the NV12 GPU surface, and waits for it to become ready.
    ///
    /// RU: Завершает рисование, записывает поверхность NV12 на GPU и дожидается её готовности.
    /// </summary>
    public void EndDraw(nint nv12Texture, uint slice)
    {
        RequireCanvasDrawing();
        if(endDraw(GetRenderer(), nv12Texture, slice) != 0)
            throw ReadError();
    }

    /// <summary>
    /// EN: Abandons the current frame, returning any CUDA mapping and waiting for submitted D3D11 work.
    ///
    /// RU: Сбрасывает текущий кадр, возвращает буфер из CUDA и дожидается запущенных команд D3D11.
    /// </summary>
    public void Reset()
    {
        RequireStagedDrawing();
        if(reset(GetRenderer()) != 0)
            throw ReadError();
    }

    void DrawCore(nint nv12Texture, uint slice, ReadOnlySpan<Direct2DRectangle> rectangles,
        delegate* unmanaged[Cdecl]<nint, nint, uint, NativeRectangle*, uint, char*, int> drawFrame)
    {
        int textLength = 0;
        foreach(Direct2DRectangle rectangle in rectangles)
            textLength = checked(textLength + (rectangle.Label?.Length ?? 0));

        NativeRectangle[] native = ArrayPool<NativeRectangle>.Shared.Rent(rectangles.Length);
        char[] text = ArrayPool<char>.Shared.Rent(textLength);
        try
        {
            int offset = 0;
            for(int index = 0; index < rectangles.Length; index++)
            {
                Direct2DRectangle rectangle = rectangles[index];
                ReadOnlySpan<char> label = rectangle.Label.AsSpan();
                label.CopyTo(text.AsSpan(offset));
                native[index] = new(rectangle, (uint)offset, (uint)label.Length);
                offset += label.Length;
            }

            fixed(NativeRectangle* boxes = native)
            fixed(char* labels = text)
            {
                if(drawFrame(GetRenderer(), nv12Texture, slice, boxes, (uint)rectangles.Length, labels) != 0)
                    throw ReadError();
            }
        }
        finally
        {
            ArrayPool<NativeRectangle>.Shared.Return(native);
            ArrayPool<char>.Shared.Return(text);
        }
    }

    #endregion

    public void Dispose()
    {
        if(renderer == 0)
            return;
        destroy(renderer);
        renderer = 0;
        NativeLibrary.Free(library);
    }

    nint GetRenderer()
    {
        ObjectDisposedException.ThrowIf(renderer == 0, this);
        return renderer;
    }

    void RequireStagedDrawing()
    {
        if(prepareBackground == null)
            throw new EntryPointNotFoundException("The Direct2D renderer library does not export the staged drawing API. Rebuild the native Direct2D renderer.");
    }

    void RequireCanvasDrawing()
    {
        if(beginDraw == null)
            throw new EntryPointNotFoundException("The Direct2D renderer library does not export the drawing canvas API. Rebuild the native Direct2D renderer.");
    }

    InvalidOperationException ReadError() => new(Marshal.PtrToStringUTF8(error()) ?? "Direct2D renderer failed.");

    sealed class Direct2DCanvas(Direct2DFrameRenderer owner, int width, int height) : IDirect2DCanvas
    {
        public int Width { get; } = width;
        public int Height { get; } = height;

        public void DrawRectangle(float centerX, float centerY, float width, float height, float angleDegrees, uint rgba,
            float strokeWidth)
        {
            if(owner.drawRectangle(owner.GetRenderer(), centerX, centerY, width, height, angleDegrees, rgba, strokeWidth) != 0)
                throw owner.ReadError();
        }

        public void DrawLine(float startX, float startY, float endX, float endY, uint rgba, float strokeWidth)
        {
            if(owner.drawLine(owner.GetRenderer(), startX, startY, endX, endY, rgba, strokeWidth) != 0)
                throw owner.ReadError();
        }

        public void DrawText(string text, float left, float top, float width, float height, uint rgba, float fontSize)
        {
            ArgumentNullException.ThrowIfNull(text);
            fixed(char* characters = text)
            {
                if(owner.drawText(owner.GetRenderer(), characters, (uint)text.Length, left, top, width, height, rgba, fontSize) != 0)
                    throw owner.ReadError();
            }
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    readonly struct NativeRectangle(Direct2DRectangle rectangle, uint labelOffset, uint labelLength)
    {
        readonly float centerX = rectangle.CenterX;
        readonly float centerY = rectangle.CenterY;
        readonly float width = rectangle.Width;
        readonly float height = rectangle.Height;
        readonly float angleDegrees = rectangle.AngleDegrees;
        readonly float strokeWidth = rectangle.StrokeWidth;
        readonly uint rgba = rectangle.Rgba;
        readonly uint offset = labelOffset;
        readonly uint length = labelLength;
    }
}
