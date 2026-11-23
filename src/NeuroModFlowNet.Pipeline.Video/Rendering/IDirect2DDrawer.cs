using NeuroModFlowNet.ONNX.Visualizer;

namespace NeuroModFlowNet.Pipeline.Video;

/// <summary>
/// EN: Draws application data from the current render VM registers on its active GPU canvas.
///
/// RU: Рисует прикладные данные из регистров текущего запуска render VM на её активной GPU-поверхности.
/// </summary>
/// <remarks>
/// EN: Use the supplied registers and canvas only during this synchronous call.
///
/// RU: Используйте переданные регистры и интерфейс рисования только в рамках этого синхронного вызова.
/// </remarks>
public interface IDirect2DDrawer
{
    void Draw(IVmRegisterReader registers, IDirect2DCanvas canvas);
}
