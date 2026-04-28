using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.ONNX.Demo.Assets;
using NeuroModFlowNet.ONNX.Diagnostics;
using OpenCvSharp;

namespace NeuroModFlowNet.ONNX.Demo.Dashboard;

internal abstract class DashboardFrameProcessorBase<TResult> : IDashboardFrameProcessor
{
    private TResult _result = default!;
    private bool _hasResult;
    private bool _disposed;

    protected DashboardFrameProcessorBase(
        ModelSlot slot,
        string title,
        string modelName,
        DashboardModelSettings settings)
    {
        Slot = slot;
        Title = title;
        ModelName = modelName;
        Settings = settings;
    }

    public ModelSlot Slot { get; }
    public string Title { get; }
    public bool IsEnabled => Runner != null;

    protected string ModelName { get; }
    protected DashboardModelSettings Settings { get; private set; }
    protected OnnxRuntimeContext? Context { get; private set; }
    protected IRunner<Mat, TResult>? Runner { get; private set; }
    protected bool HasResult => _hasResult;
    protected TResult Result => _result;

    public async Task InitializeAsync()
    {
        string modelFileName = ModelNaming.GetFileName(
            ModelName,
            Settings.InputSize,
            1,
            Settings.Precision,
            isByteBgr: Settings.IsByteBgr);

        string modelPath = await AssetsManager.GetAssetPathAsync(modelFileName);
        Context = new OnnxRuntimeContext(modelPath, Settings.Backend);
        Runner = CreateRunner(Context);
        Context.WriteInfo();
    }

    public async Task ReloadAsync(DashboardModelSettings settings)
    {
        DisposeModel();
        Settings = settings;
        _hasResult = false;
        _result = default!;
        await InitializeAsync();
    }

    public void Process(Mat modelInput, in DashboardFrameInfo frameInfo)
    {
        if(Runner == null) return;

        _result = Runner.Predict(modelInput);
        _hasResult = true;
    }

    public void Draw(Mat target, in DashboardFrameInfo frameInfo)
    {
        if(Runner == null || !_hasResult)
        {
            DashboardRenderUtils.DrawNoModel(target, $"{Title} disabled");
            return;
        }

        DrawResult(target, frameInfo, _result);
    }

    protected abstract IRunner<Mat, TResult> CreateRunner(OnnxRuntimeContext context);
    protected abstract void DrawResult(Mat target, in DashboardFrameInfo frameInfo, TResult result);

    protected string GetClassName(int classId) => Context?.GetYoloClassName(classId) ?? $"#{classId}";
    protected static float GetScaleX(Mat target, in DashboardFrameInfo frameInfo) => target.Width / frameInfo.SourceWidth;
    protected static float GetScaleY(Mat target, in DashboardFrameInfo frameInfo) => target.Height / frameInfo.SourceHeight;

    public void Dispose()
    {
        if(_disposed) return;

        DisposeModel();
        _disposed = true;
    }

    private void DisposeModel()
    {
        Runner?.Dispose();
        Runner = null;
        Context?.Dispose();
        Context = null;
    }
}
