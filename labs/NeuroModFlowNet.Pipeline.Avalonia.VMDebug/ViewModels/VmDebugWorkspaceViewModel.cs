using System.Collections.ObjectModel;
using System.Windows.Input;
using Avalonia.Media;
using Avalonia.Threading;
using NeuroModFlowNet.Pipeline.Avalonia.VMDebug.Runtime;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.Avalonia.VMDebug.ViewModels;

/// <summary>
/// Owns VM-debug runtime state, commands, trace data, and current watch values.
/// </summary>
/// <remarks>
/// UI panes are disposable views over this object. This matters for Dock: a pane can be dragged, recreated, or floated
/// without losing the running VM transaction state or the latest frame snapshot.
/// </remarks>
public sealed class VmDebugWorkspaceViewModel : ViewModelBase, IDisposable
{
    RecognitionOptions? recognitionOptions;
    PipelineRealtimeEngine? engine;
    VmRunTrace? selectedTrace;
    Mat? latestFrame;
    FrameOverlaySnapshot? latestOverlay;
    int selectedTraceIndex;
    string lastCompileDiagnostics = string.Empty;
    string vmProgramText = VmDebugProgramTextLoader.LoadDefaultProgram();
    string diagnosticsText = string.Empty;
    string runtimeStatus = "Ready";
    IBrush runtimeStatusForeground = new SolidColorBrush(Color.FromRgb(84, 96, 112));
    IBrush runtimeStatusBackground = new SolidColorBrush(Color.FromRgb(232, 237, 242));
    IBrush runtimeStatusBorder = new SolidColorBrush(Color.FromRgb(184, 194, 204));
    bool isInitialized;
    bool isDisposed;

    public VmDebugWorkspaceViewModel()
    {
        StartCommand = new AsyncRelayCommand(StartEngineAsync);
        StopCommand = new AsyncRelayCommand(StopEngineAsync);
        PauseCommand = new RelayCommand(() => engine?.Pause());
        RunCommand = new RelayCommand(() => engine?.Play());
        StepCommand = new RelayCommand(() => engine?.StepFrame());
        CompileCommand = new RelayCommand(CompileVmProgram);
        PreviousTraceCommand = new RelayCommand(() => MoveTraceSelection(-1));
        NextTraceCommand = new RelayCommand(() => MoveTraceSelection(1));
        DebugFrameCommand = new RelayCommand(() => engine?.StepFrameWithInstructionDebug());
        DebugNextInstructionCommand = new RelayCommand(() => engine?.TryResumeOneInstruction());

        SetStatus(runtimeStatus);
    }

    public event Action<Mat, FrameOverlaySnapshot>? FrameUpdated;

    public ObservableCollection<string> CompiledStepItems { get; } = [];
    public ObservableCollection<string> TraceVariableItems { get; } = [];
    public ObservableCollection<string> TraceStepItems { get; } = [];

    public ICommand StartCommand { get; }
    public ICommand StopCommand { get; }
    public ICommand PauseCommand { get; }
    public ICommand RunCommand { get; }
    public ICommand StepCommand { get; }
    public ICommand CompileCommand { get; }
    public ICommand PreviousTraceCommand { get; }
    public ICommand NextTraceCommand { get; }
    public ICommand DebugFrameCommand { get; }
    public ICommand DebugNextInstructionCommand { get; }

    public string VmProgramText
    {
        get => vmProgramText;
        set => SetProperty(ref vmProgramText, value);
    }

    public string DiagnosticsText
    {
        get => diagnosticsText;
        private set => SetProperty(ref diagnosticsText, value);
    }

    public string RuntimeStatus
    {
        get => runtimeStatus;
        private set => SetProperty(ref runtimeStatus, value);
    }

    public IBrush RuntimeStatusForeground
    {
        get => runtimeStatusForeground;
        private set => SetProperty(ref runtimeStatusForeground, value);
    }

    public IBrush RuntimeStatusBackground
    {
        get => runtimeStatusBackground;
        private set => SetProperty(ref runtimeStatusBackground, value);
    }

    public IBrush RuntimeStatusBorder
    {
        get => runtimeStatusBorder;
        private set => SetProperty(ref runtimeStatusBorder, value);
    }

    public async Task InitializeRuntimeAsync()
    {
        ThrowIfDisposed();
        if(isInitialized)
            return;

        isInitialized = true;
        AvaloniaJsonConfig jsonConfig = AvaloniaJsonConfig.Load();
        recognitionOptions = new RecognitionOptions(jsonConfig.Postprocessing.RecognitionRoi);
        InitialInferenceSelection.ApplyFromConfig(recognitionOptions.InferenceSelection);
        engine = new PipelineRealtimeEngine(RealTimeAvaloniaSettings.FromConfig(), recognitionOptions, jsonConfig);
        engine.FrameReady += OnFrameReady;
        engine.StatusChanged += OnStatusChanged;
        engine.InstructionStopped += OnInstructionStopped;

        CompileVmProgram();
        await StartEngineAsync();
    }

    public bool TryGetLatestFrame(out Mat? frame, out FrameOverlaySnapshot? overlay)
    {
        frame = latestFrame;
        overlay = latestOverlay;
        return frame is not null && overlay is not null;
    }

    public void Dispose()
    {
        if(isDisposed)
            return;

        isDisposed = true;
        if(engine is not null)
        {
            engine.FrameReady -= OnFrameReady;
            engine.StatusChanged -= OnStatusChanged;
            engine.InstructionStopped -= OnInstructionStopped;
        }

        latestFrame?.Dispose();
        engine?.Dispose();
        recognitionOptions?.Dispose();
    }

    async Task StartEngineAsync()
    {
        if(engine is null)
            return;

        SetStatus("Starting...");
        await engine.StartAsync();
    }

    async Task StopEngineAsync()
    {
        if(engine is not null)
            await engine.StopAsync();
    }

    void OnFrameReady(object? sender, PipelineFrameResultData update)
    {
        Dispatcher.UIThread.Post(() =>
        {
            using(update)
            {
                latestFrame?.Dispose();
                latestFrame = update.Frame.Clone();
                latestOverlay = update.Overlay;

                FrameUpdated?.Invoke(update.Frame, update.Overlay);
                UpdateTrace(update.Trace);
            }
        });
    }

    void OnStatusChanged(object? sender, string status)
    {
        Dispatcher.UIThread.Post(() => SetStatus(status));
    }

    void OnInstructionStopped(object? sender, OpDebugPoint point)
    {
        Dispatcher.UIThread.Post(() => ShowInstructionStop(point));
    }

    void SetStatus(string status)
    {
        RuntimeStatus = status;
        (Color foreground, Color background, Color border) = ResolveStatusColors(status);
        RuntimeStatusForeground = new SolidColorBrush(foreground);
        RuntimeStatusBackground = new SolidColorBrush(background);
        RuntimeStatusBorder = new SolidColorBrush(border);
    }

    void CompileVmProgram()
    {
        VmDebugProgramCompileResult result = VmDebugProgramCompiler.Compile(
            VmProgramText,
            engine?.KnownOperationNames ?? new HashSet<string>(StringComparer.Ordinal));

        CompiledStepItems.Clear();
        foreach(VmDebugCompiledStep step in result.Steps)
        {
            string marker = step.IsKnown ? "OK" : "ERR";
            string id = string.IsNullOrWhiteSpace(step.Id) ? string.Empty : $" [{step.Id}]";
            CompiledStepItems.Add($"{step.Index:00} {marker} {step.Operation}{id}");
        }

        lastCompileDiagnostics = result.Success
            ? $"Compiled {result.Steps.Count} visible VM steps. Runtime executes this operation list through strongly typed instruction factories."
            : string.Join(Environment.NewLine, result.Diagnostics);
        DiagnosticsText = lastCompileDiagnostics;

        if(result.Success)
            engine?.ConfigureProgram(result.Steps);
    }

    void UpdateTrace(VmRunTrace? trace)
    {
        if(trace is null)
            return;

        selectedTrace = trace;
        selectedTraceIndex = Math.Clamp(selectedTraceIndex, 0, Math.Max(0, trace.Instructions.Count - 1));
        RefreshTracePanels();
    }

    void MoveTraceSelection(int delta)
    {
        if(selectedTrace is null || selectedTrace.Instructions.Count == 0)
            return;

        selectedTraceIndex = Math.Clamp(selectedTraceIndex + delta, 0, selectedTrace.Instructions.Count - 1);
        RefreshTracePanels();
    }

    void ShowInstructionStop(OpDebugPoint point)
    {
        TraceStepItems.Clear();
        TraceVariableItems.Clear();
        TraceStepItems.Add($"> {point.InstructionIndex:00} {point.Operation,-28} waiting before ExecuteAsync");

        foreach(VarDebugInfo variable in point.VariablesBefore)
        {
            string shape = string.IsNullOrWhiteSpace(variable.Shape) ? string.Empty : $" {variable.Shape}";
            string owned = variable.IsDisposable || variable.IsAsyncDisposable ? " disposable" : string.Empty;
            TraceVariableItems.Add($"{variable.MemoryLocation,-22} {variable.Key,-28} {ShortTypeName(variable.TypeName)}{shape}{owned}");
        }

        int cpuVariables = point.VariablesBefore.Count(variable => variable.MemoryLocation == VarMemoryLocation.Cpu);
        int externalVariables = point.VariablesBefore.Count(variable =>
            variable.MemoryLocation is VarMemoryLocation.Gpu or VarMemoryLocation.ExternalDevice);
        DiagnosticsText =
            $"{lastCompileDiagnostics}{Environment.NewLine}{Environment.NewLine}" +
            $"Stopped before instruction {point.InstructionIndex:00}: {point.Operation}.{Environment.NewLine}" +
            $"Press Debug Next to execute this instruction and stop before the following instruction.{Environment.NewLine}" +
            $"Variables before instruction: CPU={cpuVariables}, device/unknown={externalVariables}.";
    }

    void RefreshTracePanels()
    {
        TraceStepItems.Clear();
        TraceVariableItems.Clear();

        if(selectedTrace is null)
            return;

        for(int index = 0; index < selectedTrace.Instructions.Count; index++)
        {
            OpTraceEntry entry = selectedTrace.Instructions[index];
            string marker = index == selectedTraceIndex ? ">" : " ";
            TraceStepItems.Add($"{marker} {entry.Index:00} {entry.Operation,-28} {entry.ElapsedMilliseconds,8:F3} ms {entry.ResultKind}");
        }

        if(selectedTrace.Instructions.Count == 0)
            return;

        OpTraceEntry selectedEntry = selectedTrace.Instructions[selectedTraceIndex];
        foreach(VarDebugInfo variable in selectedEntry.VariablesAfter)
        {
            string shape = string.IsNullOrWhiteSpace(variable.Shape) ? string.Empty : $" {variable.Shape}";
            string owned = variable.IsDisposable || variable.IsAsyncDisposable ? " disposable" : string.Empty;
            TraceVariableItems.Add($"{variable.MemoryLocation,-22} {variable.Key,-28} {ShortTypeName(variable.TypeName)}{shape}{owned}");
        }

        int cpuVariables = selectedEntry.VariablesAfter.Count(variable => variable.MemoryLocation == VarMemoryLocation.Cpu);
        int externalVariables = selectedEntry.VariablesAfter.Count(variable =>
            variable.MemoryLocation is VarMemoryLocation.Gpu or VarMemoryLocation.ExternalDevice);
        DiagnosticsText =
            $"{lastCompileDiagnostics}{Environment.NewLine}{Environment.NewLine}" +
            $"Trace: {selectedTrace.Instructions.Count} steps, VM instruction time {selectedTrace.TotalInstructionMilliseconds:F3} ms.{Environment.NewLine}" +
            $"Selected step: {selectedEntry.Operation}, {selectedEntry.ElapsedMilliseconds:F3} ms.{Environment.NewLine}" +
            $"Variables after selected step: CPU={cpuVariables}, device/unknown={externalVariables}. CPU Mats here are visible candidates for future GPU-resident replacements.";
    }

    void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(isDisposed, this);
    }

    static string ShortTypeName(string typeName)
    {
        int lastDotIndex = typeName.LastIndexOf('.');
        return lastDotIndex < 0 ? typeName : typeName[(lastDotIndex + 1)..];
    }

    static (Color Foreground, Color Background, Color Border) ResolveStatusColors(string status)
    {
        if(status.Contains("Running", StringComparison.OrdinalIgnoreCase))
            return (Color.FromRgb(38, 186, 92), Color.FromRgb(20, 48, 32), Color.FromRgb(46, 120, 72));

        if(status.Contains("Paused", StringComparison.OrdinalIgnoreCase))
            return (Color.FromRgb(84, 158, 255), Color.FromRgb(22, 38, 62), Color.FromRgb(52, 94, 150));

        if(status.Contains("Starting", StringComparison.OrdinalIgnoreCase) ||
           status.Contains("Initializing", StringComparison.OrdinalIgnoreCase))
            return (Color.FromRgb(255, 190, 62), Color.FromRgb(58, 45, 18), Color.FromRgb(130, 95, 26));

        if(status.Contains("failed", StringComparison.OrdinalIgnoreCase) ||
           status.Contains("Camera only", StringComparison.OrdinalIgnoreCase))
            return (Color.FromRgb(255, 92, 92), Color.FromRgb(62, 24, 24), Color.FromRgb(145, 45, 45));

        return (Color.FromRgb(84, 96, 112), Color.FromRgb(232, 237, 242), Color.FromRgb(184, 194, 204));
    }
}
