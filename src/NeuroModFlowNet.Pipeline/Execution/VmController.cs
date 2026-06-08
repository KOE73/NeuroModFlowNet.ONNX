using System.Collections.Concurrent;
using System.Diagnostics;

namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// EN: Controller that owns a pipeline program and manages its bounded concurrent executions.
/// RU: Контроллер, владеющий программой пайплайна и управляющий её параллельным выполнением в заданных границах.
/// </summary>
/// <remarks>
/// EN:
/// Essence: Coordinates VM execution runs, resource registries, synchronization, and runtime concurrency limits.
/// Reasons: Decouples VM execution from specific input mechanisms (cameras, directories). The host determines which items to process and submits them. By skipping files or frames before submitting, the host ensures no RunId is wasted and no synchronization gate will stall waiting for aborted items.
/// 
/// RU:
/// Суть: Координирует фоновые запуски VM, реестры общих ресурсов, механизмы синхронизации и лимиты параллельного выполнения.
/// Причины: Отделяет выполнение VM от конкретных механизмов ввода (камер, файлов). Вызывающий хост сам решает, какие элементы обрабатывать, и передает их в контроллер. Отбрасывая кадры или файлы до вызова контроллера, хост гарантирует отсутствие «пустых» RunId, из-за которых могли бы заблокироваться шлюзы синхронизации.
/// </remarks>
public sealed class VmController : IAsyncDisposable, IDisposable
{
    readonly VmProgram program;
    readonly SemaphoreSlim inFlightLimiter;
    readonly ConcurrentDictionary<long, Task<VmRunOutcome>> activeRuns = [];
    readonly CancellationTokenSource stopTokenSource = new();
    readonly object lifecycleSyncRoot = new();
    bool disposed;
    bool isWarmupCompleted;

    public VmController(
        VmControllerOptions options,
        VmProgram program,
        VmGlobalMemory? globalMemory = null)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(program);
        options.Validate();

        Options = options;
        this.program = program;
        GlobalMemory = globalMemory ?? new VmGlobalMemory();
        RunLedger = new VmRunLedger(options.InitialRunId);
        SyncGates = new VmSyncGateRegistry();
        inFlightLimiter = new SemaphoreSlim(options.MaxInFlight, options.MaxInFlight);
    }

    public VmControllerOptions Options { get; }

    public VmGlobalMemory GlobalMemory { get; }

    public VmRunLedger RunLedger { get; }

    public VmSyncGateRegistry SyncGates { get; }

    /// <summary>
    /// RU: Выполняет явный прогревочный прогон программы на тестовых данных с жесткой валидацией устройств.
    /// </summary>
    public async Task<VmRunOutcome> WarmupAsync(
        Action<VmRunContext> initializeContext,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(initializeContext);
        ThrowIfDisposed();

        VmRunIdentity identity = RunLedger.Accept(Options.SourceId + "_warmup", DateTimeOffset.UtcNow, null);
        var context = new VmRunContext(identity, GlobalMemory, SyncGates, Options.DebugGate);

        try
        {
            initializeContext(context);
        }
        catch
        {
            context.Dispose();
            throw;
        }

        try
        {
            RunLedger.MarkRunning(identity.RunId);
            await program.ExecuteWarmupAsync(context, cancellationToken, Options.CaptureVariablesInTrace).ConfigureAwait(false);
            VmRunOutput output = VmRunOutput.Capture(context, Options.OutputKeys);
            RunLedger.MarkCompleted(identity.RunId);
            
            lock (lifecycleSyncRoot)
            {
                isWarmupCompleted = true;
            }

            return new VmRunOutcome(identity, VmRunStatus.Completed, Output: output, Trace: context.Trace);
        }
        catch (Exception exception)
        {
            RunLedger.MarkFailed(identity.RunId, exception);
            return new VmRunOutcome(identity, VmRunStatus.Failed, exception, Trace: context.Trace);
        }
        finally
        {
            await context.DisposeAsync().ConfigureAwait(false);
        }
    }

    /// <summary>
    /// RU: Выполняет явный прогревочный прогон программы на тестовых данных с жесткой валидацией устройств.
    /// </summary>
    public Task<VmRunOutcome> WarmupAsync(
        VmRunInputs inputs,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        return WarmupAsync(inputs.ApplyTo, cancellationToken);
    }

    public bool TryStartRun(
        VmRunInputs inputs,
        out VmRunHandle handle,
        long? sourceFrameId = null)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        return TryStartRun(inputs.ApplyTo, out handle, sourceFrameId);
    }

    public bool TryStartRun(
        Action<VmRunContext> initializeContext,
        out VmRunHandle handle,
        long? sourceFrameId = null)
    {
        ArgumentNullException.ThrowIfNull(initializeContext);
        ThrowIfDisposed();

        lock (lifecycleSyncRoot)
        {
            if (Options.RequireWarmup && !isWarmupCompleted)
            {
                throw new InvalidOperationException("Пайплайн не был прогрет. Вызовите метод WarmupAsync перед запуском работы в горячем цикле.");
            }
        }

        if(!inFlightLimiter.Wait(0))
        {
            handle = null!;
            return false;
        }

        VmRunIdentity identity = RunLedger.Accept(Options.SourceId, DateTimeOffset.UtcNow, sourceFrameId);
        var context = new VmRunContext(identity, GlobalMemory, SyncGates, Options.DebugGate);

        try
        {
            initializeContext(context);
        }
        catch
        {
            inFlightLimiter.Release();
            context.Dispose();
            throw;
        }

        Task<VmRunOutcome> completion = Task.Run(() => ExecuteRunAsync(context, stopTokenSource.Token));
        activeRuns[identity.RunId] = completion;
        handle = new VmRunHandle(identity, completion);
        return true;
    }

    public async ValueTask StopAsync()
    {
        Task<VmRunOutcome>[] tasks;

        lock(lifecycleSyncRoot)
        {
            if(disposed)
                return;

            stopTokenSource.Cancel();
            tasks = activeRuns.Values.ToArray();
        }

        if(tasks.Length == 0)
            return;

        try
        {
            await Task.WhenAll(tasks).ConfigureAwait(false);
        }
        catch
        {
            // Individual outcomes already carry their failures; StopAsync should only finish teardown.
        }
    }

    public void Dispose() => DisposeAsync().AsTask().GetAwaiter().GetResult();

    public async ValueTask DisposeAsync()
    {
        if(disposed)
            return;

        await StopAsync().ConfigureAwait(false);

        // Освобождаем ресурсы всех инструкций программы
        foreach (IOp instruction in program.Instructions)
        {
            if (instruction is IAsyncDisposable asyncDisposable)
                await asyncDisposable.DisposeAsync().ConfigureAwait(false);
            else if (instruction is IDisposable disposable)
                disposable.Dispose();
        }

        lock(lifecycleSyncRoot)
        {
            if(disposed)
                return;

            disposed = true;
            stopTokenSource.Dispose();
            inFlightLimiter.Dispose();
        }
    }

    async Task<VmRunOutcome> ExecuteRunAsync(VmRunContext context, CancellationToken cancellationToken)
    {
        VmRunIdentity identity = context.Identity;
        VmRunOutcome? outcome = null;
        long totalStartTimestamp = Stopwatch.GetTimestamp();
        double programMilliseconds = 0;
        double outputCaptureMilliseconds = 0;
        double contextDisposeMilliseconds = 0;
        double cleanupMilliseconds = 0;

        try
        {
            RunLedger.MarkRunning(identity.RunId);

            long programStartTimestamp = Stopwatch.GetTimestamp();
            await program.ExecuteAsync(context, cancellationToken, Options.CaptureVariablesInTrace).ConfigureAwait(false);
            programMilliseconds = GetElapsedMilliseconds(programStartTimestamp);

            long outputCaptureStartTimestamp = Stopwatch.GetTimestamp();
            VmRunOutput output = VmRunOutput.Capture(context, Options.OutputKeys);
            outputCaptureMilliseconds = GetElapsedMilliseconds(outputCaptureStartTimestamp);

            RunLedger.MarkCompleted(identity.RunId);
            outcome = new VmRunOutcome(identity, VmRunStatus.Completed, Output: output, Trace: context.Trace);
        }
        catch(OperationCanceledException exception) when(cancellationToken.IsCancellationRequested)
        {
            RunLedger.MarkCanceled(identity.RunId, exception);
            outcome = new VmRunOutcome(identity, VmRunStatus.Canceled, exception, Trace: context.Trace);
        }
        catch(Exception exception)
        {
            RunLedger.MarkFailed(identity.RunId, exception);
            outcome = new VmRunOutcome(identity, VmRunStatus.Failed, exception, Trace: context.Trace);
        }
        finally
        {
            long contextDisposeStartTimestamp = Stopwatch.GetTimestamp();
            SyncGates.NotifyRunClosed(identity.RunId);
            await context.DisposeAsync().ConfigureAwait(false);
            contextDisposeMilliseconds = GetElapsedMilliseconds(contextDisposeStartTimestamp);

            long cleanupStartTimestamp = Stopwatch.GetTimestamp();
            activeRuns.TryRemove(identity.RunId, out _);
            inFlightLimiter.Release();
            cleanupMilliseconds = GetElapsedMilliseconds(cleanupStartTimestamp);
        }

        VmRunTiming timing = new(
            TotalMilliseconds: GetElapsedMilliseconds(totalStartTimestamp),
            ProgramMilliseconds: programMilliseconds,
            OutputCaptureMilliseconds: outputCaptureMilliseconds,
            ContextDisposeMilliseconds: contextDisposeMilliseconds,
            CleanupMilliseconds: cleanupMilliseconds);

        return outcome is null
            ? new VmRunOutcome(identity, VmRunStatus.Failed, Timing: timing)
            : outcome with { Timing = timing };
    }

    static double GetElapsedMilliseconds(long startTimestamp) =>
        (Stopwatch.GetTimestamp() - startTimestamp) * 1000.0 / Stopwatch.Frequency;

    void ThrowIfDisposed()
    {
        if(disposed)
            throw new ObjectDisposedException(nameof(VmController));
    }
}
