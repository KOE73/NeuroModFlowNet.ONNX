using System.Collections.Concurrent;
using System.Diagnostics;

namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// EN: Controller that owns a chain of pipeline programs for one source and manages its bounded concurrent executions.
/// RU: Контроллер, владеющий цепочкой программ пайплайна одного источника и управляющий её параллельным выполнением в заданных границах.
/// </summary>
/// <remarks>
/// EN:
/// Essence: Coordinates VM execution runs, resource registries, synchronization, and runtime concurrency limits.
/// Reasons: Decouples VM execution from specific input mechanisms (cameras, directories). The host determines which items to process and submits them. By skipping files or frames before submitting, the host ensures no RunId is wasted and no synchronization gate will stall waiting for aborted items.
/// Program chain: one accepted run executes every program of <see cref="Programs"/> in order with the same
/// <see cref="VmRunContext"/>, so the chain is semantically one concatenated program. A source-specific preparation
/// program can precede a program shared by all sources as code (each controller keeps its own instances, because
/// instructions may own mutable native state). Labels stay local to each program; <see cref="OpResultKind.Stop"/>
/// ends the whole run; a failure in any program fails the run. The chain never crosses controllers: the register
/// file, RunId, global memory and disposable outputs all belong to this controller.
///
/// RU:
/// Суть: Координирует фоновые запуски VM, реестры общих ресурсов, механизмы синхронизации и лимиты параллельного выполнения.
/// Причины: Отделяет выполнение VM от конкретных механизмов ввода (камер, файлов). Вызывающий хост сам решает, какие элементы обрабатывать, и передает их в контроллер. Отбрасывая кадры или файлы до вызова контроллера, хост гарантирует отсутствие «пустых» RunId, из-за которых могли бы заблокироваться шлюзы синхронизации.
/// Цепочка программ: один принятый запуск выполняет все программы <see cref="Programs"/> по порядку с одним и тем же
/// <see cref="VmRunContext"/>, поэтому цепочка семантически равна одной склеенной программе. Программа подготовки,
/// специфичная для источника, может предшествовать программе, общей для всех источников как код (экземпляры у каждого
/// контроллера свои, потому что инструкции могут владеть изменяемым native-состоянием). Метки локальны для каждой
/// программы; <see cref="OpResultKind.Stop"/> завершает весь запуск; ошибка в любой программе проваливает запуск.
/// Цепочка не пересекает границу контроллеров: регистры, RunId, глобальная память и disposable-выходы принадлежат
/// этому контроллеру.
/// </remarks>
public sealed class VmController : IAsyncDisposable, IDisposable
{
    readonly IReadOnlyList<VmProgram> programs;
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
        : this(options, [program ?? throw new ArgumentNullException(nameof(program))], globalMemory)
    {
    }

    /// <summary>
    /// EN: Creates a controller that executes <paramref name="programs"/> in order for every accepted run.
    ///
    /// RU: Создает контроллер, выполняющий <paramref name="programs"/> по порядку для каждого принятого запуска.
    /// </summary>
    public VmController(
        VmControllerOptions options,
        IReadOnlyList<VmProgram> programs,
        VmGlobalMemory? globalMemory = null)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(programs);
        options.Validate();

        if(programs.Count == 0)
            throw new ArgumentException("At least one program is required.", nameof(programs));

        foreach(VmProgram program in programs)
            ArgumentNullException.ThrowIfNull(program, nameof(programs));

        Options = options;
        this.programs = programs.ToArray();
        GlobalMemory = globalMemory ?? new VmGlobalMemory();
        RunLedger = new VmRunLedger(options.InitialRunId);
        SyncGates = new VmSyncGateRegistry();
        inFlightLimiter = new SemaphoreSlim(options.MaxInFlight, options.MaxInFlight);
    }

    public VmControllerOptions Options { get; }

    /// <summary>
    /// EN: Programs executed in order for one accepted run with one shared run context.
    ///
    /// RU: Программы, выполняемые по порядку для одного принятого запуска с общим контекстом запуска.
    /// </summary>
    public IReadOnlyList<VmProgram> Programs => programs;

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
            await ExecuteProgramChainAsync(context, cancellationToken, warmup: true).ConfigureAwait(false);
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

        // Освобождаем ресурсы всех инструкций всех программ цепочки
        foreach (VmProgram program in programs)
        {
            foreach (IOp instruction in program.Instructions)
            {
                if (instruction is IAsyncDisposable asyncDisposable)
                    await asyncDisposable.DisposeAsync().ConfigureAwait(false);
                else if (instruction is IDisposable disposable)
                    disposable.Dispose();
            }
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
            await ExecuteProgramChainAsync(context, cancellationToken, warmup: false).ConfigureAwait(false);
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

    /// <summary>
    /// EN:
    /// Executes the program chain with one shared context. A <see cref="VmProgramExit.Stopped"/> result ends the run
    /// and skips the remaining programs; a failure propagates as an exception exactly like in a single program.
    ///
    /// RU:
    /// Выполняет цепочку программ с одним общим контекстом. Результат <see cref="VmProgramExit.Stopped"/> завершает
    /// запуск и пропускает оставшиеся программы; ошибка пробрасывается исключением так же, как в одной программе.
    /// </summary>
    async ValueTask ExecuteProgramChainAsync(VmRunContext context, CancellationToken cancellationToken, bool warmup)
    {
        foreach(VmProgram program in programs)
        {
            VmProgramExit exit = warmup
                ? await program.ExecuteWarmupAsync(context, cancellationToken, Options.CaptureVariablesInTrace).ConfigureAwait(false)
                : await program.ExecuteAsync(context, cancellationToken, Options.CaptureVariablesInTrace).ConfigureAwait(false);

            if(exit == VmProgramExit.Stopped)
                return;
        }
    }

    static double GetElapsedMilliseconds(long startTimestamp) =>
        (Stopwatch.GetTimestamp() - startTimestamp) * 1000.0 / Stopwatch.Frequency;

    void ThrowIfDisposed()
    {
        if(disposed)
            throw new ObjectDisposedException(nameof(VmController));
    }
}
