using System.Windows.Input;

namespace NeuroModFlowNet.Pipeline.Avalonia.VMDebug.ViewModels;

/// <summary>
/// Minimal command implementation for debugger actions that call asynchronous runtime APIs.
/// </summary>
public sealed class AsyncRelayCommand : ICommand
{
    readonly Func<Task> execute;
    readonly Func<bool>? canExecute;
    bool isExecuting;

    public AsyncRelayCommand(Func<Task> execute, Func<bool>? canExecute = null)
    {
        this.execute = execute;
        this.canExecute = canExecute;
    }

    public event EventHandler? CanExecuteChanged;

    public bool CanExecute(object? parameter)
    {
        return !isExecuting && (canExecute?.Invoke() ?? true);
    }

    public async void Execute(object? parameter)
    {
        if(!CanExecute(parameter))
            return;

        isExecuting = true;
        RaiseCanExecuteChanged();
        try
        {
            await execute();
        }
        finally
        {
            isExecuting = false;
            RaiseCanExecuteChanged();
        }
    }

    public void RaiseCanExecuteChanged()
    {
        CanExecuteChanged?.Invoke(this, EventArgs.Empty);
    }
}
