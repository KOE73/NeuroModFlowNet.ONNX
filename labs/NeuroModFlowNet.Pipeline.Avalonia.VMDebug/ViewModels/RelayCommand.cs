using System.Windows.Input;

namespace NeuroModFlowNet.Pipeline.Avalonia.VMDebug.ViewModels;

/// <summary>
/// Minimal command implementation for debugger actions that complete synchronously.
/// </summary>
public sealed class RelayCommand : ICommand
{
    readonly Action execute;
    readonly Func<bool>? canExecute;

    public RelayCommand(Action execute, Func<bool>? canExecute = null)
    {
        this.execute = execute;
        this.canExecute = canExecute;
    }

    public event EventHandler? CanExecuteChanged;

    public bool CanExecute(object? parameter)
    {
        return canExecute?.Invoke() ?? true;
    }

    public void Execute(object? parameter)
    {
        execute();
    }

    public void RaiseCanExecuteChanged()
    {
        CanExecuteChanged?.Invoke(this, EventArgs.Empty);
    }
}
