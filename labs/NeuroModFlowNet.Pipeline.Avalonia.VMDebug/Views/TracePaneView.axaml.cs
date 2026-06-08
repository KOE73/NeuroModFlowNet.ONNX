using Avalonia.Controls;

namespace NeuroModFlowNet.Pipeline.Avalonia.VMDebug.Views;

/// <summary>
/// Shows per-instruction timing from the last VM transaction.
/// </summary>
public partial class TracePaneView : UserControl
{
    public TracePaneView()
    {
        InitializeComponent();
    }
}
