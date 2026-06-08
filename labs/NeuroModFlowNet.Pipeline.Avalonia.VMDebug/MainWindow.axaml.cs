using Avalonia;
using Avalonia.Collections;
using Avalonia.Controls;
using Dock.Avalonia.Controls;
using Dock.Model.Avalonia.Controls;
using Dock.Model.Core;
using NeuroModFlowNet.Pipeline.Avalonia.VMDebug.ViewModels;
using NeuroModFlowNet.Pipeline.Avalonia.VMDebug.Views;

namespace NeuroModFlowNet.Pipeline.Avalonia.VMDebug;

/// <summary>
/// Hosts the dock workspace for the VM debugger lab.
/// </summary>
/// <remarks>
/// The window owns only the docking surface. VM state, commands, trace data, and runtime lifetime live in
/// <see cref="VmDebugWorkspaceViewModel"/> so panes can be moved, recreated, and later extracted without coupling the
/// debugger logic to this top-level shell.
/// </remarks>
public partial class MainWindow : global::Avalonia.Controls.Window
{
    readonly VmDebugWorkspaceViewModel viewModel = new();
    bool isInitialized;

    public MainWindow()
    {
        InitializeComponent();
        DataContext = viewModel;
        Opened += async (_, _) => await InitializeRuntimeAsync();
    }

    async Task InitializeRuntimeAsync()
    {
        if(isInitialized)
            return;

        isInitialized = true;
        CreateDockWorkspace();
        await viewModel.InitializeRuntimeAsync();
    }

    protected override void OnClosed(EventArgs e)
    {
        viewModel.Dispose();
        base.OnClosed(e);
    }

    void CreateDockWorkspace()
    {
        DockControl dockControlWorkspace = this.FindControl<DockControl>("DockControl_Workspace")
            ?? throw new InvalidOperationException("Dock workspace was not created.");

        Tool commandBarTool = CreateTool("CommandBarTool", "Commands", () => new CommandBarPaneView());
        Tool programTool = CreateTool("VmProgramTool", "VM Program", () => new VmProgramPaneView());
        Tool compiledStepsTool = CreateTool("CompiledStepsTool", "Compiled Steps", () => new CompiledStepsPaneView());
        Tool diagnosticsTool = CreateTool("DiagnosticsTool", "Diagnostics", () => new DiagnosticsPaneView());
        Tool watchTool = CreateTool("WatchTool", "Watch Variables", () => new WatchPaneView());
        Tool traceTool = CreateTool("TraceTool", "Instruction Trace", () => new TracePaneView());
        Document previewDocument = CreateDocument("PreviewDocument", "Frame Preview", () => new PreviewPaneView());

        ToolDock commandBarDock = CreateToolDock("CommandBarDock", "Commands Dock", Dock.Model.Core.Alignment.Top, 0.085, commandBarTool);
        commandBarDock.MinHeight = 76;
        commandBarDock.MaxHeight = 76;
        ToolDock programDock = CreateToolDock("VmProgramDock", "Program Dock", Dock.Model.Core.Alignment.Left, 0.3, programTool);
        ToolDock compiledStepsDock = CreateToolDock("CompiledStepsDock", "Compiled Steps Dock", Dock.Model.Core.Alignment.Left, 0.29, compiledStepsTool);
        ToolDock diagnosticsDock = CreateToolDock("DiagnosticsDock", "Diagnostics Dock", Dock.Model.Core.Alignment.Bottom, 0.16, diagnosticsTool);
        ToolDock watchDock = CreateToolDock("WatchDock", "Watch Dock", Dock.Model.Core.Alignment.Right, 0.36, watchTool);
        ToolDock traceDock = CreateToolDock("TraceDock", "Trace Dock", Dock.Model.Core.Alignment.Right, 0.41, traceTool);

        DocumentDock previewDock = new()
        {
            Id = "PreviewDock",
            Title = "Preview Dock",
            Proportion = 0.64,
            EnableWindowDrag = true,
            VisibleDockables = Dockables(previewDocument),
            ActiveDockable = previewDocument,
            DefaultDockable = previewDocument
        };

        ProportionalDock leftTopDock = new()
        {
            Id = "LeftTopDock",
            Title = "VM Debug Editors",
            Orientation = Dock.Model.Core.Orientation.Horizontal,
            Proportion = 0.84,
            VisibleDockables = Dockables(programDock, new ProportionalDockSplitter(), compiledStepsDock, new ProportionalDockSplitter(), traceDock)
        };

        ProportionalDock leftColumnDock = new()
        {
            Id = "LeftColumnDock",
            Title = "VM Debug Left Workspace",
            Orientation = Dock.Model.Core.Orientation.Vertical,
            Proportion = 0.49,
            MinWidth = 640,
            VisibleDockables = Dockables(leftTopDock, new ProportionalDockSplitter(), diagnosticsDock)
        };

        ProportionalDock rightColumnDock = new()
        {
            Id = "RightColumnDock",
            Title = "VM Debug Preview",
            Orientation = Dock.Model.Core.Orientation.Vertical,
            Proportion = 0.51,
            MinWidth = 640,
            VisibleDockables = Dockables(watchDock, new ProportionalDockSplitter(), previewDock)
        };

        ProportionalDock workspaceDock = new()
        {
            Id = "WorkspaceDock",
            Title = "VM Debug Workspace",
            Orientation = Dock.Model.Core.Orientation.Horizontal,
            Proportion = 0.955,
            VisibleDockables = Dockables(leftColumnDock, new ProportionalDockSplitter(), rightColumnDock)
        };

        ProportionalDock mainDock = new()
        {
            Id = "MainDock",
            Title = "VM Debug Main Dock",
            Orientation = Dock.Model.Core.Orientation.Vertical,
            VisibleDockables = Dockables(commandBarDock, new ProportionalDockSplitter(), workspaceDock)
        };

        RootDock rootDock = new()
        {
            Id = "Root",
            Title = "VM Debug",
            IsFocusableRoot = true,
            VisibleDockables = Dockables(mainDock),
            ActiveDockable = mainDock,
            DefaultDockable = mainDock
        };

        dockControlWorkspace.Layout = rootDock;
    }

    Tool CreateTool(string id, string title, Func<Control> content)
    {
        return new Tool
        {
            Id = id,
            Title = title,
            Content = new Func<IServiceProvider, object>(_ => CreatePane(content))
        };
    }

    Document CreateDocument(string id, string title, Func<Control> content)
    {
        return new Document
        {
            Id = id,
            Title = title,
            CanClose = false,
            Content = new Func<IServiceProvider, object>(_ => CreatePane(content))
        };
    }

    Control CreatePane(Func<Control> content)
    {
        Control control = content();
        control.DataContext = viewModel;
        return control;
    }

    static ToolDock CreateToolDock(string id, string title, Dock.Model.Core.Alignment alignment, double proportion, Tool tool)
    {
        return new ToolDock
        {
            Id = id,
            Title = title,
            Alignment = alignment,
            Proportion = proportion,
            VisibleDockables = Dockables(tool),
            ActiveDockable = tool,
            DefaultDockable = tool
        };
    }

    static AvaloniaList<IDockable> Dockables(params IDockable[] dockables)
    {
        AvaloniaList<IDockable> result = [];
        foreach(IDockable dockable in dockables)
        {
            result.Add(dockable);
        }

        return result;
    }
}
