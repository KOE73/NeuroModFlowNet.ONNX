using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;

namespace NeuroModFlowNet.ONNX.Avalonia;

/// <summary>
/// EN: Represents the Avalonia application object for the realtime lab UI.
/// RU: Представляет объект Avalonia-приложения для realtime lab UI.
/// </summary>
/// <remarks>
/// EN: Loads application XAML resources and creates the main desktop window.
/// RU: Загружает XAML-ресурсы приложения и создает главное desktop-окно.
/// </remarks>
public partial class App : Application
{
    public override void Initialize()
    {
        AvaloniaXamlLoader.Load(this);
    }

    public override void OnFrameworkInitializationCompleted()
    {
        if(ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
            desktop.MainWindow = new MainWindow();

        base.OnFrameworkInitializationCompleted();
    }
}
