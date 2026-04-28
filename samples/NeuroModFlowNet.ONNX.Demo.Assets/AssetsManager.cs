using Spectre.Console;

namespace NeuroModFlowNet.ONNX.Demo.Assets;

public static class AssetsManager
{
    private static readonly HttpClient client = new HttpClient();

    public const string RepositoryId = "NeuroModFlowNet/NeuroModFlowNet-ONNX-Demo-Models";
    public const string PaddleOcrRepositoryId = "monkt/paddleocr-onnx";
    public const string BaseAssetsUrl = $"https://huggingface.co/{RepositoryId}/resolve/main/";
    public const string BasePaddleOcrAssetsUrl = $"https://huggingface.co/{PaddleOcrRepositoryId}/resolve/main/";
    public const string ModelsRootEnvironmentVariable = "NEUROMODFLOWNET_ONNX_MODELS";

    public static async Task<string> GetAssetPathAsync(
        string modelFileName,
        string? baseUrl = null,
        bool forceDownload = false,
        string? targetModelsRoot = null)
    {
        // Убираем ведущие слэши, чтобы Path.Combine не воспринял это как абсолютный путь от корня диска
        modelFileName = modelFileName.TrimStart('/', '\\');

        string localModelFileName = modelFileName.Replace('/', Path.DirectorySeparatorChar);

        string modelsRoot = ResolveModelsRoot(targetModelsRoot);
        string filePath = Path.Combine(modelsRoot, localModelFileName);

        if(!forceDownload && File.Exists(filePath)) return filePath;

        // Убеждаемся, что все подпапки для файла существуют
        string? fileDir = Path.GetDirectoryName(filePath);
        if(!string.IsNullOrEmpty(fileDir))
        {
            Directory.CreateDirectory(fileDir);
        }

        string url = GetAssetUrl(modelFileName, baseUrl);

        try
        {
            await AnsiConsole.Status()
                .StartAsync($"Downloading [cyan]{modelFileName}[/]...", async ctx =>
                {
                    var response = await client.GetAsync(url);
                    response.EnsureSuccessStatusCode();
                    using var fs = new FileStream(filePath, FileMode.Create);
                    await response.Content.CopyToAsync(fs);
                });
        }
        catch(Exception ex)
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] Failed to download asset [yellow]{modelFileName}[/].");
            AnsiConsole.MarkupLine($"[grey]URL:[/] {url}");
            AnsiConsole.MarkupLine($"[grey]Details:[/] {ex.Message}");

            // Удаляем битый файл, если он успел создаться
            if(File.Exists(filePath)) File.Delete(filePath);

            throw; // Пробрасываем выше для завершения работы программы
        }

        return filePath;
    }

    public static string ResolveModelsRoot(string? targetModelsRoot = null, string? startDirectory = null)
    {
        if(!string.IsNullOrWhiteSpace(targetModelsRoot))
            return Path.GetFullPath(targetModelsRoot);

        string? environmentModelsRoot = Environment.GetEnvironmentVariable(ModelsRootEnvironmentVariable);
        if(!string.IsNullOrWhiteSpace(environmentModelsRoot))
            return Path.GetFullPath(environmentModelsRoot);

        string? repositoryRoot = FindRepositoryRoot(startDirectory ?? AppContext.BaseDirectory);
        if(repositoryRoot == null && startDirectory == null)
            repositoryRoot = FindRepositoryRoot(Environment.CurrentDirectory);

        if(repositoryRoot != null)
            return Path.Combine(repositoryRoot, "models");

        return Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "NeuroModFlowNet.ONNX",
            "models");
    }

    public static string GetAssetUrl(string modelFileName, string? baseUrl = null)
    {
        string normalizedModelFileName = modelFileName.TrimStart('/', '\\').Replace('\\', '/');

        if(baseUrl != null)
            return baseUrl + normalizedModelFileName;

        if(IsOriginalPaddleOcrModel(normalizedModelFileName))
        {
            const string paddleOcrPrefix = "paddleocr/";
            string paddleOcrModelFileName = normalizedModelFileName[paddleOcrPrefix.Length..];
            return BasePaddleOcrAssetsUrl + paddleOcrModelFileName;
        }

        return BaseAssetsUrl + normalizedModelFileName;
    }

    private static bool IsOriginalPaddleOcrModel(string modelFileName)
    {
        return modelFileName.StartsWith("paddleocr/", StringComparison.OrdinalIgnoreCase) &&
               !Path.GetFileNameWithoutExtension(modelFileName).EndsWith("_bytebgr", StringComparison.OrdinalIgnoreCase);
    }

    private static string? FindRepositoryRoot(string startDirectory)
    {
        string? currentFolder = Path.GetFullPath(startDirectory);
        if(File.Exists(currentFolder))
            currentFolder = Path.GetDirectoryName(currentFolder);

        while(!string.IsNullOrEmpty(currentFolder))
        {
            if(File.Exists(Path.Combine(currentFolder, "NeuroModFlowNet.ONNX.slnx")) ||
               Directory.Exists(Path.Combine(currentFolder, ".git")) ||
               File.Exists(Path.Combine(currentFolder, "Directory.Build.props")))
                return currentFolder;

            DirectoryInfo? parent = Directory.GetParent(currentFolder);
            if(parent == null) break;
            currentFolder = parent.FullName;
        }

        return null;
    }
}
