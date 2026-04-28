using Spectre.Console;

namespace NeuroModFlowNet.ONNX.Demo.Assets;

public static class AssetsManager
{
    private static readonly HttpClient client = new HttpClient();

    public const string RepositoryId = "NeuroModFlowNet/NeuroModFlowNet-ONNX-Demo-Models";
    public const string PaddleOcrRepositoryId = "monkt/paddleocr-onnx";
    public const string BaseAssetsUrl = $"https://huggingface.co/{RepositoryId}/resolve/main/";
    public const string BasePaddleOcrAssetsUrl = $"https://huggingface.co/{PaddleOcrRepositoryId}/resolve/main/";

    public static async Task<string> GetAssetPathAsync(
        string modelFileName,
        string? baseUrl = null,
        bool forceDownload = false,
        string? targetModelsRoot = null)
    {
        // Убираем ведущие слэши, чтобы Path.Combine не воспринял это как абсолютный путь от корня диска
        modelFileName = modelFileName.TrimStart('/', '\\');

        string localModelFileName = modelFileName.Replace('/', Path.DirectorySeparatorChar);

        string filePath = targetModelsRoot == null
            ? FindFileInSharedLocations("models", localModelFileName)
            : Path.Combine(targetModelsRoot, localModelFileName);

        if(!forceDownload && File.Exists(filePath)) return filePath;

        // Если файла нет, получаем путь, куда его стоит скачать (приоритет общей папке)
        string targetDir = targetModelsRoot ?? FindSharedFolder("models");
        filePath = Path.Combine(targetDir, localModelFileName);

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

    /// <summary>
    /// Ищет путь к файлу, проверяя корневую общую папку models выше по дереву.
    /// Если файл не найден, возвращает путь, где он *должен* быть в общей папке.
    /// </summary>
    private static string FindFileInSharedLocations(string modelsFolderName, string modelFileName)
    {
        string currentFolder = AppDomain.CurrentDomain.BaseDirectory;
        string? bestFolderSoFar = null;

        while(!string.IsNullOrEmpty(currentFolder))
        {
            // Проверяем {current}/{modelsFolderName}/{fileName} -- каноническая общая папка
            string rootModelsPath = Path.Combine(currentFolder, modelsFolderName);
            if(Directory.Exists(rootModelsPath))
            {
                bool isLocal = currentFolder.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}") ||
                               currentFolder.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}");

                if(!isLocal) bestFolderSoFar ??= rootModelsPath;

                string testModelPath = Path.Combine(rootModelsPath, modelFileName);
                if(File.Exists(testModelPath))
                    return testModelPath;
            }

            DirectoryInfo? parent = Directory.GetParent(currentFolder);
            if(parent == null) break;
            currentFolder = parent.FullName;
        }

        // Если файл не нашли, возвращаем путь в лучшей найденной "общей" папке или рядом с exe
        string finalDir = bestFolderSoFar ?? Path.Combine(AppDomain.CurrentDomain.BaseDirectory, modelsFolderName);
        return Path.Combine(finalDir, modelFileName);
    }

    private static string FindSharedFolder(string folderName)
    {
        // Используем логику поиска, но без привязки к конкретному файлу
        string currentFolder = AppDomain.CurrentDomain.BaseDirectory;

        while(!string.IsNullOrEmpty(currentFolder))
        {
            string candidate = Path.Combine(currentFolder, folderName);
            if(Directory.Exists(candidate))
            {
                bool isLocal = currentFolder.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}") ||
                               currentFolder.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}");
                if(!isLocal) return candidate;
            }

            DirectoryInfo? parent = Directory.GetParent(currentFolder);
            if(parent == null) break;
            currentFolder = parent.FullName;
        }

        return Path.Combine(AppDomain.CurrentDomain.BaseDirectory, folderName);
    }
}
