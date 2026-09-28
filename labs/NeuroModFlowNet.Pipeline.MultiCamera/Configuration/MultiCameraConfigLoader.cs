using System.Configuration;
using System.Text.Json;
using System.Text.Json.Serialization;
using NeuroModFlowNet.ONNX.Demo.Assets;

namespace NeuroModFlowNet.Pipeline.MultiCamera;

/// <summary>
/// EN: Loads <see cref="MultiCameraConfig"/> from JSON, resolves model paths against the models root and expands
/// "@file.local.txt" indirections for RTSP URLs with credentials.
///
/// RU: Загружает <see cref="MultiCameraConfig"/> из JSON, разрешает пути моделей относительно корня моделей и
/// раскрывает ссылки вида "@file.local.txt" для RTSP-URL с учётными данными.
/// </summary>
internal static class MultiCameraConfigLoader
{
    const string ConfigKey = "CamerasConfig";

    static readonly JsonSerializerOptions SerializerOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true,
        Converters = { new JsonStringEnumConverter(JsonNamingPolicy.CamelCase, allowIntegerValues: false) }
    };

    public static string ResolveConfigPath(string[] args)
    {
        string? configured = args.Length > 0 && !string.IsNullOrWhiteSpace(args[0])
            ? args[0]
            : ConfigurationManager.AppSettings[ConfigKey];

        if(string.IsNullOrWhiteSpace(configured))
            throw new InvalidOperationException($"Config path is not set: pass it as the first argument or set '{ConfigKey}' in App.local.config.");

        return ResolveExistingFile(configured.Trim());
    }

    public static MultiCameraConfig Load(string configPath)
    {
        string json = File.ReadAllText(configPath);
        MultiCameraConfig config = JsonSerializer.Deserialize<MultiCameraConfig>(json, SerializerOptions)
            ?? throw new InvalidDataException($"Config '{configPath}' is empty.");

        config.Validate();

        string modelsRoot = AssetsManager.ResolveModelsRoot();
        string configDirectory = Path.GetDirectoryName(configPath) ?? Directory.GetCurrentDirectory();

        OcrConfig ocr = config.Ocr.Enabled
            ? config.Ocr with
            {
                TextObbModelPath = ResolveModelPath(config.Ocr.TextObbModelPath!, modelsRoot),
                RecognitionModelPath = ResolveModelPath(config.Ocr.RecognitionModelPath!, modelsRoot)
            }
            : config.Ocr;

        var cameras = new List<CameraConfig>(config.Cameras.Count);
        foreach(CameraConfig camera in config.Cameras)
        {
            SourceConfig source = camera.Source with
            {
                Url = camera.Source.Url is null ? null : ResolveIndirection(camera.Source.Url, configDirectory),
                Path = camera.Source.Path is null ? null : ResolveIndirection(camera.Source.Path, configDirectory),
                Files = camera.Source.Files?.Select(file => ResolveIndirection(file, configDirectory)).ToArray()
            };
            cameras.Add(camera with
            {
                Source = source,
                Transforms = camera.Transforms ?? [],
                Detector = camera.Detector with { ModelPath = ResolveModelPath(camera.Detector.ModelPath, modelsRoot) }
            });
        }

        return config with { Ocr = ocr, Cameras = cameras };
    }

    #region Path resolution

    static string ResolveModelPath(string modelPath, string modelsRoot)
    {
        string resolved = Path.IsPathRooted(modelPath) ? modelPath : Path.GetFullPath(Path.Combine(modelsRoot, modelPath));
        if(!File.Exists(resolved))
            throw new FileNotFoundException($"Model file was not found: {resolved}", resolved);

        return resolved;
    }

    /// <summary>
    /// "@name.local.txt" reads the actual value from a git-ignored file next to the config or in the search roots.
    /// </summary>
    static string ResolveIndirection(string value, string configDirectory)
    {
        if(!value.StartsWith('@'))
            return value;

        string linkPath = value[1..].Trim();
        string resolvedLink = Path.IsPathRooted(linkPath) ? linkPath : ResolveExistingFile(linkPath, configDirectory);
        string content = File.ReadAllText(resolvedLink).Trim();
        if(string.IsNullOrWhiteSpace(content))
            throw new InvalidDataException($"Indirection file '{resolvedLink}' is empty.");

        return content;
    }

    static string ResolveExistingFile(string path, string? extraRoot = null)
    {
        if(Path.IsPathRooted(path))
            return File.Exists(path) ? path : throw new FileNotFoundException($"File was not found: {path}", path);

        foreach(string root in EnumerateSearchRoots(extraRoot))
        {
            string candidate = Path.GetFullPath(Path.Combine(root, path));
            if(File.Exists(candidate))
                return candidate;
        }

        throw new FileNotFoundException($"File '{path}' was not found in the working directory, executable directory or their parents.", path);
    }

    static IEnumerable<string> EnumerateSearchRoots(string? extraRoot)
    {
        if(extraRoot is not null)
            yield return extraRoot;

        foreach(string start in new[] { Directory.GetCurrentDirectory(), AppContext.BaseDirectory })
        {
            var directory = new DirectoryInfo(start);
            while(directory is not null)
            {
                yield return directory.FullName;
                directory = directory.Parent;
            }
        }
    }

    #endregion
}
