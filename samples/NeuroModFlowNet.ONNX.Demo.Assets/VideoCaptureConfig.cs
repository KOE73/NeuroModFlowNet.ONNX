using System.Configuration;
using OpenCvSharp;

namespace NeuroModFlowNet.ONNX.Demo.Assets;

public static class VideoCaptureConfig
{
    public const string DefaultConfigKey = "VideoSource";

    public static VideoCapture CreateFromConfig(string configKey = DefaultConfigKey)
    {
        string source = ConfigurationManager.AppSettings[configKey] ?? "0";
        return Create(source);
    }

    public static VideoCaptureSourceInfo GetSourceInfo(string configKey = DefaultConfigKey)
    {
        string source = ConfigurationManager.AppSettings[configKey] ?? "0";
        string resolvedSource = ResolveSource(source, out string? linkPath, out string? linkContent);
        return new VideoCaptureSourceInfo(configKey, source, resolvedSource, linkPath, linkContent);
    }

    public static VideoCapture Create(string source)
    {
        string resolvedSource = ResolveSource(source);

        return int.TryParse(resolvedSource, out int cameraIndex)
            ? new VideoCapture(cameraIndex)
            : new VideoCapture(resolvedSource);
    }

    private static string ResolveSource(string source) =>
        ResolveSource(source, out _, out _);

    private static string ResolveSource(string source, out string? linkPath, out string? linkContent)
    {
        source = source.Trim();
        linkPath = null;
        linkContent = null;

        if(source.Length == 0)
            return "0";

        if(!source.StartsWith('@'))
            return source;

        string path = source[1..].Trim();
        string fullPath = ResolveExistingFile(path);

        string fileSource = File.ReadAllText(fullPath).Trim();
        if(fileSource.Length == 0)
            throw new InvalidOperationException($"Video source file is empty: {fullPath}");

        linkPath = fullPath;
        linkContent = fileSource;
        return fileSource;
    }

    private static string ResolveExistingFile(string path)
    {
        if(Path.IsPathRooted(path))
        {
            if(File.Exists(path))
                return path;

            throw new FileNotFoundException("Video source file was not found.", path);
        }

        foreach(string root in EnumerateSearchRoots())
        {
            string candidate = Path.GetFullPath(Path.Combine(root, path));
            if(File.Exists(candidate))
                return candidate;
        }

        throw new FileNotFoundException("Video source file was not found.", path);
    }

    private static IEnumerable<string> EnumerateSearchRoots()
    {
        yield return Environment.CurrentDirectory;
        yield return AppContext.BaseDirectory;

        string? current = AppContext.BaseDirectory;
        while(!string.IsNullOrEmpty(current))
        {
            yield return current;
            current = Directory.GetParent(current)?.FullName;
        }
    }
}
