using System.Text.Json;
using NeuroModFlowNet.ONNX;

namespace NeuroModFlowNet.Pipeline.ONNX.Tests.Common;

internal sealed class PipelineOnnxTestEnvironment
{
    static readonly Lazy<PipelineOnnxTestEnvironment> CurrentLazy = new(Create);

    PipelineOnnxTestEnvironment(PipelineOnnxTestSettings settings)
    {
        Settings = settings;
        VisualArtifactsRoot = ResolveVisualArtifactsRoot(settings);
        EnabledExecutionBackends = ResolveEnabledExecutionBackends(settings);
    }

    public static PipelineOnnxTestEnvironment Current => CurrentLazy.Value;

    public PipelineOnnxTestSettings Settings { get; }

    public string VisualArtifactsRoot { get; }

    public bool SaveVisualArtifacts =>
        Settings.SaveVisualArtifacts ||
        IsEnabled("NMFN_VISUAL_SAVE") ||
        InteractiveVisualArtifacts;

    public bool InteractiveVisualArtifacts =>
        Settings.InteractiveVisualArtifacts ||
        IsEnabled("NMFN_VISUAL_INTERACTIVE");

    public IReadOnlyList<InferenceBackend> EnabledExecutionBackends { get; }

    public string CreateOperationArtifactsDirectory(string domain, string operationName)
    {
        string[] operationPathParts = CreateOperationPathParts(domain, operationName)
            .Split([Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar], StringSplitOptions.RemoveEmptyEntries)
            .Select(MakePathSafe)
            .ToArray();
        string path = Path.Combine([VisualArtifactsRoot, MakePathSafe(domain), .. operationPathParts]);
        Directory.CreateDirectory(path);
        return path;
    }

    public string CreateOperationArtifactsDirectory(string domain, string operationName, string caseName)
    {
        string safeCaseName = MakePathSafe(caseName);
        string path = Path.Combine(CreateOperationArtifactsDirectory(domain, operationName), safeCaseName);
        Directory.CreateDirectory(path);
        return path;
    }

    public string CreateOperationArtifactPath(
        string domain,
        string operationName,
        string backendName,
        string caseName,
        string artifactKind,
        string extension = ".png")
    {
        string artifactsDirectory = CreateOperationArtifactsDirectory(domain, operationName);
        string safeExtension = extension.StartsWith('.') ? extension : "." + extension;
        string fileName =
            $"{MakePathSafe(operationName)}-{MakePathSafe(backendName)}-{MakePathSafe(caseName)}-{MakePathSafe(artifactKind)}{safeExtension}";
        return Path.Combine(artifactsDirectory, fileName);
    }

    static string CreateOperationPathParts(string domain, string operationName)
    {
        if(!string.Equals(domain, "Onnx", StringComparison.OrdinalIgnoreCase))
            return operationName;

        string typeFolder = InferOnnxTypeFolder(operationName);
        return Path.Combine(typeFolder, operationName);
    }

    static string InferOnnxTypeFolder(string operationName)
    {
        if(operationName.Contains("FP16_NCHW", StringComparison.OrdinalIgnoreCase) ||
           operationName.Contains("FP16Nchw", StringComparison.OrdinalIgnoreCase))
        {
            return "FP16_NCHW";
        }

        if(operationName.Contains("FP32_NCHW", StringComparison.OrdinalIgnoreCase) ||
           operationName.Contains("FP32Nchw", StringComparison.OrdinalIgnoreCase))
        {
            return "FP32_NCHW";
        }

        if(operationName.Contains("U8_NHWC", StringComparison.OrdinalIgnoreCase))
            return "U8_NHWC";

        if(operationName.Contains("U8Hwc", StringComparison.OrdinalIgnoreCase))
            return "U8_HWC";

        return "UnknownType";
    }

    static PipelineOnnxTestEnvironment Create()
    {
        PipelineOnnxTestSettings settings = LoadSettings();
        InitializeNativeLibraryPaths(settings);
        return new PipelineOnnxTestEnvironment(settings);
    }

    static PipelineOnnxTestSettings LoadSettings()
    {
        string configPath = Path.Combine(AppContext.BaseDirectory, "appsettings.test.json");

        if(!File.Exists(configPath))
            return new PipelineOnnxTestSettings();

        string json = File.ReadAllText(configPath);
        return JsonSerializer.Deserialize<PipelineOnnxTestSettings>(
                json,
                new JsonSerializerOptions(JsonSerializerDefaults.Web)) ??
            new PipelineOnnxTestSettings();
    }

    static void InitializeNativeLibraryPaths(PipelineOnnxTestSettings settings)
    {
        string existingPath = Environment.GetEnvironmentVariable("PATH") ?? string.Empty;
        string?[] configuredPaths =
        [
            .. settings.NativeLibrarySearchPaths,
            GetPathFromEnvironmentOrSettings("NMFN_CUDA_BIN_PATH", settings.CudaBinPath),
            GetPathFromEnvironmentOrSettings("NMFN_CUDNN_BIN_PATH", settings.CudnnBinPath),
            GetPathFromEnvironmentOrSettings("NMFN_TRT_LIB_PATH", settings.TrtLibPath)
        ];

        string[] resolvedPaths = configuredPaths
            .Where(static path => !string.IsNullOrWhiteSpace(path))
            .Select(static path => ResolveConfiguredPath(path!))
            .Where(Directory.Exists)
            .ToArray();

        if(resolvedPaths.Length == 0)
            return;

        string pathSeparator = Path.PathSeparator.ToString();
        string updatedPath = string.Join(pathSeparator, resolvedPaths.Concat([existingPath]));
        Environment.SetEnvironmentVariable("PATH", updatedPath);
    }

    static string? GetPathFromEnvironmentOrSettings(string environmentVariable, string? configuredPath)
    {
        string? environmentPath = Environment.GetEnvironmentVariable(environmentVariable);
        return string.IsNullOrWhiteSpace(environmentPath) ? configuredPath : environmentPath;
    }

    static string ResolveVisualArtifactsRoot(PipelineOnnxTestSettings settings)
    {
        string? configuredRoot = Environment.GetEnvironmentVariable("NMFN_VISUAL_ROOT");
        if(string.IsNullOrWhiteSpace(configuredRoot))
            configuredRoot = settings.VisualArtifactsRoot;

        if(!string.IsNullOrWhiteSpace(configuredRoot))
            return ResolveConfiguredPath(configuredRoot);

        return Path.Combine(Path.GetTempPath(), "NeuroModFlowNet.Pipeline.ONNX.Tests", "VisualTestResults");
    }

    static string ResolveConfiguredPath(string path)
    {
        string expandedPath = Environment.ExpandEnvironmentVariables(path);
        return Path.IsPathRooted(expandedPath)
            ? expandedPath
            : Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, expandedPath));
    }

    static bool IsEnabled(string environmentVariable)
    {
        string? value = Environment.GetEnvironmentVariable(environmentVariable);
        return string.Equals(value, "1", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(value, "true", StringComparison.OrdinalIgnoreCase);
    }

    static IReadOnlyList<InferenceBackend> ResolveEnabledExecutionBackends(PipelineOnnxTestSettings settings)
    {
        var backends = new List<InferenceBackend>();

        if(IsEnabledBySettingsOrEnvironment(settings.ExecutionBackends.Cpu, "NMFN_ONNX_TEST_CPU"))
            backends.Add(InferenceBackend.Cpu);

        if(IsEnabledBySettingsOrEnvironment(settings.ExecutionBackends.Cuda, "NMFN_ONNX_TEST_CUDA"))
            backends.Add(InferenceBackend.Cuda);

        if(IsEnabledBySettingsOrEnvironment(settings.ExecutionBackends.TensorRt, "NMFN_ONNX_TEST_TENSORRT"))
            backends.Add(InferenceBackend.TensorRt);

        return backends;
    }

    static bool IsEnabledBySettingsOrEnvironment(bool configuredValue, string environmentVariable)
    {
        string? value = Environment.GetEnvironmentVariable(environmentVariable);

        if(string.IsNullOrWhiteSpace(value))
            return configuredValue;

        return string.Equals(value, "1", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(value, "true", StringComparison.OrdinalIgnoreCase);
    }

    static string MakePathSafe(string value)
    {
        char[] invalidChars = Path.GetInvalidFileNameChars();
        return string.Join("_", value.Split(invalidChars, StringSplitOptions.RemoveEmptyEntries));
    }
}
