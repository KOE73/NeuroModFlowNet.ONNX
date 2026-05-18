using System.Text.Json;
using System.Text.Json.Serialization;

namespace NeuroModFlowNet.ONNX.Avalonia.Runtime;

/// <summary>
/// EN: Root JSON configuration for the Avalonia realtime lab — postprocessing parameters and inference thresholds.
/// RU: Корневая JSON-конфигурация Avalonia realtime lab — параметры постобработки и пороги инференса.
/// </summary>
/// <remarks>
/// EN: Load order: <c>appsettings.json</c> supplies defaults; <c>appsettings.Local.json</c> (git-ignored) overlays
/// individual keys. Missing local file is silently ignored. Call <see cref="Load"/> once at startup and pass the
/// result down the call chain; do not call it on every frame.
/// RU: Порядок загрузки: <c>appsettings.json</c> задает дефолты; <c>appsettings.Local.json</c> (в gitignore)
/// перекрывает отдельные ключи. Отсутствие локального файла игнорируется. Вызывать <see cref="Load"/> один раз при
/// старте и передавать результат по цепочке; вызывать на каждом кадре нельзя.
/// </remarks>
public sealed class AvaloniaJsonConfig
{
    private static readonly JsonSerializerOptions SerializerOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true,
        Converters = { new JsonStringEnumConverter(JsonNamingPolicy.CamelCase) },
    };

    public PaddleOCRPostprocessingOptions Postprocessing { get; set; } = new();

    public AvaloniaInferenceConfig Inference { get; set; } = new();

    /// <summary>
    /// EN: Loads configuration from <c>appsettings.json</c>, then overlays keys from <c>appsettings.Local.json</c>
    /// if that file exists. Both files are resolved relative to the application base directory.
    /// RU: Загружает конфигурацию из <c>appsettings.json</c>, затем перекрывает ключи из
    /// <c>appsettings.Local.json</c>, если файл существует. Оба файла разрешаются относительно base directory приложения.
    /// </summary>
    public static AvaloniaJsonConfig Load()
    {
        string baseDirectory = AppContext.BaseDirectory;
        string defaultPath = Path.Combine(baseDirectory, "appsettings.json");
        string localPath = Path.Combine(baseDirectory, "appsettings.Local.json");

        AvaloniaJsonConfig config = LoadFile(defaultPath) ?? new AvaloniaJsonConfig();

        if(File.Exists(localPath))
            MergeLocalOverrides(config, localPath);

        return config;
    }

    private static AvaloniaJsonConfig? LoadFile(string path)
    {
        if(!File.Exists(path))
            return null;

        string json = File.ReadAllText(path);
        return JsonSerializer.Deserialize<AvaloniaJsonConfig>(json, SerializerOptions);
    }

    private static void MergeLocalOverrides(AvaloniaJsonConfig target, string localPath)
    {
        string localJson = File.ReadAllText(localPath);

        using JsonDocument localDocument = JsonDocument.Parse(localJson, new JsonDocumentOptions
        {
            CommentHandling = JsonCommentHandling.Skip,
            AllowTrailingCommas = true,
        });

        // EN: Re-serialize the current config, merge local document on top, then re-deserialize into the target.
        // RU: Сериализуем текущий конфиг, поверх накладываем local-документ, затем десериализуем в target.
        string baseJson = JsonSerializer.Serialize(target, SerializerOptions);
        using JsonDocument baseDocument = JsonDocument.Parse(baseJson);

        string mergedJson = Merge(baseDocument.RootElement, localDocument.RootElement);
        AvaloniaJsonConfig? merged = JsonSerializer.Deserialize<AvaloniaJsonConfig>(mergedJson, SerializerOptions);

        if(merged is null)
            return;

        target.Postprocessing = merged.Postprocessing;
        target.Inference = merged.Inference;
    }

    private static string Merge(JsonElement baseElement, JsonElement overrideElement)
    {
        using var stream = new System.IO.MemoryStream();
        using var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = false });

        MergeElement(writer, baseElement, overrideElement);

        writer.Flush();
        return System.Text.Encoding.UTF8.GetString(stream.ToArray());
    }

    private static void MergeElement(Utf8JsonWriter writer, JsonElement baseElement, JsonElement overrideElement)
    {
        // EN: Only objects are merged recursively; all other types are simply replaced by the override value.
        // RU: Объекты мержатся рекурсивно; все остальные типы просто заменяются значением из override.
        if(baseElement.ValueKind != JsonValueKind.Object || overrideElement.ValueKind != JsonValueKind.Object)
        {
            overrideElement.WriteTo(writer);
            return;
        }

        writer.WriteStartObject();

        foreach(JsonProperty baseProperty in baseElement.EnumerateObject())
        {
            writer.WritePropertyName(baseProperty.Name);

            if(overrideElement.TryGetProperty(baseProperty.Name, out JsonElement overrideProperty))
                MergeElement(writer, baseProperty.Value, overrideProperty);
            else
                baseProperty.Value.WriteTo(writer);
        }

        foreach(JsonProperty overrideProperty in overrideElement.EnumerateObject())
        {
            if(!baseElement.TryGetProperty(overrideProperty.Name, out _))
            {
                writer.WritePropertyName(overrideProperty.Name);
                overrideProperty.Value.WriteTo(writer);
            }
        }

        writer.WriteEndObject();
    }
}
