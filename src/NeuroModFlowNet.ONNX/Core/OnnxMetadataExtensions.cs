using System.Text.RegularExpressions;

namespace NeuroModFlowNet.ONNX;

public static partial class OnnxMetadataExtensions
{
    [GeneratedRegex(@"(\d+):\s*['""]([^'""\[\]]+)['""]")]
    private static partial Regex PythonDictRegex();

    public static Dictionary<int, string> GetMetadataMap(this OnnxModel context, string key)
    {
        var raw = context.GetCustomMetadata(key);
        if (raw == null) return new();
        
        var result = new Dictionary<int, string>();
        var matches = PythonDictRegex().Matches(raw);
        foreach (Match m in matches) if (int.TryParse(m.Groups[1].Value, out int id)) result[id] = m.Groups[2].Value;
        return result;
    }

    public static string GetYoloClassName(this OnnxModel context, int id) 
        => context.GetMetadataMap("names").TryGetValue(id, out var name) ? name : $"#{id}";

    public static Dictionary<int, string> GetMetadataMap(this OnnxExecutionContext context, string key) =>
        context.Model.GetMetadataMap(key);

    public static string GetYoloClassName(this OnnxExecutionContext context, int id) =>
        context.Model.GetYoloClassName(id);
}
