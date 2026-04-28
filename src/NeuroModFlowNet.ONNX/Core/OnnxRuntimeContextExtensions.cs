using System.Text.RegularExpressions;

namespace NeuroModFlowNet.ONNX;

public static partial class OnnxRuntimeContextExtensions
{
    [GeneratedRegex(@"(\d+):\s*['""]([^'""\[\]]+)['""]")]
    private static partial Regex PythonDictRegex();

    public static Dictionary<int, string> GetMetadataMap(this OnnxRuntimeContext context, string key)
    {
        var raw = context.GetCustomMetadata(key);
        if (raw == null) return new();
        
        var result = new Dictionary<int, string>();
        var matches = PythonDictRegex().Matches(raw);
        foreach (Match m in matches) if (int.TryParse(m.Groups[1].Value, out int id)) result[id] = m.Groups[2].Value;
        return result;
    }

    public static string GetYoloClassName(this OnnxRuntimeContext context, int id) 
        => context.GetMetadataMap("names").TryGetValue(id, out var name) ? name : $"#{id}";
}
