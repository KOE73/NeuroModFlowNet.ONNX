namespace NeuroModFlowNet.ONNX.Demo.Assets;

public static class ModelNaming
{
    public const string Delimiter = "__";

    public static string GetFileName(
        string baseName, 
        int? imgSize = null, 
        int? batchSize = null, 
        string precision = "fp16",
        bool isByteBgr = true)
    {
        string suffix;
        
        if (imgSize.HasValue)
        {
            // Static model
            suffix = $"{imgSize.Value}_b{batchSize ?? 1}_{precision}";
        }
        else
        {
            // Dynamic model
            suffix = precision;
        }

        if (isByteBgr)
        {
            suffix += "_bytebgr";
        }

        return Path.Combine(baseName, $"{baseName}{Delimiter}{suffix}.onnx");
    }
}
