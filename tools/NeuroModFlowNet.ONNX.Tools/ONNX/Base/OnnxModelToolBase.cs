using Spectre.Console;

namespace NeuroModFlowNet.ONNX.Tools;

public class OnnxModelToolBase
{
    public  ModelProto LoadModel(string path)
    {
        // Читаем файл как поток байтов
        using var input = File.OpenRead(path);
        var model = ModelProto.Parser.ParseFrom(input);
        return model;
    }

    public  void SaveModifiedModel(ModelProto model, string originalPath, string extraName)
    {
        // Генерируем новый путь: C:\models\yolo.onnx -> C:\models\yolo_head.onnx
        string directory = Path.GetDirectoryName(originalPath) ?? "";
        string fileName = Path.GetFileNameWithoutExtension(originalPath);
        string extension = Path.GetExtension(originalPath);

        string newPath = Path.Combine(directory, $"{fileName}{extraName}{extension}");

        using var output = File.Create(newPath);
        model.WriteTo(output);

        AnsiConsole.WriteLine($"[green]SUCCESS:[/] Model saved to: {newPath}");
    }
}
