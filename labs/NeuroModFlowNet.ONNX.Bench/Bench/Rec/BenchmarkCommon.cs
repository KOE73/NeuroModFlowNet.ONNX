using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Columns;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Jobs;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;


namespace NeuroModFlowNet.ONNX.Bench;

public class Config : ManualConfig
{
    public Config()
    {
        AddColumn(StatisticColumn.OperationsPerSecond);
        SmartCpuConfig();
    }

    public void SmartCpuConfig()
    {
        var job = Job.Default;
        string cpuName = GetCpuName();
        int totalThreads = Environment.ProcessorCount;

        if(cpuName == "Intel")
        {
            // Для Intel: Пытаемся оставить только P-cores. 
            // У i5-14500 6 P-ядер (12 потоков) и 8 E-ядер.
            // Обычно P-ядра идут первыми. Маска 0xFFF = первые 12 потоков.
            // Если ядер другое количество, берем первые 12 как стандарт для i5/i7.
            long pCoreMask = (1L << Math.Min(12, totalThreads)) - 1;
            job = job.WithAffinity((IntPtr)pCoreMask);
            Console.WriteLine($"[Config] Intel detected. Using P-Cores only (Mask: 0x{pCoreMask:X})");
        }
        else if(cpuName == "AMD1")
        {
            // Для AMD: Берем два потока в самом центре
            // Например: из 40 потоков это будут 20-й и 21-й (индексы 19 и 20)
            int middle = totalThreads / 2;
            long amdMask = (1L << (middle - 1)) | (1L << middle);

            job = job.WithAffinity((IntPtr)amdMask);
        }

        AddJob(job
            .WithLaunchCount(1)
            .WithWarmupCount(5)
            .WithInvocationCount(16 * 1).WithUnrollFactor(16)
            .WithIterationCount(100)
            );
    }

    public static string GetCpuName()
    {
        // Самый простой способ получить имя процессора в .NET
        string cpuName = RuntimeInformation.OSDescription.Contains("Windows")
            ? Environment.GetEnvironmentVariable("PROCESSOR_IDENTIFIER") ?? "Unknown"
            : "Unknown";

        if(cpuName.Contains("Intel", StringComparison.OrdinalIgnoreCase)) return "Intel";
        if(cpuName.Contains("AMD", StringComparison.OrdinalIgnoreCase)) return "AMD";

        return cpuName;
    }
}

//[SimpleJob(
//    iterationCount: 3,
//    warmupCount: 3,
//    invocationCount: 250)]

//[SimpleJob(
//    launchCount: 1,     // Сколько раз перезапускать весь процесс (помогает против случайных перегревов)
//    warmupCount: 5,    // Прогрев (JIT должен успеть всё скомпилировать и оптимизировать)
//    iterationCount: 10, // Количество замеров (чем больше, тем выше точность)
//    invocationCount: 250 // Оставить авто, или задать жестко, если тест слишком быстрый
//)]

[Config(typeof(Config))]
[MemoryDiagnoser]
[ThreadingDiagnoser]
[GroupBenchmarksBy(BenchmarkLogicalGroupRule.ByCategory)]
public partial class Benchmark
{

    public string modelPath = @$"c:\paddleocr-onnx\languages\english\rec.onnx";
    public string dictionaryPath = @$"c:\paddleocr-onnx\languages\english\dict.txt";


    [Params(1, 4)]
    public int batch = 4;

    public Benchmark()
    {
    }

    bool _IsAmd = false;

    #region Images
    public const string imagePath_48_320 = "Images/image_48_320_5_65.png";
    public const string imagePath_48_240 = "Images/image_48_240_5_65.png";

    Mat _image_48_320 = default!;
    List<Mat> _images_48_320 = default!;
    Mat _image_48_240 = default!;
    List<Mat> _images_48_240 = default!;


    [GlobalSetup()]
    public void SetupImage()
    {
        _image_48_320 = LoadMat_48_320();
        _image_48_240 = LoadMat_48_240();
        _images_48_320 = Enumerable.Range(1, batch).Select(_ => _image_48_320.Clone()).ToList();
        _images_48_240 = Enumerable.Range(1, batch).Select(_ => _image_48_240.Clone()).ToList();

        _IsAmd = Config.GetCpuName() == "AMD";

        //modelPath = $"Models/best_half_{batch}.onnx";
    }

    [GlobalCleanup()]
    public void CleanupImage()
    {
        _image_48_320?.Dispose(); _image_48_320 = null!;
        _image_48_240?.Dispose(); _image_48_240 = null!;
        _images_48_320?.ForEach(o => o.Dispose()); _images_48_320 = null!;
        _images_48_240?.ForEach(o => o.Dispose()); _images_48_240 = null!;
    }

    static Mat LoadMat_48_320()
    {
        var image = Cv2.ImRead(imagePath_48_320, ImreadModes.Color);
        return image;
    }
    static Mat LoadMat_48_240()
    {
        var image = Cv2.ImRead(imagePath_48_240, ImreadModes.Color);
        return image;
    }
    #endregion


    private static void PrintMeta(NodeMetadata meta)
    {
        var dimensions = string.Join(',', meta.Dimensions.Select(o => o.ToString()));
        var symbolicDimensions = string.Join(',', meta.SymbolicDimensions.Select(o => o));
        Console.WriteLine($"  Dimensions: {meta.ElementDataType.ToString()} Names:Values=[{symbolicDimensions}]:[{dimensions}]  OnnxValueType:{meta.OnnxValueType}  IsTensor:{meta.IsTensor} IsString:{meta.IsString}");
    }

    InferenceSession CreateInferenceSession()
    {
        // Настраиваем GPU (CUDA)
        var options = new SessionOptions();
        options.AppendExecutionProvider_CUDA();
        var session = new InferenceSession(modelPath, options);
        return session;
    }


}


//public readonly struct XChar(float pos, float conf)
//{
//    public readonly float X1 = x1, Y1 = y1, X2 = x2, Y2 = y2, Conf = conf;
//}

//public readonly struct BoxF16(Float16 X1, Float16 Y1, Float16 X2, Float16 Y2, Float16 Conf);
//public readonly struct BoxHalf(Half X1, Half Y1, Half X2, Half Y2, Half Conf);

