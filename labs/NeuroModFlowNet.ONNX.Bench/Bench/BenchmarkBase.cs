using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Columns;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Jobs;
using OpenCvSharp;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;


namespace NeuroModFlowNet.ONNX.Bench;

public class ConfigDet : ManualConfig
{
    public ConfigDet()
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
            .WithWarmupCount(4)
            .WithInvocationCount(32 * 1).WithUnrollFactor(16)
            .WithIterationCount(10)
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

[Config(typeof(ConfigDet))]
[MemoryDiagnoser]
[ThreadingDiagnoser]
[GroupBenchmarksBy(BenchmarkLogicalGroupRule.ByCategory)]
public class BenchmarkBase
{
    public string ModelRootDir { get; set; } = @"C:\Models\det-to-obb";

    public bool BenchDebug { get; set; } = false;


    [Params(1, 4)]
    public int Batch = 4;

    #region Images

    List<Mat> matsToDispose = new();

    protected Mat _Image_Text1 = default!; List<Mat> _List_Image_Text1 = default!;
    protected Mat _Image_Text2 = default!; List<Mat> _List_Image_Text2 = default!;
    protected Mat _Image_Text3 = default!; List<Mat> _List_Image_Text3 = default!;
    protected Mat _Image_Text4 = default!; List<Mat> _List_Image_Text4 = default!;

    protected List<Mat> _List_Text_1_Image = default!;
    protected List<Mat> _List_Text_2_Image = default!;
    protected List<Mat> _List_Text_3_Image = default!;
    protected List<Mat> _List_Text_4_Image = default!;




    [GlobalSetup()]
    public void SetupImage()
    {
        var path = Environment.GetEnvironmentVariable("DET_MODEL_PATH");


        LoadMat(@"Images/Image_Text1.jpg", ref _Image_Text1, ref _List_Image_Text1, 512, 512);
        LoadMat(@"Images/Image_Text2.jpg", ref _Image_Text2, ref _List_Image_Text2, 512, 512);
        LoadMat(@"Images/Image_Text3.jpg", ref _Image_Text3, ref _List_Image_Text3, 512, 512);
        LoadMat(@"Images/Image_Text4.jpg", ref _Image_Text4, ref _List_Image_Text4, 512, 512);

        _List_Text_1_Image = [_Image_Text1];
        _List_Text_2_Image = [_Image_Text1, _Image_Text2];
        _List_Text_3_Image = [_Image_Text1, _Image_Text2, _Image_Text3];
        _List_Text_4_Image = [_Image_Text1, _Image_Text2, _Image_Text3, _Image_Text4];

        //LoadMat(@"Images/frame_000046.png", ref , ref , 512, 512);
        //LoadMat(@"Images/frame_000062.png", ref , ref , 512, 512);
        //LoadMat(@"Images/frame_000198.png", ref , ref , 512, 512);
        //LoadMat(@"Images/image_48_240_5_65.png", ref , ref , 512, 512);
        //LoadMat(@"Images/image_48_320_5_65.png", ref , ref , 512, 512);
    }

    [GlobalCleanup()]
    public void CleanupImage()
    {
        foreach(var item in matsToDispose) item.Dispose();
    }

    void LoadMat(string path, ref Mat image, ref List<Mat> images, int? width, int? height)
    {
        image = Cv2.ImRead(path, ImreadModes.Color);

        if(width.HasValue && height.HasValue)
        {
            Mat resize = new();
            Cv2.Resize(image, resize, new Size(width.Value, height.Value));
            image.Dispose();
            image = resize;
        }
        var tmp = image;
        images = Enumerable.Range(1, Batch).Select(_ => tmp.Clone()).ToList();

        matsToDispose.Add(image);
        matsToDispose.AddRange(images);
    }


    #endregion



}


//public readonly struct XChar(float pos, float conf)
//{
//    public readonly float X1 = x1, Y1 = y1, X2 = x2, Y2 = y2, Conf = conf;
//}

//public readonly struct BoxF16(Float16 X1, Float16 Y1, Float16 X2, Float16 Y2, Float16 Conf);
//public readonly struct BoxHalf(Half X1, Half Y1, Half X2, Half Y2, Half Conf);

