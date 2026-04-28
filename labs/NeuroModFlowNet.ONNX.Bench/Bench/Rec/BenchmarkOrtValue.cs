using BenchmarkDotNet.Attributes;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX.Diagnostics;
using OpenCvSharp;
using System.Numerics.Tensors;
using System.Text;


namespace NeuroModFlowNet.ONNX.Bench;

public partial class Benchmark
{
    long[] _InputOrtValueShape;
    long[] _OutputOrtValueShape;


    const long trt_max_workspace_size_Gb = 4L;


    object _OnnxPaddleOCRRecModel = default!;

    //ConvertersBase? _OrtValueHelper;


    #region Mats2

    //[GlobalSetup(Targets = [
    //    //nameof(Mats2Ort_DivHalfTensor_ReorderPtr),
    //    //nameof(Mats2Ort_ReorderDivPtr_HalfTensor),
    //    //nameof(Mats2Ort_ReorderPtr_DivHalfTensor),
    //    //nameof(Mats2Ort_ReorderDivHalfPtr),
    //    //nameof(Mats2Ort_OpenCV_Blob)
    //    ])]
    public void SetupOrt()
    {
        SetupImage();

   


    }


    //[GlobalCleanup(Targets = [
    //    //nameof(Mats2Ort_DivHalfTensor_ReorderPtr),
    //    //nameof(Mats2Ort_ReorderDivPtr_HalfTensor),
    //    //nameof(Mats2Ort_ReorderPtr_DivHalfTensor),
    //    //nameof(Mats2Ort_ReorderDivHalfPtr),
    //    //nameof(Mats2Ort_OpenCV_Blob)
    //    ])]
    public void CleanupOrt()
    {
        CleanupImage();



    }

    //[Benchmark(OperationsPerInvoke = 1), BenchmarkCategory("Ort")]
    //public void Mats2Ort_DivHalfTensor_ReorderPtr()
    //{
    //    Span<Float16> buffer = _InputOrtValue.GetTensorMutableDataAsSpan<Float16>();
    //    _OrtValueHelper!.Mats2Ort_DivHalfTensor_ReorderPtr(_images, buffer, batch);
    //}

    #endregion


    #region Run

    /// <summary>
    /// Specifies the inference backend to use for model execution during benchmarking.
    /// 
    /// On NVidia DML work but not compatible with TensorRT.
    /// </summary>
    /// <remarks>This field is typically set by the benchmarking framework to control which hardware or
    /// software backend is used for inference. Supported values may include TensorRT, CUDA, or DirectML, depending on
    /// the available hardware and configuration.</remarks>
    [Params([InferenceBackend.TensorRt, InferenceBackend.Cuda])]
    //[Params([InferenceBackend.DML])]
    public InferenceBackend _InferenceBackend = InferenceBackend.Cuda;

    string[] _Alphabet;

    int _Device = 0;

    [Params(false, true)] public bool EnableFp16 = true;
    [Params(false, true)] public bool EnableBf16 = true;

    [GlobalSetup(Targets = [
            nameof(Inference_Run),
            nameof(Inference_RunWithBinding)
            //,nameof(InferenceOrt_ByMatBlob)
            ])]
    public void SetupInferenceOrtModel()
    {
        SetupOrt();

        SessionOptions sessionOptions;

        switch(_InferenceBackend)
        {
            case InferenceBackend.Rocm:
                {
                    sessionOptions = SessionOptions.MakeSessionOptionWithRocmProvider(0);
                    break;
                }
            case InferenceBackend.Cuda:
                {
                    using var cudaOptions = new OrtCUDAProviderOptions();
                    var tt = cudaOptions.GetOptions();
                    sessionOptions = SessionOptions.MakeSessionOptionWithCudaProvider(cudaOptions);
                    break;
                }
            case InferenceBackend.TensorRt:
                {
                    using var trtOptions = new OrtTensorRTProviderOptions();
                    TrtConfig providerOptions = TrtConfig.FromDefaultOptions(trtOptions.GetOptions());
                    providerOptions.MaxWorkspaceSizeGb = trt_max_workspace_size_Gb;
                    providerOptions.EnableFp16 = EnableFp16;
                    providerOptions.EnableBf16 = EnableBf16;
                    providerOptions.EnableSparsity = true;
                    providerOptions.EnableEngineCache = true;
                    providerOptions.EngineCachePath = @"C:\TRT_Cache";
                    providerOptions.BuilderOptimizationLevel = 4;
                    providerOptions.DumpSubgraphs = true;

                    var providerOptions1 = providerOptions.ToDictionary();
                    trtOptions.UpdateOptions(providerOptions);
                    sessionOptions = SessionOptions.MakeSessionOptionWithTensorrtProvider(trtOptions);
                    break;
                }
            case InferenceBackend.DML:
                sessionOptions = new SessionOptions();
                sessionOptions.AppendExecutionProvider_DML(0);
                break;
            case InferenceBackend.Cpu:
            default:
                sessionOptions = new SessionOptions();
                break;
        }

        //sessionOptions.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE;
        //sessionOptions.LogVerbosityLevel = (int)OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO;


        // _OnnxPaddleOCRRecModel = new (modelPath);
        // _OnnxPaddleOCRRecModel.LoadAlphabet(dictionaryPath);

        //_Session.PrintInferenceSessionInfo();



    }

    [GlobalCleanup(Targets = [
        nameof(Inference_Run)    ,
        nameof(Inference_RunWithBinding)
        //,nameof(InferenceOrt_ByMatBlob)
        ])]
    public void CleanupInferenceOrtModel()
    {
        CleanupImage();

    }

    [Benchmark(OperationsPerInvoke = 1), BenchmarkCategory("OCR")]
    public void Inference_Run()
    {
        //Span<float> buffer = _InputOrtValue.GetTensorMutableDataAsSpan<float>();

        //OrtValueHelpers ortValueHelper = new OrtValueHelpers();
        //ortValueHelper.Symmetric_ReorderNormPtr(_images_48_320, buffer, batch);

        //using var outputs = _Session.Run(_RunOptions, (IReadOnlyCollection<string>)[_ModelInputName], (IReadOnlyCollection<OrtValue>)[_InputOrtValue], (IReadOnlyCollection<string>)[_ModelOutputName]);

        //var output = outputs.First();
        //var tensorSpan = output.GetTensorDataAsTensorSpan<float>();
        //AnalizeResult_PrintData(tensorSpan, true);

    }

    [Benchmark(OperationsPerInvoke = 1), BenchmarkCategory("OCR")]
    public void Inference_RunWithBinding()
    {


        //Span<float> buffer = _InferenceSessionPPOcrRecContext.InputOrtValue.GetTensorMutableDataAsSpan<float>();

        //_OrtValueHelper!.Symmetric_ReorderNormPtr(_images_48_320, buffer, batch);

        //var result = _InferenceSessionPPOcrRecContext.RunInferenceAndAnalyse();

        //if(result[0].Standard != "5-6550KG")
        //    throw new Exception($"Не работает как надо. resStandard = '{result[0].Standard}' должно быть '5-6550KG'");

    }


    //[Benchmark(OperationsPerInvoke = 1), BenchmarkCategory("Ort")]
    public void InferenceOrt_ByMatBlob()
    {
        //Span<float> buffer = _InputOrtValue.GetTensorMutableDataAsSpan<float>();

        //_OrtValueHelper!.Symmetric_OpenCV_Blob(_images_48_320, buffer, batch);

        //_OrtIoBinding.BindInput(_Session.InputNames[0], _InputOrtValue);
        //_OrtIoBinding.BindOutput(_Session.OutputNames[0], _OutputOrtValue);

        //_Session.RunWithBinding(_RunOptions, _OrtIoBinding);
        //OrtValue resultFirst = _OutputOrtValue.Value;
        //ReadOnlyTensorSpan<float> tensorSpan = resultFirst.GetTensorDataAsTensorSpan<float>();
        //AnalizeResult_PrintData(tensorSpan, true);
    }

    #endregion


    #region Main

    public void MainCuda_Run()
    {
        //_InferenceBackend = InferenceBackend.Cuda;

        //SetupInferenceOrtModel();

//_OnnxPaddleOCRRecModel.WriteInfo();

        //using Mat resized = CutMatFromFile();

        ////_OrtValueHelper!.Mats2Ort_OpenCV_Blob(_images, buffer, batch);

        ////_OrtIoBinding.BindInput(_Session.InputNames[0], _InputOrtValue);


        //// Создаем OrtValue. Он будет "контейнером" для памяти.
        //using OrtValue inputOrtValue = OrtValue.CreateAllocatedTensorValue(
        //    OrtAllocator.DefaultInstance, // Уточнить кроме CPU что еще есть
        //    TensorElementType.Float,
        //    _InputOrtValueShape);



        //Span<float> buffer = inputOrtValue.GetTensorMutableDataAsSpan<float>();
        //var _images = Enumerable.Range(1, batch).Select(_ => resized.Clone()).ToList();

        //OrtValueHelpers ortValueHelper = new OrtValueHelpers();
        //ortValueHelper.Symmetric_ReorderNormPtr(_images, buffer, batch);

        //using var outputs = _Session.Run(new RunOptions(), (IReadOnlyCollection<string>)[_ModelInputName], (IReadOnlyCollection<OrtValue>)[inputOrtValue], (IReadOnlyCollection<string>)[_ModelOutputName]);

        //OrtValue output = outputs.First();
        //OrtTensorTypeAndShapeInfo ttsh = output.GetTensorTypeAndShape();
//ttsh.WriteInfo("output");

        //AnalizeResult_GetText(output.GetTensorDataAsTensorSpan<float>());

        //CleanupInferenceOrtModel();
    }

    public void MainTensorRt_Run()
    {
        //_InferenceBackend = InferenceBackend.TensorRt;

        //SetupInferenceOrtModel();

        //_Session.PrintInferenceSessionInfo();


        //using OrtValue inputOrtValue = OrtValue.CreateAllocatedTensorValue(
        //    OrtAllocator.DefaultInstance, // Уточнить кроме CPU что еще есть
        //    TensorElementType.Float,
        //    _InputOrtValueShape);


        //Span<float> buffer = inputOrtValue.GetTensorMutableDataAsSpan<float>();

        //using Mat resized = CutMatFromFile();
        //var _images = Enumerable.Range(1, batch).Select(_ => resized.Clone()).ToList();

        //OrtValueHelpers ortValueHelper = new OrtValueHelpers();
        //ortValueHelper.Symmetric_ReorderNormPtr(_images, buffer, batch);

        //using var outputs = _Session.Run(new RunOptions(), (IReadOnlyCollection<string>)[_ModelInputName], (IReadOnlyCollection<OrtValue>)[inputOrtValue], (IReadOnlyCollection<string>)[_ModelOutputName]);

        //OrtValue output = outputs.First();
        //OrtTensorTypeAndShapeInfo ttsh = output.GetTensorTypeAndShape();
//ttsh.WriteInfo("output");

        //AnalizeResult_GetText(output.GetTensorDataAsTensorSpan<float>());

        //CleanupInferenceOrtModel();
    }

    public void MainTensorRt_RunWithBinding()
    {
        //_InferenceBackend = InferenceBackend.TensorRt;

        //SetupInferenceOrtModel();

        //_Session.PrintInferenceSessionInfo();

        //using Mat resized = CutMatFromFile();

        //using OrtValue inputOrtValue = OrtValue.CreateAllocatedTensorValue(
        //    OrtAllocator.DefaultInstance,
        //    TensorElementType.Float,
        //    _InputOrtValueShape);

        //using OrtValue outputOrtValue = OrtValue.CreateAllocatedTensorValue(
        //    OrtAllocator.DefaultInstance,
        //    TensorElementType.Float,
        //    _OutputOrtValueShape);



        //Span<float> buffer = inputOrtValue.GetTensorMutableDataAsSpan<float>();
        //var _images = Enumerable.Range(1, batch).Select(_ => resized.Clone()).ToList();

        //OrtValueHelpers ortValueHelper = new OrtValueHelpers();
        //ortValueHelper.Symmetric_ReorderNormPtr(_images, buffer, batch);

        //using var binding = _Session.CreateIoBinding();
        //binding.BindInput(_ModelInputName, inputOrtValue);
        //binding.BindOutput(_ModelOutputName, outputOrtValue);

        //_Session.RunWithBinding(new RunOptions(), binding);

        //AnalizeResult_GetText(outputOrtValue.GetTensorDataAsTensorSpan<float>());

        //CleanupInferenceOrtModel();
    }

    public void MainTensorRt_RunWithBinding_Context()
    {
        //_InferenceBackend = InferenceBackend.TensorRt;

        //SetupInferenceOrtModel();

        //_Session.PrintInferenceSessionInfo();

        //using Mat resized = CutMatFromFile();

        //var _images = Enumerable.Range(1, _OnnxPaddleOCRRecModel.Batch).Select(_ => resized.Clone()).ToList();

        //OrtValueHelpers ortValueHelper = new OrtValueHelpers();
        //ortValueHelper.Symmetric_ReorderNormPtr(_images,
        //    _OnnxPaddleOCRRecModel.InputOrtValue.GetTensorMutableDataAsSpan<float>(),
        //    _OnnxPaddleOCRRecModel.Batch);


        //var result = _OnnxPaddleOCRRecModel.RunInferenceAndAnalyse();

        //foreach(var res in result)
        //    Console.WriteLine($"{res}");

        //CleanupInferenceOrtModel();
    }


    private void AnalizeResult_GetText(ReadOnlyTensorSpan<float> tensorSpan3D, bool throwException = false)
    {
        nint resultsBatchCount = tensorSpan3D.Lengths[0];
        for(int outBatch = 0; outBatch < resultsBatchCount; outBatch++)
        {
            var tensorSpan3D_batch1 = tensorSpan3D[outBatch..(outBatch + 1), .., ..];
            var tensorSpan2D = tensorSpan3D_batch1.Squeeze();

            nint posCount = tensorSpan2D.Lengths[0];
            nint symCount = tensorSpan2D.Lengths[1];

            StringBuilder resStandard = new StringBuilder();
            StringBuilder resSpaceBias = new StringBuilder();

            int lastIdxStd = -1;
            int lastIdxBias = -1;

            const int BlankIdx = 0;
            int SpaceIdx = _Alphabet.Length - 1; // Твой индекс пробела
            const float SpaceThreshold = 0.1f; //0.15f; // Порог, при котором мы верим в пробел

            for(int pos = 0; pos < posCount; pos++)
            {
                int bestStd = 0;
                float maxConfStd = 0;

                int bestBias = 0;
                float maxConfBias = 0;

                for(int sym = 0; sym < symCount; sym++)
                {
                    float conf = tensorSpan2D[pos, sym];

                    // 1. Стандартный поиск максимума
                    if(conf > maxConfStd)
                    {
                        maxConfStd = conf;
                        bestStd = sym;
                    }

                    // 2. Логика с приоритетом пробела
                    // Если это пробел и он выше порога — даем ему шанс
                    if(sym == SpaceIdx && conf > SpaceThreshold)
                    {
                        // Если у пробела есть хоть какая-то значимая вероятность, 
                        // и текущий лидер — это "пустота" (0), принудительно берем пробел
                        bestBias = SpaceIdx;
                        maxConfBias = conf;
                    }
                    else if(conf > maxConfBias)
                    {
                        // В остальном ищем максимум как обычно
                        maxConfBias = conf;
                        bestBias = sym;
                    }
                }

                // Если в Bias логике пробел все еще не побил максимум, а максимум — не 0,
                // то фиксируем стандартного лидера
                if(bestBias == 0 && maxConfStd > maxConfBias) bestBias = bestStd;

                // --- CTC Декодирование для обоих вариантов ---

                // Вариант 1: Standard
                if(bestStd != BlankIdx && bestStd != lastIdxStd)
                    resStandard.Append(_Alphabet[bestStd]);
                lastIdxStd = bestStd;

                // Вариант 2: Space Bias
                if(bestBias != BlankIdx && bestBias != lastIdxBias)
                    resSpaceBias.Append(_Alphabet[bestBias]);
                lastIdxBias = bestBias;
            }
            if(throwException)
            {
                if(resStandard.ToString() != "5-6550KG")
                    throw new Exception($"Не работает как надо. resStandard = '{resStandard}' должно быть '5-6550KG'");
            }
            else
            {
                Console.WriteLine($"--- Batch {outBatch} ---");
                Console.WriteLine($"Standard:   [{resStandard}]");
                Console.WriteLine($"Space Bias: [{resSpaceBias}]");
            }
        }
    }



    private void AnalizeResult_PrintData(ReadOnlyTensorSpan<float> tensorSpan3D, bool throwException = false)
    {
        //Span<XChar> boxesStack = stackalloc XChar[300];

        nint resultsBatchCount = tensorSpan3D.Lengths[0];
        for(int outBatch = 0; outBatch < resultsBatchCount; outBatch++)
        {
            var tensorSpan3D_batch1 = tensorSpan3D[outBatch..(outBatch + 1), .., ..];
            var tensorSpan2D = tensorSpan3D_batch1.Squeeze();

            int boxesCount = 0;

            // Цикл по позициям
            nint posCount = tensorSpan2D.Lengths[0];
            nint symCount = tensorSpan2D.Lengths[1];

            for(int pos = 0; pos < posCount; pos++)
            {
                int indexMaxConf = 0;
                float maxConf = 0;
                string str = "";
                for(int sym = 0; sym < symCount; sym++)
                {
                    float conf = (float)tensorSpan2D[pos, sym];
                    if(conf < 0.1)
                        continue;

                    if(conf > maxConf)
                    {
                        maxConf = conf;
                        indexMaxConf = sym;
                    }

                    str += $"{sym,-3}:{conf:F3} ";
                }

                var ch = _Alphabet[indexMaxConf];
                if(!throwException)
                    Console.WriteLine($"Pos:{pos,-3}  | {str,-40} {ch}");

            }

            //if(throwException)
            //{
            //    if(boxesCount != 4)
            //        throw new Exception($"Не работает как надо. Смотри подготовку. Count = {boxesCount}");
            //}
            //else
            //    Console.WriteLine($"Count+: {boxesCount}");


        }
    }
    #endregion


    private static Mat CutMatFromFile()
    {
        string filePath = "Images/frame_000046.png";
        using Mat src = new Mat(filePath, ImreadModes.Color);

        // 2. Вырезаем прямоугольник (ROI)
        var cropRect = new Rect(420, 300, 320, 100); // Пример: x=50, y=50, ширина=200, высота=100
        using Mat cropped = new Mat(src, cropRect);

        // 3. Показываем вырезанный кусок на экране
        Cv2.ImShow("Cropped Image", cropped);
        Cv2.WaitKey(1);

        Mat resized = new Mat();
        Cv2.Resize(cropped, resized, new Size(320, 48), interpolation: InterpolationFlags.Linear);

        Cv2.ImShow("Resized 48x320", resized);
        Cv2.WaitKey(1);

        return resized;
    }


}

