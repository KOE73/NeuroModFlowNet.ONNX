using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX.Diagnostics;
using OpenCvSharp;
using System.Diagnostics;
using System.Numerics.Tensors;
using System.Text;


namespace NeuroModFlowNet.ONNX.Bench;

public class BenchmarkConverters : BenchmarkBase
{
    const long trt_max_workspace_size_Gb = 4L;


    OnnxYoloObbNmsModel YoloImageObbNms = default!;
    OnnxYoloObbNmsByteBgrModel OnnxYoloObbNmsByteBgr = default!;

    AdapterBuilder<ConvertersPositiveNormalized.ToFloat, OnnxYoloObbNmsModel, List<Mat>, float> floatAdapter = default!;
    InputAdapter<OnnxYoloObbNmsModel, List<Mat>> floatAdapterInput = default!;

    AdapterBuilder<ConvertersPositiveNormalized.ToFloat16, OnnxYoloObbNmsModel, List<Mat>, Float16> Float16Adapter = default!;
    InputAdapter<OnnxYoloObbNmsModel, List<Mat>> Float16AdapterInput = default!;

    InferenceRunner<Mat, Dictionary<int, OBB_single_XYWHSCA[]>> SingleMatRunner = default!;
    InferenceRunner<List<Mat>, Dictionary<int, OBB_single_XYWHSCA[]>> BatchMatRunner = default!;
    InferenceRunner<List<Mat>, Dictionary<int, OBB_single_XYWHSCA[]>> BatchMatRunner_Best = default!;
    //InferenceRunner<List<Mat>, Dictionary<int, OBB_single_XYWHSCA[]>> Batch4atRunner = default!;


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



    private static void EPConfig(ExecutionProviderConfig epConfig)
    {
        switch(epConfig)
        {
            case CudaConfig cudaConfig:
                //cudaConfig.MaxWorkspaceSizeGb = 4;
                break;
            case TrtConfig trtConfig:
                trtConfig.MaxWorkspaceSizeGb = 4;
                trtConfig.EnableFp16 = true;
                trtConfig.EnableBf16 = true;
                trtConfig.EnableSparsity = true;
                trtConfig.EnableEngineCache = true;
                trtConfig.EngineCachePath = @"C:\TRT_Cache";
                trtConfig.BuilderOptimizationLevel = 2;
                break;
        }
    }
    private static void EPConfig2(ExecutionProviderConfig epConfig)
    {
        switch(epConfig)
        {
            case CudaConfig cudaConfig:
                //cudaConfig.MaxWorkspaceSizeGb = 4;
                break;
            case TrtConfig trtConfig:
                trtConfig.MaxWorkspaceSizeGb = 4;
                trtConfig.EnableFp16 = true;
                trtConfig.EnableBf16 = true;
                trtConfig.EnableSparsity = true;
                trtConfig.EnableEngineCache = true;
                trtConfig.EngineCachePath = @"C:\TRT_Cache";
                trtConfig.BuilderOptimizationLevel = 2;

                // Оптимизируем под батч 8
                trtConfig.ProfileMinShapes = "pixel_sequence:1x1x512x512x3,pixel_values:1x512x512x3";
                trtConfig.ProfileMaxShapes = "pixel_sequence:16x1x512x512x3,pixel_values:16x512x512x3";
                trtConfig.ProfileOptShapes = "pixel_sequence:8x1x512x512x3,pixel_values:8x512x512x3";

                break;
        }
    }


    [Params(false, true)] public bool FixedModel = true;
    bool DynamicModel => !FixedModel;

    //[Params(false, true)] public bool EnableFp16 = true;
    //[Params(false, true)] public bool EnableBf16 = true;

    #region Standard Float

    [GlobalSetup(Targets = [nameof(Inference_StdFloat)])]
    public void Setup_StdFloat()
    {
        SetupImage();

        string fileName = DynamicModel ? "26n_imgtext2obb_512_dyn.onnx" :
                            Batch == 1 ? "26n_imgtext2obb_512_b1.onnx" :
                                         "26n_imgtext2obb_512_b4.onnx";

        string fullPath = Path.Combine(ModelRootDir, fileName);

        YoloImageObbNms = new(fullPath, _InferenceBackend, EPConfig);
        YoloImageObbNms.InitSimple([Batch, 3, 512, 512], [Batch, 300, 7]);
YoloImageObbNms.WriteInfo();

        floatAdapter = new();

        BatchMatRunner = YoloImageObbNms.MakeBatchBlobConverterFloatRunner(Batch);

    }



    [GlobalCleanup(Targets = [nameof(Inference_StdFloat),])]
    public void Cleanup_StdFloat()
    {
        CleanupImage();
        YoloImageObbNms.Dispose();
    }

    [Benchmark(/*OperationsPerInvoke = 1*/), BenchmarkCategory("Image2RBox")]
    public void Inference_StdFloat()
    {
        var result = BatchMatRunner(Batch == 1 ? _List_Text_1_Image : _List_Text_4_Image, Batch);

        //var result = YoloImageObbNms.PredictAndResult(
        //    (InAdapter: floatAdapterInput,
        //      Model: YoloImageObbNms,
        //      Images: Batch == 1 ? _List_Text_1_Image : _List_Text_4_Image,
        //      Batch: Batch),
        //    // Статическая лямбда: нет захвата переменных (запрещено ключевым словом static)
        //    static state => state.InAdapter(state.Model, state.Images, state.Batch),
        //    static state => ExtractorOBBNMS.GetOutputAsDictionary(state.Model, 0.5f)
        //    );


        //var result = batch == 1 ?
        //    YoloImageObbNms.PredictAndResult(
        //        (Model: YoloImageObbNms, Image: _Image_Text1),
        //        static state => Adatpers.PrepareInputFromMatAsFloat(state.Model, state.Image),
        //        static state => ExtractorOBBNMS.GetOutputAsDictionary(state.Model, 0.5f)) :
        //    YoloImageObbNms.PredictAndResult(
        //        (Model: YoloImageObbNms, Images: _List_Text_Images),
        //        static state => Adatpers.PrepareInputFromMatAsFloat(state.Model, state.Images),
        //        static state => ExtractorOBBNMS.GetOutputAsDictionary(state.Model, 0.5f));

        if(BenchDebug)
            for(int b = 0; b < Batch; b++)
                Cv2.ImShow($"Inference_StdFloat {b}", DrawOnMat(_List_Text_4_Image[b], result[b], Scalar.Green));
    }

    #endregion


    #region Standard FP16

    [GlobalSetup(Targets = [
        nameof(Inference_StdFP16_OpenCV_Blob),
        nameof(Inference_StdFP16_ReorderDivPtr_HalfTensor)
        ])]
    public void Setup_StdFP16()
    {
        SetupImage();

        string fileName = DynamicModel ? "26n_imgtext2obb_512_dyn_fp16.onnx" :
                            Batch == 1 ? "26n_imgtext2obb_512_b1_fp16.onnx" :
                                         "26n_imgtext2obb_512_b4_fp16.onnx";

        string fullPath = Path.Combine(ModelRootDir, fileName);

        YoloImageObbNms = new(fullPath, _InferenceBackend, EPConfig);
        YoloImageObbNms.InitSimple([Batch, 3, 512, 512], [Batch, 300, 7]);
YoloImageObbNms.WriteInfo();

        Float16Adapter = new();

        BatchMatRunner = YoloImageObbNms.MakeBatchBlobConverterFP16Runner(Batch);
        BatchMatRunner_Best = YoloImageObbNms.MakeBatchBestConverterFP16Runner(Batch);
    }


    [GlobalCleanup(Targets = [
        nameof(Inference_StdFP16_OpenCV_Blob),
        nameof(Inference_StdFP16_ReorderDivPtr_HalfTensor),
        ])]
    public void Cleanup_StdFP16()
    {
        CleanupImage();
        YoloImageObbNms.Dispose();
    }

    [Benchmark(), BenchmarkCategory("Image2RBox")]
    public void Inference_StdFP16_OpenCV_Blob()
    {
        var result = BatchMatRunner(Batch == 1 ? _List_Text_1_Image : _List_Text_4_Image, Batch);

        //Float16AdapterInput ??= Float16Adapter
        //    .WithConverter(o => o.Positive_OpenCV_Blob)
        //    //.WithAutoTuneConverter(_List_Text_4_Image, Batch)
        //    .Build(removed AdapterForT);

        //var result = YoloImageObbNms.PredictAndResult(
        //    (InAdapter: Float16AdapterInput,
        //      Model: YoloImageObbNms,
        //      Images: Batch == 1 ? _List_Text_1_Image : _List_Text_4_Image,
        //      Batch: Batch),
        //    // Статическая лямбда: нет захвата переменных (запрещено ключевым словом static)
        //    static state => state.InAdapter(state.Model, state.Images, state.Batch),
        //    static state => ExtractorOBBNMS.GetOutputAsDictionary(state.Model, 0.5f)
        //    );

        //var result = batch == 1 ?
        //    YoloImageObbNms.PredictAndResult(
        //        (Model: YoloImageObbNms, Image: _Image_Text1),
        //        static state => Adatpers.PrepareInputFromMatAsFloat16(state.Model, state.Image),
        //        static state => ExtractorOBBNMS.GetOutputAsDictionary(state.Model, 0.5f)) :
        //    YoloImageObbNms.PredictAndResult(
        //        (Model: YoloImageObbNms, Images: _List_Text_4_Image),
        //        static state => Adatpers.PrepareInputFromMatAsFloat16(state.Model, state.Images),
        //        static state => ExtractorOBBNMS.GetOutputAsDictionary(state.Model, 0.5f));

        if(BenchDebug)
            for(int b = 0; b < Batch; b++)
                Cv2.ImShow($"Inference_StdFP16_OpenCV_Blob {b}", DrawOnMat(_List_Text_4_Image[b], result[b], Scalar.Green));
    }

    [Benchmark(), BenchmarkCategory("Image2RBox")]
    public void Inference_StdFP16_ReorderDivPtr_HalfTensor()
    {
        //Float16AdapterInput ??= Float16Adapter
        //    .WithConverter(o => o.Positive_ReorderDivPtr_HalfTensor)
        //    .Build(removed AdapterForT);

        //var result = YoloImageObbNms.PredictAndResult(
        //    (InAdapter: Float16AdapterInput,
        //      Model: YoloImageObbNms,
        //      Images: Batch == 1 ? _List_Text_1_Image : _List_Text_4_Image,
        //      Batch: Batch),
        //    static state => state.InAdapter(state.Model, state.Images, state.Batch),
        //    static state => ExtractorOBBNMS.GetOutputAsDictionary(state.Model, 0.5f)
        //    );

        var result = BatchMatRunner_Best(Batch == 1 ? _List_Text_1_Image : _List_Text_4_Image, Batch);


        if(BenchDebug)
            for(int b = 0; b < Batch; b++)
                Cv2.ImShow($"Inference_StdFP16_ReorderDivPtr_HalfTensor {b}", DrawOnMat(_List_Text_4_Image[b], result[b], Scalar.Green));
    }

    #endregion


    #region Mod model

    //[GlobalSetup(Targets = [nameof(Inference_ModHead),])]
    //public void SetupInference_ModHead()
    //{
    //    SetupImage();

    //    string fileName = DynamicModel ? "26n_imgtext2obb_512_dyn_fp16_head.onnx" :
    //                    Batch == 1 ? "26n_imgtext2obb_512_b1_fp16_head.onnx" :
    //                                 "26n_imgtext2obb_512_b4_fp16_head.onnx";

    //    string fullPath = Path.Combine(ModelRootDir, fileName);

    //    YoloImageObbNms_mod = new(fullPath, _InferenceBackend, EPConfig);

    //    YoloImageObbNms_mod.InitSimple([Batch, 512, 512, 3], [Batch, 300, 7]);
//    YoloImageObbNms_mod.WriteInfo();

    //    var MyYoloRunner = InferenceBuilderT<OnnxYoloObbNmsModel, Mat, byte, Dictionary<int, OBB_single_XYWHSCA[]>>
    //        .Create(YoloImageObbNms_mod, Batch)
    //        .UsingAdapter(LabMatInputAdapters.BgrDirectU8)
    //        .WithExtractor(ExtractorOBBNMS.GetOutputAsDictionary, threshold: 0.5f)
    //        .Compile();

    //    var aa = MyYoloRunner(_Image_Text1, Batch);

    //}

    //[GlobalCleanup(Targets = [nameof(Inference_ModHead),])]
    //public void CleanupInference_ModHead()
    //{
    //    CleanupImage();
    //    YoloImageObbNms_mod.Dispose();
    //}

    ////[Benchmark(/*OperationsPerInvoke = 1*/), BenchmarkCategory("Image2RBox")]
    //public void Inference_ModHead()
    //{
    //    var result = Batch == 1 ?
    //        YoloImageObbNms_mod.PredictAndResult(
    //            (Model: YoloImageObbNms_mod, Image: _Image_Text1),
    //            static state => LabMatInputAdapters.BgrDirectU8(state.Model, state.Image),
    //            static state => ExtractorOBBNMS.GetOutputAsDictionary(state.Model, 0.5f)) :
    //        YoloImageObbNms_mod.PredictAndResult(
    //            (Model: YoloImageObbNms_mod, Images: _List_Text_4_Image),
    //            static state => TODO ByDirectSequence(state.Model, state.Images),
    //            static state => ExtractorOBBNMS.GetOutputAsDictionary(state.Model, 0.5f));


    //    //var result = batch == 1 ?
    //    //    YoloImageObbNms_mod.PredictAndResult(
    //    //        (Model: YoloImageObbNms_mod, Image: _Image_Text1),
    //    //        static state => OnnxYoloObbNmsModel.PrepareInputFromMatBGRAsByte(state.Model, state.Image),
    //    //        static state => ExtractorOBBNMS.GetOutputAsDictionary(state.Model)) :
    //    //    YoloImageObbNms_mod.PredictAndResult(
    //    //        (Model: YoloImageObbNms_mod, Images: _List_Text_Images),
    //    //        static state => OnnxYoloObbNmsModel.PrepareInputFromMatBGRAsByte(state.Model, state.Images),
    //    //        static state => ExtractorOBBNMS.GetOutputAsDictionary(state.Model));



    //    if(BenchDebug)
    //        for(int b = 0; b < Batch; b++)
    //            Cv2.ImShow($"Inference_ModHead {b}", DrawOnMat(_List_Text_4_Image[b], result[b], Scalar.Green));
    //}

    #endregion


    #region Mod model InferenceRunner


    [GlobalSetup(Targets = [nameof(Inference_ModHead_InferenceRunner),])]
    public void SetupInference_ModHead_InferenceRunner()
    {
        SetupImage();

        string fileName = DynamicModel ? "26n_imgtext2obb_512_dyn_fp16_head.onnx" :
                Batch == 1 ? "26n_imgtext2obb_512_b1_fp16_head.onnx" :
                             "26n_imgtext2obb_512_b4_fp16_head.onnx";

        string fullPath = Path.Combine(ModelRootDir, fileName);


        OnnxYoloObbNmsByteBgr = new(fullPath, _InferenceBackend, EPConfig);
        OnnxYoloObbNmsByteBgr.InitSimple([Batch, 512, 512, 3], [Batch, 300, 7]);
OnnxYoloObbNmsByteBgr.WriteInfo();


        SingleMatRunner = OnnxYoloObbNmsByteBgr.MakeSingleBGRBytesDirectRunner();
        // !!! BatchMatRunner = OnnxYoloObbNmsByteBgr.MakeBatchBlobConverterFP16Runner(Batch);

        //SingleMatRunner = InferenceBuilder
        //    .For(YoloImageObbNms_mod, Batch)
        //    .UsingByte<Mat>(LabMatInputAdapters.BgrDirectU8)
        //    .WithExtractor(ExtractorOBBNMS.GetOutputAsDictionary, threshold: 0.5f)
        //    .Compile();
    }

    [GlobalCleanup(Targets = [nameof(Inference_ModHead_InferenceRunner),])]
    public void CleanupInference_ModHead_InferenceRunner()
    {
        CleanupImage();
        OnnxYoloObbNmsByteBgr.Dispose();
    }

    [Benchmark(), BenchmarkCategory("Image2RBox")]
    public void Inference_ModHead_InferenceRunner()
    {
        var result = Batch == 1 ? SingleMatRunner(_Image_Text1, Batch) : BatchMatRunner(_List_Text_4_Image, Batch);



        //YoloImageObbNms_mod.PredictAndResult(
        //        (Model: YoloImageObbNms_mod, Images: _List_Text_4_Image),
        //        static state => TODO ByDirectSequence(state.Model, state.Images),
        //        static state => ExtractorOBBNMS.GetOutputAsDictionary(state.Model, 0.5f));
        //var result = batch == 1 ?
        //    YoloImageObbNms_mod.PredictAndResult(
        //        (Model: YoloImageObbNms_mod, Image: _Image_Text1),
        //        static state => OnnxYoloObbNmsModel.PrepareInputFromMatBGRAsByte(state.Model, state.Image),
        //        static state => ExtractorOBBNMS.GetOutputAsDictionary(state.Model)) :
        //    YoloImageObbNms_mod.PredictAndResult(
        //        (Model: YoloImageObbNms_mod, Images: _List_Text_Images),
        //        static state => OnnxYoloObbNmsModel.PrepareInputFromMatBGRAsByte(state.Model, state.Images),
        //        static state => ExtractorOBBNMS.GetOutputAsDictionary(state.Model));



        if(BenchDebug)
            for(int b = 0; b < Batch; b++)
                Cv2.ImShow($"Inference_ModHead {b}", DrawOnMat(_List_Text_4_Image[b], result[b], Scalar.Green));
    }

    #endregion

    #region Mod Seq model

    [GlobalSetup(Targets = [nameof(Inference_ModHeadSeq),])]
    public void SetupInference_ModHeadSeq()
    {
        SetupImage();

        OnnxYoloObbNmsByteBgr = new(
            @"C:\Models\det-to-obb\26n_imgtext2obb_512_dyn_fp16_head_seg.onnx",
            _InferenceBackend,
            EPConfig2);
        OnnxYoloObbNmsByteBgr.InitSimple([Batch, 512, 512, 3], [Batch, 300, 7]);
OnnxYoloObbNmsByteBgr.WriteInfo();

    }

    [GlobalCleanup(Targets = [nameof(Inference_ModHeadSeq),])]
    public void CleanupInference_ModHeadSeq()
    {
        CleanupImage();
        OnnxYoloObbNmsByteBgr.Dispose();
    }

    //[Benchmark(/*OperationsPerInvoke = 1*/), BenchmarkCategory("Image2RBox")]
    public void Inference_ModHeadSeq()
    {
        throw new NotSupportedException("ByDirectSequence is temporarily disabled until sequence-input support is reworked.");
        //var result = Batch == 1 ?
        //    OnnxYoloObbNmsByteBgr.PredictAndResult(
        //        (Model: OnnxYoloObbNmsByteBgr, Image: _Image_Text1),
        //        static state => /* TODO ByDirectSequence */ throw new NotSupportedException(),
        //        static state => ExtractorOBBNMS.GetOutputAsDictionary(state.Model, 0.5f)) :
        //    OnnxYoloObbNmsByteBgr.PredictAndResult(
        //        (Model: OnnxYoloObbNmsByteBgr, Images: _List_Text_4_Image),
        //        static state => /* TODO ByDirectSequence */ throw new NotSupportedException(),
        //        static state => ExtractorOBBNMS.GetOutputAsDictionary(state.Model, 0.5f));
        //var result = batch == 1 ?
        //    YoloImageObbNms_mod.PredictAndResult(
        //        (Model: YoloImageObbNms_mod, Image: _Image_Text1),
        //        static state => OnnxYoloObbNmsModel.PrepareInputFromMatBGRAsByte(state.Model, state.Image),
        //        static state => ExtractorOBBNMS.GetOutputAsDictionary(state.Model)) :
        //    YoloImageObbNms_mod.PredictAndResult(
        //        (Model: YoloImageObbNms_mod, Images: _List_Text_Images),
        //        static state => OnnxYoloObbNmsModel.PrepareInputFromMatBGRAsByte(state.Model, state.Images),
        //        static state => ExtractorOBBNMS.GetOutputAsDictionary(state.Model));
        //if(BenchDebug)
        //    for(int b = 0; b < Batch; b++)
        //        Cv2.ImShow($"Inference_ModHeadSeq {b}", DrawOnMat(_List_Text_4_Image[b], result[b], Scalar.Green));
    }

    #endregion

    #region Main

    //public unsafe void MainCuda_Run()
    //{
    //    _InferenceBackend = InferenceBackend.Cuda;
    //    batch = 1;

    //    SetupInferenceOrtModel();

    //    _Session.PrintInferenceSessionInfo();

    //    int height = _image_1_512_512.Height;
    //    int width = _image_1_512_512.Width;

    //    long[] _InputOrtValueShape = [batch, 3, height, width];
    //    long[] _OutputOrtValueShape = [batch, 320 / 8 /*?*/, 438];


    //    using var _InputOrtValue = OrtValue.CreateAllocatedTensorValue(
    //           OrtAllocator.DefaultInstance,
    //           TensorElementType.Float,
    //           _InputOrtValueShape);

    //    //using var _OutputOrtValue = OrtValue.CreateAllocatedTensorValue(
    //    //       OrtAllocator.DefaultInstance,
    //    //       TensorElementType.Float,
    //    //       _OutputOrtValueShape);


    //    Span<float> buffer = _InputOrtValue.GetTensorMutableDataAsSpan<float>();

    //    OrtValueHelpers ortValueHelper = new OrtValueHelpers();
    //    ortValueHelper.Symmetric_ReorderNormPtr(_images_1_512_512, buffer, batch);

    //    using var outputs = _Session.Run(new RunOptions(), (IReadOnlyCollection<string>)[_ModelInputName], (IReadOnlyCollection<OrtValue>)[_InputOrtValue], (IReadOnlyCollection<string>)[_ModelOutputName]);

    //    OrtValue output = outputs.First();
    //    OrtTensorTypeAndShapeInfo ttsh = output.GetTensorTypeAndShape();
//    ttsh.WriteInfo("output");

    //    Span<byte> dataPtr = output.GetTensorMutableRawData();

    //    // Получаем размерности (обязательно приводим к int)
    //    var shape = output.GetTensorTypeAndShape().Shape;
    //    int rows = (int)shape[2];
    //    int cols = (int)shape[3];

    //    fixed(byte* ptr = dataPtr)
    //    {
    //        // Создаем Mat, который "смотрит" прямо в память OrtValue
    //        Mat probMap = Mat.FromPixelData(rows, cols, MatType.CV_32FC1, (IntPtr)ptr);
    //        Cv2.ImShow("floatMap", probMap);
    //        Cv2.WaitKey();


    //        //probMap.Dilate();
    //        var pp = GetContours(probMap, 1080, 1080);

    //    }


    //    //AnalizeResult_GetText(output.GetTensorDataAsTensorSpan<float>());

    //    CleanupInferenceOrtModel();
    //}




    //// tensorData — это твой выход [1, 1, 512, 512]
    //public List<Point[]> PostProcess(ReadOnlyTensorSpan<float> tensor, int originalWidth, int originalHeight)
    //{


    //    int rows = (int)tensor.Lengths[2]; // 512
    //    int cols = (int)tensor.Lengths[3]; // 512
    //    float threshold = 0.3f; // Порог уверенности для пикселя

    //    // Создаем ч/б маску (используем OpenCVSharp для поиска контуров)
    //    using Mat binaryMap = new Mat(rows, cols, MatType.CV_8UC1);
    //    using Mat binaryMap2 = new Mat(rows, cols, MatType.CV_8UC1);

    //    unsafe
    //    {
    //        byte* pBinary = (byte*)binaryMap.Data;
    //        byte* pBinary2 = (byte*)binaryMap2.Data;
    //        fixed(float* pTensor = tensor)
    //        {
    //            for(int i = 0; i < rows * cols; i++)
    //            {
    //                // Если вероятность > 0.3, ставим 255 (белый), иначе 0
    //                pBinary[i] = pTensor[i] > threshold ? (byte)255 : (byte)0;
    //                pBinary2[i] = (byte)(pTensor[i] * 255);
    //            }
    //        }
    //    }

    //    Cv2.ImShow("ss2", binaryMap2);
    //    //Cv2.ImShow("ss", binaryMap);
    //    Cv2.WaitKey(1);
    //    // 
    //    return GetContours(binaryMap, originalWidth, originalHeight);
    //}

    private List<Point[]> GetContours(Mat binaryMap, int srcW, int srcH)
    {
        Point[][] contours;
        HierarchyIndex[] hierarchy;

        // Находим все белые пятна
        Cv2.FindContours(binaryMap, out contours, out hierarchy,
                         RetrievalModes.List, ContourApproximationModes.ApproxSimple);

        List<Point[]> boxes = new();
        float unclipRatio = 1.5f; // На сколько расширить рамку (стандарт для v5)

        foreach(var contour in contours)
        {
            // Считаем площадь и периметр
            float area = (float)Cv2.ContourArea(contour);
            if(area < 16) continue; // Игнорируем шум

            // Получаем минимальный прямоугольник (может быть под углом!)
            RotatedRect rect = Cv2.MinAreaRect(contour);

            // Расширяем рамку (Unclip)
            Point2f[] points = GetUnclippedBox(rect, unclipRatio);

            // Масштабируем точки обратно под размер исходного фото мешка
            for(int i = 0; i < 4; i++)
            {
                points[i].X = Math.Clamp(points[i].X * srcW / 512f, 0, srcW);
                points[i].Y = Math.Clamp(points[i].Y * srcH / 512f, 0, srcH);
            }

            boxes.Add(points.Select(p => new Point((int)p.X, (int)p.Y)).ToArray());
        }

        return boxes;
    }

    private Point2f[] GetUnclippedBox(RotatedRect rect, float ratio)
    {
        float area = rect.Size.Width * rect.Size.Height;
        float perimeter = (rect.Size.Width + rect.Size.Height) * 2;
        // Насколько нужно отодвинуть границы
        float distance = area * ratio / perimeter;

        // В OpenCVSharp это делается через расширение сторон Rect
        // Или более простой вариант — чуть увеличить Size:
        var newSize = new Size2f(rect.Size.Width + distance, rect.Size.Height + distance);
        return Cv2.BoxPoints(new RotatedRect(rect.Center, newSize, rect.Angle));
    }

    //private void AnalizeResult_GetText(ReadOnlyTensorSpan<float> tensorSpan3D, bool throwException = false)
    //{
    //    nint resultsBatchCount = tensorSpan3D.Lengths[0];
    //    for(int outBatch = 0; outBatch < resultsBatchCount; outBatch++)
    //    {
    //        var tensorSpan3D_batch1 = tensorSpan3D[outBatch..(outBatch + 1), .., ..];
    //        var tensorSpan2D = tensorSpan3D_batch1.Squeeze();

    //        nint posCount = tensorSpan2D.Lengths[0];
    //        nint symCount = tensorSpan2D.Lengths[1];

    //        StringBuilder resStandard = new StringBuilder();
    //        StringBuilder resSpaceBias = new StringBuilder();

    //        int lastIdxStd = -1;
    //        int lastIdxBias = -1;

    //        const int BlankIdx = 0;
    //        int SpaceIdx = _Alphabet.Length - 1; // Твой индекс пробела
    //        const float SpaceThreshold = 0.1f; //0.15f; // Порог, при котором мы верим в пробел

    //        for(int pos = 0; pos < posCount; pos++)
    //        {
    //            int bestStd = 0;
    //            float maxConfStd = 0;

    //            int bestBias = 0;
    //            float maxConfBias = 0;

    //            for(int sym = 0; sym < symCount; sym++)
    //            {
    //                float conf = tensorSpan2D[pos, sym];

    //                // 1. Стандартный поиск максимума
    //                if(conf > maxConfStd)
    //                {
    //                    maxConfStd = conf;
    //                    bestStd = sym;
    //                }

    //                // 2. Логика с приоритетом пробела
    //                // Если это пробел и он выше порога — даем ему шанс
    //                if(sym == SpaceIdx && conf > SpaceThreshold)
    //                {
    //                    // Если у пробела есть хоть какая-то значимая вероятность, 
    //                    // и текущий лидер — это "пустота" (0), принудительно берем пробел
    //                    bestBias = SpaceIdx;
    //                    maxConfBias = conf;
    //                }
    //                else if(conf > maxConfBias)
    //                {
    //                    // В остальном ищем максимум как обычно
    //                    maxConfBias = conf;
    //                    bestBias = sym;
    //                }
    //            }

    //            // Если в Bias логике пробел все еще не побил максимум, а максимум — не 0,
    //            // то фиксируем стандартного лидера
    //            if(bestBias == 0 && maxConfStd > maxConfBias) bestBias = bestStd;

    //            // --- CTC Декодирование для обоих вариантов ---

    //            // Вариант 1: Standard
    //            if(bestStd != BlankIdx && bestStd != lastIdxStd)
    //                resStandard.Append(_Alphabet[bestStd]);
    //            lastIdxStd = bestStd;

    //            // Вариант 2: Space Bias
    //            if(bestBias != BlankIdx && bestBias != lastIdxBias)
    //                resSpaceBias.Append(_Alphabet[bestBias]);
    //            lastIdxBias = bestBias;
    //        }
    //        if(throwException)
    //        {
    //            if(resStandard.ToString() != "5-6550KG")
    //                throw new Exception($"Не работает как надо. resStandard = '{resStandard}' должно быть '5-6550KG'");
    //        }
    //        else
    //        {
    //            Console.WriteLine($"--- Batch {outBatch} ---");
    //            Console.WriteLine($"Standard:   [{resStandard}]");
    //            Console.WriteLine($"Space Bias: [{resSpaceBias}]");
    //        }
    //    }
    //}



    //private void AnalizeResult_PrintData(ReadOnlyTensorSpan<float> tensorSpan3D, bool throwException = false)
    //{
    //    //Span<XChar> boxesStack = stackalloc XChar[300];

    //    nint resultsBatchCount = tensorSpan3D.Lengths[0];
    //    for(int outBatch = 0; outBatch < resultsBatchCount; outBatch++)
    //    {
    //        var tensorSpan3D_batch1 = tensorSpan3D[outBatch..(outBatch + 1), .., ..];
    //        var tensorSpan2D = tensorSpan3D_batch1.Squeeze();

    //        int boxesCount = 0;

    //        // Цикл по позициям
    //        nint posCount = tensorSpan2D.Lengths[0];
    //        nint symCount = tensorSpan2D.Lengths[1];

    //        for(int pos = 0; pos < posCount; pos++)
    //        {
    //            int indexMaxConf = 0;
    //            float maxConf = 0;
    //            string str = "";
    //            for(int sym = 0; sym < symCount; sym++)
    //            {
    //                float conf = (float)tensorSpan2D[pos, sym];
    //                if(conf < 0.1)
    //                    continue;

    //                if(conf > maxConf)
    //                {
    //                    maxConf = conf;
    //                    indexMaxConf = sym;
    //                }

    //                str += $"{sym,-3}:{conf:F3} ";
    //            }

    //            var ch = _Alphabet[indexMaxConf];
    //            if(!throwException)
    //                Console.WriteLine($"Pos:{pos,-3}  | {str,-40} {ch}");

    //        }

    //        //if(throwException)
    //        //{
    //        //    if(boxesCount != 4)
    //        //        throw new Exception($"Не работает как надо. Смотри подготовку. Count = {boxesCount}");
    //        //}
    //        //else
    //        //    Console.WriteLine($"Count+: {boxesCount}");


    //    }
    //}
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

    private Mat DrawOnMat(
        Mat matIn,
        OBB_single_XYWHSCA[]? listBox,
        Scalar lineColor)
    {
        if(listBox is null)
            return matIn.Clone();

        var matOut = matIn.Clone();

        float scaleX = matOut.Width / (float)matOut.Width;
        float scaleY = matOut.Height / (float)matOut.Height;

        for(int i = 0; i < listBox.Length; i++)
        {
            var box = listBox[i];
            if(!box.IsValid(0.3f)) continue;


            float centerX = box.X * scaleX;
            float centerY = box.Y * scaleY;
            float width = box.W * scaleX;
            float height = box.H * scaleY;
            float angleDegree = (float)(box.Angle * 180.0 / Math.PI);
            //if(angleDegree > 45)
            //{
            //    angleDegree -= 90;
            //    // Меняем местами W и H, так как бокс "лег на бок"
            //    float temp = width;
            //    width = height;
            //    height = temp;
            //}
            //// Если угол слишком отрицательный (зависит от формата модели)
            //else if(angleDegree < -45)
            //{
            //    angleDegree += 90;
            //    float temp = width;
            //    width = height;
            //    height = temp;
            //}

            Cv2.Circle(matOut, new Point((int)(centerX), (int)(centerY)), 3, Scalar.Red, -1);

            // Создаем повернутый прямоугольник и Получаем 4 вершины
            var rotatedRect = new RotatedRect(new Point2f(centerX, centerY), new Size2f(width, height), angleDegree);
            var vertices = rotatedRect.Points().Select(o => o.ToPoint()).ToArray();
            Cv2.Polylines(matOut, [vertices], isClosed: true, color: lineColor, thickness: 2);
        }
        return matOut;
    }

}

