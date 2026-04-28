using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.ONNX.Diagnostics;

using NeuroModFlowNet.ONNX.Demo.Assets;
using NeuroModFlowNet.ONNX.Visualizer;
using OpenCvSharp;
using Spectre.Console;


namespace OnnxTestLoader;

/// <summary>
/// https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#cuda-12x
/// </summary>
public class RealTimeView2 : IDisposable
{
    bool showSeg = false;

    readonly RealTimeViewSettings _settings = RealTimeViewSettings.FromConfig();


    public void Dispose()
    {
        //throw new NotImplementedException();
    }

    public async void ForAI()
    {
        string modelBox_Path = await AssetsManager.GetAssetPathAsync(ModelNaming.GetFileName("yolo26s", _settings.InputSize, 1, _settings.ModelPrecision, isByteBgr: _settings.IsByteBgr));
        string modelObb_Path = await AssetsManager.GetAssetPathAsync(ModelNaming.GetFileName("yolo26s-obb", _settings.InputSize, 1, _settings.ModelPrecision, isByteBgr: _settings.IsByteBgr));

        var modelBox_ = new OnnxRuntimeContext(modelBox_Path);
        var modelObb_ = new OnnxRuntimeContext(modelObb_Path);

        var runnerBox_ = YoloBoxFactory.CreateRunner<IDetectionResult<YoloBox>>(modelBox_);
        var runnerObb_ = YoloObbFactory.CreateRunner<IDetectionResult<YoloObb>>(modelObb_);
    }

    public async void Run()
    {
        string modelBox_Path = await AssetsManager.GetAssetPathAsync(ModelNaming.GetFileName("yolo26s", _settings.InputSize, 1, _settings.ModelPrecision, isByteBgr: _settings.IsByteBgr));
        //string modelObb_Path = await AssetsManager.GetAssetPathAsync(ModelNaming.GetFileName("yolo26s-obb", _settings.InputSize, 1, _settings.ModelPrecision, isByteBgr: _settings.IsByteBgr));
        string modelObb_Path = await AssetsManager.GetAssetPathAsync(ModelNaming.GetFileName("img-text-to-obb", _settings.InputSize, 1, _settings.ModelPrecision, isByteBgr: _settings.IsByteBgr));
        string modelPosePath = await AssetsManager.GetAssetPathAsync(ModelNaming.GetFileName("yolo26s-pose", _settings.InputSize, 1, _settings.ModelPrecision, isByteBgr: _settings.IsByteBgr));
        string modelSeg_Path = await AssetsManager.GetAssetPathAsync(ModelNaming.GetFileName("yolo26s-seg", _settings.InputSize, 1, _settings.ModelPrecision, isByteBgr: _settings.IsByteBgr));

        string modelDv3_Path = await AssetsManager.GetAssetPathAsync("/paddleocr/detection/v3/det.onnx");
        string modelDv5_Path = await AssetsManager.GetAssetPathAsync("/paddleocr/detection/v5/det.onnx");



        var modelBox_ = new OnnxRuntimeContext(modelBox_Path, _settings.InferenceBackend);
        var modelObb_ = new OnnxRuntimeContext(modelObb_Path, _settings.InferenceBackend);
        var modelPose = new OnnxRuntimeContext(modelPosePath, _settings.InferenceBackend);
        var modelSeg_ = new OnnxRuntimeContext(modelSeg_Path, _settings.InferenceBackend);
        var modelDv3_ = new OnnxRuntimeContext(modelDv3_Path, _settings.InferenceBackend);
        var modelDv5_ = new OnnxRuntimeContext(modelDv5_Path, _settings.InferenceBackend);

        //modelDv3_

        modelBox_.WriteInfo();
        modelObb_.WriteInfo();
        modelPose.WriteInfo();
        modelSeg_.WriteInfo();
        modelDv3_.WriteInfo();
        modelDv5_.WriteInfo();


        IRunner<Mat, IDetectionResult<YoloBox>> runnerBox_ = YoloBoxFactory.CreateRunner<IDetectionResult<YoloBox>>(modelBox_);
        IRunner<Mat, IDetectionResult<YoloObb>> runnerObb_ = YoloObbFactory.CreateRunner<IDetectionResult<YoloObb>>(modelObb_);
        //IRunner<Mat, IDetectionResult<YoloPose_FP32_Size57_Keypoint17>> runnerPose = YoloPoseFactory.CreateRunner17(modelPose);
        IRunner<Mat, IDetectionResult<YoloPose>> runnerPose = YoloPoseFactory.CreateRunner(modelPose);
        //IRunner<Mat, YoloSegResult_FP16_Mask32> runnerSeg_ = (IRunner<Mat, YoloSegResult_FP16_Mask32>)YoloSegFactory.CreateRunner(modelSeg_);

        runnerObb_.OutAs<IExtractorThreshold>()?.Threshold = 0.3f;


        var runnerDv3 = PaddleOCRDetFactory.CreateRunner<Mat, Mat>(modelDv3_, MatType.CV_8UC1);
        var runnerDv5 = PaddleOCRDetFactory.CreateRunner<Mat, Mat>(modelDv5_, MatType.CV_8UC1);



        // ── Video capture ────────────────────────────────────────────
        using var capture = VideoCaptureConfig.CreateFromConfig();

        if(!capture.IsOpened())
            AnsiConsole.MarkupLine("[yellow]Warning: Camera not found — simulation (black frame) mode.[/]");

        using var window = new Window("AI DASHBOARD [NeuroModFlowNet.ONNX]");
        using var windowPose = new Window("AI DASHBOARD [NeuroModFlowNet.ONNX]");

        int modelInputW = _settings.InputSize, modelInputH = _settings.InputSize;
        int windowOutW = 640;

        using var sourceMat = new Mat();
        using Mat bgraMat = new Mat();

        long lastTick = System.Diagnostics.Stopwatch.GetTimestamp();
        var fpsMonitor = new FpsConsoleMonitor();

        AnsiConsole.Live(fpsMonitor.Render())
            .AutoClear(false)
            .Start(liveContext =>
        {
            while(true)
            {
            // ── Capture ──────────────────────────────────────────────
            capture.Read(sourceMat);
            if(sourceMat.Empty())
            {
                using var dummy = new Mat(720, 1280, MatType.CV_8UC3, Scalar.Black);
                Cv2.PutText(dummy, "NO SIGNAL", new Point(480, 360),
                                    HersheyFonts.HersheySimplex, 1.5, new Scalar(60, 60, 60), 2);
                dummy.CopyTo(sourceMat);
            }

            // ── Input Handling ──────────────────────────────────────────────
            int key = Cv2.WaitKey(1);
            if(key == 27 || key == 'q' || key == 'Q') break;

            // Изменение ширины окна по нажатию + и -
            if(key == '+' || key == '=') windowOutW = Math.Min(windowOutW + 40, 1920);
            if(key == '-' || key == '_') windowOutW = Math.Max(windowOutW - 40, 320);


            // ── Resize ──────────────────────────────────────────────────────
            // Приводим размер картинки пропорционально к ширине окна windowOutW
            double aspectRatio = (double)sourceMat.Height / sourceMat.Width;
            int targetHeight = (int)(windowOutW * aspectRatio);
            using var resizedMat = new Mat();
            Cv2.Resize(sourceMat, resizedMat, new OpenCvSharp.Size(windowOutW, targetHeight));

            // ── Pre-process (shared letterbox) ────────────────────────
            using var letterboxed = resizedMat.Letterbox(modelInputW, modelInputH, out var info);


            if(!modelDv3_.IsInputPersistentValueInitialized(modelDv3_.PrimaryInputName))
            {
                modelDv3_.InitInputPersistentValue(modelDv3_.PrimaryInputName, [1, 3, letterboxed.Width, letterboxed.Height]);
                modelDv3_.InitOutputPersistentValue(modelDv3_.PrimaryOutputName, [1, 1, letterboxed.Width, letterboxed.Height]);
            }

            if(!modelDv5_.IsInputPersistentValueInitialized(modelDv5_.PrimaryInputName))
            {
                modelDv5_.InitInputPersistentValue(modelDv5_.PrimaryInputName, [1, 3, letterboxed.Width, letterboxed.Height]);
                modelDv5_.InitOutputPersistentValue(modelDv5_.PrimaryOutputName, [1, 1, letterboxed.Width, letterboxed.Height]);
            }

            // ── Inference ────────────────────────────
            IDetectionResult<YoloBox> resultBox_ = runnerBox_.Predict(letterboxed);
            IDetectionResult<YoloObb> resultObb_ = runnerObb_.Predict(letterboxed);
            IDetectionResult<YoloPose> resultPose = runnerPose.Predict(letterboxed);
            Mat resultDv3 = runnerDv3.Predict(letterboxed);
            Mat resultDv5 = runnerDv5.Predict(letterboxed);

            YoloSegResult_FP16_Mask32 resultSeg_ = default!;


            Cv2.CvtColor(resizedMat, bgraMat, ColorConversionCodes.BGR2BGRA);

            BoxPainter.DrawBoxSkia(bgraMat, resultBox_.GetBatch(0).ToArray(), info, 1f, 1f, modelBox_.GetYoloClassName);
            ObbPainter.DrawObb(bgraMat, resultObb_.GetBatch(0).ToArray(), info, 1f, 1f, nameResolver: modelObb_.GetYoloClassName);

            YoloPosePainter.DrawPose(bgraMat, resultPose.GetBatch(0).ToArray(), info, 1f, 1f, modelPose.GetYoloClassName);

            if(showSeg)
            {
                // resultSeg_ = runnerSeg_.Predict(letterboxed);
                //
                // YoloSeg_FP16_XYWHSC_Mask32[] dets = resultSeg_.Values;
                // Half[][] masks = resultSeg_.Masks;
                // //SegPainter.DrawSeg(bgraMat, dets, masks, info, 1f, 1f, modelSeg_.GetYoloClassName);
            }

            //var tt = dets.Length > 0 ? dets[0].ToString() : "";

            window.ShowImage(bgraMat);

            Cv2.ImShow("resultDv3", resultDv3);
            Cv2.ImShow("resultDv5", resultDv5);


            //Cv2.ImShow("letterboxed", letterboxed);

            long elapsedTicks = System.Diagnostics.Stopwatch.GetTimestamp() - lastTick;
            double fps = System.Diagnostics.Stopwatch.Frequency / (double)elapsedTicks;
            lastTick = System.Diagnostics.Stopwatch.GetTimestamp();

            fpsMonitor.AddFrame(fps);
            if(fpsMonitor.ShouldRender)
            {
                liveContext.UpdateTarget(fpsMonitor.Render());
                liveContext.Refresh();
                fpsMonitor.RestartRenderTimer();
            }

            //windowPose.ShowImage(bgraMat);

            (resultBox_ as IDisposable)?.Dispose();
            (resultObb_ as IDisposable)?.Dispose();
            (resultPose as IDisposable)?.Dispose();
            (resultSeg_ as IDisposable)?.Dispose();
        }
        });

    }

}
