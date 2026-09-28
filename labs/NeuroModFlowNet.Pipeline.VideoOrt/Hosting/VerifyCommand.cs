using NeuroModFlowNet.Pipeline.Video.Nvdec;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline.ONNX;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.VideoOrt;

/// <summary>
/// EN: Correctness check of the whole chain NVDEC → zero-copy OrtValue → ONNX NV12→BGR → download, frame by frame against
/// FFmpeg/OpenCV CPU decoding of the same file. Reports per-frame mean/max absolute difference and writes side-by-side
/// PNGs of the first and the worst frame.
///
/// RU: Проверка корректности всей цепочки NVDEC → zero-copy OrtValue → ONNX NV12→BGR → выгрузка, покадрово против
/// CPU-декода того же файла через FFmpeg/OpenCV. Печатает среднюю/максимальную разницу по кадрам и пишет PNG
/// «рядом» для первого и худшего кадра.
/// </summary>
internal static class VerifyCommand
{
    public static async Task<int> RunAsync(VideoOrtOptions options)
    {
        string file = options.Sources.Single();
        using var decoder = new NvdecVideoDecoder(file, loop: false, useFence: options.UseFence);
        Console.WriteLine($"verify {file}: {decoder.CodecName} {decoder.Width}x{decoder.Height} @ {decoder.Fps:F1} fps, backend {options.Backend}, fence {(options.UseFence ? "on" : "off")}");

        VmProgram program = new VmProgramBuilder("video.verify")
            .Step(new Op_Onnx_Nv12_To_BgrU8Nhwc(VideoOrtKeys.Nv12, VideoOrtKeys.Bgr, decoder.Width, decoder.Height, options.Matrix, isFinal: true, executionBackend: options.Backend))
            .Step(new Copy_OrtValue_To_MatImage(VideoOrtKeys.Bgr, VideoOrtKeys.Mat))
            .Build();

        await using var controller = new VmController(
            new VmControllerOptions("verify", MaxInFlight: 1, OutputKeys: [VideoOrtKeys.Mat], CaptureVariablesInTrace: false),
            program);

        using var reference = new VideoCapture(file, VideoCaptureAPIs.FFMPEG);
        if(!reference.IsOpened())
            throw new InvalidOperationException($"OpenCV cannot open '{file}'.");

        Directory.CreateDirectory(options.OutputDirectory);
        using var referenceFrame = new Mat();
        double worstMean = -1;
        double sumMean = 0;
        int compared = 0;
        double maxOverall = 0;

        for(int index = 0; index < options.Frames; index++)
        {
            NvdecFrame? frame = decoder.ReadNext();
            if(frame is null || !reference.Read(referenceFrame) || referenceFrame.Empty())
            {
                frame?.Dispose();
                break;
            }

            var inputs = new VmRunInputs()
                .Add(VideoOrtKeys.Frame, frame, disposeWithContext: true)
                .Add(VideoOrtKeys.Nv12, frame.CreateNv12Tensor(), disposeWithContext: true);

            if(!controller.TryStartRun(inputs, out VmRunHandle handle, sourceFrameId: index))
                throw new InvalidOperationException("Controller rejected the run.");

            VmRunOutcome outcome = await handle.Completion.ConfigureAwait(false);
            if(outcome.Status != VmRunStatus.Completed)
                throw new InvalidOperationException($"Run {index} failed.", outcome.Exception);

            using Mat gpu = outcome.Output!.Get<Mat>(VideoOrtKeys.Mat);
            using var difference = new Mat();
            Cv2.Absdiff(gpu, referenceFrame, difference);
            double mean = Cv2.Mean(difference).Val0 + Cv2.Mean(difference).Val1 + Cv2.Mean(difference).Val2;
            mean /= 3.0;
            Cv2.MinMaxLoc(difference.Reshape(1), out _, out double max);

            sumMean += mean;
            compared++;
            maxOverall = Math.Max(maxOverall, max);

            if(index == 0 || mean > worstMean)
            {
                if(index != 0)
                    worstMean = mean;
                else
                    worstMean = Math.Max(worstMean, mean);

                SaveComparison(options.OutputDirectory, index == 0 ? "first" : "worst", index, gpu, referenceFrame, difference);
            }

            if(index < 5 || index % 50 == 0)
                Console.WriteLine($"frame {index,4}: mean |diff| {mean,6:F3}  max {max,3:F0}");
        }

        double average = compared == 0 ? double.NaN : sumMean / compared;
        Console.WriteLine($"compared {compared} frames: average mean |diff| {average:F3}, worst frame mean {worstMean:F3}, max pixel diff {maxOverall:F0}");
        Console.WriteLine($"artifacts: {options.OutputDirectory}");

        // Chroma is upsampled nearest here and by swscale filters on the CPU side: a small mean difference on edges is
        // expected. A torn/stale frame (missing fence) shows up as a large mean difference.
        bool passed = compared > 0 && average < options.MaxMeanDifference;
        Console.WriteLine(passed ? "VERIFY PASSED" : $"VERIFY FAILED (threshold {options.MaxMeanDifference})");
        return passed ? 0 : 2;
    }

    static void SaveComparison(string directory, string label, int index, Mat gpu, Mat reference, Mat difference)
    {
        using var amplified = new Mat();
        difference.ConvertTo(amplified, MatType.CV_8UC3, 8.0);
        using var row = new Mat();
        Cv2.HConcat([gpu, reference, amplified], row);
        using var small = new Mat();
        Cv2.Resize(row, small, new Size(row.Width / 4, row.Height / 4), interpolation: InterpolationFlags.Area);
        Cv2.ImWrite(Path.Combine(directory, $"{label}_frame{index:D5}_gpu_cpu_diffx8.png"), small);
    }
}
