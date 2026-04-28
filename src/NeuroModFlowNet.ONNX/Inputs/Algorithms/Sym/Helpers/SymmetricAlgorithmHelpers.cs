using Microsoft.ML.OnnxRuntime;
using System.Runtime.CompilerServices;

namespace NeuroModFlowNet.ONNX.Converters.Algorithms;

public static class SymmetricAlgorithmHelpers
{
    public static bool CheckAlgorithms(Action<string>? log = null)
    {
        const int width = 128;
        const int height = 128;
        const int batch = 4;

        using Mat mat = PositiveAlgorithmHelpers.CreateFastRealMat(width, height);
        return CheckAlgorithms(mat, batch, log);
    }

    public static bool CheckAlgorithms(Mat mat, int batch, Action<string>? log = null)
    {
        List<Mat> mats = PositiveAlgorithmHelpers.CloneMatBatch(mat, batch);
        try
        {
            return
                CheckFP32Algorithms(mats, mat.Width, mat.Height, batch, log) &&
                CheckFP16Algorithms(mats, mat.Width, mat.Height, batch, log);
        }
        finally
        {
            foreach(Mat item in mats)
                item.Dispose();
        }
    }

    public static InputDataToSpanBufConverter<List<Mat>, float> AutoTuneFP32(Mat mat, int batch, Action<string>? log = null)
    {
        int sizeOne = mat.Width * mat.Height * 3;
        int pixelsCount = mat.Width * mat.Height;
        return AutoTune(
            mat,
            batch,
            log,
            ("SymCvdnnFP32", Wrap<float, SymCvdnnFP32>(sizeOne, pixelsCount)),
            ("SymReorderNormPtrFP32", Wrap<float, SymReorderNormPtrFP32>(sizeOne, pixelsCount)));
    }

    public static InputDataToSpanBufConverter<List<Mat>, Float16> AutoTuneFP16(Mat mat, int batch, Action<string>? log = null)
    {
        int sizeOne = mat.Width * mat.Height * 3;
        int pixelsCount = mat.Width * mat.Height;
        return AutoTune(
            mat,
            batch,
            log,
            ("SymCvdnnFP16", Wrap<Float16, SymCvdnnFP16>(sizeOne, pixelsCount)),
            ("SymReorderNormPtrHalfTensorFP16", Wrap<Float16, SymReorderNormPtrHalfTensorFP16>(sizeOne, pixelsCount)));
    }

    private static bool CheckFP32Algorithms(List<Mat> mats, int width, int height, int batch, Action<string>? log)
    {
        int sizeOne = width * height * 3;
        int pixelsCount = width * height;

        byte[] asBlob = Call(Wrap<float, SymCvdnnFP32>(sizeOne, pixelsCount));
        byte[] asReorderNormPtr = Call(Wrap<float, SymReorderNormPtrFP32>(sizeOne, pixelsCount));

        int compareReorderNormPtr = asBlob.SequenceCompareTo(asReorderNormPtr);
        log?.Invoke($"SymReorderNormPtrFP32 {(compareReorderNormPtr == 0 ? "OK" : "Error")}");

        return compareReorderNormPtr == 0;

        byte[] Call(InputDataToSpanBufConverter<List<Mat>, float> action)
        {
            byte[] bytes = new byte[sizeOne * batch * sizeof(float)];
            action(mats, MemoryMarshal.Cast<byte, float>(bytes.AsSpan()), batch);
            return bytes;
        }
    }

    private static bool CheckFP16Algorithms(List<Mat> mats, int width, int height, int batch, Action<string>? log)
    {
        int sizeOne = width * height * 3;
        int pixelsCount = width * height;

        byte[] asBlob = Call(Wrap<Float16, SymCvdnnFP16>(sizeOne, pixelsCount));
        byte[] asReorderNormPtrHalfTensor = Call(Wrap<Float16, SymReorderNormPtrHalfTensorFP16>(sizeOne, pixelsCount));

        int compareReorderNormPtrHalfTensor = asBlob.SequenceCompareTo(asReorderNormPtrHalfTensor);
        log?.Invoke($"SymReorderNormPtrHalfTensorFP16 {(compareReorderNormPtrHalfTensor == 0 ? "OK" : "Error")}");

        return compareReorderNormPtrHalfTensor == 0;

        byte[] Call(InputDataToSpanBufConverter<List<Mat>, Float16> action)
        {
            byte[] bytes = new byte[sizeOne * batch * sizeof(ushort)];
            action(mats, MemoryMarshal.Cast<byte, Float16>(bytes.AsSpan()), batch);
            return bytes;
        }
    }

    private static InputDataToSpanBufConverter<List<Mat>, TBuffer> AutoTune<TBuffer>(
        Mat mat,
        int batch,
        Action<string>? log,
        params (string Name, InputDataToSpanBufConverter<List<Mat>, TBuffer> Proc)[] algorithms)
        where TBuffer : unmanaged
    {
        List<Mat> mats = PositiveAlgorithmHelpers.CloneMatBatch(mat, batch);
        try
        {
            int sizeOne = mat.Width * mat.Height * 3;
            byte[] bytes = new byte[sizeOne * batch * Unsafe.SizeOf<TBuffer>()];
            Span<TBuffer> buffer = MemoryMarshal.Cast<byte, TBuffer>(bytes.AsSpan());

            string bestName = "";
            InputDataToSpanBufConverter<List<Mat>, TBuffer> bestDelegate = algorithms[0].Proc;
            double minMs = double.MaxValue;

            foreach(var algorithm in algorithms)
            {
                for(int i = 0; i < 50; i++)
                    algorithm.Proc(mats, buffer, batch);

                Stopwatch stopwatch = Stopwatch.StartNew();
                const int iterations = 50;

                for(int i = 0; i < iterations; i++)
                    algorithm.Proc(mats, buffer, batch);

                stopwatch.Stop();

                double avgMs = stopwatch.Elapsed.TotalMilliseconds / iterations;
                double fps = 1000.0 / avgMs;
                log?.Invoke($"{algorithm.Name,-40} | {avgMs,8:F5} ms | {fps,8:F0} fps");

                if(avgMs < minMs)
                {
                    minMs = avgMs;
                    bestName = algorithm.Name;
                    bestDelegate = algorithm.Proc;
                }
            }

            log?.Invoke(new string('-', 68));
            log?.Invoke($"[AutoTune] Best result: {bestName} ({1000.0 / minMs:F0} FPS) on {mat.Width}x{mat.Height} batch {batch}");

            return bestDelegate;
        }
        finally
        {
            foreach(Mat item in mats)
                item.Dispose();
        }
    }

    private static InputDataToSpanBufConverter<List<Mat>, TBuffer> Wrap<TBuffer, TAlgorithm>(
        int sizeOne,
        int pixelsCount)
        where TBuffer : unmanaged
        where TAlgorithm : struct, IMatListFillAlgorithm<TBuffer>
    {
        return (mats, buffer, batch) =>
        {
            if(mats.Count > batch)
                throw new ArgumentException("Buffer length is greater than batch size.");

            TAlgorithm.Fill(mats, buffer, mats.Count, sizeOne, pixelsCount);
        };
    }
}
