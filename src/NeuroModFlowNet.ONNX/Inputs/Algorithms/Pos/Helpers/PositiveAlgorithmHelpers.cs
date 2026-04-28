using Microsoft.ML.OnnxRuntime;

namespace NeuroModFlowNet.ONNX.Converters.Algorithms;

public static class PositiveAlgorithmHelpers
{
    public static bool CheckAlgorithms(Action<string>? log = null)
    {
        const int width = 128;
        const int height = 128;
        const int batch = 4;

        using Mat mat = CreateFastRealMat(width, height);
        return CheckAlgorithms(mat, batch, log);
    }

    public static bool CheckAlgorithms(Mat mat, int batch, Action<string>? log = null)
    {
        List<Mat> mats = CloneMatBatch(mat, batch);
        try
        {
            return CheckFP16Algorithms(mats, mat.Width, mat.Height, batch, log);
        }
        finally
        {
            foreach(Mat item in mats)
                item.Dispose();
        }
    }

    public static InputDataToSpanBufConverter<List<Mat>, Float16> AutoTuneFP16(Mat mat, int batch, Action<string>? log = null)
    {
        List<Mat> mats = CloneMatBatch(mat, batch);
        try
        {
            int sizeOne = mat.Width * mat.Height * 3;
            byte[] bytes = new byte[sizeOne * batch * sizeof(ushort)];
            Span<Float16> buffer = MemoryMarshal.Cast<byte, Float16>(bytes.AsSpan());
            return AutoTuneFP16(mats, buffer, mat.Width, mat.Height, batch, log);
        }
        finally
        {
            foreach(Mat item in mats)
                item.Dispose();
        }
    }

    public static InputDataToSpanBufConverter<List<Mat>, Float16> AutoTuneFP16(int width, int height, int batch, Action<string>? log = null)
    {
        using Mat mat = CreateFastRealMat(width, height);
        return AutoTuneFP16(mat, batch, log);
    }

    public static unsafe Mat CreateFastRealMat(int width, int height)
    {
        Mat mat = new(height, width, MatType.CV_8UC3);
        int totalBytes = (int)(mat.Step() * mat.Rows);
        Span<byte> dataSpan = new(mat.DataPointer, totalBytes);

        Random random = new();
        for(int i = 0; i < dataSpan.Length; i++)
            dataSpan[i] = (byte)random.Next(0, 256);

        return mat;
    }

    public static List<Mat> CloneMatBatch(Mat mat, int batch) =>
        Enumerable.Range(0, batch).Select(_ => mat.Clone()).ToList();

    private static bool CheckFP16Algorithms(List<Mat> mats, int width, int height, int batch, Action<string>? log)
    {
        int sizeOne = width * height * 3;
        int pixelsCount = width * height;

        byte[] asBlob = Call(Wrap<Float16, PosCvdnnFP16>(sizeOne, pixelsCount));
        byte[] asDivHalfTensorReorderPtr = Call(Wrap<Float16, PosDivHalfTensorReorderPtrFP16>(sizeOne, pixelsCount));
        byte[] asReorderDivPtrHalfTensor = Call(Wrap<Float16, PosReorderDivPtrHalfTensorFP16>(sizeOne, pixelsCount));
        byte[] asReorderPtrDivHalfTensor = Call(Wrap<Float16, PosReorderPtrDivHalfTensorFP16>(sizeOne, pixelsCount));
        byte[] asReorderDivHalfPtr = Call(Wrap<Float16, PosReorderDivHalfPtrFP16>(sizeOne, pixelsCount));

        int compareDivHalfTensorReorderPtr = asBlob.SequenceCompareTo(asDivHalfTensorReorderPtr);
        log?.Invoke($"PosDivHalfTensorReorderPtrFP16 {(compareDivHalfTensorReorderPtr == 0 ? "OK" : "Error")}");

        int compareReorderDivPtrHalfTensor = asBlob.SequenceCompareTo(asReorderDivPtrHalfTensor);
        log?.Invoke($"PosReorderDivPtrHalfTensorFP16 {(compareReorderDivPtrHalfTensor == 0 ? "OK" : "Error")}");

        int compareReorderPtrDivHalfTensor = asBlob.SequenceCompareTo(asReorderPtrDivHalfTensor);
        log?.Invoke($"PosReorderPtrDivHalfTensorFP16 {(compareReorderPtrDivHalfTensor == 0 ? "OK" : "Error")}");

        int compareReorderDivHalfPtr = asBlob.SequenceCompareTo(asReorderDivHalfPtr);
        log?.Invoke($"PosReorderDivHalfPtrFP16 {(compareReorderDivHalfPtr == 0 ? "OK" : "Error")}");

        return
            compareDivHalfTensorReorderPtr == 0 &&
            compareReorderDivPtrHalfTensor == 0 &&
            compareReorderPtrDivHalfTensor == 0 &&
            compareReorderDivHalfPtr == 0;

        byte[] Call(InputDataToSpanBufConverter<List<Mat>, Float16> action)
        {
            byte[] bytes = new byte[sizeOne * batch * sizeof(ushort)];
            action(mats, MemoryMarshal.Cast<byte, Float16>(bytes.AsSpan()), batch);
            return bytes;
        }
    }

    private static InputDataToSpanBufConverter<List<Mat>, Float16> AutoTuneFP16(
        List<Mat> mats,
        Span<Float16> buffer,
        int width,
        int height,
        int batch,
        Action<string>? log)
    {
        int sizeOne = width * height * 3;
        int pixelsCount = width * height;

        var algorithms = new (string Name, InputDataToSpanBufConverter<List<Mat>, Float16> Proc)[]
        {
            ("PosCvdnnFP16", Wrap<Float16, PosCvdnnFP16>(sizeOne, pixelsCount)),
            ("PosDivHalfTensorReorderPtrFP16", Wrap<Float16, PosDivHalfTensorReorderPtrFP16>(sizeOne, pixelsCount)),
            ("PosReorderDivPtrHalfTensorFP16", Wrap<Float16, PosReorderDivPtrHalfTensorFP16>(sizeOne, pixelsCount)),
            ("PosReorderPtrDivHalfTensorFP16", Wrap<Float16, PosReorderPtrDivHalfTensorFP16>(sizeOne, pixelsCount)),
            ("PosReorderDivHalfPtrFP16", Wrap<Float16, PosReorderDivHalfPtrFP16>(sizeOne, pixelsCount))
        };

        string bestName = "";
        InputDataToSpanBufConverter<List<Mat>, Float16> bestDelegate = algorithms[0].Proc;
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
        log?.Invoke($"[AutoTune] Best result: {bestName} ({1000.0 / minMs:F0} FPS) on {width}x{height} batch {batch}");

        return bestDelegate;
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
