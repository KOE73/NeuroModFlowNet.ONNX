namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: YOLO Pose extractor for FP32.
/// <br/>
/// RU: Экстрактор YOLO Pose для FP32.
/// </summary>
public class YoloPoseFP32Keypoint17Extractor : YoloPoseFP32ExtractorBase<IDetectionResult<YoloPose_FP32_Size57_Keypoint17>>
{
    public override IDetectionResult<YoloPose_FP32_Size57_Keypoint17> Extract()
    {
        return GetOutput();
    }

    public IDetectionResult<YoloPose_FP32_Size57_Keypoint17> GetOutput()
    {
        var data = Model.GetTensorDataAsSpan<float>();
        var allDetections = MemoryMarshal.Cast<float, YoloPose_FP32_Size57_Keypoint17>(data);

        var result = new BatchedResult<YoloPose_FP32_Size57_Keypoint17>(BatchCount, BatchCount * ItemCount);

        for(int batch = 0; batch < BatchCount; batch++)
        {
            result.MoveNext();

            var batchSpan = allDetections.Slice(batch * ItemCount, ItemCount);

            for(int itemIndex = 0; itemIndex < ItemCount; itemIndex++)
                if(batchSpan[itemIndex].Score >= Threshold)
                    result.Add(batchSpan[itemIndex]);
        }
        return result;
    }
}
