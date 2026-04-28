namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: YOLO Pose extractor for FP32.
/// <br/>
/// RU: Экстрактор YOLO Pose для FP32.
/// </summary>
public class YoloPoseFP32UniversalExtractor : YoloPoseFP32ExtractorBase<IDetectionResult<YoloPose>>
{
    public override IDetectionResult<YoloPose> Extract()
    {
        return GetOutput();
    }



    // ------------------------------------------------------------
    // УНИВЕРСАЛЬНЫЙ ВАРИАНТ ДЛЯ ЛЮБОГО ЧИСЛА ТОЧЕК
    //
    // ВАЖНО:
    // Тут уже нельзя делать BatchedResult<YoloPose_FP32_RowView>, потому что ref struct
    // нельзя хранить в heap-объектах.
    // Поэтому сохраняем уже "обычный" managed DTO.
    // ------------------------------------------------------------

    public IDetectionResult<YoloPose> GetOutput()
    {
        var data = Model.GetTensorDataAsSpan<float>();

        // allRows содержит все detection подряд по всем batch.
        var allRows = new YoloPose_FP32_RowsView(data, KeypointsCount);

        Debug.Assert(allRows.Count == BatchCount * ItemCount);

        var result = new BatchedResult<YoloPose>(BatchCount, BatchCount * ItemCount);

        for(int batch = 0; batch < BatchCount; batch++)
        {
            result.MoveNext();

            int batchOffset = batch * ItemCount;

            for(int itemIndex = 0; itemIndex < ItemCount; itemIndex++)
            {
                YoloPose_FP32_RowView row = allRows[batchOffset + itemIndex];

                if(row.Score < Threshold)
                    continue;

                var keypoints = new YoloPoseKeypointXYV[KeypointsCount];

                for(int k = 0; k < KeypointsCount; k++)
                {
                    var kp = row.GetKeypoint(k);
                    keypoints[k] = new YoloPoseKeypointXYV(kp.X, kp.Y, kp.V);
                }

                result.Add(new YoloPose
                {
                    X = row.X,
                    Y = row.Y,
                    W = row.W,
                    H = row.H,
                    Score = row.Score,
                    ClassId = row.ClassId,
                    Keypoints = keypoints
                });
            }
        }

        return result;
    }
}
