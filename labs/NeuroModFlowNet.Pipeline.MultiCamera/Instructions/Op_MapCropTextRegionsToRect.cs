using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline.ONNX;

namespace NeuroModFlowNet.Pipeline.MultiCamera;

/// <summary>
/// EN: Maps text OBBs found on letterboxed crops back into rect space using the per-crop back transforms written by
/// <c>Op_Onnx_ExtractObbToLetterboxBatch_FP32_NCHW</c>, and resolves each crop index to its track id. Outputs a plain
/// <c>YoloObb[]</c> (rect space, feeds <c>Op_Onnx_ExtractObbToPaddleRec</c>) and the parallel <c>int[]</c> of track ids.
/// Regions from padding slots (crop index beyond the actual crop count) are dropped.
///
/// RU: Переводит текстовые OBB с letterbox-вырезок обратно в rect-пространство по обратным преобразованиям,
/// записанным <c>Op_Onnx_ExtractObbToLetterboxBatch_FP32_NCHW</c>, и сопоставляет индекс вырезки с id трека. Выход:
/// плоский <c>YoloObb[]</c> (rect-пространство, вход для <c>Op_Onnx_ExtractObbToPaddleRec</c>) и параллельный
/// <c>int[]</c> с id треков. Регионы из слотов-заполнителей (индекс вырезки за пределами реального числа) отбрасываются.
/// </summary>
internal sealed class Op_MapCropTextRegionsToRect : OpBase
{
    readonly string regionsKey;
    readonly string transformsKey;
    readonly string cropCountKey;
    readonly string trackIdsKey;
    readonly string outputKey;
    readonly string outputTrackIdsKey;

    public Op_MapCropTextRegionsToRect(
        string regionsKey,
        string transformsKey,
        string cropCountKey,
        string trackIdsKey,
        string outputKey,
        string outputTrackIdsKey)
        : base(OpDescriptor.Create(
            "Op_MapCropTextRegionsToRect",
            "op.map.cropTextRegions.toRect",
            [
                VarRequirement.Read<CropTextRegion[]>(regionsKey),
                VarRequirement.Read<ICoordinateBackTransform[]>(transformsKey),
                VarRequirement.Read<int>(cropCountKey),
                VarRequirement.Read<int[]>(trackIdsKey)
            ],
            [VarRequirement.Write<YoloObb[]>(outputKey), VarRequirement.Write<int[]>(outputTrackIdsKey)]))
    {
        this.regionsKey = regionsKey;
        this.transformsKey = transformsKey;
        this.cropCountKey = cropCountKey;
        this.trackIdsKey = trackIdsKey;
        this.outputKey = outputKey;
        this.outputTrackIdsKey = outputTrackIdsKey;
    }

    public override ValueTask<OpResult> ExecuteAsync(VmRunContext context, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        CropTextRegion[] regions = context.Get<CropTextRegion[]>(regionsKey);
        ICoordinateBackTransform[] transforms = context.Get<ICoordinateBackTransform[]>(transformsKey);
        int cropCount = context.Get<int>(cropCountKey);
        int[] trackIds = context.Get<int[]>(trackIdsKey);

        if(cropCount > transforms.Length || cropCount > trackIds.Length)
            return ValueTask.FromResult(OpResult.Fail($"Crop count {cropCount} exceeds transforms ({transforms.Length}) or track ids ({trackIds.Length})."));

        var mapper = default(YoloObbCoordinatePayloadMapper);
        var boxes = new List<YoloObb>(regions.Length);
        var boxTrackIds = new List<int>(regions.Length);

        foreach(CropTextRegion region in regions)
        {
            if(region.CropIndex < 0 || region.CropIndex >= cropCount)
                continue;

            boxes.Add(mapper.Map(region.Box, transforms[region.CropIndex], CoordinateMappingShapePolicy.BoundingOBB));
            boxTrackIds.Add(trackIds[region.CropIndex]);
        }

        context.Set(outputKey, boxes.ToArray());
        context.Set(outputTrackIdsKey, boxTrackIds.ToArray());
        return ValueTask.FromResult(OpResult.Continue);
    }
}
