using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline;
using NeuroModFlowNet.Pipeline.ONNX;
using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.Avalonia.VMDebug.Runtime;

/// <summary>
/// Runs PaddleOCR recognition over prepared ROI batches.
/// </summary>
/// <remarks>
/// The recognition model has a fixed persistent batch shape. This instruction preserves the old batching behavior by
/// slicing the per-frame ROI list into model-sized chunks while keeping each chunk inside the same VM run context.
/// </remarks>
internal sealed class PaddleRecognitionBatchesInstruction : OpBase
{
    readonly RecognitionOptions recognitionOptions;
    readonly ReloadableOnnxRunnerResource<List<Mat>, List<PaddleOCRRecExtractor.OcrResult>> resource;

    public PaddleRecognitionBatchesInstruction(
        RecognitionOptions recognitionOptions,
        ReloadableOnnxRunnerResource<List<Mat>, List<PaddleOCRRecExtractor.OcrResult>> resource)
        : base(OpDescriptor.Create(
            "recognize-ocr-regions",
            "ocr.recognizeBatches",
            [VarRequirement.Read<List<PreparedRecognitionRoi>>(PipelineAvaloniaKeys.RecognitionRois)],
            [VarRequirement.Write<List<RecognitionTextRow>>(PipelineAvaloniaKeys.RecognitionRows)],
            hasSideEffects: true))
    {
        this.recognitionOptions = recognitionOptions;
        this.resource = resource;
    }

    public override async ValueTask<OpResult> ExecuteAsync(VmRunContext context, CancellationToken cancellationToken)
    {
        List<PreparedRecognitionRoi> preparedImages = context.Get<List<PreparedRecognitionRoi>>(PipelineAvaloniaKeys.RecognitionRois);
        PipelineFrameTiming timing = context.Get<PipelineFrameTiming>(PipelineAvaloniaKeys.FrameTiming);
        timing.StartRecognition();

        var outputResults = new List<RecognitionTextRow>();

        for(int offset = 0; offset < preparedImages.Count; offset += recognitionOptions.BatchSize)
        {
            cancellationToken.ThrowIfCancellationRequested();

            int itemCount = Math.Min(recognitionOptions.BatchSize, preparedImages.Count - offset);
            List<Mat> batchImages = new(itemCount);

            for(int index = 0; index < itemCount; index++)
                batchImages.Add(preparedImages[offset + index].Roi);

            List<PaddleOCRRecExtractor.OcrResult> batchResults = await resource.ExecuteAsync(batchImages, cancellationToken).ConfigureAwait(false);
            int resultCount = Math.Min(itemCount, batchResults.Count);

            for(int index = 0; index < resultCount; index++)
            {
                PaddleOCRRecExtractor.OcrResult recognitionResult = batchResults[index];
                if(recognitionResult.IsEmpty)
                    continue;

                PreparedRecognitionRoi prepared = preparedImages[offset + index];
                outputResults.Add(new RecognitionTextRow(
                    recognitionOptions.FormatRecognitionText(recognitionResult),
                    prepared.LabelPoint,
                    prepared.Roi,
                    prepared.RoiHeightDebug));
            }
        }

        timing.RecognitionItemCount = preparedImages.Count;
        timing.StopRecognition();
        context.Set(PipelineAvaloniaKeys.RecognitionRows, outputResults);
        return OpResult.Continue;
    }
}

