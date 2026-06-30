using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using System.Text;

namespace NeuroModFlowNet.Pipeline.ONNX;

public sealed class PaddleOCRRecOrtValueOutputDecoder : IOrtValueBatchOutputDecoder<PaddleOCRRecExtractor.OcrResult>
{
    readonly PaddleOCRRecExtractor extractor = new();
    OnnxModel? initializedModel;

    public string? PriorityChars
    {
        get => extractor.PriorityChars;
        set => extractor.PriorityChars = value;
    }

    public IReadOnlyList<PaddleOCRRecExtractor.OcrResult[]> Decode(
        OrtValue output,
        OnnxModel model,
        string outputName,
        int requestCount,
        IReadOnlyList<int>? requestItemCounts)
    {
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(model);

        if(requestItemCounts is null)
            throw new InvalidOperationException($"{nameof(PaddleOCRRecOrtValueOutputDecoder)} requires request item counts to split recognition output.");

        if(requestItemCounts.Count != requestCount)
            throw new InvalidOperationException($"Request item count metadata has {requestItemCounts.Count} entries for {requestCount} requests.");

        EnsureInitialized(model);

        var outputInfo = output.GetTensorTypeAndShape();
        if(outputInfo.ElementDataType != TensorElementType.Float)
            throw new NotSupportedException($"PaddleOCR Rec output element type is not supported: {outputInfo.ElementDataType}. Expected FP32.");

        long[] outputShape = outputInfo.Shape;
        if(outputShape.Length != 3)
            throw new InvalidOperationException($"PaddleOCR Rec output must be rank 3, actual: [{string.Join(", ", outputShape)}].");

        int expectedItemCount = requestItemCounts.Sum();
        if(outputShape[0] < expectedItemCount)
            throw new InvalidOperationException($"PaddleOCR Rec output batch {outputShape[0]} is smaller than requested item count {expectedItemCount}.");

        List<PaddleOCRRecExtractor.OcrResult> flatResults = DecodeFlat(
            output.GetTensorDataAsSpan<float>(),
            checked((int)outputShape[0]),
            checked((int)outputShape[1]),
            checked((int)outputShape[2]));
        if(flatResults.Count < expectedItemCount)
            throw new InvalidOperationException($"PaddleOCR Rec decoder produced {flatResults.Count} items for requested item count {expectedItemCount}.");

        var results = new PaddleOCRRecExtractor.OcrResult[requestCount][];
        int offset = 0;
        for(int requestIndex = 0; requestIndex < requestCount; requestIndex++)
        {
            int itemCount = requestItemCounts[requestIndex];
            if(itemCount < 0)
                throw new InvalidOperationException($"Request item count at index {requestIndex} must be non-negative, actual: {itemCount}.");

            results[requestIndex] = flatResults.GetRange(offset, itemCount).ToArray();
            offset += itemCount;
        }

        return results;
    }

    void EnsureInitialized(OnnxModel model)
    {
        if(ReferenceEquals(initializedModel, model))
            return;

        extractor.SetModel(model);
        initializedModel = model;
    }

    List<PaddleOCRRecExtractor.OcrResult> DecodeFlat(
        ReadOnlySpan<float> data,
        int batchCount,
        int positionCount,
        int symbolCount)
    {
        const int BlankIndex = 0;
        const float SpaceThreshold = 0.1f;
        const float CandidateThreshold = 0.05f;

        string[] alphabet = extractor.Alphabet;
        if(!extractor.IsAlphabetLoaded)
            throw new InvalidOperationException("PaddleOCR Rec alphabet is not loaded.");

        if(symbolCount != alphabet.Length)
            throw new InvalidOperationException($"PaddleOCR Rec symbol count {symbolCount} differs from alphabet length {alphabet.Length}.");

        int spaceIndex = alphabet.Length - 1;
        var results = new List<PaddleOCRRecExtractor.OcrResult>(batchCount);
        var standard = new StringBuilder();
        var withSpaces = new StringBuilder();
        var fullCandidates = new StringBuilder();
        var candidates = new List<(int SymbolIndex, float Confidence)>(16);

        for(int batchIndex = 0; batchIndex < batchCount; batchIndex++)
        {
            int lastStandardSymbolIndex = -1;
            int lastWithSpacesSymbolIndex = -1;
            int lastBestCandidateSymbolIndex = -1;
            standard.Clear();
            withSpaces.Clear();
            fullCandidates.Clear();

            ReadOnlySpan<float> batchData = data.Slice(batchIndex * positionCount * symbolCount, positionCount * symbolCount);
            for(int positionIndex = 0; positionIndex < positionCount; positionIndex++)
            {
                candidates.Clear();
                ReadOnlySpan<float> positionData = batchData.Slice(positionIndex * symbolCount, symbolCount);
                int bestStandardSymbolIndex = 0;
                int bestWithSpacesSymbolIndex = 0;
                float maxStandardConfidence = 0;
                float maxWithSpacesConfidence = 0;

                for(int symbolIndex = 0; symbolIndex < symbolCount; symbolIndex++)
                {
                    float confidence = positionData[symbolIndex];
                    string currentCharacter = alphabet[symbolIndex];
                    if(symbolIndex != BlankIndex &&
                        extractor.PriorityChars is not null &&
                        !extractor.PriorityChars.Contains(currentCharacter, StringComparison.Ordinal))
                    {
                        continue;
                    }

                    if(confidence > maxStandardConfidence)
                    {
                        bestStandardSymbolIndex = symbolIndex;
                        maxStandardConfidence = confidence;
                    }

                    if(symbolIndex == spaceIndex && confidence > SpaceThreshold)
                    {
                        bestWithSpacesSymbolIndex = spaceIndex;
                        maxWithSpacesConfidence = confidence;
                    }
                    else if(confidence > maxWithSpacesConfidence)
                    {
                        bestWithSpacesSymbolIndex = symbolIndex;
                        maxWithSpacesConfidence = confidence;
                    }

                    if(symbolIndex != BlankIndex && confidence > CandidateThreshold)
                        candidates.Add((symbolIndex, confidence));
                }

                if(bestWithSpacesSymbolIndex == BlankIndex && maxStandardConfidence > maxWithSpacesConfidence)
                    bestWithSpacesSymbolIndex = bestStandardSymbolIndex;

                if(bestStandardSymbolIndex != BlankIndex && bestStandardSymbolIndex != lastStandardSymbolIndex)
                    standard.Append(alphabet[bestStandardSymbolIndex]);
                lastStandardSymbolIndex = bestStandardSymbolIndex;

                if(bestWithSpacesSymbolIndex != BlankIndex && bestWithSpacesSymbolIndex != lastWithSpacesSymbolIndex)
                    withSpaces.Append(alphabet[bestWithSpacesSymbolIndex]);
                lastWithSpacesSymbolIndex = bestWithSpacesSymbolIndex;

                if(candidates.Count > 0)
                {
                    candidates.Sort(static (left, right) => right.Confidence.CompareTo(left.Confidence));
                    int currentBest = candidates[0].SymbolIndex;

                    if(currentBest != lastBestCandidateSymbolIndex)
                    {
                        fullCandidates.Append('|');
                        foreach((int symbolIndex, float _) in candidates)
                            fullCandidates.Append(alphabet[symbolIndex]);

                        lastBestCandidateSymbolIndex = currentBest;
                    }
                }
                else
                {
                    lastBestCandidateSymbolIndex = BlankIndex;
                }
            }

            results.Add(new PaddleOCRRecExtractor.OcrResult(
                standard.ToString(),
                withSpaces.ToString(),
                fullCandidates.ToString()));
        }

        return results;
    }
}
