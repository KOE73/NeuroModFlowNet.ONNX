namespace NeuroModFlowNet.ONNX;

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

public class PaddleOCRRecExtractor : ResultExtractorBase<List<PaddleOCRRecExtractor.OcrResult>>
{
    public const int OutputWidthStride = 8;

    public int ModelDetectionAttributes { get; private set; }
    public int DetectionWidth { get; private set; }

    public string[] Alphabet { get; private set; } = Array.Empty<string>();
    public bool IsAlphabetLoaded { get; private set; } = false;

    public string? PriorityChars { get; set; }

    protected override void Init()
    {
        long[] inputShape = Model.IsInputPersistentValueInitialized(Model.PrimaryInputName)
            ? Model.GetRealInputShape(Model.PrimaryInputName)
            : Model.ModelInputShapes[Model.PrimaryInputName];

        DetectionWidth = (int)inputShape[3] / OutputWidthStride;
        ModelDetectionAttributes = (int)Model.ModelOutputShapes[Model.PrimaryOutputName][2];

        // Auto-load dictionary
        string expectedDictPath = Path.Combine(Path.GetDirectoryName(Model.ModelPath) ?? "", "dict.txt");
        LoadAlphabet(expectedDictPath);
    }

    public void LoadAlphabet(string filePath)
    {
        if(!File.Exists(filePath))
        {
            IsAlphabetLoaded = false;
            throw new FileNotFoundException("Dictionary file not found.", filePath);
        }

        string[] fileLines = File.ReadAllLines(filePath, Encoding.UTF8);
        string[] alphabet = ["", .. fileLines, " "];

        if(alphabet.Length != ModelDetectionAttributes)
            throw new ArgumentException($"Alphabet length must be DetectionAttributes + 2 (for blank and space). Expected: {ModelDetectionAttributes + 2}, Actual: {alphabet.Length}");

        Alphabet = alphabet;
        IsAlphabetLoaded = true;
    }

    public override List<OcrResult> Extract()
    {
        if(!IsAlphabetLoaded)
        {
            // If the user hasn't loaded a valid alphabet, return empty results to prevent crash, since they might load it later or ignore.
            // Still we can raise an informative message or throw if strictness is required.
            // As per user request: "сделай чтобы если нет файла не падало а выставляло флаг доступности. и возможность ручного запуска."
            return new List<OcrResult>();
        }

        var tensorSpan = Model.GetOutputValue(Model.PrimaryOutputName).GetTensorDataAsTensorSpan<float>();
        return AnalizeResult_GetText(tensorSpan);
    }

    public List<OcrResult> AnalizeResult_GetText(ReadOnlyTensorSpan<float> tensorSpan3D)
    {
        const int BlankIndex = 0;
        const float SpaceThreshold = 0.1f; // Порог, при котором мы верим в пробел
        const float CandidateThreshold = 0.05f; // Порог для включения в список [6G]

        int SpaceIndex = Alphabet.Length - 1;

        StringBuilder sbStandard = new StringBuilder();
        StringBuilder sbWithSpace = new StringBuilder();
        StringBuilder sbFull = new StringBuilder();

        List<(int SymIndex, float SymConf)> _candidates = new(16);
        List<OcrResult> result = new();

        nint batchCount = tensorSpan3D.Lengths[0];
        for(int batch = 0; batch < batchCount; batch++)
        {
            var tensorSpan3D_batch1 = tensorSpan3D[batch..(batch + 1), .., ..];
            var tensorSpan2D = tensorSpan3D_batch1.Squeeze();

            nint posCount = tensorSpan2D.Lengths[0];
            nint symCount = tensorSpan2D.Lengths[1];

            int lastSymIndexStandard = -1;
            int lastSymIndexWithSpaces = -1;
            int lastSymIndexBestCandidate = -1;

            sbStandard.Clear();
            sbWithSpace.Clear();
            sbFull.Clear();

            for(int pos = 0; pos < posCount; pos++)
            {
                _candidates.Clear();
                int bestSymIndexStandard = 0, bestSymIndexWithSpaces = 0;
                float maxSymConfStandard = 0, maxSymConfWithSpaces = 0;

                for(int symIndex = 0; symIndex < symCount; symIndex++)
                {
                    float conf = tensorSpan2D[pos, symIndex];

                    string currentChar = Alphabet[symIndex];
                    if(symIndex != BlankIndex &&
                        PriorityChars is not null &&
                        !PriorityChars.Contains(currentChar, StringComparison.Ordinal))
                    {
                        continue;
                    }

                    if(conf > maxSymConfStandard)
                    {
                        bestSymIndexStandard = symIndex;
                        maxSymConfStandard = conf;
                    }

                    if(symIndex == SpaceIndex && conf > SpaceThreshold)
                    {
                        bestSymIndexWithSpaces = SpaceIndex;
                        maxSymConfWithSpaces = conf;
                    }
                    else if(conf > maxSymConfWithSpaces)
                    {
                        bestSymIndexWithSpaces = symIndex;
                        maxSymConfWithSpaces = conf;
                    }

                    if(symIndex != BlankIndex && conf > CandidateThreshold)
                    {
                        _candidates.Add((symIndex, conf));
                    }
                }

                if(bestSymIndexWithSpaces == 0 && maxSymConfStandard > maxSymConfWithSpaces)
                    bestSymIndexWithSpaces = bestSymIndexStandard;

                // Вариант 1: Standard
                if(bestSymIndexStandard != BlankIndex && bestSymIndexStandard != lastSymIndexStandard)
                    sbStandard.Append(Alphabet[bestSymIndexStandard]);
                lastSymIndexStandard = bestSymIndexStandard;

                // Вариант 2: Space Bias
                if(bestSymIndexWithSpaces != BlankIndex && bestSymIndexWithSpaces != lastSymIndexWithSpaces)
                    sbWithSpace.Append(Alphabet[bestSymIndexWithSpaces]);
                lastSymIndexWithSpaces = bestSymIndexWithSpaces;

                if(_candidates.Count > 0)
                {
                    _candidates.Sort((a, b) => b.SymConf.CompareTo(a.SymConf));
                    int currentBest = _candidates[0].SymIndex;

                    if(currentBest != lastSymIndexBestCandidate)
                    {
                        sbFull.Append('|');
                        foreach(var cand in _candidates)
                            sbFull.Append(Alphabet[cand.SymIndex]);

                        lastSymIndexBestCandidate = currentBest;
                    }
                }
                else
                {
                    lastSymIndexBestCandidate = BlankIndex;
                }
            }

            result.Add(new OcrResult(sbStandard.ToString(), sbWithSpace.ToString(), sbFull.ToString()));
        }

        return result;
    }

    public readonly struct OcrResult
    {
        public readonly string Standard;
        public readonly string WithSpaces;
        public readonly string FullCandidates;

        public OcrResult(string standard, string withSpaces, string fullCandidates)
        {
            Standard = standard;
            WithSpaces = withSpaces;
            FullCandidates = fullCandidates;
        }

        public bool IsEmpty => string.IsNullOrEmpty(Standard);

        public override string ToString() => $"{Standard} | {WithSpaces} | {FullCandidates}";
    }
}
