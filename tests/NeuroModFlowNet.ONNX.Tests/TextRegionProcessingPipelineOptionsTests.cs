using System.Text.Json;
using NeuroModFlowNet.ONNX;
using Xunit;

namespace NeuroModFlowNet.ONNX.Tests.PaddleOCR;

public class TextRegionProcessingPipelineOptionsTests
{
    [Fact]
    public void Deserialize_CreatesConfiguredProcessingPipeline()
    {
        const string json = """
        {
          "Enabled": true,
          "Stages": [
            {
              "Kind": "BrightnessContrast",
              "BrightnessContrast": {
                "Brightness": 12,
                "ContrastPercent": 140
              }
            },
            {
              "Kind": "Gamma",
              "Gamma": {
                "Gamma": 1.7
              }
            },
            {
              "Kind": "Sharpen",
              "Sharpen": {
                "KernelSize": 5,
                "Sigma": 0.8,
                "Amount": 1.2
              }
            }
          ]
        }
        """;

        TextRegionProcessingPipelineOptions options = JsonSerializer.Deserialize<TextRegionProcessingPipelineOptions>(json)!;

        using TextRegionProcessingPipeline pipeline = Assert.IsType<TextRegionProcessingPipeline>(
            TextRegionProcessingStageFactory.CreateProcessingStage(options));
        IReadOnlyList<ITextRegionProcessingStage> stages = pipeline.GetStages();

        Assert.Equal(3, stages.Count);
        ITextRegionBrightnessContrastSettings brightnessContrast = Assert.IsAssignableFrom<ITextRegionBrightnessContrastSettings>(stages[0]);
        Assert.Equal(12, brightnessContrast.Brightness);
        Assert.Equal(140, brightnessContrast.ContrastPercent);

        ITextRegionGammaCorrectionSettings gamma = Assert.IsAssignableFrom<ITextRegionGammaCorrectionSettings>(stages[1]);
        Assert.Equal(1.7, gamma.Gamma);

        ITextRegionSharpenSettings sharpen = Assert.IsAssignableFrom<ITextRegionSharpenSettings>(stages[2]);
        Assert.Equal(5, sharpen.KernelSize);
        Assert.Equal(0.8, sharpen.Sigma);
        Assert.Equal(1.2, sharpen.Amount);
    }

    [Fact]
    public void CreateProcessingStage_ReturnsNullForDisabledPipeline()
    {
        var options = new TextRegionProcessingPipelineOptions
        {
            Enabled = false,
            Stages =
            [
                new TextRegionProcessingStageOptions
                {
                    Kind = TextRegionProcessingStageKind.AutoContrast,
                },
            ],
        };

        Assert.Null(TextRegionProcessingStageFactory.CreateProcessingStage(options));
    }

    [Fact]
    public void CreateProcessingStage_ReturnsNullForEmptyPipeline()
    {
        var options = new TextRegionProcessingPipelineOptions
        {
            Enabled = true,
        };

        Assert.Null(TextRegionProcessingStageFactory.CreateProcessingStage(options));
    }

    [Fact]
    public void Deserialize_PostprocessingOptions_MapsNullableAnalyzerLimitsToRuntimeOptions()
    {
        const string json = """
        {
          "DetMask": {
            "BitmapThreshold": 0.35,
            "BoxScoreThreshold": 0.75
          },
          "Analyzer": {
            "MinRegionHeight": 8,
            "MaxRegionHeight": 80,
            "MinRegionWidth": 12,
            "OverlapSuppression": {
              "Enabled": true,
              "Ratio": 0.8
            },
            "LineMerge": {
              "Enabled": true,
              "MaxMergedRegionWidth": 320
            }
          },
          "RecognitionRoi": {
            "TargetWidth": 320,
            "TargetHeight": 48,
            "Processing": {
              "Enabled": true,
              "Stages": [
                {
                  "Kind": "AutoContrast"
                }
              ]
            }
          }
        }
        """;

        PaddleOCRPostprocessingOptions options = JsonSerializer.Deserialize<PaddleOCRPostprocessingOptions>(json)!;

        Assert.Equal(0.35f, options.DetMask.BitmapThreshold);
        Assert.Equal(0.75f, options.DetMask.BoxScoreThreshold);

        OcrRegionPostprocessorOptions runtimeAnalyzer = options.CreateAnalyzerRuntimeOptions();
        Assert.Equal(8, runtimeAnalyzer.MinRegionHeight);
        Assert.Equal(80, runtimeAnalyzer.MaxRegionHeight);
        Assert.Equal(12, runtimeAnalyzer.MinRegionWidth);
        Assert.True(float.IsPositiveInfinity(runtimeAnalyzer.MaxRegionWidth));
        Assert.True(runtimeAnalyzer.EnableOverlapSuppression);
        Assert.Equal(0.8f, runtimeAnalyzer.OverlapSuppressionRatio);
        Assert.True(runtimeAnalyzer.EnableLineMerge);
        Assert.Equal(320, runtimeAnalyzer.MaxMergedRegionWidth);
        Assert.Equal(320f / 48f, runtimeAnalyzer.MaxMergedRegionAspectRatio);

        using TextRegionProcessingPipeline pipeline = Assert.IsType<TextRegionProcessingPipeline>(
            options.RecognitionRoi.CreateProcessingStage());
        Assert.Single(pipeline.GetStages());
        Assert.IsType<TextRegionAutoContrastStage>(pipeline.GetStages()[0]);
    }
}
