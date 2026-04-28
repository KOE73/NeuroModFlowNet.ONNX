using Xunit;
using NeuroModFlowNet.ONNX;
using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuroModFlowNet.ONNX.Tests.Extractors;

public class YoloObbNmsFP32SingleExtractorTests
{
    // Размер структуры OBB_single_XYWHSCA в float (обычно 7: X, Y, W, H, Score, ClassId, Angle)
    private const int FieldCount = 7;

    [Fact]
    public void ExtractFromSpan_ShouldFilterByScore_AndParseCorrectly()
    {
        // Arrange
        int batchCount = 1;
        int detectionCount = 3;
        float threshold = 0.5f;

        // Создаем данные для 3-х детекций.
        // Предполагаемая структура (на основе индекса Score=4): X, Y, W, H, Score, ClassId, Angle
        float[] data = new float[detectionCount * FieldCount];

        // Box 1: Valid (Score 0.9)
        SetBoxData(data, 0, 10, 20, 100, 50, 0.9f, 1, 0.5f);

        // Box 2: Invalid (Score 0.1 - ниже порога)
        SetBoxData(data, 1, 5, 5, 10, 10, 0.1f, 0, 0.0f);

        // Box 3: Valid (Score 0.6)
        SetBoxData(data, 2, 30, 40, 200, 60, 0.6f, 2, 1.5f);

        // Act
        var result = YoloObbNmsFP32Extractor.ExtractFromSpan(data, batchCount, detectionCount, FieldCount, threshold);

        // Assert
        Assert.Single(result); // 1 батч
        Assert.True(result.ContainsKey(0));
        
        var boxes = result[0];
        Assert.Equal(2, boxes.Length); // Только 2 бокса должны остаться (Box 1 и Box 3)

        // Проверяем первый бокс (Box 1)
        Assert.Equal(10, boxes[0].X);
        Assert.Equal(20, boxes[0].Y);
        Assert.Equal(0.9f, boxes[0].Score);
        Assert.Equal(1, boxes[0].Class);

        // Проверяем второй бокс (Box 3)
        Assert.Equal(30, boxes[1].X);
        Assert.Equal(0.6f, boxes[1].Score);
        Assert.Equal(2, boxes[1].Class);
    }

    [Fact]
    public void ExtractFromSpan_ShouldHandleMultipleBatches()
    {
        // Arrange
        int batchCount = 2;
        int detectionCount = 2; // По 2 детекции на батч
        float threshold = 0.5f;

        float[] data = new float[batchCount * detectionCount * FieldCount];

        // --- Batch 0 ---
        // Box 1 (Valid)
        SetBoxData(data, 0, 10, 10, 10, 10, 0.8f, 0, 0);
        // Box 2 (Invalid)
        SetBoxData(data, 1, 20, 20, 20, 20, 0.2f, 0, 0);

        // --- Batch 1 ---
        // Box 3 (Valid)
        SetBoxData(data, 2, 30, 30, 30, 30, 0.7f, 1, 0); // Индекс 2 в общем массиве
        // Box 4 (Valid)
        SetBoxData(data, 3, 40, 40, 40, 40, 0.9f, 1, 0); // Индекс 3 в общем массиве

        // Act
        var result = YoloObbNmsFP32Extractor.ExtractFromSpan(data, batchCount, detectionCount, FieldCount, threshold);

        // Assert
        Assert.Equal(2, result.Count);
        
        Assert.Single(result[0]); // В 0-м батче 1 бокс
        Assert.Equal(10, result[0][0].X);

        Assert.Equal(2, result[1].Length); // В 1-м батче 2 бокса
        Assert.Equal(30, result[1][0].X);
        Assert.Equal(40, result[1][1].X);
    }

    [Fact]
    public void ExtractFromTensor_ShouldMatchSpanExtractionLogic()
    {
        // Arrange
        int batchCount = 1;
        int detectionCount = 2;
        float threshold = 0.5f;

        float[] data = new float[detectionCount * FieldCount];
        
        // Box 1: Valid
        SetBoxData(data, 0, 100, 200, 50, 60, 0.8f, 5, 0.1f);
        // Box 2: Invalid
        SetBoxData(data, 1, 0, 0, 0, 0, 0.4f, 0, 0);

        // Act
        var result = YoloObbNmsFP32Extractor.ExtractFromTensor(data, batchCount, detectionCount, FieldCount, threshold);

        // Assert
        Assert.Single(result);
        var boxes = result[0];
        
        Assert.Single(boxes);
        var box = boxes[0];

        Assert.Equal(100, box.X);
        Assert.Equal(200, box.Y);
        Assert.Equal(50, box.W);
        Assert.Equal(60, box.H);
        Assert.Equal(0.8f, box.Score);
        Assert.Equal(5, box.Class);
        Assert.Equal(0.1f, box.Angle);
    }

    [Fact]
    public void ExtractFromSpan_EmptyInput_ShouldReturnEmptyDictionaries()
    {
        // Arrange
        int batchCount = 1;
        int detectionCount = 0;
        float[] data = Array.Empty<float>();

        // Act
        var result = YoloObbNmsFP32Extractor.ExtractFromSpan(data, batchCount, detectionCount, FieldCount, 0.5f);

        // Assert
        Assert.Single(result);
        Assert.Empty(result[0]);
    }

    private void SetBoxData(float[] data, int index, float x, float y, float w, float h, float score, float classId, float angle)
    {
        int offset = index * FieldCount;
        // Порядок полей должен совпадать с layout структуры OBB_single_XYWHSCA
        // 0:X, 1:Y, 2:W, 3:H, 4:Score, 5:Class, 6:Angle
        data[offset + 0] = x;
        data[offset + 1] = y;
        data[offset + 2] = w;
        data[offset + 3] = h;
        data[offset + 4] = score;
        data[offset + 5] = classId;
        data[offset + 6] = angle;
    }
}
