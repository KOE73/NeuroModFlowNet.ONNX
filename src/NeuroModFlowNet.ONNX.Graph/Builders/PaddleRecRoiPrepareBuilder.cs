using System.Buffers.Binary;
using static NeuroModFlowNet.ONNX.Graph.Builders.OnnxGraphBuilderHelpers;

namespace NeuroModFlowNet.ONNX.Graph.Builders;

/// <summary>
/// EN: Builds generated ONNX graphs that rectify OCR ROIs from one RGB FP32 NCHW image into a PaddleOCR Rec batch.
///
/// RU: Строит генерируемые ONNX-графы, которые выпрямляют OCR ROI из одного RGB FP32 NCHW изображения в batch для PaddleOCR Rec.
/// </summary>
/// <remarks>
/// EN: The builder accepts ready target-to-source matrices so it stays independent from detector payload types and ROI
/// policies. Domain code decides how OBB/quad regions, absolute padding, and proportional padding become matrices.
///
/// RU: Builder принимает готовые матрицы target-to-source, чтобы не зависеть от detector payload типов и ROI policy.
/// Доменный код отдельно решает, как OBB/quad регионы, абсолютный padding и коэффициентный padding превращаются в матрицы.
/// </remarks>
public static class PaddleRecRoiPrepareBuilder
{
    public const string InputName = "rgb_fp32_nchw";
    public const string MatrixInputName = "roi_target_to_source_matrices";
    public const string OutputName = "paddle_rec_fp32_nchw";

    public static byte[] BuildPerRegionGridSampleConcatFP32Nchw(
        int sourceWidth,
        int sourceHeight,
        int channels,
        int targetWidth,
        int targetHeight,
        IReadOnlyList<float[]> targetToSourceMatrices)
    {
        ValidateArguments(sourceWidth, sourceHeight, channels, targetWidth, targetHeight, targetToSourceMatrices);

        int regionCount = targetToSourceMatrices.Count;
        var graph = CreateBaseGraph(
            "paddle_rec_roi_prepare_per_region_grid_sample_concat",
            sourceWidth,
            sourceHeight,
            channels,
            targetWidth,
            targetHeight,
            regionCount);

        string[] roiOutputs = new string[regionCount];
        for(int regionIndex = 0; regionIndex < regionCount; regionIndex++)
        {
            string gridName = $"grid_{regionIndex}";
            string roiOutputName = $"roi_{regionIndex}";
            graph.Initializer.Add(GridTensor(
                gridName,
                sourceWidth,
                sourceHeight,
                targetWidth,
                targetHeight,
                batch: 1,
                targetToSourceMatrices[regionIndex]));

            AddGridSample(graph, InputName, gridName, roiOutputName, regionIndex);
            roiOutputs[regionIndex] = roiOutputName;
        }

        string batchName = AddBatchConcat(graph, roiOutputs);
        AddPaddleNormalization(graph, batchName, OutputName);

        return CreateModel("neuromodflownet-paddle-rec-roi-prepare-per-region-grid-sample-concat", graph).ToByteArray();
    }

    public static byte[] BuildBatchedGridSampleFP32Nchw(
        int sourceWidth,
        int sourceHeight,
        int channels,
        int targetWidth,
        int targetHeight,
        IReadOnlyList<float[]> targetToSourceMatrices)
    {
        ValidateArguments(sourceWidth, sourceHeight, channels, targetWidth, targetHeight, targetToSourceMatrices);

        int regionCount = targetToSourceMatrices.Count;
        var graph = CreateBaseGraph(
            "paddle_rec_roi_prepare_batched_grid_sample",
            sourceWidth,
            sourceHeight,
            channels,
            targetWidth,
            targetHeight,
            regionCount);

        graph.Initializer.Add(Int64Tensor("expanded_source_shape", [regionCount, channels, sourceHeight, sourceWidth]));
        graph.Initializer.Add(GridTensor(
            "batch_grid",
            sourceWidth,
            sourceHeight,
            targetWidth,
            targetHeight,
            regionCount,
            targetToSourceMatrices));

        graph.Node.Add(new NodeProto
        {
            Name = "expand_source_to_roi_batch",
            OpType = "Expand",
            Input = { InputName, "expanded_source_shape" },
            Output = { "source_batch" }
        });

        AddGridSample(graph, "source_batch", "batch_grid", "roi_batch", regionIndex: null);
        AddPaddleNormalization(graph, "roi_batch", OutputName);

        return CreateModel("neuromodflownet-paddle-rec-roi-prepare-batched-grid-sample", graph).ToByteArray();
    }

    public static byte[] BuildBatchedGridSampleFromMatricesFP32Nchw(
        int sourceWidth,
        int sourceHeight,
        int channels,
        int targetWidth,
        int targetHeight,
        int regionCount)
    {
        ValidateStaticArguments(sourceWidth, sourceHeight, channels, targetWidth, targetHeight, regionCount);

        var graph = CreateBaseGraph(
            "paddle_rec_roi_prepare_batched_grid_sample_from_matrices",
            sourceWidth,
            sourceHeight,
            channels,
            targetWidth,
            targetHeight,
            regionCount);
        graph.Input.Add(TensorInfo(MatrixInputName, TensorProto.Types.DataType.Float, regionCount, 9));

        graph.Initializer.Add(Int64Tensor("expanded_source_shape", [regionCount, channels, sourceHeight, sourceWidth]));
        graph.Initializer.Add(Int64Tensor("matrix_component_shape", [regionCount, 1, 1]));
        graph.Initializer.Add(Int64Tensor("grid_component_axes", [3]));
        graph.Initializer.Add(FloatTensor("target_x", [1, targetHeight, targetWidth], CreateTargetCoordinateValues(targetWidth, targetHeight, useX: true)));
        graph.Initializer.Add(FloatTensor("target_y", [1, targetHeight, targetWidth], CreateTargetCoordinateValues(targetWidth, targetHeight, useX: false)));
        graph.Initializer.Add(FloatTensor("normalize_x_scale", [2f / (sourceWidth - 1)]));
        graph.Initializer.Add(FloatTensor("normalize_y_scale", [2f / (sourceHeight - 1)]));
        graph.Initializer.Add(FloatTensor("normalize_bias", [-1f]));

        for(int componentIndex = 0; componentIndex < 9; componentIndex++)
        {
            graph.Initializer.Add(Int64Tensor($"matrix_component_{componentIndex}_index", [componentIndex]));
            graph.Node.Add(new NodeProto
            {
                Name = $"gather_m{componentIndex}",
                OpType = "Gather",
                Input = { MatrixInputName, $"matrix_component_{componentIndex}_index" },
                Output = { $"m{componentIndex}_n1" },
                Attribute = { IntAttribute("axis", 1) }
            });
            graph.Node.Add(new NodeProto
            {
                Name = $"reshape_m{componentIndex}",
                OpType = "Reshape",
                Input = { $"m{componentIndex}_n1", "matrix_component_shape" },
                Output = { $"m{componentIndex}" }
            });
        }

        AddMul(graph, "m0", "target_x", "m0x");
        AddMul(graph, "m1", "target_y", "m1y");
        AddAdd(graph, "m0x", "m1y", "source_x_num_xy");
        AddAdd(graph, "source_x_num_xy", "m2", "source_x_num");

        AddMul(graph, "m3", "target_x", "m3x");
        AddMul(graph, "m4", "target_y", "m4y");
        AddAdd(graph, "m3x", "m4y", "source_y_num_xy");
        AddAdd(graph, "source_y_num_xy", "m5", "source_y_num");

        AddMul(graph, "m6", "target_x", "m6x");
        AddMul(graph, "m7", "target_y", "m7y");
        AddAdd(graph, "m6x", "m7y", "denominator_xy");
        AddAdd(graph, "denominator_xy", "m8", "denominator");

        AddDiv(graph, "source_x_num", "denominator", "source_x");
        AddDiv(graph, "source_y_num", "denominator", "source_y");
        AddMul(graph, "source_x", "normalize_x_scale", "source_x_scaled");
        AddMul(graph, "source_y", "normalize_y_scale", "source_y_scaled");
        AddAdd(graph, "source_x_scaled", "normalize_bias", "grid_x");
        AddAdd(graph, "source_y_scaled", "normalize_bias", "grid_y");

        graph.Node.Add(new NodeProto
        {
            Name = "unsqueeze_grid_x",
            OpType = "Unsqueeze",
            Input = { "grid_x", "grid_component_axes" },
            Output = { "grid_x_nhw1" }
        });
        graph.Node.Add(new NodeProto
        {
            Name = "unsqueeze_grid_y",
            OpType = "Unsqueeze",
            Input = { "grid_y", "grid_component_axes" },
            Output = { "grid_y_nhw1" }
        });
        graph.Node.Add(new NodeProto
        {
            Name = "concat_grid_xy",
            OpType = "Concat",
            Input = { "grid_x_nhw1", "grid_y_nhw1" },
            Output = { "runtime_grid" },
            Attribute = { IntAttribute("axis", 3) }
        });

        graph.Node.Add(new NodeProto
        {
            Name = "expand_source_to_roi_batch",
            OpType = "Expand",
            Input = { InputName, "expanded_source_shape" },
            Output = { "source_batch" }
        });

        AddGridSample(graph, "source_batch", "runtime_grid", "roi_batch", regionIndex: null);
        AddPaddleNormalization(graph, "roi_batch", OutputName);

        return CreateModel("neuromodflownet-paddle-rec-roi-prepare-batched-grid-sample-from-matrices", graph).ToByteArray();
    }

    static GraphProto CreateBaseGraph(
        string name,
        int sourceWidth,
        int sourceHeight,
        int channels,
        int targetWidth,
        int targetHeight,
        int regionCount)
    {
        var graph = new GraphProto { Name = name };
        graph.Input.Add(TensorInfo(InputName, TensorProto.Types.DataType.Float, 1, channels, sourceHeight, sourceWidth));
        graph.Output.Add(TensorInfo(OutputName, TensorProto.Types.DataType.Float, regionCount, channels, targetHeight, targetWidth));
        graph.Initializer.Add(FloatTensor("paddle_scale", [2f]));
        graph.Initializer.Add(FloatTensor("paddle_bias", [-1f]));
        return graph;
    }

    static void AddGridSample(GraphProto graph, string inputName, string gridName, string outputName, int? regionIndex)
    {
        string nodeName = regionIndex.HasValue
            ? $"grid_sample_roi_{regionIndex.Value}"
            : "grid_sample_roi_batch";

        graph.Node.Add(new NodeProto
        {
            Name = nodeName,
            OpType = "GridSample",
            Input = { inputName, gridName },
            Output = { outputName },
            Attribute =
            {
                IntAttribute("align_corners", 1),
                StringAttribute("mode", "bilinear"),
                StringAttribute("padding_mode", "zeros")
            }
        });
    }

    static string AddBatchConcat(GraphProto graph, IReadOnlyList<string> roiOutputs)
    {
        if(roiOutputs.Count == 1)
            return roiOutputs[0];

        graph.Node.Add(new NodeProto
        {
            Name = "concat_roi_batch",
            OpType = "Concat",
            Input = { roiOutputs },
            Output = { "roi_batch" },
            Attribute = { IntAttribute("axis", 0) }
        });
        return "roi_batch";
    }

    static void AddPaddleNormalization(GraphProto graph, string inputName, string outputName)
    {
        graph.Node.Add(new NodeProto
        {
            Name = "scale_0_1_to_0_2",
            OpType = "Mul",
            Input = { inputName, "paddle_scale" },
            Output = { "roi_batch_scaled" }
        });

        graph.Node.Add(new NodeProto
        {
            Name = "shift_0_2_to_minus1_1",
            OpType = "Add",
            Input = { "roi_batch_scaled", "paddle_bias" },
            Output = { outputName }
        });
    }

    static TensorProto GridTensor(
        string name,
        int sourceWidth,
        int sourceHeight,
        int targetWidth,
        int targetHeight,
        int batch,
        IReadOnlyList<float[]> targetToSourceMatrices)
    {
        if(targetToSourceMatrices.Count != batch)
            throw new ArgumentException("Matrix count must be equal to grid batch size.", nameof(targetToSourceMatrices));

        int itemCount = checked(batch * targetHeight * targetWidth * 2);
        var bytes = new byte[checked(itemCount * sizeof(float))];
        int byteOffset = 0;

        for(int batchIndex = 0; batchIndex < batch; batchIndex++)
            WriteGrid(bytes, ref byteOffset, sourceWidth, sourceHeight, targetWidth, targetHeight, targetToSourceMatrices[batchIndex]);

        return CreateGridTensor(name, bytes, batch, targetHeight, targetWidth);
    }

    static TensorProto GridTensor(
        string name,
        int sourceWidth,
        int sourceHeight,
        int targetWidth,
        int targetHeight,
        int batch,
        ReadOnlySpan<float> targetToSourceMatrix)
    {
        if(batch != 1)
            throw new ArgumentOutOfRangeException(nameof(batch), "Single-matrix grid tensor must have batch size 1.");

        var bytes = new byte[checked(targetHeight * targetWidth * 2 * sizeof(float))];
        int byteOffset = 0;
        WriteGrid(bytes, ref byteOffset, sourceWidth, sourceHeight, targetWidth, targetHeight, targetToSourceMatrix);
        return CreateGridTensor(name, bytes, batch, targetHeight, targetWidth);
    }

    static TensorProto CreateGridTensor(string name, byte[] bytes, int batch, int targetHeight, int targetWidth)
    {
        var tensor = new TensorProto
        {
            Name = name,
            DataType = (int)TensorProto.Types.DataType.Float,
            RawData = ByteString.CopyFrom(bytes)
        };
        tensor.Dims.Add(batch);
        tensor.Dims.Add(targetHeight);
        tensor.Dims.Add(targetWidth);
        tensor.Dims.Add(2);
        return tensor;
    }

    static void WriteGrid(
        byte[] bytes,
        ref int byteOffset,
        int sourceWidth,
        int sourceHeight,
        int targetWidth,
        int targetHeight,
        ReadOnlySpan<float> matrix)
    {
        if(matrix.Length < 9)
            throw new ArgumentException("Perspective matrix must contain 9 values.", nameof(matrix));

        for(int y = 0; y < targetHeight; y++)
        {
            for(int x = 0; x < targetWidth; x++)
            {
                float denominator = matrix[6] * x + matrix[7] * y + matrix[8];
                float sourceX = (matrix[0] * x + matrix[1] * y + matrix[2]) / denominator;
                float sourceY = (matrix[3] * x + matrix[4] * y + matrix[5]) / denominator;
                float normalizedX = NormalizeCoordinate(sourceX, sourceWidth);
                float normalizedY = NormalizeCoordinate(sourceY, sourceHeight);

                BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(byteOffset, sizeof(float)), normalizedX);
                byteOffset += sizeof(float);
                BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(byteOffset, sizeof(float)), normalizedY);
                byteOffset += sizeof(float);
            }
        }
    }

    static void ValidateArguments(
        int sourceWidth,
        int sourceHeight,
        int channels,
        int targetWidth,
        int targetHeight,
        IReadOnlyList<float[]> targetToSourceMatrices)
    {
        if(sourceWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(sourceWidth), "Source width must be positive.");

        if(sourceHeight <= 0)
            throw new ArgumentOutOfRangeException(nameof(sourceHeight), "Source height must be positive.");

        if(channels <= 0)
            throw new ArgumentOutOfRangeException(nameof(channels), "Channel count must be positive.");

        if(targetWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(targetWidth), "Target width must be positive.");

        if(targetHeight <= 0)
            throw new ArgumentOutOfRangeException(nameof(targetHeight), "Target height must be positive.");

        ArgumentNullException.ThrowIfNull(targetToSourceMatrices);

        if(targetToSourceMatrices.Count == 0)
            throw new ArgumentException("At least one ROI matrix is required.", nameof(targetToSourceMatrices));

        for(int index = 0; index < targetToSourceMatrices.Count; index++)
        {
            if(targetToSourceMatrices[index] is null || targetToSourceMatrices[index].Length < 9)
                throw new ArgumentException($"ROI matrix {index} must contain 9 values.", nameof(targetToSourceMatrices));
        }
    }

    static void ValidateStaticArguments(
        int sourceWidth,
        int sourceHeight,
        int channels,
        int targetWidth,
        int targetHeight,
        int regionCount)
    {
        if(sourceWidth <= 1)
            throw new ArgumentOutOfRangeException(nameof(sourceWidth), "Source width must be greater than 1.");

        if(sourceHeight <= 1)
            throw new ArgumentOutOfRangeException(nameof(sourceHeight), "Source height must be greater than 1.");

        if(channels <= 0)
            throw new ArgumentOutOfRangeException(nameof(channels), "Channel count must be positive.");

        if(targetWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(targetWidth), "Target width must be positive.");

        if(targetHeight <= 0)
            throw new ArgumentOutOfRangeException(nameof(targetHeight), "Target height must be positive.");

        if(regionCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(regionCount), "Region count must be positive.");
    }

    static float[] CreateTargetCoordinateValues(int targetWidth, int targetHeight, bool useX)
    {
        var values = new float[checked(targetWidth * targetHeight)];
        int index = 0;
        for(int y = 0; y < targetHeight; y++)
        {
            for(int x = 0; x < targetWidth; x++)
                values[index++] = useX ? x : y;
        }

        return values;
    }

    static TensorProto FloatTensor(string name, ReadOnlySpan<long> dimensions, ReadOnlySpan<float> values)
    {
        long itemCount = 1;
        var tensor = new TensorProto
        {
            Name = name,
            DataType = (int)TensorProto.Types.DataType.Float
        };

        foreach(long dimension in dimensions)
        {
            tensor.Dims.Add(dimension);
            itemCount = checked(itemCount * dimension);
        }

        if(itemCount != values.Length)
            throw new ArgumentException("Value count must match tensor dimensions.", nameof(values));

        var bytes = new byte[checked(values.Length * sizeof(float))];
        for(int index = 0; index < values.Length; index++)
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(index * sizeof(float), sizeof(float)), values[index]);

        tensor.RawData = ByteString.CopyFrom(bytes);
        return tensor;
    }

    static TensorProto FloatTensor(string name, ReadOnlySpan<float> values) =>
        OnnxGraphBuilderHelpers.FloatTensor(name, values);

    static void AddMul(GraphProto graph, string firstInput, string secondInput, string output) =>
        graph.Node.Add(new NodeProto
        {
            Name = $"mul_{output}",
            OpType = "Mul",
            Input = { firstInput, secondInput },
            Output = { output }
        });

    static void AddAdd(GraphProto graph, string firstInput, string secondInput, string output) =>
        graph.Node.Add(new NodeProto
        {
            Name = $"add_{output}",
            OpType = "Add",
            Input = { firstInput, secondInput },
            Output = { output }
        });

    static void AddDiv(GraphProto graph, string firstInput, string secondInput, string output) =>
        graph.Node.Add(new NodeProto
        {
            Name = $"div_{output}",
            OpType = "Div",
            Input = { firstInput, secondInput },
            Output = { output }
        });

    static float NormalizeCoordinate(float value, int size) =>
        size <= 1 ? 0f : 2f * value / (size - 1) - 1f;
}
