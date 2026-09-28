using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeuroModFlowNet.ONNX;
using NeuroModFlowNet.Pipeline.ONNX;

namespace NeuroModFlowNet.Pipeline.Video.Nvdec;

/// <summary>
/// EN: Colour/layout macro-op at the video boundary: pitched NV12 device surface <c>[allocHeight * 3 / 2, pitch] U8</c>
/// → <c>BGR U8 NHWC [1, H, W, 3]</c> on the model device. The output is the same bridge format that
/// <c>Wrap_MatImage_To_OrtTensor</c> + <c>Copy_OrtTensor_To_ModelDevice</c> produce from a Mat, so downstream programs
/// (prepare/common) are unchanged. Frame size and colour matrix are explicit and part of the kernel cache key.
///
/// RU: Color/layout macro-op на границе видео: NV12-поверхность устройства с pitch <c>[allocHeight * 3 / 2, pitch] U8</c>
/// → <c>BGR U8 NHWC [1, H, W, 3]</c> на устройстве модели. Выход в том же bridge-формате, что дают
/// <c>Wrap_MatImage_To_OrtTensor</c> + <c>Copy_OrtTensor_To_ModelDevice</c> из Mat, поэтому программы prepare/common не
/// меняются. Размер кадра и цветовая матрица явные и входят в ключ кэша kernel-а.
/// </summary>
public sealed class Op_Onnx_Nv12_To_BgrU8Nhwc : Op_Onnx_TensorTransformBase
{
    readonly int width;
    readonly int height;
    readonly Nv12ColorMatrix matrix;

    public Op_Onnx_Nv12_To_BgrU8Nhwc(
        string inputKey,
        string outputKey,
        int width,
        int height,
        Nv12ColorMatrix matrix,
        bool isFinal = false,
        InferenceBackend? executionBackend = null)
        : base(
            OpDescriptor.Create(
                "Op_Onnx_Nv12_To_BgrU8Nhwc",
                "op.onnx.nv12.toBgrU8Nhwc",
                reads: [VarRequirement.Read<OrtValue>(inputKey)],
                writes: [VarRequirement.Write<OrtValue>(outputKey)]),
            inputKey,
            outputKey,
            isFinal,
            executionBackend)
    {
        if(width <= 0 || height <= 0 || width % 2 != 0 || height % 2 != 0)
            throw new ArgumentOutOfRangeException(nameof(width), "NV12 frame width and height must be positive and even.");

        this.width = width;
        this.height = height;
        this.matrix = matrix;
    }

    protected override string GraphInputName => Nv12ToBgrGraphBuilder.InputName;

    protected override string GraphOutputName => Nv12ToBgrGraphBuilder.OutputName;

    protected override string DisplayName => "nv12-to-bgr-u8-nhwc.onnx";

    protected override string GetCacheSemanticKey(long[] inputShape, TensorElementType inputElementType) =>
        $"frame={width}x{height};matrix={matrix}";

    protected override string? ValidateInput(long[] inputShape, TensorElementType inputElementType)
    {
        if(inputElementType != TensorElementType.UInt8)
            return $"NV12 input must be UInt8, actual: {inputElementType}.";

        if(inputShape.Length != 2 || inputShape[0] % 3 != 0)
            return $"NV12 input must be [allocHeight * 3 / 2, pitch], actual: [{string.Join(", ", inputShape)}].";

        return null;
    }

    protected override byte[] BuildModel(long[] inputShape, TensorElementType inputElementType) =>
        Nv12ToBgrGraphBuilder.Build(checked((int)inputShape[0]), checked((int)inputShape[1]), width, height, matrix);

    protected override long[] CreateOutputShape(long[] inputShape, TensorElementType inputElementType) =>
        [1, height, width, 3];
}
