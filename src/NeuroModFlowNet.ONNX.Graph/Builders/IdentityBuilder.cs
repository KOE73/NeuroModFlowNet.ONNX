using static NeuroModFlowNet.ONNX.Graph.Builders.OnnxGraphBuilderHelpers;

namespace NeuroModFlowNet.ONNX.Graph.Builders;

/// <summary>
/// EN: Dynamic ONNX model builder for an Identity operator.
/// RU: Динамический построитель ONNX-модели с оператором Identity.
/// </summary>
public static class IdentityBuilder
{
    public const string InputName = "input";
    public const string OutputName = "output";

    public static byte[] Build(
        TensorProto.Types.DataType dataType,
        params long[] shape)
    {
        var graph = new GraphProto { Name = "identity_only" };

        graph.Input.Add(TensorInfo(InputName, dataType, shape));
        graph.Output.Add(TensorInfo(OutputName, dataType, shape));

        graph.Node.Add(new NodeProto
        {
            Name = "identity",
            OpType = "Identity",
            Input = { InputName },
            Output = { OutputName }
        });

        return CreateModel("neuromodflownet-runtime-identity-builder", graph).ToByteArray();
    }
}
