namespace NeuroModFlowNet.ONNX.Tools;

public abstract class OnnxModelModifier : OnnxModelToolBase
{
    public OnnxModelModifier(string? extraName = null)
    {
        ExtraName = string.IsNullOrWhiteSpace(extraName) ? GetDefaultExtraName() : extraName;
    }

    public string ExtraName { get; set; }

    public abstract string GetDefaultExtraName();

    public void Inject(string path)
    {
        ModelProto model = LoadModel(path);
        Inject(model);
        SaveModifiedModel(model, path, ExtraName);
    }

    public abstract void Inject(ModelProto model);



    /// <summary>
    /// Creates a new instance of <see cref="ValueInfoProto"/> that describes a tensor with
    /// the specified name, data type, and shape.
    /// 
    /// </summary>
    /// <remarks>The returned <see cref="ValueInfoProto"/> includes a tensor type with the specified element type and
    /// shape. This method is useful for constructing model metadata when defining inputs,
    /// outputs, or intermediate tensors in ONNX models.</remarks>
    /// <param name="name">The name that identifies the tensor within the model.</param>
    /// <param name="type">The data type of the tensor elements, specified as a <see cref="TensorProto.Types.DataType"/> value.</param>
    /// <param name="dims">An array of dimensions that defines the shape of the tensor. 
    ///   Each dimension is represented by a <see cref="TensorShapeProto.Types.Dimension"/>.</param>
    /// <returns>A <see cref="ValueInfoProto"/> object containing the provided name, data type, and
    /// shape information for the tensor.</returns>
    protected ValueInfoProto CreateValueInfo(
        string name,
        TensorProto.Types.DataType type,
        params TensorShapeProto.Types.Dimension[] dims)
    {
        ValueInfoProto valueInfo = new ValueInfoProto { Name = name };
        TypeProto.Types.Tensor tensor = new TypeProto.Types.Tensor
        {
            ElemType = (int)type,
            Shape = new TensorShapeProto()
        };

        foreach(var dim in dims) tensor.Shape.Dim.Add(new TensorShapeProto.Types.Dimension(dim));
        valueInfo.Type = new TypeProto { TensorType = tensor };
        return valueInfo;
    }

}
