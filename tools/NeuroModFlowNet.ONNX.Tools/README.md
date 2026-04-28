# NeuroModFlowNet.ONNX.Tools

`NeuroModFlowNet.ONNX.Tools` is a command line tool and helper library for working with ONNX model files.

The tool is intended for model inspection and small graph changes that make runtime inference easier and faster. It is separate from the core runtime library because most applications do not need ONNX graph editing code during normal inference.

## What It Does

- Shows ONNX model structure in a readable console view.
- Prints model inputs, outputs, values, initializers, attributes, and graph nodes.
- Lets you limit the graph view to the first and last nodes, or show the full graph.
- Injects preprocessing heads into ONNX models.
- Registers the tool in the operating system for `.onnx` files and terminal aliases.

## Commands

### View Model Structure

```powershell
dotnet run --project tools/NeuroModFlowNet.ONNX.Tools -- view path\to\model.onnx
```

Useful options:

```powershell
dotnet run --project tools/NeuroModFlowNet.ONNX.Tools -- view path\to\model.onnx --all
dotnet run --project tools/NeuroModFlowNet.ONNX.Tools -- view path\to\model.onnx --top 20 --bottom 20
dotnet run --project tools/NeuroModFlowNet.ONNX.Tools -- view path\to\model.onnx --filter IOTA
dotnet run --project tools/NeuroModFlowNet.ONNX.Tools -- view path\to\model.onnx --names
```

The filter letters are:

- `I` - graph inputs;
- `O` - graph outputs;
- `V` - intermediate values;
- `T` - tensors / weights;
- `A` - node attributes.

### Inject a Preprocessing Head

```powershell
dotnet run --project tools/NeuroModFlowNet.ONNX.Tools -- inject ByteBGR_FP32 path\to\model.onnx
```

Available head types:

- `ByteBGR_FP16` - NHWC byte BGR input, converted to FP16 RGB with `0..1` normalization.
- `ByteBGR_FP32` - NHWC byte BGR input, converted to FP32 RGB with `0..1` normalization.
- `SequenceByteBGR` - sequence byte BGR preprocessing head.

You can also choose the output path or suffix:

```powershell
dotnet run --project tools/NeuroModFlowNet.ONNX.Tools -- inject ByteBGR_FP32 path\to\model.onnx --output path\to\model_head.onnx
dotnet run --project tools/NeuroModFlowNet.ONNX.Tools -- inject ByteBGR_FP32 path\to\model.onnx --extra _byte_bgr
```

Preprocessing heads are used to move simple input conversion into the ONNX graph. This can reduce external preprocessing work and make the runtime pipeline simpler, especially when frames already arrive as byte BGR data.

### Register File Association

```powershell
dotnet run --project tools/NeuroModFlowNet.ONNX.Tools -- register
```

The register command can add system integration for `.onnx` files and a terminal alias. On Windows it writes file association data under the current user's registry hive. On Linux it creates or removes command alias integration where supported by the current implementation.

Options:

```powershell
dotnet run --project tools/NeuroModFlowNet.ONNX.Tools -- register --alias nmf-onnx
dotnet run --project tools/NeuroModFlowNet.ONNX.Tools -- register --unregister
```

## Extending Head Injection

New model heads can be added by creating an `OnnxModelModifier` implementation and marking it with `OnnxHeadAttribute`.

At startup, the tool scans the current assembly for non-abstract `OnnxModelModifier` classes with this attribute and registers them in `OnnxHeadRegistry`. After that, the head can be used by the `inject` command.

## ONNX Proto Source

The project contains generated C# types for the ONNX protobuf schema in `OnnxReflection/Onnx.cs`.

To regenerate it:

1. Clone the ONNX repository.
2. Download `protoc` from the Protobuf releases.
3. Copy `protoc.exe` into the folder that contains `onnx.proto`.
4. Run:

```powershell
protoc --csharp_out=. onnx.proto
```

5. Add the generated `Onnx.cs` file to this project.

## Why This Project Is Separate

The core runtime should stay focused on inference. Graph inspection, graph modification, file association, and model preparation utilities are useful during development and deployment, but they are not part of the hot inference path.
