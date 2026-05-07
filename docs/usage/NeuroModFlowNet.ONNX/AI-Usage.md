# NeuroModFlowNet.ONNX Package Usage Rules

This file is for coding agents writing public usage documentation for package consumers.

## Audience

Write for a developer who installed the NuGet package and wants to use it in their own application.

Do not describe repository internals, source file locations, sample project layout, lab projects, build scripts, or internal maintenance workflow in human usage documents.

## Canonical Flow

Document the package flow in this order:

1. Create `OnnxRuntimeContext`.
2. Create an `IRunner<TIn, TOut>` through a factory.
3. Call `Predict(input)`.
4. Read the typed result.
5. Dispose `OnnxRuntimeContext`, runners, and `Mat` values.

## Required Concepts

Human usage docs should cover:

- model loading with `OnnxRuntimeContext`;
- backend choices: `Cpu`, `Cuda`, `TensorRt`, `DML`, `Rocm`;
- fine provider configuration through `ExecutionProviderConfig`;
- `CudaConfig` examples;
- `TrtConfig` examples;
- native DLL visibility for CUDA/cuDNN/TensorRT;
- the runner principle: input preparation, inference, output extraction;
- exact factory methods for fixed scenarios;
- automatic `CreateRunner` methods for metadata-based selection;
- `Predict` input requirements;
- `IDetectionResult<T>` result access.

## Do Not Write

Do not include:

- repository file paths;
- internal project names;
- references to labs, samples, tests, or tools as project structure;
- instructions for developing the library itself;
- undocumented APIs;
- guessed behavior.

If a behavior is unclear, add a TODO instead of inventing it.

## Preferred Examples

Use compact package-consumer examples:

```csharp
using var context = new OnnxRuntimeContext("model.onnx", InferenceBackend.Cuda);
using var runner = YoloBoxFactory.CreateRunner<IDetectionResult<YoloBox>>(context);
IDetectionResult<YoloBox> result = runner.Predict(image);
```
Show exact factory methods when the user must control preprocessing or precision:

```csharp
var runner = YoloBoxFactory.Single_PosCvdnn_FP32(context);
```

Show automatic factory methods when model metadata should drive converter/extractor selection:

```csharp
var runner = YoloObbFactory.CreateRunner<IDetectionResult<YoloObb>>(context);
```

Show provider tuning through typed config checks:

```csharp
using var context = new OnnxRuntimeContext(
    "model.onnx",
    InferenceBackend.TensorRt,
    config =>
    {
        if (config is TrtConfig trt)
        {
            trt.EnableFp16 = true;
            trt.EnableEngineCache = true;
        }
    });
```
