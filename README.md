# NeuroModFlowNet.ONNX

## Overview

**NeuroModFlowNet.ONNX** is a modular high-performance library built on top of [ONNX Runtime](https://onnxruntime.ai/) for computer vision workloads. Its main focus is real-time inference, where stable latency, low allocation pressure, and strongly typed data flow are more important than a minimal wrapper around `InferenceSession`.

The library covers the common but usually repetitive ONNX Runtime workflow: prepare input data, run inference, decode raw output tensors, and return convenient typed results. Instead of writing the same glue code around `InferenceSession`, `OrtValue`, tensor layouts, and postprocessing in every project, these responsibilities are split into explicit layers.

The main data flow is:

```text
Input data -> Converter -> OnnxRuntimeContext / Runner -> Extractor -> Typed result
```

## Main Components

### Core

`Core` contains the runtime layer. The central type is `OnnxRuntimeContext`: it owns the `InferenceSession`, backend settings, model metadata, and the resources needed to run inference.

This layer also contains strongly typed execution provider configuration, including CUDA and TensorRT. Keeping these settings close to runtime context creation avoids spreading provider-specific string options through application code.

### Inputs

`Inputs` is responsible for input preparation. Converters receive external data, such as `OpenCvSharp.Mat` or a list of `Mat`, and transform it into the format expected by the model.

This layer contains:

- common converter contracts;
- implementations for `BGR`, `NCHW`, positive normalization, and PaddleOCR inputs;
- low-level preparation algorithms under `Inputs/Algorithms`;
- helper classes for checking and auto-selecting algorithms.

For the hot path, the important rule is simple: strategy selection and expensive validation should happen before frame processing, not inside every frame.

### Runners

`Runners` connects an input converter, a runtime context, and an extractor into an executable pipeline. The base contract is `IRunner<TIn, TOut>`.

A runner receives input data, prepares the input buffers, runs ONNX Runtime, and returns a typed result. Image-based scenarios use `ImageRunner`, while the common execution flow is implemented by `StrategyRunner`.

### Outputs

`Outputs` is responsible for interpreting model outputs. It contains extractors, packed-layout structures, result structures, and factories.

YOLO outputs are grouped by task family:

- `Box`;
- `Cls`;
- `Obb`;
- `Pose`;
- `Seg`.

Where applicable, each family uses the following folders:

- `InnerData` — structures that match output tensor layout and can be used with `MemoryMarshal.Cast`;
- `OutData` — independent result structures;
- `Extractors` — logic that converts raw model outputs into typed results;
- `Factories` — fixed factory methods and metadata-based runner creation.

PaddleOCR outputs are also supported, including detection, recognition, and related tasks.

### Visualizer

`NeuroModFlowNet.ONNX.Visualizer` is a separate project for rendering inference results. Visualization is intentionally kept outside the core library, so runtime code does not depend on UI or debug-only concerns.

It contains painter classes for YOLO Box, OBB, Seg, Pose, Cls, plus helper methods for working with `Mat`.

### Diagnostics

`NeuroModFlowNet.ONNX.Diagnostics` is a separate project for displaying information about a runtime context and model structure. For example, `WriteInfo(...)` can print the backend, model name, inputs, outputs, and selected metadata when needed.

This code is separated for the same reason as visualization: production runtime code should stay lean, while development and investigation still need quick, readable diagnostics.

## Supported Workloads

The main workloads currently target YOLO models:

- detection / box;
- oriented bounding boxes;
- segmentation;
- pose;
- classification.

The project also includes PaddleOCR support, including detection and recognition.

For YOLO26, see:

- `src/NeuroModFlowNet.ONNX/Doc/Yolo26_2.md` — real existing model exports and output variants;
- `src/NeuroModFlowNet.ONNX/Doc/Factores.md` — comparison between current factory/extractor implementations and those real variants.

## Runtime Requirements

The library targets `.NET 10` and uses ONNX Runtime as the inference engine. The core project references `Microsoft.ML.OnnxRuntime.Gpu` when `UseGpu=true`, and falls back to the CPU ONNX Runtime package when GPU support is disabled at build time.

CPU execution does not require NVIDIA runtime libraries. CUDA and TensorRT execution do require native NVIDIA dependencies that match the ONNX Runtime GPU package used by the application.

For CUDA execution, install:

- a compatible NVIDIA driver;
- CUDA Toolkit runtime libraries;
- cuDNN runtime libraries;
- `Microsoft.ML.OnnxRuntime.Gpu` or `Microsoft.ML.OnnxRuntime.Gpu.Windows` in the final application.

For TensorRT execution, install everything required for CUDA execution, plus TensorRT libraries built for the same CUDA major/minor line. TensorRT is also more sensitive to model shape support and operator coverage, and may need time to build an engine on the first run.

On Windows, the native library directories must be discoverable before creating `InferenceSession`. Use either the system `PATH`, or add the paths to the executable `App.config` and call `OnnxRuntimePathHelper.InitFromConfig()` before creating any runtime context:

```xml
<appSettings>
  <add key="CudaBinPath" value="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin" />
  <add key="CudnnBinPath" value="C:\cuDNN\9.10.2.21_cuda12\bin" />
  <add key="TrtLibPath" value="C:\TensorRT\TensorRT-10.12.0.36-cuda12.9\lib" />
</appSettings>
```

These values are local machine paths, not portable project defaults. Keep them aligned with the CUDA/cuDNN/TensorRT versions installed on the target machine.

TensorRT engine cache is enabled by default in `OnnxRuntimeContext` when `InferenceBackend.TensorRt` is selected without custom provider configuration. The cache path is resolved in this order:

- `TrtConfigDefaults.CustomEngineCachePath`;
- `NMFN_ONNX_TRT_CACHE_PATH` environment variable;
- `%LOCALAPPDATA%\NeuroModFlowNet\TRT_Cache`.

## Factories and Automatic Selection

Fixed factory methods exist for common combinations of input format, output format, and precision. They allow explicit runner creation for scenarios such as FP32, FP16, direct BGR input, or a specific YOLO task family.

Where possible, metadata-based creation is available through `CreateRunner(...)`. This path reads model metadata and automatically selects a compatible converter and extractor. It is useful for groups of similar ONNX exports where the differences are input tensor type, output precision, or result shape.

## Performance Model

The project is designed around practical constraints of real-time systems.

First, the hot path should be short and predictable. Algorithm selection, metadata analysis, buffer creation, and extractor choice should happen during initialization. During frame processing, the runner should execute an already selected sequence of operations.

Second, the library tries to avoid unnecessary allocations. It uses prepared buffers, `Span<T>`, `ReadOnlySpan<T>`, packed structures, and direct data mapping where the model format allows it.

Third, when a model accepts `U8 BGR` input, CPU-side preprocessing can be much simpler. In that scenario, part of preprocessing is moved into the ONNX graph, and the external pipeline can stay closer to zero-copy execution.

## Repository Structure

- `src/NeuroModFlowNet.ONNX` — the core library.
- `src/NeuroModFlowNet.ONNX.Visualizer` — rendering of inference results.
- `src/NeuroModFlowNet.ONNX.Diagnostics` — diagnostic output for models and runtime contexts.
- `models/` — shared local model assets used by samples, labs, and manual library checks. Each network uses its own folder, for example `models/yolo26s/yolo26s__640_b1_fp32.onnx`.
- `ModelPrepare/` — maintenance tooling for shared model assets. `Preparation/` contains download/export scripts, while `HuggingFace/` contains scripts for publishing prepared ONNX artifacts to external model storage.
- `samples/NeuroModFlowNet.ONNX.Demo.Dashboard` — demo dashboard for parallel frame processing by multiple pipeline processors.
- `samples/NeuroModFlowNet.ONNX.Demo.Assets` — shared assets for demo projects.
- `labs/NeuroModFlowNet.ONNX.Lab` — interactive experiments and camera scenarios.
- `labs/NeuroModFlowNet.ONNX.Lab.Algorithms` — visual checks and demonstrations for input algorithms.
- `labs/NeuroModFlowNet.ONNX.Bench` — benchmark project and measurements.
- `tests/NeuroModFlowNet.ONNX.Tests` — unit and integration tests.
- `tools/NeuroModFlowNet.ONNX.Tools` — tools for inspecting and modifying ONNX models.

## Tools

`NeuroModFlowNet.ONNX.Tools` is a standalone CLI and library for working with ONNX models at graph level. It is used to inspect model structure, inputs, outputs, and attributes, and also for deeper transformations such as adding preprocessing heads or preparing a model for a specific backend.

In practice, this layer can be as important as runtime code: when preprocessing can be moved into the model, the external pipeline becomes simpler and faster.

## Further Reading

- `src/NeuroModFlowNet.ONNX/Doc/Architecture_ru.md` — architecture notes.
- `src/NeuroModFlowNet.ONNX/Doc/ConvertersTranslators_ru.md` — historical and technical notes about converters and algorithms.
- `src/NeuroModFlowNet.ONNX/Doc/Yolo26_2.md` — real YOLO26 exports.
- `src/NeuroModFlowNet.ONNX/Doc/Factores.md` — factory/extractor implementation comparison.
- `labs/NeuroModFlowNet.ONNX.Bench/Doc/ReadMe.md` — current benchmark measurements.

## License

The NeuroModFlowNet.ONNX source code is licensed under the Apache License 2.0. See `LICENSE`.

Third-party dependencies keep their own licenses. CUDA, cuDNN and TensorRT are not distributed with this repository and must be installed separately according to NVIDIA license terms.

## Model License

Model weights and prepared model artifacts are not covered by the Apache-2.0 license of this source code repository unless explicitly stated.

YOLO model weights and YOLO-derived prepared artifacts are provided under AGPL-3.0, consistent with the Ultralytics YOLO licensing terms.

The author provides these weights "as is", without warranty. Users are responsible for ensuring that their use complies with the licenses of:

- Ultralytics YOLO;
- the base/pretrained model, if any;
- the training dataset;
- any deployment/runtime libraries.
