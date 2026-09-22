# NeuroModFlowNet.Pipeline.ONNX

ONNX integration layer for the experimental `NeuroModFlowNet.Pipeline` runtime.

This project keeps ONNX Runtime placement, generated ONNX operators, model endpoints, batching, native output access,
and decode logic out of the base pipeline project. VM instructions read named registers, call a typed endpoint or run a
small tensor operator, and write named registers back. Domain-specific model internals stay in services/resources.

## Current MVP scope

- generic `IOnnxInferenceEndpoint<TInput, TOutput>` / `Model_Inference<TInput, TOutput>` call path;
- `Model_OrtValueInference<TOutput>` path for GPU hot paths where the VM passes one prepared `OrtValue` into a service;
- bounded batched endpoint/resource base with typed request-to-result routing back to the VM execution that submitted the request;
- ONNX Runtime tensor operators for geometry, layout conversion, normalization, upload, and OCR ROI preparation;
- OrtValue batch assembly through ONNX Runtime graphs for GPU-resident inputs;
- typed decoders that return only request-owned results such as `YoloObb[]` or `PaddleOCRRecExtractor.OcrResult[]`;
- compatibility wrappers over existing `IRunner<TIn,TOut>` assets where a legacy Mat-based path is still used.

Large raw tensors should stay inside ONNX Runtime resources or VM `OrtValue` registers. VM-visible service results should
be typed arrays, even when a current model normally returns one object.

## Project Layout

- `Resources/` - inference endpoints, bounded batching, OrtValue batch assembly, output shape and decode contracts.
- `Runtime/` - runtime-generated ONNX operator kernel cache.
- `Instructions/` - generic inference calls, tensor operators, and memory bridge instructions.
- `Coordinates/` - VM coordinate back-transform application for typed payloads.
- `Yolo/` - YOLO output decoders plus legacy/domain wrappers.
- `PaddleOCR/` - PaddleOCR output decoders plus legacy/domain wrappers.

## Lifetime Note

Some existing `IRunner<TIn,TOut>` implementations dispose their `OnnxRuntimeContext`. `ReloadableOnnxRunnerResource`
therefore allows replacement without disposing the previous runner immediately. This is needed for PaddleOCR Rec, where
the runner is rebuilt after UI shape changes but the model context stays the same resource until shutdown.

Generated tensor operators and explicit upload steps own mutable ONNX Runtime binding state. They must serialize each
instruction instance internally when a VM controller runs several executions in parallel.
