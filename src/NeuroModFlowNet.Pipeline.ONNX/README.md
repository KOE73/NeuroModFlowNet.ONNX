# NeuroModFlowNet.Pipeline.ONNX

ONNX integration layer for the experimental `NeuroModFlowNet.Pipeline` runtime.

This project keeps ONNX Runtime and existing NeuroModFlowNet.ONNX runner/extractor types out of the base pipeline
project. The base VM sees domain instructions such as YOLO OBB or YOLO Box, while this project owns model runners,
batching resources, and compact detection results.

## Current MVP scope

- bounded batched ONNX resource base;
- typed request-to-result routing back to the VM execution that submitted the request;
- YOLO OBB and YOLO Box batch resources using the existing `IRunner<List<Mat>, IDetectionResult<T>>` contracts;
- serialized runner resources for current batch-1 assets;
- reloadable runner resources for dynamic PaddleOCR Rec shape changes;
- `Model_Yolo*` commands for Box, OBB, Seg, Cls and Pose inference;
- `Model_PaddleDet` and `Model_PaddleRec` commands for PaddleOCR inference;
- compact CPU detection results written to transaction variables.

Large raw tensors should stay inside resources unless a later GPU instruction explicitly needs them.

## Project Layout

- `Resources/` - generic ONNX resource wrappers, bounded batching and reloadable runner lifetime.
- `Instructions/` - reusable runner instructions independent from YOLO/Paddle naming.
- `Yolo/` - domain commands and compact result containers for YOLO-family outputs.
- `PaddleOCR/` - PaddleOCR detector and recognition commands.

## Lifetime Note

Some existing `IRunner<TIn,TOut>` implementations dispose their `OnnxRuntimeContext`. `ReloadableOnnxRunnerResource`
therefore allows replacement without disposing the previous runner immediately. This is needed for PaddleOCR Rec, where
the runner is rebuilt after UI shape changes but the model context stays the same resource until shutdown.
