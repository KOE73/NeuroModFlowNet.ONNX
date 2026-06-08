# NeuroModFlowNet.Pipeline

Experimental base runtime for the new timeline/VM pipeline model.

The project intentionally has no dependency on ONNX Runtime. It owns only the orchestration primitives:

- accepted `RunId` per source;
- string-key transaction registers;
- compiled instruction timeline;
- `Continue`, `Jump`, `Stop`, and controlled failure results;
- ordered sync gates for stateful instructions;
- global memory and named resources;
- coordinate-space transform graph.
- neutral VM run inputs/outputs and instruction trace diagnostics.

The key design rule is: one accepted run executes one program from start to finish, while slow or stateful resources are called through explicit instructions.

## Boundaries

`NeuroModFlowNet.Pipeline` must not own camera/file/RTSP capture and must not know about ONNX model paths, YOLO,
PaddleOCR, CUDA, DirectML, or visualization.

ONNX-specific instructions and batched model resources live in `NeuroModFlowNet.Pipeline.ONNX`.

The base VM accepts named input variables through `PipelineRunInputs` and returns selected named values through
`PipelineRunOutput`. This keeps the same VM usable as a standalone realtime loop or as one module inside an outer
event-based system.

Model and auxiliary asset lookup is outside the VM. ONNX-related assets should be resolved through the ONNX core
`IOnnxAssetResolver` boundary before resources are created.

## Project Layout

- `Execution/` - VM admission, dense accepted `RunId`, named inputs/outputs, run ledger and completion handles.
- `Diagnostics/` - per-instruction timing and conservative variable memory placement snapshots.
- `Transactions/` - per-run local memory with string keys and owned resource cleanup.
- `Instructions/` - VM instruction contracts, program builder, conditional steps, jumps and ordered sync instruction.
- `Synchronization/` - reusable ordered gates for stateful external services such as trackers.
- `Memory/` and `Resources/` - global per-source memory and named shared resources.
- `Coordinates/` - transform graph for reversible coordinate mapping without fixed `scale/pad` metadata.

## Current Constraints

The public variable contract is string-key based by design. A later compiler can map keys to faster internal slots, but
JSON configs, scripts and diagnostics should keep using the same names.
