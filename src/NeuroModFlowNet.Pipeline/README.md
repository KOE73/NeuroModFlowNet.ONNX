# NeuroModFlowNet.Pipeline

Experimental base runtime for the new timeline/VM pipeline model.

The project intentionally has no dependency on ONNX Runtime. It owns only the orchestration primitives:

- accepted `RunId` per source;
- string-key transaction registers;
- compiled instruction timeline;
- `Continue`, `Jump`, `Stop`, and controlled failure results;
- ordered sync gates for stateful instructions;
- global memory and named resources;
- named coordinate back-transform registers.
- neutral VM run inputs/outputs and instruction trace diagnostics.

The key design rule is: one accepted run executes one program chain from start to finish with one shared run context, while slow or stateful resources are called through explicit instructions.

## Program chain

`VmController` accepts one program or an ordered list of programs. Every accepted run executes the programs in order
with the same `VmRunContext`, so the chain is semantically one concatenated program: same `RunId`, same global memory,
same sync gates, same disposable outputs. Typical use is `[prepare, common]`: a source-specific preparation program
(perspective, undistort, resize with per-camera parameters) followed by a program that is identical code for every
source. Labels stay local to each program, `Stop` ends the whole run, a failure in any program fails the run, and
`OpTraceEntry.Program` tells which program a trace sample belongs to.

The chain never crosses controllers, and program instances are not shared between controllers: instructions may own
mutable native state, so "identical for every source" means the same builder output, not the same object. Expensive
state (sessions, model endpoints) is already shared through resources and the kernel cache. Rationale:
`docs/architecture/ADR-001_program_chain_per_controller.ru.md`.

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
- `Synchronization/` - reusable ordered gates that release runs in dense `RunId` order for stateful stages such as trackers. Ordering only: a stage that also needs per-frame mutual exclusion must serialize itself or be run inside an ordered critical section (see docs/architecture/tracker_and_source_config.ru.md).
- `Memory/` and `Resources/` - global per-source memory and named shared resources.
- `Coordinates/` - small back-transform contracts stored as normal named VM registers.

## Current Constraints

The public variable contract is string-key based by design. A later compiler can map keys to faster internal slots, but
JSON configs, scripts and diagnostics should keep using the same names.
