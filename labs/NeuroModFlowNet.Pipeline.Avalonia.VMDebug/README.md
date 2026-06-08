# NeuroModFlowNet.Pipeline.Avalonia.VMDebug

Avalonia lab for visually testing and debugging the new pipeline VM direction.

Current scope:

- opens exactly one configured source outside the VM, with camera `0` as the default path;
- passes accepted source data to one `PipelineVmController` run through named VM inputs;
- prepares source/model frames through VM instructions;
- runs the current strongly typed VM program through resources; model selection UI is intentionally removed from this lab;
- prepares OCR regions and recognition batches in VM transaction memory;
- returns a neutral frame-result DTO as a selected VM output;
- shows one UI-owned frame, overlays and metrics;
- provides an editable YAML-like program surface, compile diagnostics, trace stepping, per-instruction
  timing, variable list, and conservative CPU/GPU/unknown placement markers;
- reserves the right side for future dynamic watch/debug outputs generated from VM variables and `debugOut`.

## Project Layout

- `Controls/` - minimal scene and metrics controls for the VM debugger.
- `Rendering/` - Skia frame and overlay drawing helpers.
- `Runtime/Instructions/` - VM commands that replace the old monolithic realtime loop.
- `Runtime/Debug/` - lab-only textual program parser and debug surface helpers.
- `Runtime/ModelResources/` - ONNX model/context/resource ownership for this lab, resolved through `IOnnxAssetResolver`.
- `Runtime/Ocr/` - OCR-specific transaction payloads.
- `docs/` - rationale and architecture notes for the lab.

The lab is deliberately a consumer of `NeuroModFlowNet.Pipeline` and `NeuroModFlowNet.Pipeline.ONNX`. New base VM
concepts should move into those projects instead of being hidden in the Avalonia application.
