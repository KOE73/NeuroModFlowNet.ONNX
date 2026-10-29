# Multi-camera VM web debugger

Separate ASP.NET Core laboratory host for the existing MultiCamera pipeline. No VM/ONNX core changes, no Avalonia dependency, no replacement inference path. Programs, sources, transforms, shared endpoints and strict backend policies come from `NeuroModFlowNet.Pipeline.MultiCamera`.

One compact browser overview provides per-camera pause, continue, single-instruction step, next accepted frame, program-local breakpoints, register metadata, typed values and explicit intermediate image inspection. Each camera retains its `[prepare:<id>, common]` chain and one `VmRunContext`. Only this debugger limits concurrency to one in-flight run per camera. The ordinary MultiCamera host keeps its original concurrency and does not create decorators.

## Launch

Use an existing multi-camera JSON configuration. Local native dependency paths go in git-ignored `App.local.config` beside the executable, following the existing MultiCamera `App.config` convention. The configuration and media files are read, never modified.

```powershell
./Run-WebDebug.ps1 -CamerasConfig C:/private/cameras.json `
  -FFmpegLibrariesPath C:/ffmpeg/bin -ModelsRoot C:/models
```

Default: `http://127.0.0.1:5264`. The script checks the port and fails if occupied. Port 443 is prohibited. The web server starts with the cameras stopped; **Start** initializes the actual sources and kernels. Warmup runs without stopping. The first working frame pauses before its first instruction. New TensorRT kernels can take several minutes on first use; the UI remains available throughout.

### Explicit CUDA 12 runtime

The NuGet ONNX Runtime 1.30 GPU package defaults to CUDA 13. For an existing CUDA 12.9 installation, select the official **same-version CUDA 12** native build explicitly. This affects only this debugger; it does not downgrade the managed packages or modify the normal lab's output. The installer verifies the release archive's SHA-256 and leaves licenses beside the native libraries.

```powershell
$native = ./Install-Cuda12Runtime.ps1
./Run-WebDebug.ps1 -CamerasConfig C:/private/cameras.json `
  -FFmpegLibrariesPath C:/ffmpeg/bin -ModelsRoot C:/models -NativeRuntimeRoot $native
```

The explicit build input is `-p:WebDebugNativeRoot=<native lib directory>`. It copies the selected native DLLs into this host's output. Use the same option on each build that will be installed. Omission selects the normal NuGet runtime. The launcher installs a separate copy outside the checkout so a running UI does not lock solution builds. It never stops another process to deploy.

### Remote browser

The default loopback endpoint can be forwarded over an existing SSH/VPN connection. To bind directly to a LAN interface, set `WebDebug__AccessToken` (at least 24 characters) and explicitly pass `-ListenHost <server IP>`. Open `http://<server IP>:5264` and enter the same token in **Settings**. The browser keeps it in session storage for that tab. API calls use `Authorization: Bearer ...`; tokens never go into URLs. Use a trusted LAN/VPN or HTTPS with an explicit nonstandard port when a transport needs encryption. The launcher does not change firewall rules. A non-loopback bind without a token fails startup.

## Inspection and ownership

- The debug-only instruction decorator preserves descriptors, results, branch semantics, device validation and disposal of the underlying instruction. The VM and ONNX core have no dependency on the debugger.
- Stop snapshots contain metadata and small scalar values. A live context/OrtValue/Mat is never handed to an HTTP thread. Inspection requests are bounded and are executed by the waiting VM task. Stale stop revisions are rejected.
- **Step / F6** executes one instruction and stops after it. **Continue / F8** resumes to a breakpoint or pause. **Next frame** finishes the current run and stops before the first instruction of the next accepted run; configured breakpoints remain effective.
- Each register row exposes only supported actions: the Tabler list icon reads a typed JSON value; the eye icon explicitly requests an image. The server advertises `canInspectValue` and `canPreviewImage` in the paused register metadata, including batch/layout restrictions. Native resources with no supported inspection have empty action cells.
- The instruction pointer has its own fixed column before `#`; moving it never changes the instruction name position. The row's **Run to** icon sets a temporary program-local target and stops **before** it executes. Existing breakpoints remain active; any earlier stop clears the temporary target. A passed target may be reached on a later frame. Pause, Continue, Step, Next frame and Detach cancel a pending run-to target.
- Capture continues into the existing two-frame drop-oldest queue while a VM is paused. Next frame means next VM-accepted frame, not every consecutive frame in a recording. Other cameras remain independently controllable.
- Image inspection is explicit on pause: NV12 is converted on the configured GPU backend, images are resized there and explicitly downloaded by the existing ONNX operator's `isFinal` path. Native GPU pointers are never read in managed code. No CPU/provider fallback is introduced.
- Register checkboxes enable live observation of up to eight images or typed JSON values together. The inspection pane has a master observation switch, a per-register frequency limit (0.2–15 FPS) and a maximum image edge (100–2000 pixels). Tiles retain a fixed content size while new samples arrive; data and diagnostics never insert banners or shift the workspace. Charts/history are deliberately deferred.
- Observation captures at the last executed writer of the selected register at configuration time; source registers without an executed writer are captured before the first instruction. Captions identify the exact program, instruction, phase and run. Separate registers can show different runs; this is a latest-sample view, not a synchronized recording or lossless replay.
- Each register has one occupied slot from reservation through CPU encoding and browser acknowledgement. Frequency limits, busy slots and a global maximum of eight in-flight preparations/encoders are checked **before** serialization, GPU resize or download. There is no frame queue. The VM creates an independent CPU image or serialized JSON; only this owned snapshot reaches a background PNG encoder. Contexts, live arrays, OrtValues and provider allocators never reach HTTP/encoder threads.
- Browser polling acknowledges a sample only after decoding/display. Without a browser lease, no new preparation occurs; the lease expires after five seconds. Turning observation off disables the hook immediately, without preparation, serialization, GPU transfers or encoding of further samples. Already admitted work can finish and is discarded after configuration changes. The normal MultiCamera host is untouched and has no observation hooks.
- While running, register rows retain metadata from the last pause, including after reconnecting. Observed data refreshes in the adjacent tiles. Counters show prepared samples (including errors) and skipped capture-point visits since the last configuration change. Errors stay in the affected tile's existing status text with details on hover; they never trigger provider fallback or fail the VM run.
- Image contracts: raw NV12 surfaces use the source's declared dimensions and color matrix; U8 images are BGR NHWC; FP16/FP32 images are RGB NCHW in 0..1. Single images only; batch/unsupported layouts stay metadata-only and fail explicit image requests. Paddle-normalized batch data is never guessed to be 0..1.
- Pin an input above an output to compare transform stages. Captions retain camera, run, register and stop location. Result frame requests show the last completed annotated preview when that preview is enabled in the camera configuration.
- Instruction times exclude pauses and inspection; total run times include them. A detached debug session does not capture register snapshots, but this is still an instrumented host, not a benchmark replacement for the ordinary lab.
- Closing a browser preserves a pause. Reconnect to resume, or call the API. Stop releases pauses, drains the camera workers, disposes run contexts before instruction allocators and then disposes shared endpoints.

## HTTP API

All mutation requests require `X-VM-Debug: 1`. Remote bindings additionally require the bearer token. Successful commands are visible in `GET /api/status`. A missing camera returns 404; stale/unavailable stops return 409; unsupported inspection returns 400. Do not concurrently issue a pause-relative command from multiple clients without the current revision.

| Method | Route | Body / result |
|---|---|---|
| GET | `/api/status` | Host state, cameras, statistics and debug snapshots |
| POST | `/api/start` | Start a separate debug run; 202 |
| POST | `/api/stop` | Stop the debug run and release resources |
| GET | `/api/cameras/{id}/instructions` | Program name, local instruction index, operation and register dependencies |
| GET | `/api/cameras/{id}/debug` | State, revision, location, variables, breakpoints and timings |
| POST | `/api/cameras/{id}/command` | `{"command":"step","revision":123}`; also pause, continue, next-frame, attach, detach; run-to uses `{"command":"run-to","revision":123,"program":"common","index":3}` |
| PUT | `/api/cameras/{id}/breakpoints` | `[{"program":"common","index":3}]`; replaces the camera's breakpoint set |
| POST | `/api/cameras/{id}/inspect` | `{"key":"image.rect","revision":123,"format":"png","maxEdge":1000}` or format json |
| GET | `/api/cameras/{id}/preview.png` | Last completed frame preview; 404 until one exists |
| GET | `/api/cameras/{id}/watch` | Observation generation, configuration, targets, counters, latest metadata and last-pause register catalog; no image payload |
| PUT | `/api/cameras/{id}/watch` | `{"enabled":true,"framesPerSecond":3,"maxEdge":640,"registers":[{"key":"image.rect","format":"png"},{"key":"det.rect","format":"json"}]}`; replaces configuration and invalidates old-generation samples |
| POST | `/api/cameras/{id}/watch/poll` | `{"generation":1,"after":{"image.rect":12,"det.rect":13}}`; renews a five-second lease, acknowledges displayed sequences, and returns newer samples as content type plus base64 bytes |

The register JSON endpoint allows typed arrays (first 256 items, with explicit truncation metadata), scalars and value types. Arbitrary native resource graphs are metadata-only. No register mutation or arbitrary program upload is exposed.

## UI provenance

Shell resources, palettes, local Tabler icons and tooltip behavior were copied from `UI.TechnicalShell`; its complete owner standard is retained in `docs/ui-principles.md` and `REQUIREMENTS.md`. Appearance remains the combined theme/density menu, with RU/EN and per-application browser preferences. Icon licensing is in `wwwroot/shell/icons/LICENSE.txt`.

## Verified 2026-10-29

- Complete solution builds, including samples, both Avalonia labs and all test projects. Existing unrelated warnings remain.
- 19 debugger/renderer/access-policy tests (including run-to and register action capabilities) and all 36 existing VM tests pass.
- Actual recordings from both new cameras: NVDEC surfaces in CUDA memory; H.264 2688×1520 verified with FFprobe; 4K 3840×2160 second source. Single-step color conversion, undistort and perspective, real detector outputs, independent camera pauses, program-local breakpoints, next accepted frame and explicit PNG inspection were exercised through the running host.
- More than 1,200 completed VM runs across the two cameras without VM failures, followed by successful pause and OCR inspection: fixed recognition batch `[16,3,48,640]`, actual ROI counts 2 and 1, typed recognition strings available through the register API. Model recognition accuracy is outside this debugger change.
- Stop while both cameras are paused releases the debug run cleanly. Stale revision requests return 409; requests without the mutation header return 403.
- Real browser checks: Cream/Nano, Dark/Nano, RU/EN, a 1600×1000 desktop and 700×850 narrow viewport; register/image panes retain independent scroll areas, and the document stays at viewport height.
- Sources tested here are recordings, not a live RTSP connection. Direct remote access and Linux native deployment have not been exercised on another machine. Remote bind/token policy is covered by tests.
