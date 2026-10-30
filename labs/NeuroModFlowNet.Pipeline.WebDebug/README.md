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
- Register checkboxes enable live observation of up to eight images or typed JSON values together. The inspection pane has a master observation switch, a per-register frequency limit (0.2–15 FPS) and a maximum image edge (100–8192 pixels, or Original). Tiles retain a fixed content size while new samples arrive; data and diagnostics never insert banners or shift the workspace. Charts/history are deliberately deferred.
- The browser uses one shared capture point, checked in the instruction table's **Capture / Съём** column, after that instruction. Register checkboxes choose the values. Captions identify the exact program, instruction, phase and run. Separate registers can show different runs; this is a latest-sample view, not a synchronized recording or lossless replay. The API still supports explicit per-register points.
- Each register has one occupied slot from reservation through CPU encoding and browser acknowledgement. Frequency limits, busy slots and a global maximum of eight in-flight preparations/encoders are checked **before** serialization, GPU resize or download. There is no frame queue. The VM creates an independent CPU image or serialized JSON; only this owned snapshot reaches a background PNG encoder. Contexts, live arrays, OrtValues and provider allocators never reach HTTP/encoder threads.
- Browser polling acknowledges a sample only after decoding/display. Without a browser lease, no new preparation occurs; the lease expires after five seconds. Turning observation off disables the hook immediately, without preparation, serialization, GPU transfers or encoding of further samples. Already admitted work can finish and is discarded after configuration changes. The normal MultiCamera host is untouched and has no observation hooks.
- Register names accumulate across frames while a camera page or observation window renews its catalog lease. Independent windows choose their own shared instruction capture point. Images/JSON refresh in adjacent tiles. Counters show prepared samples (including errors) and skipped capture-point visits since the last configuration change. Errors stay in the affected tile's existing status text with details on hover; they never trigger provider fallback or fail the VM run.
- Image contracts: raw NV12 surfaces use the source's declared dimensions and color matrix; U8 images are BGR NHWC; FP16/FP32 images are RGB NCHW in 0..1. Rank-4 image batches up to 64 slots can be inspected as numbered grids. Unsupported layouts remain metadata-only. Paddle normalization is explicit for ocr.recInput; its [-1,1] values are restored to display colors, never interpreted as 0..1.
- Pin an input above an output to compare transform stages. Captions retain camera, run, register and stop location. **Result frame** requests one fresh completed annotated frame while running, or explicitly inspects current `image.rect` at a pause (without completed-frame overlays). If preparation has not reached `image.rect`, the paused request reports that absence; it neither resumes the VM nor substitutes an older frame.
- Instruction times exclude pauses and inspection; total run times include them. A detached debug session does not capture register snapshots, but this is still an instrumented host, not a benchmark replacement for the ordinary lab.
- Closing a browser preserves a pause. Reconnect to resume, or call the API. Stop releases pauses, drains the camera workers, disposes run contexts before instruction allocators and then disposes shared endpoints.

### Demand-only result preview

The WebDebug decorator gates the existing `image.preview.tensor` resize/download and `image.preview` Mat commands. Their descriptors, instruction numbers and disposal remain intact, but neither command body runs without demand; warmup also skips them. There is no perpetual preview download behind the result-frame button. Ordinary register inspection and selected observations retain their bounded pre-GPU admission rules. A watch of another GPU image does not enable the result preview.

An explicit result-frame request coalesces concurrent callers into one pending capture, admits one preview and encodes only an owned CPU clone after normal overlays are drawn. Cancellation/timeout removes unused demand; session failure/disposal rejects waiters. Requests time out after ten seconds rather than retaining GPU frames. The debug host retains the preview command contract even if the ordinary camera configuration disables continuous preview; the demand gate determines execution. Ordinary MultiCamera preview policy remains controlled by its own configuration.

Watching `image.preview.tensor` or `image.preview` enables its producer only when that viewer has an active lease, available capture/acknowledgement slot, a due frequency limit, and a capture point at/after production. This readiness check runs on the camera's single VM task; asynchronous encoders may release capacity but cannot admit another VM capture between the check and the producer. Observation of an already downloaded CPU tensor stays on the CPU and does not feed it back through a GPU resize. Preview registers first appear in the accumulated catalog after an actual requested preview.

`/api/status` includes per-camera `preview.prepared`, `preview.encoded` and `preview.pending`: admitted result-preview command executions, completed one-shot PNG encodes and explicit waiters. These do not count independent register captures. They allow idle behavior to be checked without GPU profiling or periodic screenshots. The only MultiCamera addition is an optional callback after publishing an already owned CPU preview; VM/ONNX core and decoder behavior are unchanged.

## HTTP API

### Register catalog and reused names

The catalog is metadata, not value history. The VM task compares existing local names before and after each instruction, retaining all observed boundaries and the last disappearance location. Removed names and names from completed frames remain selectable for a later run. A watched value is captured at its explicitly selected boundary, while it still exists; selecting an old name never resurrects an old GPU tensor. Captions retain the actual program/index/phase and run. A name created and removed entirely inside one instruction is invisible at these boundaries.

Each catalog entry has a one-based `additionOrder`: the order in which the debugger first discovered that name. The compact **№** (discovery order) column sorts numerically in either direction and preserves the existing sorting preference. Numbers survive overwrites, removal, later frames and pause refreshes; they reset with the camera debug session. Discovery starts when the client connects, so a conditional branch reached later appends new names even if its instructions are earlier in the program. Mutation order within one instruction or a first snapshot containing several existing names is not reconstructed.

Register columns are: creating operation index, discovery order, name, observation checkbox, one content-dependent inspection action, memory and shape. The compact **Op. / Оп.** cell's tooltip identifies the creating program and instruction. `createdAt` is recorded only when consecutive before/after boundaries prove that the name appeared during that instruction; a declared write or overwriting an existing value is not creation. External inputs and creation not yet established show `—`. A late attachment can learn the creator on a later run; overwrites and later recreations preserve the first confirmed creator. The single inspection cell uses an eye for an image or a list for data, preserving pause/availability checks.

Selecting an instruction continues to highlight its declared output registers. Their entire rows, including action cells, use the separate theme extension role `--vm-register-write-highlight`; provider text colors and the existing input/output semantics are retained.

Only names and immutable metadata leave the VM task; no payload, live context, Mat, OrtValue or resource ownership is retained. Each boundary's type/shape/capabilities refresh at most once per second. Metadata failures remain visible as unavailable metadata and do not fail the VM. The catalog is bounded at 512 names per camera, with a visible `+` count and `truncated` flag on overflow; boundary count is bounded by the installed program. At a pause the table merges the exact current snapshot with historical names, disabling manual inspection for absent values while keeping dynamic observation selectable.

`POST /registers/poll` renews collection for five seconds; the browser calls it once per second only on a visible camera page. GET never renews collection. After a lease gap, missing keys are not attributed to a particular instruction; ending a frame is distinguished from deleting a name mid-frame. The ordinary MultiCamera host has no catalog wrappers. An inactive WebDebug catalog takes its volatile-flag fast path without enumeration, descriptions or allocations. Existing debug-host pause/timing overhead remains separate.

Watch selections now accept `program`, `index`, `phase` in addition to `key` and `format`, for example `{"key":"image.rect","format":"png","program":"common","index":0,"phase":"before"}`. The browser selects up to eight register names in the register table and one shared capture point using the **Capture / Съём** checkbox on an instruction. All selected values are sampled after that operation; checking another instruction moves the point, unchecking it disables observation while retaining register selections. There is no per-register capture dropdown. The point is stored separately per camera and viewer. Existing multi-point selections are normalized to one after-instruction point when the browser opens. An installed instruction boundary may be selected before creation or after removal of a known name; missing values are explicitly unavailable, and another instruction or an older payload is never substituted. If the name is present, metadata at that boundary must support the selected format. A legacy selection without a point resolves once to an observed writer/source point; the response returns the resolved explicit configuration. Reconfiguring clears the old displayed sample so data from another point is not presented as current. Rate, backpressure, pre-GPU admission and ownership rules are unchanged.

`VmRunContext.Remove` removes a name only. Resource ownership is registered separately and is released when the context ends. Another name may still refer to the same owned object, so automatic disposal on removing one alias would be unsafe. No early resource-release contract is introduced by the catalog; the existing ownership behavior is covered by an alias/disposal test.

### Catalog cost measurements (2026-10-30)

Reproduce the CPU comparison with the Release debug-host DLL: `dotnet NeuroModFlowNet.Pipeline.WebDebug.dll --measure-register-catalog <absolute-output.json>`. This mode opens no server or cameras. The workload is 32 scalar names, 40 instructions/80 boundaries, seven samples of 10,000 frames, compared with the same input/update/disposal loop without catalog calls. On this machine (.NET 10.0.13), median added time was 0.17 microseconds per frame inactive, with zero added bytes; active steady-state was 29.24 microseconds and 24 bytes per frame. Cold discovery was 0.42 ms / 832 KB, and a frame explicitly refreshing all point metadata was 0.37 ms / 576 KB. Refreshes are rate-limited; these cold/refresh costs are separate from steady-state. HTTP/serialization, GPU metadata and actual image/value capture are outside this CPU comparison.

The installed Debug host was also measured for 20 seconds on both real camera recordings, renewing the metadata catalog at 1 Hz without register image/value captures. Camera 1: 28 names, 18 instructions, 11,844 visits, 35.97 ms total catalog time (3.04 microseconds/boundary; approximately 0.109 ms/frame). Camera 2: 26 names, 17 instructions, 14,926 visits, 44.03 ms total (2.95 microseconds/boundary; approximately 0.100 ms/frame). This includes discovery, GPU metadata calls, lock waiting and measurement probes, but excludes HTTP snapshot construction/serialization and PNG encoding. Maximum observed individual visits were 0.244 ms and 1.573 ms; these are measurements under the current workload, not latency guarantees.

The 26-name metadata response was 193,710 bytes; ten separate PowerShell loopback GETs had a median end-to-end latency of 3.77 ms, including the client/network/HTTP path. This is separate from the VM visit cost and is requested at 1 Hz by the UI. Leaving the camera page was verified to expire both discovery leases: catalog revisions stayed unchanged while both cameras kept running without failures. The full catalog also remained visible on an early pause with only source registers present; historical manual inspection actions stayed disabled.

For explicit live profiling, POST `/registers/measure` starts a twenty-second per-camera counter window; GET reads calls, elapsed time, maximum duration and allocated bytes. It does not renew collection. Outside that requested window no timer/allocation probes are performed. Raw CPU and live results are saved locally under the installed host's `Logs/register-catalog-benchmark.json` and `Logs/register-catalog-live-cost.json`.

### Shared inference batches

`#batches` opens a separate shared-service overview: actual backend and batching policy, observed prepared/completed counters, last request/tensor counts and processing time. Select a service row for two charts and its execution history. Each bar is an observed batch in assembly order, not a uniform time interval. The request limit and queue capacity are configured limits, not current queue depth. One VM request can already contain a complete ROI tensor batch; the input's first dimension includes upstream padding and must not be presented as an actual object count or useful-slot percentage.

Collection is opt-in through the visible page's checkbox and a five-second lease, renewed once per second. Leaving the page, hiding/closing the tab or unchecking stops renewal; any other active observer may still renew the shared lease. At most 256 samples per service are retained in a ring; cumulative counters cover observed periods until cameras restart. GET reads never renew collection. An admitted batch can finish recording after its lease expires. The ordinary MultiCamera host creates no decorators; the debugger's inactive wrappers forward directly without metadata capture or GPU work.

The optional factory decorators live only in the MultiCamera/WebDebug labs. The existing input assembler retains tensor ownership and assembly semantics. A sample is admitted only after successful assembly; result decoder return confirms completion. Processing time spans assembly through decoding, excluding queue wait and lazy session creation. A failure before decoding remains unconfirmed, then becomes incomplete when the endpoint next assembles or disposes; decoder failures are recorded as such and rethrown unchanged. No scheduler, VM, model or core ONNX interfaces are changed. Per-camera contribution identities, actual padding occupancy and live queue depth are not exposed by these existing component boundaries.

### Dockview workspace and independent windows

The root URL and `#overview` open the initial camera overview with aggregate statistics and host start/stop controls. A camera row or sidebar entry opens `#<camera-id>` with only that camera's debug toolbar and Dockview workspace. Browser back/forward and direct camera links are supported; switching to the overview preserves the workspace layout and stops renewing the hidden main viewer's observation lease. Autonomous viewers retain their own leases. Sidebar camera icons show the current debug state (green play, amber pause, red error) with a translated tooltip; selecting the icon opens the camera without issuing a VM command.

The default workspace retains the three adjacent instruction, register and inspection panes. Drag a pane header to move it, combine tabs or split horizontally/vertically; drag the separators to resize. The bare maximize icon floats a pane inside the page and docks it back; move a floating window by its narrow top grip. A watched image/value has the same action beside its caption, so individual samples can be arranged separately. **Panels → Reset layout** restores the initial overview. Layout and sizes are saved in browser storage, separately for the main page and each observation window.

The debug command icons sit on the left before the selected camera. The common appearance menu has three independent axes: color, row density and panel framing. **Joined** preserves the original view and is the default; **Rounded** adds 8px gaps, 8px corners, a 1px content inset and a soft shadow offset 2px right/3px down; **Square** keeps the same gaps/inset/shadow with straight corners. The inner surface clips scrolling and sticky content to all four corners; the outer frame owns the shadow so it remains outside the clip. Framing is saved as a browser preference and changes the live Dockview theme gap without rebuilding or resetting the workspace. All additions are isolated in `wwwroot/panel-schemes.css`; base palettes and row sizes are unchanged.

The external-link icon opens an ordinary popup with a complete, independent Dockview page. It copies the chosen camera and either that register or the current observation selection. The popup has its own register selection, frequency, image size and layout; it can add the register table through **Panels**. Closing a watched pane in that window unsubscribes only its register. Closing the main page does not stop the window. The window talks directly to the same HTTP API, without opener messaging or a parent-window lifetime dependency. Its compact UI omits VM execution controls. The instruction panel can be opened through **Panels** to choose its own capture point; breakpoints and execution controls remain disabled there. PWA installation is deferred. Browser popup policy determines whether this is a separate window or tab and how much browser address chrome remains visible; allow popups for this origin when needed.

Observation subscriptions are isolated by the optional `viewer` query parameter on all three `/watch` routes (omitted means `main`). Simultaneously ready register/location/format/size requests share one owned snapshot and encoding. Each viewer has its own acknowledgement slot, rate and five-second lease: a slow viewer cannot block another viewer or accumulate a queue. Limits per camera are eight viewer subscriptions, eight registers per viewer, sixteen distinct register/size targets and eight in-flight captures/encoders globally across those viewers. Unseen non-main subscriptions expire after ten minutes. Idle/closed windows lose their lease before any further GPU preparation; disabled observation retains the allocation-free hook fast path.

Register headers sort by register name, memory or shape/value; clicking again reverses the order. Natural sorting keeps numeric name components together and compares scalar numbers numerically. Provider colors are an additive extension in `wwwroot/theme-debug-extensions.css`, leaving the base palettes untouched: readable NVIDIA greens for CUDA/TensorRT, the theme's red for Managed and information blue for DirectML. Unknown placement stays neutral; no GPU manufacturer is guessed from a generic provider name.

All mutation requests require `X-VM-Debug: 1`. Remote bindings additionally require the bearer token. Successful commands are visible in `GET /api/status`. A missing camera returns 404; stale/unavailable stops return 409; unsupported inspection returns 400. Do not concurrently issue a pause-relative command from multiple clients without the current revision.

| Method | Route | Body / result |
|---|---|---|
| GET | `/api/status` | Host state, cameras, statistics and debug snapshots |
| GET | `/api/batches` | Shared-service configuration, bounded batch history and observed-period counters; does not enable collection |
| POST | `/api/batches/poll` | Renew a five-second metadata collection lease and return the same snapshot |
| POST | `/api/start` | Start a separate debug run; 202 |
| POST | `/api/stop` | Stop the debug run and release resources |
| GET | `/api/cameras/{id}/instructions` | Program name, local instruction index, operation and register dependencies |
| GET | `/api/cameras/{id}/debug` | State, revision, location, variables, breakpoints and timings |
| GET | `/api/cameras/{id}/registers` | Retained metadata catalog, observed points and removal locations; no lease renewal |
| POST | `/api/cameras/{id}/registers/poll` | Renew five-second discovery lease and return metadata catalog |
| POST / GET | `/api/cameras/{id}/registers/measure` | Start a twenty-second explicit catalog cost measurement / read its result; no lease renewal |
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

Dockview Core 8.2.0 is vendored locally under `wwwroot/vendor/dockview` with its MIT license. It is the same dependency version already used by the local SeMaps editor. No CDN, frontend framework or bundler is required; the workspace adapter is isolated in `workspace.js`/`workspace.css`.

## Verified 2026-10-30

- Complete solution builds; all 63 debugger tests and ten Node register-table/observation tests pass. New tests also cover cross-frame register catalogs, removal/lease gaps, zero-allocation disabled discovery, bounded metadata failures, exact reused-name capture points in two viewers and alias ownership.
- The new batch page was checked against the actual two-camera host: four TensorRT services, one request per batch, detector inputs of one row, text detection inputs of four rows and recognition inputs of sixteen rows. Both cameras continue running without VM failures. These are tensor sizes, not actual ROI counts.
- History remains at 256 samples after thousands of observations. Disabling collection expires the lease and freezes counters while camera frame counters continue increasing. Overview/camera/batch navigation retains the camera workspace. Checkbox and collection-state changes keep table/chart/panel geometry unchanged.

- Complete solution builds, including samples, both Avalonia labs and all test projects. Existing unrelated warnings remain.
- Dockview changes: all 35 debugger tests pass, including capture sharing and independent/slow viewers; three Node tests cover register sorting and provider classification. Browser checks cover header drag/split, persisted layout after reload, reset, individual floating images, isolated window settings and continued samples after the opener closes. Physical transfer to a second monitor has not been exercised.
- 19 debugger/renderer/access-policy tests (including run-to and register action capabilities) and all 36 existing VM tests pass.
- Actual recordings from both new cameras: NVDEC surfaces in CUDA memory; H.264 2688×1520 verified with FFprobe; 4K 3840×2160 second source. Single-step color conversion, undistort and perspective, real detector outputs, independent camera pauses, program-local breakpoints, next accepted frame and explicit PNG inspection were exercised through the running host.
- More than 1,200 completed VM runs across the two cameras without VM failures, followed by successful pause and OCR inspection: fixed recognition batch `[16,3,48,640]`, actual ROI counts 2 and 1, typed recognition strings available through the register API. Model recognition accuracy is outside this debugger change.
- Stop while both cameras are paused releases the debug run cleanly. Stale revision requests return 409; requests without the mutation header return 403.
- Real browser checks: Cream/Nano, Dark/Nano, RU/EN, a 1600×1000 desktop and 700×850 narrow viewport; register/image panes retain independent scroll areas, and the document stays at viewport height.
- Sources tested here are recordings, not a live RTSP connection. Direct remote access and Linux native deployment have not been exercised on another machine. Remote bind/token policy is covered by tests.

### Original-resolution inspection

The image-size setting limits the longest transmitted PNG edge; it does not modify model inputs. Numeric presets are UI conveniences, not decoder/backend limits. `maxEdge: 0` (Original) preserves the selected register's dimensions, with a 16-megapixel snapshot ceiling and a 64-MiB encoded sample ceiling. Existing smaller preview registers cannot recover the resolution of their original source.

For single images that need downscaling, GPU resize precedes download. Original NV12 inspection performs the required NV12-to-BGR conversion with a direct CPU output and skips resize. Other original GPU image tensors use a debug-only ORT Identity graph on the explicitly selected backend with a bound CPU output, preserving shape and pixel values without resampling. CPU images stay on the CPU. Encoding uses the already prepared geometry and never applies a second hidden size limit. The paused register-inspection eye uses the same size selection; the completed annotated Result frame retains its separately configured preview size.

The bare 1:1 control on each image switches display between fitting the panel and native CSS-pixel dimensions with internal scrolling; it does not change transmission size or panel geometry. The existing counts line shows actual decoded width and height. Size changes invalidate pending PNG decodes before their results or acknowledgement sequence can enter the new generation, including rejected decodes. Frequency, leases and pre-download admission remain unchanged.
An enabled common capture point received from the server takes precedence over a browser tab's older saved point. Refreshing another main-viewer tab or changing image size must not silently move an active observation back to that tab's old instruction.

### Image-batch grids

`ocr.recInput` and other supported BGR U8 NHWC / RGB FP16/FP32 NCHW batches expose the same image eye and observation checkbox as single images. A batch is captured once at the selected VM boundary and displayed as one numbered PNG grid, with zero-based slot indices and actual/capacity in the header. It retains the existing float/popup, original-resolution and 1:1 display controls; no new VM instructions or inference changes are introduced.

For `ocr.recInput`, actual count comes from `ocr.roiCount` and colors are explicitly restored from PaddleOCR RGB [-1,1]. For `ocr.crop.batch`, count comes from `ocr.crop.count` and RGB remains 0..1. Missing/inconsistent named counts are reported instead of presenting padding as real ROIs. Zero actual count generates an empty CPU grid without downloading tensor data. Generic image batches without a named actual-count contract show all tensor slots.

Admission still precedes GPU transfer. One explicitly selected bounded batch (up to 128 MiB) is downloaded through ORT on the selected backend; only the owned CPU grid reaches encoding. Cells are reduced on CPU during assembly because existing geometry commands require a single image; their contracts are preserved. The size setting limits each cell, not the complete grid. The grid remains bounded to 16 megapixels and the PNG sample to 64 MiB. Closing/disabling an observer retains the existing no-preparation path. The grid renderer and explicitly indexed CPU item renderer live in the isolated Visualizer project.