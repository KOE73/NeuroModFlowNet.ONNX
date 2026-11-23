# GPU video output

`VideoOutputService` runs optional video outputs in the host process. Each profile consumes a CUDA BGR U8 NHWC tensor with a fixed input shape and image-space rectangles with optional labels. The device, dimensions, FPS, bitrate, output factories and queue limits are explicit. Raw pixels stay on the GPU.

The render worker keeps the latest input per profile and runs a separate VM program named `render`. Each run has its own register context and executes `Op_Video_PrepareSurface`, `Op_Video_DrawDirect2D`, and `Op_PostOwned<NvencSurface>` in order. The first operation prepares a GPU surface from the tensor and the second draws the supplied scene and produces D3D11 NV12. `POST MOVE` transfers the completed surface to the encoding worker's inbox. The service takes ownership of each posted frame, including frames it drops or replaces.

The encoding worker takes the latest completed surface at its configured FPS and can repeat it until it expires. NVENC produces H.264 for the configured packet destinations. Dispose the service before disposing the sessions or allocators that supplied its input tensors.

`Op_PostOwned<T>` moves an owned VM register into a per-run `VmOwnedHandoff<T>`. The host publishes the staged value after a successful run and disposes the handoff after every outcome. The renderer and packet outputs have no dependencies on application register names or payload types.

## Build on Windows x64

From the repository root with a C++20 toolchain, Windows SDK and CUDA Toolkit installed:

```powershell
cmake -S src/NeuroModFlowNet.ONNX.Visualizer/Direct2D/Native -B "$env:TEMP/nmfn-direct2d" -A x64
cmake --build "$env:TEMP/nmfn-direct2d" --config Release
dotnet build NeuroModFlowNet.ONNX.slnx
```

Pass the built `NeuroModFlowNet.Direct2D.dll` and the FFmpeg 9 shared `bin` directory to the service. The host must configure its matching ONNX Runtime CUDA dependencies. D3D11 selects the adapter that matches the explicitly configured CUDA device. Unsupported placement, formats or hardware fail with a diagnostic.

## Packet destinations

Use `FfmpegVideoOutputs.Mp4` for a new MP4 file and `FfmpegVideoOutputs.Rtsp` for publishing to an existing RTSP server. A profile can include both. Each destination runs its own worker, timeout and bounded queue. The packet layer shares encoded buffer references and preserves PTS and DTS. MP4 finalizes on graceful stop. RTSP reconnects on an upcoming IDR with the stream's codec configuration. The callback receives output state, gaps and errors.

Custom destinations provide factories for `IVideoPacketSink`. A sink can read the encoded bytes or retain an independent reference for its own bounded processing stage. Its lifetime and I/O belong to its output worker. The first integration provides one input per coded profile and independent MP4 or RTSP destinations. [Architecture plan](../../docs/architecture/vm_post_gpu_video_plan.ru.md)
