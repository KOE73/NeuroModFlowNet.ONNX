# NeuroModFlowNet.Pipeline.MultiCamera

Lab that runs N cameras through the Pipeline VM the way ADR-001 describes: **one `VmController` per camera** executing
the program chain `[prepare:<camera>, common]` with a single `VmRunContext` per frame.

- `prepare:<camera>` is built per camera from its `transforms` (undistort / perspective / rotate / crop / resize as
  ONNX operators on the model device) and ends in the register `image.rect` (U8 NHWC BGR, device memory).
- `common` is the same code for every camera: the whole rectified strip resized to the camera's detector input (RGB
  NCHW) -> bag detector -> coordinates back to rect space -> `Op_Track` (state in the controller's `VmGlobalMemory`,
  ordered critical section) -> crops of confirmed tracks (GPU letterbox batch) -> text OBB on the crops -> text regions
  back to rect space with track ids -> PaddleOCR recognition from the full-resolution rect frame -> optional CPU preview.
  This is the old BlazorNNServer order (detect bag -> crop bag -> OCR the crop).
- Only the batched model endpoints (`OrtValueBatchedInferenceEndpoint`) and the runtime kernel cache are shared between
  cameras. Programs, geometry ops and tracker state are per controller.

Capture (video folders, RTSP, USB) lives in this lab (`Sources/`), not in `NeuroModFlowNet.Pipeline`.

## Run

```powershell
dotnet run --project labs/NeuroModFlowNet.Pipeline.MultiCamera -- [cameras.json] [--no-preview] [--plain] [--snapshots <dir>]
```

- Config path: first positional argument, otherwise `CamerasConfig` from `App.config` / `App.local.config`
  (git-ignored; also holds `CudaBinPath`, `CudnnBinPath`, `TrtLibPath`).
- `--no-preview`: no OpenCV windows. `--plain`: line-per-camera status instead of the live table (automatic when
  stdout is redirected). `--snapshots <dir>`: write annotated previews as PNG (run 1, then every 100 runs).
- Esc / Q in a preview window or Ctrl+C stops the lab.

## Config (`cameras.sample.json`)

| Section | Meaning |
| --- | --- |
| `backend`, `modelInputSize` | Execution backend for geometry ops and detectors; square model input (multiple of 32). |
| `detector` | Shared detector settings: `kind` `box` (NMS inside, `[1,300,6]`) or `obb`; `backend` override (TensorRT for the CGP model, strict CUDA leaves nodes on CPU); `precision`; `inputMode` `stretch` or `letterbox`; `scoreThreshold`; `trackedClassId` (0 = BagTop). |
| `tracker` | `IouTrackerOptions` values. |
| `ocr` | Text OBB model (FP32, fixed batch = `maxCrops`) run on bag crops of `detectionInputSize`, crop padding, PaddleOCR rec model and backend (TensorRT is the proven GPU path for `rec.onnx`), ROI size/padding, fixed `maxRoiCount`. |
| `preview` | GPU-resized preview downloaded to a Mat (`maxEdge`). |
| `cameras[]` | `id`, `source`, declared `resolution`, `maxInFlight`, `transforms[]`, `detector` (`modelPath`, `inputWidth`, `inputHeight`: a static model exported for this strip), `tracking` zones. |

`source.kind`: `folder` (loops over `*.ts;*.mp4;...` files sorted by name, skips empty/unopenable ones), `files`
(explicit playlist in the given order, for hand-picked segments with bags), `file`, `rtsp` (`url` or `@file.local.txt` with credentials, TCP transport), `camera` (index). `realtime: true` paces file
playback to the file FPS; the host keeps a 2-frame drop-oldest channel per camera, so a slow VM never accumulates lag.

`transforms[]` mirrors `T:\CGPCam2026\cameras.json` (`radialUndistort`, `perspective`) plus `rotate`, `crop`, `resize`.
`radialUndistort` reproduces the CGPCam2026 DistortionCorrector exactly: K is `fx = fy = width * focalLengthFactor`,
`cx = width / 2`, `cy = height / 2`, and the `getOptimalNewCameraMatrix(alpha)` result becomes the operator's output
intrinsics (`RadialTangentialDistortionParameters.Output*`). Perspective points calibrated on the tool output are
therefore used unchanged.

Tracking zones, detections, tracks and OCR regions are all in **rect space** (the output of the last transform).

## Register contract

| Register | Producer | Type |
| --- | --- | --- |
| `image.source` | host | `Mat` (BGR, declared resolution) |
| `image.rect` | `prepare:<camera>` | `OrtValue` U8 NHWC on the model device |
| `det.rect`, `track.det.in`, `track.out` | `common` | `YoloBox[]` or `YoloObb[]`, `TrackDetection[]`, `TrackedObject[]` |
| `ocr.crop.boxes`, `ocr.crop.trackIds`, `ocr.crop.batch`, `ocr.crop.transforms`, `ocr.crop.count` | `common` (OCR) | crops of confirmed tracks: `YoloObb[]`, `int[]`, `OrtValue [maxCrops,3,S,S]`, `ICoordinateBackTransform[]`, `int` |
| `text.obb.rect`, `text.trackIds`, `ocr.roiCount`, `ocr.recognition` | `common` (OCR, only when crops / text exist) | `YoloObb[]`, `int[]`, `int`, `OcrResult[]` |
| `image.preview` | `common` (preview enabled) | `Mat` |

## Decoder: cpu or nvdec

`source.decoder` per camera, or `--decoder cpu|nvdec` on the command line for all cameras.

| | `cpu` | `nvdec` |
| --- | --- | --- |
| Decode | OpenCV (OpenCvSharp5 FFmpeg) into a `Mat`, optional `hwAcceleration: d3d11` | FFmpeg 9 NVDEC (`labs/NeuroModFlowNet.Pipeline.Video.Nvdec`) into CUDA memory |
| Frame in the run | `image.source` (Mat) | `image.source.nv12` (zero-copy NV12 tensor over the surface) |
| Ingest steps of prepare | `Wrap_MatImage_To_OrtTensor` + `Copy_OrtTensor_To_ModelDevice` | `Op_Onnx_Nv12_To_BgrU8Nhwc` |
| Source kinds | folder, files, file, rtsp, camera | folder, files, file, rtsp |

Both variants leave BGR U8 NHWC in model device memory, so every transform and the whole common program are the
same code. `source.colorMatrix` (default `Bt601Full`, what yuvj420p cameras use) selects the NV12 conversion.
`FFmpegLibrariesPath` in `App.config` points to the FFmpeg 9 shared build. The NVDEC source opens its first file at
startup so that FFmpeg creates the CUDA primary context before ONNX Runtime does.

Measured unpaced on RTX 5090 (TensorRT, per-caller sessions):

| Camera | cpu | nvdec |
| --- | --- | --- |
| cam_202_01 (4K HEVC) | ~55 fps | ~120-135 fps |
| cam_202_02 (2688x1520 H.264) | ~180 fps | ~120-135 fps |

With NVDEC both cameras are bounded by the shared GPU work (the detector call is the largest item in `--profile`), not
by decode or upload.

## Detector input size

The detector sees the whole strip once, reduced so that a bag has the pixel size of the training data (median BagTop
175x246 px in the 512 training input). Bag size on the rect strips was measured from white bag blobs:

| Camera | Rect strip | Bag in rect | Scale | Model input |
| --- | --- | --- | --- | --- |
| cam_202_01 | 1120x5408 | ~890x1310 | 0.19 | 224x1024 |
| cam_202_02 | 1472x4224 | ~1130x1440 | 0.16 | 224x672 |

Models: `NeuroModFlowNet.ONNX.Private\models\cgp2510-multi9\cgp2510-multi9__{W}x{H}_b1_fp16.onnx`, exported from
`C:\NN\CGP2510_Multi\CGP2510_Multi\CGP2510_Multi9\weights\best.pt` with
`yolo export format=onnx imgsz=[H,W] batch=1 half=True nms=True dynamic=False opset=20`. The naming convention is
extended with `{W}x{H}` for non-square inputs. Tracking only needs approximate positions; OCR crops are cut from the
full-resolution rect frame, so the small detector input does not reduce OCR quality.

Compared on 1200 frames per camera with 512 tiles (removed): higher confidence (0.86-0.90 vs 0.72-0.73), far fewer
track ids (27 vs 68 and 10 vs 32) because bags are no longer cut by tile seams, comparable detection rate.

## Known limitations

- The bag model was trained on the old cameras (61/62); retrain on `T:\CGPCam2026` rect strips at the input sizes above
  for production quality.
- The detector call takes 8-13 ms including endpoint overhead for a 224x1024 input, which is worth profiling next.
- OCR steps are skipped on frames without confirmed tracks / text regions, so their kernels and TensorRT engines are
  built on the first frame that has them, not during warmup.
