# MultiCamera VM lab

Runs one controller per video source with a `[prepare:<id>, common]` program chain. It supports camera geometry, object detection, tracking and optional text detection and recognition on tracked regions. Controllers own their programs and tracking state. Cameras using the same model file share an inference endpoint.

## Public live sources

`cameras.sample.json` uses two public HLS traffic cameras from [DelDOT](https://deldot.gov/map/). The profile selects CPU decoding and CPU ONNX inference, uses YOLO26n to track cars, and disables OCR and frame preview.

Place `yolo26n/yolo26n__640_b1_fp32.onnx` under the configured models root. The default models root in a checkout is the top-level `models` directory. Set `NEUROMODFLOWNET_ONNX_MODELS` to point to a different root. The demo assets resolver uses the repository's published demo assets.

From the repository root:

    dotnet run --project labs/NeuroModFlowNet.Pipeline.MultiCamera -- labs/NeuroModFlowNet.Pipeline.MultiCamera/cameras.sample.json --plain --no-preview

The console reports frames received and completed VM runs, detections, track counts and failures. To debug the same profile, use the [WebDebug launcher](../NeuroModFlowNet.Pipeline.WebDebug/README.md).

## Configuration

- `cameras[].source.kind` accepts `file`, `files`, `folder`, `rtsp`, `hls` and `camera`. An HLS URL is an HTTP(S) playlist and currently uses the OpenCV/FFmpeg CPU decode path. RTSP credentials may be supplied using an ignored `@name.local.txt` file next to the configuration.
- `resolution` declares the expected source frame size. `transforms` can apply undistortion, perspective, rotation, crop and resize. Detection and tracking coordinates use the output of the last transform.
- `detector` selects box or OBB output, precision, stretch or letterbox, score threshold and optional class ID. Each camera declares its model and static input shape.
- `ocr` optionally crops confirmed tracks, detects text on the crops and recognizes regions extracted from the full-resolution rectified frame. Text detection and recognition are conditional. Both the crop batch and the recognition ROI capacity are explicit.
- `preview.enabled` controls whether the preview instructions are built. `--no-preview` controls console display.
- Use `--decoder nvdec` with supported file and RTSP sources to decode into CUDA memory. Set `FFmpegLibrariesPath` in the ignored `App.local.config` to the FFmpeg shared-library directory.

The frame, detections, tracks and OCR appear in `image.rect`, `det.rect`, `track.out`, `text.obb.rect`, `text.trackIds` and `ocr.recognition` when their steps run.
