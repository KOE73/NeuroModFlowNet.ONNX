# Все в одной таблице

В целом верно но...

| Task  | Variant           | Input tensor         | Output tensor(s)                            | File                                     |
| ----- | ----------------- | ---------------------| --------------------------------------------| ---------------------------------------- |
| Box   | FP32 → FP32       | `FP32 [1×3×640×640]` | `FP32 [1×300×6]`                            | `yolo26s__640_b1_fp32.onnx`              |
| Box   | FP16 → FP32       | `FP16 [1×3×640×640]` | `FP32 [1×300×6]`                            | `yolo26s__640_b1_fp16.onnx`              |
| Box   | BGR → FP32 → FP32 | `U8   [1×640×640×3]` | `FP32 [1×300×6]`                            | `yolo26s__640_b1_fp32_bytebgr.onnx`      |
| Box   | BGR → FP16 → FP32 | `U8   [1×640×640×3]` | `FP32 [1×300×6]`                            | `yolo26s__640_b1_fp16_bytebgr.onnx`      |
| OBB   | FP32 → FP32       | `FP32 [1×3×640×640]` | `FP32 [1×300×7]`                            | `yolo26s-obb__640_b1_fp32.onnx`          |
| OBB   | FP16 → FP32       | `FP16 [1×3×640×640]` | `FP32 [1×300×7]`                            | `yolo26s-obb__640_b1_fp16.onnx`          |
| OBB   | BGR → FP32 → FP32 | `U8   [1×640×640×3]` | `FP32 [1×300×7]`                            | `yolo26s-obb__640_b1_fp32_bytebgr.onnx`  |
| OBB   | BGR → FP16 → FP32 | `U8   [1×640×640×3]` | `FP32 [1×300×7]`                            | `yolo26s-obb__640_b1_fp16_bytebgr.onnx`  |
| Class | FP32 → FP32       | `FP32 [1×3×640×640]` | `FP32 [1×1000]`                             | `yolo26s-cls__640_b1_fp32.onnx`          |
| Class | FP16 → FP16       | `FP16 [1×3×640×640]` | `FP16 [1×1000]`                             | `yolo26s-cls__640_b1_fp16.onnx`          |
| Class | BGR → FP32        | `U8   [1×640×640×3]` | `FP32 [1×1000]`                             | `yolo26s-cls__640_b1_fp32_bytebgr.onnx`  |
| Class | BGR → FP16        | `U8   [1×640×640×3]` | `FP16 [1×1000]`                             | `yolo26s-cls__640_b1_fp16_bytebgr.onnx`  |
| Pose  | FP32 → FP32       | `FP32 [1×3×640×640]` | `FP32 [1×300×57]`                           | `yolo26s-pose__640_b1_fp32.onnx`         |
| Pose  | FP16 → FP32       | `FP16 [1×3×640×640]` | `FP32 [1×300×57]`                           | `yolo26s-pose__640_b1_fp16.onnx`         |
| Pose  | BGR → FP32        | `U8   [1×640×640×3]` | `FP32 [1×300×57]`                           | `yolo26s-pose__640_b1_fp32_bytebgr.onnx` |
| Pose  | BGR → FP16 → FP32 | `U8   [1×640×640×3]` | `FP32 [1×300×57]`                           | `yolo26s-pose__640_b1_fp16_bytebgr.onnx` |
| Seg   | FP32 → FP32       | `FP32 [1×3×640×640]` | `FP32 [1×300×38]`<br/>`FP32 [1×32×160×160]` | `yolo26s-seg__640_b1_fp32.onnx`          |
| Seg   | FP16 → mixed      | `FP16 [1×3×640×640]` | `FP32 [1×300×38]`<br/>`FP16 [1×32×160×160]` | `yolo26s-seg__640_b1_fp16.onnx`          |
| Seg   | BGR → FP32        | `U8   [1×640×640×3]` | `FP32 [1×300×38]`<br/>`FP32 [1×32×160×160]` | `yolo26s-seg__640_b1_fp32_bytebgr.onnx`  |
| Seg   | BGR → mixed       | `U8   [1×640×640×3]` | `FP32 [1×300×38]`<br/>`FP16 [1×32×160×160]` | `yolo26s-seg__640_b1_fp16_bytebgr.onnx`  |
