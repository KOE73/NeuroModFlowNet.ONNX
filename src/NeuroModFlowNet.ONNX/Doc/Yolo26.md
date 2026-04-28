
## Box `yolo26s`

### FP32 -> FP32 `yolo26s__640_b1_fp32.onnx`

| Name    | Type | Shape               |
|---------|------|---------------------|
| images  | FP32 | [1 x 3 x 640 x 640] |
| output0 | FP32 | [1 x 300 x 6]       |

### FP16 -> FP32 `yolo26s__640_b1_fp16.onnx`

| Name    | Type | Shape               |
|---------|------|---------------------|
| images  | FP16 | [1 x 3 x 640 x 640] |
| output0 | FP32 | [1 x 300 x 6]       |

### BGR -> FP32 -> FP32 `yolo26s__640_b1_fp32_bytebgr.onnx`

| Name         | Type | Shape               |
|--------------|------|---------------------|
| pixel_values | U8   | [1 x 640 x 640 x 3] |
| output0      | FP32 | [1 x 300 x 6]       |

### BGR -> FP16 -> FP32 `yolo26s__640_b1_fp16_bytebgr.onnx`

| Name         | Type | Shape               |
|--------------|------|---------------------|
| pixel_values | U8   | [4 x 640 x 640 x 3] |
| output0      | FP32 | [4 x 300 x 6]       |





## Obb `yolo26s-obb`

### FP32 -> FP32 `yolo26s-obb__640_b1_fp32.onnx`

| Name    | Type | Shape               |
|---------|------|---------------------|
| images  | FP32 | [1 x 3 x 640 x 640] |
| output0 | FP32 | [1 x 300 x 7]       |

### FP16 -> FP32 `yolo26s-obb__640_b1_fp16.onnx`

| Name    | Type | Shape               |
|---------|------|---------------------|
| images  | FP16 | [1 x 3 x 640 x 640] |
| output0 | FP32 | [1 x 300 x 7]       |

### BGR -> FP32 -> FP32 `yolo26s-obb__640_b1_fp32_bytebgr.onnx`

| Name         | Type | Shape               |
|--------------|------|---------------------|
| pixel_values | U8   | [1 x 640 x 640 x 3] |
| output0      | FP32 | [1 x 300 x 7]       |

### BGR -> FP16 -> FP32 `yolo26s-obb__640_b1_fp16_bytebgr.onnx`

| Name         | Type | Shape               |
|--------------|------|---------------------|
| pixel_values | U8   | [1 x 640 x 640 x 3] |
| output0      | FP32 | [1 x 300 x 7]       |




## Class `yolo26s-cls`

### FP32 -> FP32 `yolo26s-cls__640_b1_fp32.onnx`

| Name    | Type | Shape               |
|---------|------|---------------------|
| images  | FP32 | [1 x 3 x 640 x 640] |
| output0 | FP32 | [1 x 1000]          |

### FP16 -> FP16 `yolo26s-cls__640_b1_fp16.onnx`

| Name    | Type | Shape               |
|---------|------|---------------------|
| images  | FP16 | [1 x 3 x 640 x 640] |
| output0 | FP16 | [1 x 1000]          |

### BGR -> FP32 `yolo26s-cls__640_b1_fp32_bytebgr.onnx`

| Name         | Type | Shape               |
|--------------|------|---------------------|
| pixel_values | U8   | [1 x 640 x 640 x 3] |
| output0      | FP32 | [1 x 1000]          |

### BGR -> FP16 `yolo26s-cls__640_b1_fp16_bytebgr.onnx`

| Name         | Type | Shape               |
|--------------|------|---------------------|
| pixel_values | U8   | [1 x 640 x 640 x 3] |
| output0      | FP16 | [1 x 1000]          |





## Pose `yolo26s-pose`

### FP32 -> FP32 `yolo26s-pose__640_b1_fp32.onnx`

| Name    | Type | Shape                |
|---------|------|----------------------|
| images  | FP32 | [1 x 3 x 640 x 640]  |
| output0 | FP32 | [1 x 300 x 57]       |

### FP16 -> FP16 `yolo26s-pose__640_b1_fp16.onnx`

| Name    | Type | Shape                |
|---------|------|----------------------|
| images  | FP16 | [1 x 3 x 640 x 640]  |
| output0 | FP32 | [1 x 300 x 57]       |

### BGR -> FP32 `yolo26s-pose__640_b1_fp32_bytebgr.onnx`

| Name         | Type | Shape               |
|--------------|------|---------------------|
| pixel_values | U8   | [1 x 640 x 640 x 3] |
| output0      | FP32 | [1 x 300 x 57]      |

### BGR -> FP16 `yolo26s-pose__640_b1_fp16_bytebgr.onnx`

| Name         | Type | Shape               |
|--------------|------|---------------------|
| pixel_values | U8   | [4 x 640 x 640 x 3] |
| output0      | FP32 | [4 x 300 x 57]      |



## Seg `yolo26s-seg`

### FP32 -> FP32 `yolo26s-seg__640_b1_fp32.onnx`

| Name    | Type | Shape                |
|---------|------|----------------------|
| images  | FP32 | [1 x 3 x 640 x 640]  |
| output0 | FP32 | [1 x 300 x 38]       |
| output1 | FP32 | [1 x 32 x 160 x 160] |

### FP16 -> FP16 `yolo26s-seg__640_b1_fp16.onnx`

| Name    | Type | Shape                |
|---------|------|----------------------|
| images  | FP16 | [1 x 3 x 640 x 640]  |
| output0 | FP32 | [1 x 300 x 38]       |
| output1 | FP16 | [1 x 32 x 160 x 160] |

### BGR -> FP32 `yolo26s-seg__640_b1_fp32_bytebgr.onnx`

| Name         | Type | Shape                |
|--------------|------|----------------------|
| pixel_values | U8   | [1 x 640 x 640 x 3]  |
| output0      | FP32 | [1 x 300 x 38]       |
| output1      | FP32 | [1 x 32 x 160 x 160] |

### BGR -> FP16 `yolo26s-seg__640_b1_fp16_bytebgr.onnx`

| Name         | Type | Shape                |
|--------------|------|----------------------|
| pixel_values | U8   | [1 x 640 x 640 x 3]  |
| output0      | FP32 | [1 x 300 x 38]       |
| output1      | FP16 | [1 x 32 x 160 x 160] |
