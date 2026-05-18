# NeuroModFlowNet.ONNX JSON Configuration Guide

This document describes all possible configuration parameters that can be used in your `appsettings.json` for the **NeuroModFlowNet.ONNX** postprocessing and inference pipelines.

## Configuration Structure

The configuration is divided into two main sections:
- `inference`: Settings related to model execution thresholds.
- `postprocessing`: Settings for detection, region analysis, extraction, and image processing.

---

### 1. Inference Settings (`inference`)

Global thresholds applied directly during model inference.

```json
{
  "inference": {
    "obbThreshold": 0.3
  }
}
```

* **`obbThreshold`** (`float`): Confidence threshold for YOLO OBB (Oriented Bounding Box) detection. Boxes with a score lower than this will be ignored. Default is `0.3`.

---

### 2. Postprocessing Settings (`postprocessing`)

Settings that control how raw model outputs are converted into final text regions and recognized text.

#### 2.1 PaddleOCR Detection Mask (`detMask`)
Controls how the DB-style PaddleOCR detection mask is converted into text bounding boxes.

```json
"detMask": {
  "bitmapThreshold": 0.3,
  "boxScoreThreshold": 0.7,
  "minimumBoxSide": 3.0,
  "maxCandidateCount": 1000,
  "enableUnclip": true,
  "unclipRatio": 2.0
}
```
* **`bitmapThreshold`** (`float`): Binarization threshold for the probability map.
* **`boxScoreThreshold`** (`float`): Minimum average score of the region to be considered a valid text box.
* **`minimumBoxSide`** (`float`): Minimum size (in pixels) for the shortest side of a detected bounding box.
* **`maxCandidateCount`** (`int`): Maximum number of candidate contours to process per frame.
* **`enableUnclip`** (`bool`): Whether to expand (unclip) the detected text polygons to their full size.
* **`unclipRatio`** (`float`): The expansion ratio used during the unclip process.

#### 2.2 Region Analyzer (`analyzer`)
Filters and merges the text regions before they are cropped and recognized.

```json
"analyzer": {
  "minRegionHeight": 0.0,
  "maxRegionHeight": null,
  "minRegionWidth": 0.0,
  "maxRegionWidth": null,
  "minRegionAspectRatio": 0.0,
  "maxRegionAspectRatio": null,
  "maxRegions": 64,
  "overlapSuppression": { ... },
  "lineMerge": { ... }
}
```
* **`minRegionHeight` / `maxRegionHeight`** (`float`, `null`): Height limits for valid text regions.
* **`minRegionWidth` / `maxRegionWidth`** (`float`, `null`): Width limits for valid text regions.
* **`minRegionAspectRatio` / `maxRegionAspectRatio`** (`float`, `null`): Aspect ratio (Width / Height) limits.
* **`maxRegions`** (`int`, `null`): Hard limit on the number of regions to process per frame.

**Overlap Suppression (`overlapSuppression`)**
Removes duplicate or highly overlapping bounding boxes.
* **`enabled`** (`bool`): Toggles overlap suppression.
* **`ratio`** (`float`): The Intersection-over-Union (IoU) or coverage ratio above which boxes are considered overlapping. Default is `0.8`.

**Line Merge (`lineMerge`)**
Combines adjacent words or characters into single continuous lines.
* **`enabled`** (`bool`): Toggles line merging.
* **`angleDeltaDegrees`** (`float`): Maximum difference in angles for boxes to be merged.
* **`normalOffsetInHeights`** (`float`): Maximum vertical offset between boxes (relative to their height).
* **`heightRatio`** (`float`): Maximum ratio of heights between two boxes to be merged.
* **`gapInHeights`** (`float`): Maximum horizontal gap between boxes (relative to height).
* **`minimumCoverageRatio`** (`float`): Minimum coverage when checking intersections during merge.
* **`maxMergedRegionWidth`** (`float`, `null`): Maximum allowed width for a newly merged region.
* **`maxMergedRegionAspectRatio`** (`float`, `null`): Maximum allowed aspect ratio for a merged region.

#### 2.3 Recognition ROI & Processing (`recognitionRoi`)
Controls how the image is cropped and pre-processed before passing into the PaddleOCR Recognition model.

```json
"recognitionRoi": {
  "targetWidth": 320,
  "targetHeight": 48,
  "heightScale": 2.0,
  "adaptiveHeightEnabled": true,
  "adaptiveBasePad": 1.0,
  "adaptivePadRatio": 0.25,
  "adaptiveMaxPad": 8.0,
  "processing": { ... }
}
```
* **`targetWidth` / `targetHeight`** (`int`): The normalized dimensions the crop is resized to before recognition.
* **`heightScale`** (`float`): A static scale factor applied to the region height (if adaptive height is disabled).
* **`adaptiveHeightEnabled`** (`bool`): If true, dynamically calculates padding based on the region's actual height.
* **`adaptiveBasePad`** (`float`): Base vertical padding added (in pixels).
* **`adaptivePadRatio`** (`float`): Additional padding added relative to the region's height.
* **`adaptiveMaxPad`** (`float`): The maximum allowed padding.

**Processing Pipeline (`processing`)**
A sequence of image processing stages applied to the cropped region to improve OCR accuracy.

```json
"processing": {
  "enabled": true,
  "stages": [
    {
      "kind": "BrightnessContrast",
      "brightnessContrast": {
        "brightness": 0.0,
        "contrastPercent": 100.0
      }
    },
    {
      "kind": "Gamma",
      "gamma": {
        "gamma": 1.0
      }
    },
    {
      "kind": "Grayscale",
      "grayscale": {
        "useRed": true,
        "useGreen": true,
        "useBlue": true
      }
    },
    {
      "kind": "Threshold",
      "threshold": {
        "threshold": 128.0,
        "useOtsu": false
      }
    },
    {
      "kind": "GaussianBlur",
      "gaussianBlur": {
        "kernelSize": 3,
        "sigma": 0.0
      }
    },
    {
      "kind": "Sharpen",
      "sharpen": {
        "kernelSize": 3,
        "sigma": 0.0,
        "amount": 1.0
      }
    },
    {
      "kind": "AutoContrast"
    }
  ]
}
```
* **`enabled`** (`bool`): Globally toggles the entire processing pipeline.
* **`stages`** (`array`): A list of processing filters applied in order.

**Available Stage Kinds:**
1. **`BrightnessContrast`**: Adjusts brightness (`-255` to `255`) and contrast percentage.
2. **`Gamma`**: Applies non-linear gamma correction (`gamma` value).
3. **`Grayscale`**: Converts to grayscale, allowing you to include/exclude specific color channels.
4. **`Threshold`**: Binarizes the image using a fixed `threshold` (`0-255`) or dynamically via `useOtsu`.
5. **`GaussianBlur`**: Blurs the image to reduce noise (`kernelSize`, `sigma`).
6. **`Sharpen`**: Enhances edges using an unsharp mask (`kernelSize`, `sigma`, `amount`).
7. **`AutoContrast`**: Automatically stretches the histogram. (No additional parameters needed).
