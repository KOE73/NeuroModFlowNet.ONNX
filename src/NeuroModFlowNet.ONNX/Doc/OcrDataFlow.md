# OCR Data Flow

This diagram illustrates the top-down data flow from an input image to the final recognized text.

> [!TIP]
> **Layout Controls:**
> - **Spacing:** Tune `rankSpacing` (vertical) and `nodeSpacing` (horizontal) in the `%%{init: ...}%%` block.
> - **No-Wrap:** Handled by `white-space: nowrap` in `classDef` statements to keep text inline.
> - **Class Widths:** To style groups of nodes, add `min-width: 200px;` directly into the `classDef` definitions.
> - **Individual Styling:** You can override style for an individual node by adding a `style` declaration anywhere in the diagram:
>   ```mermaid
>   style B1 width:250px,height:45px;
>   ```

```mermaid
---
config:
  theme: base
  htmlLabels: true
  flowchart:
    nodeSpacing: 15
    rankSpacing: 24
    diagramPadding: 4
    padding: 1
    wrappingWidth: 120
    curve: linear1
    subGraphTitleMargin:
      top: 0
      bottom: 16
---
graph TD
    %% Styling Definitions (Add min-width/width here to enforce sizes)
    classDef default    fill:#2D3748,stroke:#4A5568,stroke-width:2px,color:#E2E8F0,white-space:nowrap;
    classDef clsInput   fill:#2B6CB0,stroke:#2C5282,stroke-width:2px,color:#E2E8F0,font-weight:bold,white-space:nowrap;
    classDef clsModel   fill:#805AD5,stroke:#553C9A,stroke-width:2px,color:#E2E8F0,font-weight:bold,white-space:nowrap;
    classDef clsProcess fill:#319795,stroke:#285E61,stroke-width:2px,color:#E2E8F0,white-space:nowrap;
    classDef clsMerge   fill:#DD6B20,stroke:#C05621,stroke-width:2px,color:#E2E8F0,font-weight:bold,white-space:nowrap;
    classDef clsOutput  fill:#38A169,stroke:#276749,stroke-width:2px,color:#E2E8F0,font-weight:bold,white-space:nowrap;

    A([Input Image]):::clsInput

    %% YOLO OBB Branch
    subgraph YoloObbDetectionPhase [YOLO OBB Detection Phase]
        B1{{YOLO OBB Model}}:::clsModel
        C1[YoloObbOcrRegionMapper: Coordinate Mapping & Scale]:::clsProcess
        B1 --> C1
    end

    %% PaddleDet Branch
    subgraph PaddleDetectionPhase [Paddle Detection Phase]
        B2{{PaddleDet Model}}:::clsModel
        
        subgraph PaddlePostProcessing [PaddleOCRDetMaskRegionExtractor: CV2 Extraction Pipeline]
            C2_1[Probability Map / Mask]:::clsProcess
            C2_2[CV2 Threshold: Create Binary Mask]:::clsProcess
            C2_3[CV2 FindContours: Detect Text Components]:::clsProcess
            C2_4[CV2 MinAreaRect: Find Enclosing Rects]:::clsProcess
            C2_5[Unclip & Expand: Apply Polygon Offset]:::clsProcess
            
            C2_1 --> C2_2
            C2_2 --> C2_3
            C2_3 --> C2_4
            C2_4 --> C2_5
        end
        B2 --> C2_1
    end

    A --> B1
    A --> B2

    C1 --> D([Raw Text Regions]):::clsMerge
    C2_5 --> D

    %% Region Analysis & Postprocessing
    subgraph RegionAnalysis [Region Analysis & Postprocessing]
        D --> E[OcrRegionPostprocessor]:::clsProcess
        E --> E1[Domain Filters: Width, Height, Ratio Constraints]:::clsProcess
        E1 --> E2[Overlap Suppression: Remove Duplicate Boxes]:::clsProcess
        E2 --> E3[Line Merge: Combine Nearby Blocks]:::clsProcess
    end

    E3 --> F([Filtered Text Regions]):::clsMerge

    %% ROI Extraction & Image Processing
    subgraph RoiExtraction [ROI Extraction & Image Processing]
        F --> G[NaiveTextRegionExtractor: Crop, Scale & Pad]:::clsProcess
        
        subgraph Pipeline [Image Processing Pipeline: Dynamic Array of Handlers]
            H1[Stage 1: e.g. Brightness / Contrast]:::clsProcess
            H2[Stage 2: e.g. Gamma / Threshold]:::clsProcess
            H3[Stage N: e.g. Blur / Sharpen]:::clsProcess
            
            H1 --> H2
            H2 -.-> H3
        end
        G --> H1
    end

    H3 --> I([Processed ROI Crops]):::clsMerge

    %% Recognition Phase
    subgraph RecognitionPhase [Recognition Phase]
        I --> J{{PaddleOCR Rec Model}}:::clsModel
        J --> K[Format Options: Standard / Spaces]:::clsProcess
    end

    K --> L([Final Recognized Text]):::clsOutput
```
