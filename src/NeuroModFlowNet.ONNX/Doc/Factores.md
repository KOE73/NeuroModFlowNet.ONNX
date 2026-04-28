# ⚠️ Generated Document

This file is generated/maintained from the current factory/extractor implementations and `Yolo26_2.md`.

Do not remove this warning or the regeneration prompt below when updating the document.

Regeneration prompt:

> Refresh this document without removing the generated-document warning or this prompt. Compare current YOLO factory/extractor implementations with `Yolo26_2.md`, where real existing YOLO26 output variants are listed. Correct the tables and notes. Mark factory/extractor variants that exist in code but have no matching real output variant in `Yolo26_2.md` with a separate symbol.

# YOLO Factories vs real Yolo26 exports

Legend:

| Mark | Meaning |
| ---- | ------- |
| ✅ | Factory/extractor exists and matches at least one real output variant from `Yolo26_2.md`. |
| 🧪 | Code exists, but `Yolo26_2.md` does not list a matching real ONNX export/output variant. |
| ⚙️ | Supported by metadata-based `CreateRunner(...)`, but no fixed named factory method exists. |
| — | Not implemented. |

Notes:

- `Yolo26_2.md` lists only `b1` model files. `List_*` factory methods are therefore marked 🧪 when they exist, because this document does not confirm a real multi-batch YOLO26 export.
- `Pos` and `Sym` are preprocessing algorithm families. `Yolo26_2.md` describes tensor dtype/shape, not normalization semantics, so `Pos`/`Sym` are checked against output compatibility.
- `BgrDirect` means `U8 [1 x H x W x 3]` model input. For YOLO26 byte-BGR exports the graph still decides whether the internal path is FP32 or FP16.

## Real output variants from `Yolo26_2.md`

| Task | Real output variant(s) |
| ---- | ---------------------- |
| Box | `FP32 [1 x 300 x 6]` |
| OBB | `FP32 [1 x 300 x 7]` |
| Class | `FP32 [1 x 1000]`, `FP16 [1 x 1000]` |
| Pose | `FP32 [1 x 300 x 57]` |
| Seg | `FP32 [1 x 300 x 38] + FP32 [1 x 32 x 160 x 160]`, `FP32 [1 x 300 x 38] + FP16 [1 x 32 x 160 x 160]` |

## Current fixed factory methods

| Factory | Single Pos FP32 | Single Sym FP32 | Single Bgr FP32 | Single Pos FP16 | Single Sym FP16 | Single Bgr FP16 | List Pos FP32 | List Sym FP32 | List Pos FP16 | List Sym FP16 | Metadata `CreateRunner` |
| ------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | ------------- | ------------- | ------------- | ------------- | ----------------------- |
| `YoloBoxFactory` | ✅ | ✅ | ✅ | 🧪 | 🧪 | — | 🧪 | 🧪 | 🧪 | 🧪 | ✅ |
| `YoloClsFactory` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🧪 | 🧪 | 🧪 | 🧪 | ✅ |
| `YoloSegFactory` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | — | — | ✅ |
| `YoloObbFactory` | ✅ | ✅ | ✅ | ⚙️ | ⚙️ | ⚙️ | 🧪 | 🧪 | — | — | ✅ |
| `YoloPoseFactory` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🧪 | 🧪 | 🧪 | 🧪 | ✅ |

## Factory details

### `YoloBoxFactory`

Real YOLO26 Box exports always produce `FP32 [1 x 300 x 6]`, including FP16-input and byte-BGR FP16 exports.

| Code path | Status | Reason |
| --------- | ------ | ------ |
| `Single_PosCvdnn_FP32`, `Single_SymCvdnn_FP32`, `Single_BgrDirect_FP32` | ✅ | Fixed methods use FP32 output extractors and match real Box output. |
| `Single_PosCvdnn_FP16`, `Single_SymCvdnn_FP16` | 🧪 | These use FP16 output extractors, but `Yolo26_2.md` lists no Box model with FP16 output. |
| `List_*` methods | 🧪 | Implemented, but `Yolo26_2.md` lists only `b1` exports. |
| `CreateRunner<TOut>(context)` | ✅ | Selects converter from input metadata and extractor from output metadata, so real FP16-input/FP32-output models are handled correctly. |

Extractors with no matching real YOLO26 Box output listed:

| Extractor | Status |
| --------- | ------ |
| `YoloBoxNmsFP16Extractor` | 🧪 |
| `YoloBoxNmsFP16StdExtractor` | 🧪 |

### `YoloClsFactory`

Real YOLO26 Class exports include both FP32 and FP16 outputs.

| Code path | Status | Reason |
| --------- | ------ | ------ |
| `Single_*_FP32` | ✅ | Matches FP32 classification output. |
| `Single_*_FP16` | ✅ | Matches FP16 classification output. |
| `List_*` methods | 🧪 | Implemented, but `Yolo26_2.md` lists only `b1` exports. |
| `CreateRunner(context)` | ✅ | Selects FP32 or FP16 extractor from output metadata. |

### `YoloSegFactory`

Real YOLO26 Seg exports include FP32 prototypes and FP16 prototypes. Detection rows stay FP32 in both variants.

| Code path | Status | Reason |
| --------- | ------ | ------ |
| `Single_*_FP32` | ✅ | Matches `FP32 + FP32` seg output. |
| `Single_*_FP16` | ✅ | Matches mixed `FP32 + FP16` seg output. |
| `List_*` methods | — | No fixed list factory methods exist. |
| `CreateRunner(context)` | ✅ | Selects FP32 or FP16 seg extractor from prototype tensor metadata. |

### `YoloObbFactory`

Real YOLO26 OBB exports always produce `FP32 [1 x 300 x 7]`, including FP16-input and byte-BGR FP16 exports.

| Code path | Status | Reason |
| --------- | ------ | ------ |
| `Single_PosCvdnn_FP32`, `Single_SymCvdnn_FP32`, `Single_BgrDirect_FP32` | ✅ | Fixed methods match real FP32 OBB output. |
| `SingleInternal_PosCvdnn_FP32` | ✅ | Internal FP32 output type over the same real output tensor. |
| Fixed `Single_*_FP16` methods | ⚙️ | Real FP16-input OBB exports exist, but fixed FP16 methods are not defined. Use metadata `CreateRunner(...)`. |
| `List_PosCvdnn_FP32`, `List_SymCvdnn_FP32` | 🧪 | Implemented, but `Yolo26_2.md` lists only `b1` exports. |
| `CreateRunner<TOut>(context)` | ✅ | Selects FP16 input converter when model input is FP16 and keeps FP32 OBB extractor. |

### `YoloPoseFactory`

Real YOLO26 Pose exports always produce `FP32 [1 x 300 x 57]`, including FP16-input and byte-BGR FP16 exports.

| Code path | Status | Reason |
| --------- | ------ | ------ |
| `Single_*_FP32` | ✅ | Matches FP32 pose output. |
| `Single_*_FP16` | ✅ | Uses FP16 input converter with FP32 pose extractor, matching real FP16-input/FP32-output exports. |
| `List_*` methods | 🧪 | Implemented, but `Yolo26_2.md` lists only `b1` exports. |
| `CreateRunner(context)`, `CreateRunner17(context)` | ✅ | Select converter from input metadata and use FP32 pose extractors. |
