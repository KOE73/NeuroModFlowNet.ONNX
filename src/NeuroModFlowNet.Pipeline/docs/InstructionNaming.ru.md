# Система именования pipeline-инструкций

Этот файл фиксирует систему именования pipeline-инструкций.
Поводом стало имя `PrepareInputTensorInstruction`: оно слишком общее и не говорит, что инструкция фактически делает
zero-copy обертку `OpenCvSharp.Mat` в `OrtValue` поверх неуправляемой CPU-памяти.

Решение: имя инструкции должно быть видно в pipeline program и trace как технически однозначная команда. В имени должны
читаться семейство инструкции, источник и приемник данных, размещение памяти и характер операции. Название не должно
описывать намерение верхнего сценария вроде `Prepare`, `Process` или `Upload`, если фактическая операция точнее.

## Главные правила

- Суффикс `Instruction` не использовать. Namespace, папка, descriptor/opcode и base class уже показывают, что это
  pipeline-инструкция.
- Семейство команды задается явным префиксом: `Vm_`, `Model_`, `Op_`, `Copy_`, `Wrap_`, ~~`Collect_`~~, ~~`Resource_`~~.
- Для низкоуровневых операций использовать `_`, если PascalCase сливает смысл.
- Descriptor name и opcode должны повторять смысл класса. Для descriptor name допустим человекочитаемый вариант с `_`,
  для opcode - стабильный lower/camel token через `.`.

Пример:

```csharp
public sealed class Op_Onnx_Resize : PipelineInstructionBase
{
    public Op_Onnx_Resize(string inputKey, string outputKey)
        : base(PipelineInstructionDescriptor.Create(
            "Op_Onnx_Resize",
            "op.onnx.resize",
            ...))
    {
    }
}
```

## Семейства инструкций

Примечание: `Collect_` и `Resource_` пока выглядят искусственно и требуют доработки. Они оставлены ниже как
черновые варианты, но не считаются утвержденной частью системы.

- [`Vm_`](#vm_) - управление выполнением VM: ветвления, переходы, синхронизация, scope/run control.
- [`Model_`](#model_) - запуск доменной модели без раскрытия внутреннего runner/resource механизма.
- [`Op_`](#op_) - вычислительные операции над уже внутренними данными pipeline.
- [`Copy_`](#copy_) - явное копирование данных между представлениями или placement.
- [`Wrap_`](#wrap_) - zero-copy представление одних данных как другого типа.
- ~~[`Collect_`](#collect_) - сборка итоговых результатов из register values и промежуточных структур.~~
- ~~[`Resource_`](#resource_) - операции с долгоживущими ресурсами: ensure, reload, warmup, acquire/release.~~

### Vm_

`Vm_` - инструкции управления выполнением VM. Они не выполняют обработку данных, не запускают модели и не описывают
memory placement.

Примеры:`Vm_BranchIf` `Vm_JumpToLabel` `Vm_OrderedSync` `Vm_StatefulResourceSync` `Vm_StopRun` `Vm_Return` `Vm_Fork` `Vm_Join` `Vm_BeginScope` `Vm_EndScope` 

### Model_

`Model_` - инструкции запуска модели как доменной операции pipeline. Имя не должно раскрывать, какой внутренний механизм
используется: `OnnxRunner`, reloadable runner, batching resource, file-backed model или runtime-created context. Это
деталь реализации инструкции.

Не использовать в имени модели служебные слова `Run`, `Runner`, `OnnxRunner`, `Batched`, `Reloadable`, если они не
являются частью публичного смысла команды.

Примеры: `Model_YoloBox` `Model_YoloObb` `Model_YoloSeg` `Model_YoloPose` `Model_YoloCls` `Model_PaddleDet` `Model_PaddleRec` 

Если когда-то политика выполнения станет именно публичным различием поведения, ее можно добавить после домена:

```text
Model_YoloObb_Batched;
Model_PaddleRec_ReloadableBatch;
```

Но базовое правило - не выносить policy в имя, пока она остается внутренней реализацией.

### Op_

`Op_` - вычислительные операции над уже внутренними данными pipeline. После `Op_` указывается движок/механизм выполнения:
`Onnx`, `Cv`, `Cuda`, `Halide`, `Simd` и т.п. Тип входного объекта и placement в имени `Op_` не указываются: это видно
из descriptor requirements и контролируется глобальной диагностикой/валидацией pipeline.

Примеры:

```text
Op_Onnx_Resize;
Op_Onnx_Crop;
Op_Onnx_BgrU8Hwc_To_RgbFP32Nchw_Div255;
Op_Onnx_BgrU8Hwc_To_RgbFP16Nchw_Div255;
Op_Cv_Resize;
Op_Cv_Crop;
Op_Cuda_Crop;
Op_Halide_Crop;
Op_Simd_Normalize;
```

### Copy_

`Copy_` - явное копирование данных. Инструкция не должна иметь скрытых fast-path по договоренности о ключах или типах:
если имя начинается с `Copy_`, данные копируются всегда.

Примеры:

```text
Copy_OrtValue_To_MatImage;
Copy_OrtTensor_To_ModelDevice;
```

### Wrap_

`Wrap_` - zero-copy представление уже существующей памяти как другого типа. Инструкция должна явно удерживать lifetime
исходной памяти, если это требуется для безопасности результирующего представления.

Примеры:

```text
Wrap_MatImage_To_OrtTensor;
Wrap_UnmanagedMem_To_OrtTensorBase;
```

### ~~Collect_~~

~~`Collect_` - сборка результата из нескольких register values или промежуточных структур. Это финализация результата, а
не вычислительный operator.~~

Примеры:

~~`Collect_FrameResult;`~~
~~`Collect_OcrRows;`~~
~~`Collect_ModelOutputs;`~~

### ~~Resource_~~

~~`Resource_` - операции с долгоживущими ресурсами pipeline: проверка, обновление, reload, warmup, acquire/release.
Не использовать этот префикс для обычного запуска модели: запуск модели относится к `Model_`.~~

Примеры:

~~`Resource_ModelShape_Ensure;`~~
~~`Resource_PaddleRec_Reload;`~~
~~`Resource_ModelSession_Warmup;`~~

## Терминология

`HostMem` - память, доступная CPU. Она может быть managed или unmanaged.

`ManagedMem` - память под управлением .NET GC.

`UnmanagedMem` - pointer + length + owner/lifetime.

`ModelDeviceMem` - память execution provider модели. Для CUDA это GPU VRAM, для TensorRT/DirectML соответствующая device
memory, для CPU backend - CPU memory. В имени не использовать `Gpu`, потому что pipeline должен оставаться переносимым
между CUDA, TensorRT, DirectML и CPU backend.

`MatImage` - `OpenCvSharp.Mat`, рассматриваемый как изображение.

`OrtTensor` - tensor value ONNX Runtime как pipeline-понятие, когда важна именно тензорная природа значения.

`OrtValue` - CLR/API-носитель ONNX Runtime. Использовать в имени тогда, когда важно, что инструкция принимает или
возвращает именно `OrtValue`, а не просто абстрактный tensor. Пример: `Copy_OrtValue_To_MatImage`, потому что входом
является `OrtValue`, а выходом `MatImage`.

`Bgr`, `Rgb`, `Bgra`, `Rgba` - порядок цветовых каналов внутри image tensor. Для image preprocessing порядок каналов
является частью контракта и должен быть виден в имени команды.

`U8`, `FP32`, `FP16` - тип элементов tensor. `U8` означает byte/uint8 image data, `FP32` - float, `FP16` - half.

`Hwc`, `Nchw` - layout image tensor. `Hwc` означает height/width/channels, `Nchw` - batch/channels/height/width.
Для одиночного кадра batch все равно остается частью tensor shape, но в имени layout пишется без отдельного `N`, если
важное преобразование происходит между image-layout формами `HWC` и `NCHW`.

`Div255` - нормализация диапазона byte image `[0..255]` в float range `[0..1]` через умножение на `1/255`.
Использовать это имя вместо общего `Normalize`, если именно эта математика является публичным смыслом команды.

`Wrap` - zero-copy представление поверх существующей памяти.

`Copy` - реальное копирование данных.

`Ensure` - проверка или приведение предусловия.

`Collect` - сборка результата.

`Branch` - условное изменение control flow.

`Jump` - безусловный переход выполнения.

`Sync` - синхронизация выполнения.

## Порядок частей имени

Порядок частей имени является частью стандарта.

### Преобразование данных

```text
Copy_<Source>_To_<Target>;
Wrap_<Source>_To_<Target>;
```

Порядок чтения:

```text
что и где сейчас -> во что и где должно оказаться -> каким способом
```

Примеры:

```text
Copy_OrtValue_To_MatImage;
Copy_OrtTensor_To_ModelDevice;
Wrap_MatImage_To_OrtTensor;
Wrap_UnmanagedMem_To_OrtTensorBase;
```

### Применение оператора

```text
Op_<Engine>_<Operator>;
```

Порядок чтения:

```text
каким движком выполняется -> какой оператор
```

Примеры:

```text
Op_Onnx_Resize;
Op_Onnx_Crop;
Op_Cv_Resize;
Op_Simd_Normalize;
```

Для image tensor preprocessing, где сама операция состоит в смене формата, layout, типа и диапазона, используется
расширенная форма:

```text
Op_<Engine>_<SourceColor><SourceType><SourceLayout>_To_<TargetColor><TargetType><TargetLayout>_<RangeTransform>;
```

Такие команды не должны превращаться в одну CISC-инструкцию с большим набором параметров. Если меняется публичный
контракт данных, лучше добавить отдельную элементарную команду, чтобы trace и pipeline program читались как
последовательность конкретных преобразований.

Примеры:

```text
Op_Onnx_BgrU8Hwc_To_RgbFP32Nchw_Div255;
Op_Onnx_BgrU8Hwc_To_RgbFP16Nchw_Div255;
Op_Onnx_BgraU8Hwc_To_RgbFP32Nchw_DropAlphaDiv255;
Op_Onnx_BgrU8Hwc_To_RgbFP32Nchw_Sub127p5Div127p5;
```

В имени не писать `Tensor`, `OrtTensor`, `ModelDeviceMem` и похожие слова, если речь идет об `Op_`: для операторов
тип register value и placement проверяются descriptor-ами и общей диагностикой pipeline. В имени остается то, что
меняет смысл вычисления: engine, color order, element type, layout и range transform.

### Управление VM

```text
Vm_<Command>;
```

Примеры:

```text
Vm_BranchIf;
Vm_JumpToLabel;
Vm_OrderedSync;
```

### Запуск модели

```text
Model_<DomainModelCommand>;
```

Примеры:

```text
Model_YoloObb;
Model_YoloSeg;
Model_PaddleDet;
Model_PaddleRec;
```

### ~~Сборка результата~~

~~`Collect_<ResultName>;`~~

Примеры:

~~`Collect_FrameResult;`~~
~~`Collect_OcrRows;`~~

### ~~Работа с ресурсом~~

~~`Resource_<ResourceName>_<Action>;`~~

Примеры:

~~`Resource_ModelShape_Ensure;`~~
~~`Resource_ModelSession_Warmup;`~~
~~`Resource_PaddleRec_Reload;`~~

## Правило AnyMem

`AnyMem` использовать только если инструкция действительно поддерживает несколько размещений памяти без изменения
внешнего поведения.

Плохо:

~~`Copy_OrtTensorAnyMem_To_MatImageHostMem;`~~

если `AnyMem` просто скрывает неявный CPU/GPU перенос.

Хорошо:

```text
Op_Onnx_Shape_Ensure;
```

если операция действительно не провоцирует скрытое копирование.

## База для Mat и unmanaged memory

Операция `Wrap_MatImage_To_OrtTensor` ориентирована на `Mat`, но ее ядро шире: получить
pointer/length/shape/type из неуправляемой памяти и создать `OrtTensor` без копирования.

Разбивать это можно внутри `NeuroModFlowNet.Pipeline.ONNX`:

```text
Wrap_UnmanagedMem_To_OrtTensorBase;
Wrap_MatImage_To_OrtTensor;
```

`Wrap_MatImage_To_OrtTensor` может отвечать только за получение memory view:

```text
Mat -> ensure contiguous -> pointer + byte length + shape + element kind -> optional owned clone
```

База ONNX-слоя отвечает за:

```text
validate source memory
create OrtTensor wrapper
register owned resource if needed
set output register with disposeWithContext
```

Не выносить эту базу в `NeuroModFlowNet.Pipeline`. Нейтральная unmanaged-memory модель без создания `OrtTensor` не дает
завершенной pipeline-операции и создает лишнюю прослойку.

## Размещение по проектам

- `NeuroModFlowNet.Pipeline` - `Vm_`, базовая VM, descriptors, tracing, branching, synchronization;
- `NeuroModFlowNet.Pipeline.ONNX` - `Model_`, `Op_`, `Copy_` и `Wrap_` для ONNX/OrtTensor, dynamic ONNX operators,
unmanaged memory -> `OrtTensor` bridge;
- `NeuroModFlowNet.ONNX` - `IRunner<TIn,TOut>`, model context, converters, extractors, без VM control-flow;
- `labs/*` - эксперименты, временная диагностика, `Console.Write*`, UI workflow.

## Имена runtime graph builder-ов

В `NeuroModFlowNet.ONNX.Graph` builder-класс называется по операции, а не по механизму хранения или рантайму.
Слова `Runtime` и `Onnx` в имени builder-а обычно лишние: проект и namespace уже говорят, что это ONNX graph builder,
создающий модель для выполнения в runtime.

Примеры:

```text
CropBuilder;
ResizeBuilder;
IdentityBuilder;
CropResizeBuilder;
BgrU8Hwc_To_RgbNchw_Div255Builder;
```
