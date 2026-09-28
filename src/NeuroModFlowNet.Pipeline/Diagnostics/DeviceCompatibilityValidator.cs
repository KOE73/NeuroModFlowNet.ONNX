namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// Validates memory location compatibility of instruction inputs and outputs during the warmup run.
/// </summary>
public static class DeviceCompatibilityValidator
{
    public static void Validate(IOp instruction, VmRunContext context)
    {
        ArgumentNullException.ThrowIfNull(instruction);
        ArgumentNullException.ThrowIfNull(context);

        if(instruction is not IHasExecutionDevice device)
            return;

        bool isGpu = device.IsGpuExecution;
        string deviceName = device.ExecutionDeviceName;
        VarMemoryLocation expectedLocation = isGpu ? VarMemoryLocation.Gpu : VarMemoryLocation.Cpu;

        OpDescriptor descriptor = instruction.Descriptor;

        foreach(VarRequirement read in descriptor.Reads)
        {
            if(!context.TryGetObject(read.Key, out object? value) || value is null || !IsTensorLike(value))
                continue;

            VarMemoryLocation actual = VarDebugInspector.DetectMemoryLocation(value, value.GetType());

            if(expectedLocation == VarMemoryLocation.Gpu && actual == VarMemoryLocation.Cpu)
                throw new InvalidOperationException(
                    $"[COMPATIBILITY ERROR] Инструкция '{descriptor.Name}' ({descriptor.Operation}) выполняется на GPU ({deviceName}), " +
                    $"но входящая переменная '{read.Key}' расположена на CPU. " +
                    $"Добавьте явный шаг переноса (например, UploadToDevice).");

            if(expectedLocation == VarMemoryLocation.Cpu && actual == VarMemoryLocation.Gpu)
                throw new InvalidOperationException(
                    $"[COMPATIBILITY ERROR] Инструкция '{descriptor.Name}' ({descriptor.Operation}) выполняется на CPU, " +
                    $"но входящая переменная '{read.Key}' расположена на GPU. " +
                    $"Добавьте явный шаг выгрузки (например, DownloadImage).");
        }

        foreach(VarRequirement write in descriptor.Writes)
        {
            if(!context.TryGetObject(write.Key, out object? value) || value is null || !IsTensorLike(value))
                continue;

            VarMemoryLocation actual = VarDebugInspector.DetectMemoryLocation(value, value.GetType());

            if(expectedLocation == VarMemoryLocation.Gpu && actual == VarMemoryLocation.Cpu)
                throw new InvalidOperationException(
                    $"[COMPATIBILITY ERROR] Инструкция '{descriptor.Name}' ({descriptor.Operation}) выполняется на GPU ({deviceName}), " +
                    $"но создала выходной тензор '{write.Key}' на CPU. " +
                    $"Возможно, при создании OrtValue был ошибочно передан CPU-аллокатор.");

            if(expectedLocation == VarMemoryLocation.Cpu && actual == VarMemoryLocation.Gpu)
                throw new InvalidOperationException(
                    $"[COMPATIBILITY ERROR] Инструкция '{descriptor.Name}' ({descriptor.Operation}) выполняется на CPU, " +
                    $"но создала выходной тензор '{write.Key}' на GPU.");
        }
    }

    /// <summary>
    /// EN: Placement is validated for tensor payloads only. Scalars, typed results, back transforms and other metadata
    /// written by a GPU instruction legitimately live in managed memory and do not indicate an implicit copy.
    ///
    /// RU: Размещение проверяется только у тензорных значений. Скаляры, typed результаты, обратные преобразования и
    /// прочие метаданные, записанные GPU-инструкцией, законно живут в managed-памяти и не означают неявного копирования.
    /// </summary>
    static bool IsTensorLike(object value)
    {
        string fullName = value.GetType().FullName ?? string.Empty;
        return fullName == "Microsoft.ML.OnnxRuntime.OrtValue" || fullName == "OpenCvSharp.Mat";
    }
}
