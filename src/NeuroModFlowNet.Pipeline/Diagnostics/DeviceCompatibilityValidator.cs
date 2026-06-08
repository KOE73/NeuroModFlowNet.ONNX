using System;
using System.Reflection;

namespace NeuroModFlowNet.Pipeline;

/// <summary>
/// EN: Validates memory location compatibility of instruction inputs and outputs during the warmup run.
/// RU: Валидирует совместимость расположения переменных в памяти (CPU/GPU) с бэкендами исполнения на этапе прогрева.
/// </summary>
public static class DeviceCompatibilityValidator
{
    public static void Validate(IOp instruction, VmRunContext context)
    {
        ArgumentNullException.ThrowIfNull(instruction);
        ArgumentNullException.ThrowIfNull(context);

        // 1. Попытка рефлексивного извлечения onnxContext для определения бэкенда шага
        object? onnxContext = TryGetOnnxContext(instruction);
        if (onnxContext is null)
        {
            // Если это не ONNX-инструкция (нет контекста ONNX), то бэкенд по умолчанию считается CPU,
            // либо мы не навязываем жестких проверок для вспомогательных шагов.
            return;
        }

        // 2. Извлекаем бэкенд из onnxContext
        string backendStr = "Cpu";
        try
        {
            dynamic contextDyn = onnxContext;
            backendStr = contextDyn.InferenceBackend.ToString();
        }
        catch
        {
            // Игнорируем ошибки извлечения бэкенда
        }

        bool isGpuBackend = !backendStr.Equals("Cpu", StringComparison.OrdinalIgnoreCase);
        VarMemoryLocation expectedLocation = isGpuBackend
            ? VarMemoryLocation.Gpu
            : VarMemoryLocation.Cpu;

        var descriptor = instruction.Descriptor;

        // 3. Валидируем входы (Reads)
        foreach (var read in descriptor.Reads)
        {
            if (context.TryGetObject(read.Key, out object? value) && value is not null)
            {
                var actualLocation = VarDebugInspector.DetectMemoryLocation(value, value.GetType());

                // Проверяем несовпадение: если ожидаем GPU, но тензор на CPU (или наоборот)
                if (expectedLocation == VarMemoryLocation.Gpu && actualLocation == VarMemoryLocation.Cpu)
                {
                    throw new InvalidOperationException(
                        $"[COMPATIBILITY ERROR] Инструкция '{descriptor.Name}' ({descriptor.Operation}) выполняется на GPU ({backendStr}), " +
                        $"но входящая переменная '{read.Key}' расположена на CPU. " +
                        $"Это приведет к скрытому неявному копированию памяти (Host-to-Device) и падению FPS. " +
                        $"Добавьте явный шаг переноса (например, UploadToDevice).");
                }

                if (expectedLocation == VarMemoryLocation.Cpu && actualLocation == VarMemoryLocation.Gpu)
                {
                    throw new InvalidOperationException(
                        $"[COMPATIBILITY ERROR] Инструкция '{descriptor.Name}' ({descriptor.Operation}) выполняется на CPU, " +
                        $"но входящая переменная '{read.Key}' расположена на GPU. " +
                        $"Это приведет к скрытому неявному копированию памяти (Device-to-Host). " +
                        $"Добавьте явный шаг выгрузки (например, DownloadImage).");
                }
            }
        }

        // 4. Валидируем выходы (Writes)
        foreach (var write in descriptor.Writes)
        {
            if (context.TryGetObject(write.Key, out object? value) && value is not null)
            {
                var actualLocation = VarDebugInspector.DetectMemoryLocation(value, value.GetType());

                if (expectedLocation == VarMemoryLocation.Gpu && actualLocation == VarMemoryLocation.Cpu)
                {
                    throw new InvalidOperationException(
                        $"[COMPATIBILITY ERROR] Инструкция '{descriptor.Name}' ({descriptor.Operation}) выполняется на GPU ({backendStr}), " +
                        $"но создала выходной тензор '{write.Key}' на CPU. " +
                        $"Возможно, при создании OrtValue был ошибочно передан CPU-аллокатор.");
                }

                if (expectedLocation == VarMemoryLocation.Cpu && actualLocation == VarMemoryLocation.Gpu)
                {
                    throw new InvalidOperationException(
                        $"[COMPATIBILITY ERROR] Инструкция '{descriptor.Name}' ({descriptor.Operation}) выполняется на CPU, " +
                        $"но создала выходной тензор '{write.Key}' на GPU.");
                }
            }
        }
    }

    private static object? TryGetOnnxContext(IOp instruction)
    {
        Type type = instruction.GetType();
        while (type != null && type != typeof(object))
        {
            var field = type.GetField("onnxContext", BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Public);
            if (field is not null)
            {
                return field.GetValue(instruction);
            }
            type = type.BaseType!;
        }
        return null;
    }
}
