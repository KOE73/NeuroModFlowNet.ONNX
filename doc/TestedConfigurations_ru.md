# Проверенные GPU-конфигурации ONNX Runtime

Этот документ фиксирует конфигурации, на которых реально запускался NeuroModFlowNet.ONNX.
- Версии `Microsoft.ML.OnnxRuntime.Gpu`, драйвера, CUDA, cuDNN и TensorRT
- Режимов работы

Указанные конкретные сочетания не означают что другие сочетания не будут работать, просто их не проверяли. 

## Проверено

OS - Windows
ONNX = `Microsoft.ML.OnnxRuntime.` = NuGet version

| GPU         | ONNX NuGet | NVIDIA driver | CUDA | cuDNN            | TensorRT           | Backend          | 
| ----------- | ---------- | ------------- | ---  | -----------------| ------------------ | ---------------- | 
| RTX 5090    | Gpu 1.25.1 | 595.79        | 12.9 | 9.10.2.21 cu12   | 10.12.0.36 cu12.9  | CUDA,TensorRT    | 
| GTX 1080 Ti | Gpu 1.25.1 | 581.57        | 12.9 | 9.10.2    cu12.9 | -                  | CUDA             |

### Общие замечания
Эти утверждения не гарантия, но могут помочь при выборе версий драйверов и т.п. для запуска ONNX Runtime:

- Серия GTX10 поддерживаается cuDNN до 9.10.2

## Официальные ссылки

- [NVIDIA cuDNN downloads](https://developer.nvidia.com/cudnn-downloads)
- [ONNX Runtime CUDA Execution Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [ONNX Runtime install requirements](https://onnxruntime.ai/docs/install/)
- [NVIDIA legacy CUDA GPU compute capability](https://developer.nvidia.com/cuda/gpus/legacy)

## Что присылать для пополнения таблицы

Если у вас NeuroModFlowNet.ONNX запустился на другой машине, пришлите конфигурацию:

- модель GPU;
- Windows/Linux и версия ОС;
- версия NVIDIA driver;
- версия `Microsoft.ML.OnnxRuntime.Gpu` или `Microsoft.ML.OnnxRuntime.Gpu.Windows`;
- версия CUDA;
- версия cuDNN;
- версия TensorRT, если запускался TensorRT;
- backend: `CPU`, `CUDA`, `TensorRT`, `DML`;
- нужен ли был `App.local.config` или достаточно системного `PATH`;
- коротко: какая demo/lab/sample запускалась и какой результат.

## Как получить версии на Windows

```powershell
nvidia-smi
nvcc --version
```

## Важно

Это список практических проверок проекта, а не гарантия NVIDIA или Microsoft. 
Перед обновлением ONNX Runtime, CUDA, cuDNN, TensorRT или драйвера сверяйтесь с официальной документацией ONNX Runtime и NVIDIA.
