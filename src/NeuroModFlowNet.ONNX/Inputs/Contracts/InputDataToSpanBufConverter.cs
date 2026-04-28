namespace NeuroModFlowNet.ONNX;

/// <summary>
/// Сигнатура конвертора входных данных в подготовленный буфер модели.
/// </summary>
/// <typeparam name="TInput">Тип входных данных, например Mat или List&lt;Mat&gt;.</typeparam>
/// <typeparam name="TModelElement">Тип элемента входного тензора модели.</typeparam>
/// <param name="inputData">Входные данные.</param>
/// <param name="outputBuffer">
/// Буфер для выходных данных. Должен точно соответствовать размеру выходных данных.
/// Проверки не предусмотрены, так как конвертеры работают с буферами модели и знают их размер заранее.
/// </param>
/// <param name="batch">Размер батча.</param>
public delegate void InputDataToSpanBufConverter<in TInput, TModelElement>(
    TInput inputData,
    Span<TModelElement> outputBuffer,
    int batch)
    where TModelElement : unmanaged;
