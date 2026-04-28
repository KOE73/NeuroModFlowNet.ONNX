using System;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: General interface for all inference runners.
/// RU: Общий интерфейс для всех раннеров инференса.
/// </summary>
public interface IRunner<in TIn, out TOut> : IDisposable
{
    TOut Predict(TIn input);

    /// <summary>
    /// EN: Accesses additional capabilities (e.g. IImageInfo) by searching both input and output paths.
    /// RU: Доступ к дополнительным возможностям (напр. IImageInfo) через поиск во входных и выходных путях.
    /// </summary>
    T? As<T>() where T : class;

    /// <summary>
    /// EN: Accesses additional capabilities specifically from the input adapter.
    /// RU: Доступ к дополнительным возможностям напрямую из адаптера ввода.
    /// </summary>
    T? InAs<T>() where T : class;

    /// <summary>
    /// EN: Accesses additional capabilities specifically from the result extractor.
    /// RU: Доступ к дополнительным возможностям напрямую из экстрактора результатов.
    /// </summary>
    T? OutAs<T>() where T : class;
}
