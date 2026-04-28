using System;

namespace NeuroModFlowNet.ONNX;

/// <summary>
/// EN: Interface providing metadata about the adapter implementation and expected input type.
/// RU: Интерфейс, предоставляющий метаданные о реализации адаптера и ожидаемом типе входных данных.
/// </summary>
public interface IConverterInfo
{
    Type InputType { get; }
}
