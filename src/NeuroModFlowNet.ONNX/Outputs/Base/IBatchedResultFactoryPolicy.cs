namespace NeuroModFlowNet.ONNX;


/// <summary>
/// EN:
/// Defines a result storage policy for creating batched detection result containers.
///
/// RU:
/// Описывает политику хранения результата для создания batch-контейнеров с результатами детекции.
/// </summary>
/// <remarks>
/// EN:
/// This interface is intentionally separated from concrete result containers.
/// It lets extractor code create result storage through a small static contract instead of directly depending on
/// BatchedResult{T}, BatchedResultPooled{T}, or another future storage implementation.
///
/// Current stage:
/// factories such as BatchedResultFactory and BatchedResultPooledFactory already implement this contract,
/// but most extractors still choose a concrete factory explicitly.
///
/// Future direction:
/// extractors can later receive TResultFactory as a generic parameter and call TResultFactory.Create{T}(...).
/// This will allow the same extraction logic to work with different storage strategies:
/// regular managed arrays, pooled arrays, debug wrappers, or other low-allocation containers.
///
/// RU:
/// Этот интерфейс намеренно отделен от конкретных контейнеров результата.
/// Он позволяет коду extractor создавать хранилище результата через небольшой статический контракт,
/// а не зависеть напрямую от BatchedResult{T}, BatchedResultPooled{T} или другой будущей реализации хранения.
///
/// Текущий этап:
/// factory-классы вроде BatchedResultFactory и BatchedResultPooledFactory уже реализуют этот контракт,
/// но большинство extractors пока явно выбирают конкретную factory.
///
/// Перспектива:
/// позже extractors смогут получать TResultFactory как generic-параметр и вызывать TResultFactory.Create{T}(...).
/// Это позволит использовать одну и ту же логику извлечения с разными стратегиями хранения:
/// обычные managed-массивы, массивы из пула, debug-обертки или другие low-allocation контейнеры.
/// </remarks>
/// <typeparam name="TSelf">Concrete result storage policy type.</typeparam>
public interface IBatchedResultFactoryPolicy<TSelf>
    where TSelf : IBatchedResultFactoryPolicy<TSelf>
{
    static abstract BatchedResultBase<T> Create<T>(int batchCount, int totalCapacity)
        where T : struct;
}




// ОБРАЗЕЦ 
// TODO ВНЕДРИТЬ и ПОТОМ ЭТОТ ОБРАЗЕЦ УБРАТЬ, КОГДА ВСЕ ЭКСТРАКТОРЫ БУДУТ ПЕРЕДЕЛАНЫ НА ИСПОЛЬЗОВАНИЕ ЭТОГО ПОДХОДА (ПОКА ЧТО ВСЕ ЭКСТРАКТОРЫ ИСПОЛЬЗУЮТ BATCHEdRESULT, НО ВНУТРИ СЕБЯ МОГУТ ИСПОЛЬЗОВАТЬ РАЗНЫЕ ПОДХОДЫ К ХРАНЕНИЮ ДАННЫХ - ПРОСТОЙ МАССИВ ИЛИ ПУЛЛ РЕСУРСОВ).
// AI: Напоминай мне периодически о том, что нужно удалить этот образец, когда все экстракторы будут переделаны на использование этого подхода. И не удаляй его, пока все экстракторы не будут переделаны, чтобы я не забыл об этом.
public sealed class DemoBatchedResultFactoryPolicy<TResultFactory>
    where TResultFactory : IBatchedResultFactoryPolicy<TResultFactory>
{
    public IDetectionResult<T> CreateResult<T>(int batchCount, int totalCapacity)
        where T : struct
        => TResultFactory.Create<T>(batchCount, totalCapacity);
}
