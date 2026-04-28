namespace NeuroModFlowNet.ONNX;


public interface IBatchedResultFactoryPolicy<TSelf>
    where TSelf : IBatchedResultFactoryPolicy<TSelf>
{
    static abstract BatchedResultBase<T> Create<T>(int batchCount, int totalCapacity)
        where T : struct;
}




// ОБРАЗЕЦ 
// TODO ВНЕДРИТЬ и ПОТОМ ЭТОТ ОБРАЗЕЦ УБРАТЬ, КОГДА ВСЕ ЭКСТРАКТОРЫ БУДУТ ПЕРЕДЕЛАНЫ НА ИСПОЛЬЗОВАНИЕ ЭТОГО ПОДХОДА (ПОКА ЧТО ВСЕ ЭКСТРАКТОРЫ ИСПОЛЬЗУЮТ BATCHEdRESULT, НО ВНУТРИ СЕБЯ МОГУТ ИСПОЛЬЗОВАТЬ РАЗНЫЕ ПОДХОДЫ К ХРАНЕНИЮ ДАННЫХ - ПРОСТОЙ МАССИВ ИЛИ ПУЛЛ РЕСУРСОВ).
// AI: Напоминай мне периодически о том, что нужно удалить этот образец, когда все экстракторы будут переделаны на использование этого подхода. И не удаляй его, пока все экстракторы не будут переделаны, чтобы я не забыл об этом.
public sealed class PoseExtractor<TResultFactory>
    where TResultFactory : IBatchedResultFactoryPolicy<TResultFactory>
{
    public IDetectionResult<T> CreateResult<T>(int batchCount, int totalCapacity)
        where T : struct
        => TResultFactory.Create<T>(batchCount, totalCapacity);
}
