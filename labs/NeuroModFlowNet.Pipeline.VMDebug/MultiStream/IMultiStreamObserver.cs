namespace NeuroModFlowNet.Pipeline.VMDebug;

/// <summary>
/// Observer used by the multi-stream host to notify the dashboard of runtime events.
/// </summary>
internal interface IMultiStreamObserver
{
    void CameraOpened(int cameraIndex, int width, int height, double fps);

    void AllWorkersWarmedUp();
}
