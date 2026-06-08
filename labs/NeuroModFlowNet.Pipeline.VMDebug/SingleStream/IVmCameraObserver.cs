namespace NeuroModFlowNet.Pipeline.VMDebug;

/// <summary>
/// Observer used by the single-stream camera host to report VM progress without knowing how it is rendered.
/// </summary>
internal interface IVmCameraObserver
{
    void CameraOpened(int cameraIndex, int width, int height, double fps);

    void WarmupCompleted(VmRunOutcome outcome);

    void FrameCompleted(int frameIndex, VmRunOutcome outcome, TimeSpan awaitTime);
}
