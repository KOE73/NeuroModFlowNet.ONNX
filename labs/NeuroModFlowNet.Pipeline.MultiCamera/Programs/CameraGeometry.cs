namespace NeuroModFlowNet.Pipeline.MultiCamera;

/// <summary>Size of the rectified frame that the common program receives; tracking zones live in this space.</summary>
internal readonly record struct CameraGeometry(int Width, int Height);
