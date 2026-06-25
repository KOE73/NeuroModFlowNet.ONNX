using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.ONNX.Tests.Common;

internal static class VisualTestOutput
{
    public static void SaveOrShow(string title, Mat image, string outputPath)
    {
        PipelineOnnxTestEnvironment environment = PipelineOnnxTestEnvironment.Current;

        if(environment.SaveVisualArtifacts)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
            Cv2.ImWrite(outputPath, image);
        }

        if(!environment.InteractiveVisualArtifacts)
            return;

        Cv2.ImShow(title, image);
        Cv2.WaitKey(0);
        Cv2.DestroyWindow(title);
    }
}
