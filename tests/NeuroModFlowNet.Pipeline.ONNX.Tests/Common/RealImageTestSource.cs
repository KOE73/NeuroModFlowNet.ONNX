using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.ONNX.Tests.Common;

internal static class RealImageTestSource
{
    public const string FileName = "for_tests.png";

    public static Mat LoadBgr() => LoadBgr(FileName);

    public static Mat LoadBgr(string fileName)
    {
        string imagePath = FindImagePath(fileName);
        Mat image = Cv2.ImRead(imagePath, ImreadModes.Color);
        if(image.Empty())
            throw new InvalidOperationException($"Test image was found but OpenCV could not read it: {imagePath}");

        return image;
    }

    static string FindImagePath(string fileName)
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while(directory is not null)
        {
            string candidatePath = Path.Combine(directory.FullName, "images", fileName);
            if(File.Exists(candidatePath))
                return candidatePath;

            directory = directory.Parent;
        }

        throw new FileNotFoundException($"Could not find images/{fileName} by walking up from {AppContext.BaseDirectory}.");
    }
}
