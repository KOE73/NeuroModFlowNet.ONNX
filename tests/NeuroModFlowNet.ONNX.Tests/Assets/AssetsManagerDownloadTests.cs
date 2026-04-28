using NeuroModFlowNet.ONNX.Demo.Assets;

namespace NeuroModFlowNet.ONNX.Tests.Assets;

public class AssetsManagerDownloadTests
{
    private const string ModelFileName = "yolo26n/yolo26n__640_b1_fp16.onnx";

    [Fact]
    public async Task GetAssetPathAsync_DownloadsPreparedModelFromConfiguredStorage()
    {
        string targetModelsRoot = Path.Combine(
            Path.GetTempPath(),
            "NeuroModFlowNet.ONNX.Tests",
            "Assets",
            Guid.NewGuid().ToString("N"));

        try
        {
            string assetPath = await AssetsManager.GetAssetPathAsync(
                ModelFileName,
                forceDownload: true,
                targetModelsRoot: targetModelsRoot);

            Assert.True(File.Exists(assetPath));
            Assert.EndsWith(ModelFileName.Replace('/', Path.DirectorySeparatorChar), assetPath);

            var fileInfo = new FileInfo(assetPath);
            Assert.True(fileInfo.Length > 0);
        }
        finally
        {
            if(Directory.Exists(targetModelsRoot))
                Directory.Delete(targetModelsRoot, recursive: true);
        }
    }

    [Fact]
    public async Task GetAssetPathAsync_DownloadsPreparedModelUsingModelNaming()
    {
        string modelFileName = ModelNaming.GetFileName(
            baseName: "yolo26n",
            imgSize: 640,
            batchSize: 1,
            precision: "fp16",
            isByteBgr: false);

        string targetModelsRoot = Path.Combine(
            Path.GetTempPath(),
            "NeuroModFlowNet.ONNX.Tests",
            "Assets",
            Guid.NewGuid().ToString("N"));

        try
        {
            string assetPath = await AssetsManager.GetAssetPathAsync(
                modelFileName,
                forceDownload: true,
                targetModelsRoot: targetModelsRoot);

            Assert.True(File.Exists(assetPath));
            Assert.EndsWith(ModelFileName.Replace('/', Path.DirectorySeparatorChar), assetPath);

            var fileInfo = new FileInfo(assetPath);
            Assert.True(fileInfo.Length > 0);
        }
        finally
        {
            if(Directory.Exists(targetModelsRoot))
                Directory.Delete(targetModelsRoot, recursive: true);
        }
    }
}
