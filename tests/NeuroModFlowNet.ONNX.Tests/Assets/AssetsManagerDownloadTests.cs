using NeuroModFlowNet.ONNX.Demo.Assets;

namespace NeuroModFlowNet.ONNX.Tests.Assets;

public class AssetsManagerDownloadTests
{
    private const string ModelFileName = "yolo26n/yolo26n__640_b1_fp16.onnx";

    [Theory]
    [InlineData("paddleocr/detection/v3/det.onnx", "https://huggingface.co/monkt/paddleocr-onnx/resolve/main/detection/v3/det.onnx")]
    [InlineData("/paddleocr/detection/v5/det.onnx", "https://huggingface.co/monkt/paddleocr-onnx/resolve/main/detection/v5/det.onnx")]
    [InlineData(@"paddleocr\detection\v3\det.onnx", "https://huggingface.co/monkt/paddleocr-onnx/resolve/main/detection/v3/det.onnx")]
    public void GetAssetUrl_UsesOriginalPaddleOcrRepositoryForUnmodifiedPaddleOcrModels(
        string modelFileName,
        string expectedUrl)
    {
        string assetUrl = AssetsManager.GetAssetUrl(modelFileName);

        Assert.Equal(expectedUrl, assetUrl);
    }

    [Theory]
    [InlineData("paddleocr/detection/v3/det_bytebgr.onnx", "https://huggingface.co/NeuroModFlowNet/NeuroModFlowNet-ONNX-Demo-Models/resolve/main/paddleocr/detection/v3/det_bytebgr.onnx")]
    [InlineData("/paddleocr/detection/v5/det_bytebgr.onnx", "https://huggingface.co/NeuroModFlowNet/NeuroModFlowNet-ONNX-Demo-Models/resolve/main/paddleocr/detection/v5/det_bytebgr.onnx")]
    public void GetAssetUrl_UsesPreparedRepositoryForByteBgrPaddleOcrModels(
        string modelFileName,
        string expectedUrl)
    {
        string assetUrl = AssetsManager.GetAssetUrl(modelFileName);

        Assert.Equal(expectedUrl, assetUrl);
    }

    [Fact]
    public void GetAssetUrl_UsesPreparedRepositoryForNonPaddleOcrModels()
    {
        string assetUrl = AssetsManager.GetAssetUrl(ModelFileName);

        Assert.Equal(
            "https://huggingface.co/NeuroModFlowNet/NeuroModFlowNet-ONNX-Demo-Models/resolve/main/yolo26n/yolo26n__640_b1_fp16.onnx",
            assetUrl);
    }

    [Fact]
    public void GetAssetUrl_UsesExplicitBaseUrlWhenProvided()
    {
        string assetUrl = AssetsManager.GetAssetUrl(
            "paddleocr/detection/v3/det.onnx",
            "https://example.test/assets/");

        Assert.Equal("https://example.test/assets/paddleocr/detection/v3/det.onnx", assetUrl);
    }

    [Fact]
    public void ResolveModelsRoot_UsesExplicitTargetModelsRoot()
    {
        string targetModelsRoot = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));

        string modelsRoot = AssetsManager.ResolveModelsRoot(targetModelsRoot);

        Assert.Equal(Path.GetFullPath(targetModelsRoot), modelsRoot);
    }

    [Fact]
    public void ResolveModelsRoot_UsesEnvironmentModelsRoot()
    {
        string? previousModelsRoot = Environment.GetEnvironmentVariable(AssetsManager.ModelsRootEnvironmentVariable);
        string environmentModelsRoot = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));

        try
        {
            Environment.SetEnvironmentVariable(AssetsManager.ModelsRootEnvironmentVariable, environmentModelsRoot);

            string modelsRoot = AssetsManager.ResolveModelsRoot();

            Assert.Equal(Path.GetFullPath(environmentModelsRoot), modelsRoot);
        }
        finally
        {
            Environment.SetEnvironmentVariable(AssetsManager.ModelsRootEnvironmentVariable, previousModelsRoot);
        }
    }

    [Fact]
    public void ResolveModelsRoot_UsesSolutionRootModelsFolder()
    {
        string? previousModelsRoot = Environment.GetEnvironmentVariable(AssetsManager.ModelsRootEnvironmentVariable);
        string temporaryRoot = Path.Combine(Path.GetTempPath(), "NeuroModFlowNet.ONNX.Tests", Guid.NewGuid().ToString("N"));
        string repositoryRoot = Path.Combine(temporaryRoot, "repo");
        string startDirectory = Path.Combine(repositoryRoot, "samples", "Demo", "bin", "Debug", "net10.0");

        try
        {
            Environment.SetEnvironmentVariable(AssetsManager.ModelsRootEnvironmentVariable, null);
            Directory.CreateDirectory(Path.Combine(temporaryRoot, "models"));
            Directory.CreateDirectory(startDirectory);
            File.WriteAllText(Path.Combine(repositoryRoot, "NeuroModFlowNet.ONNX.slnx"), string.Empty);

            string modelsRoot = AssetsManager.ResolveModelsRoot(startDirectory: startDirectory);

            Assert.Equal(Path.Combine(repositoryRoot, "models"), modelsRoot);
        }
        finally
        {
            Environment.SetEnvironmentVariable(AssetsManager.ModelsRootEnvironmentVariable, previousModelsRoot);
            if(Directory.Exists(temporaryRoot))
                Directory.Delete(temporaryRoot, recursive: true);
        }
    }

    [Fact]
    public void ResolveModelsRoot_FallsBackToLocalApplicationDataWhenRepositoryRootIsMissing()
    {
        string? previousModelsRoot = Environment.GetEnvironmentVariable(AssetsManager.ModelsRootEnvironmentVariable);
        string temporaryRoot = Path.Combine(Path.GetTempPath(), "NeuroModFlowNet.ONNX.Tests", Guid.NewGuid().ToString("N"));
        string startDirectory = Path.Combine(temporaryRoot, "published", "app");

        try
        {
            Environment.SetEnvironmentVariable(AssetsManager.ModelsRootEnvironmentVariable, null);
            Directory.CreateDirectory(Path.Combine(temporaryRoot, "models"));
            Directory.CreateDirectory(startDirectory);

            string modelsRoot = AssetsManager.ResolveModelsRoot(startDirectory: startDirectory);

            Assert.Equal(
                Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                    "NeuroModFlowNet.ONNX",
                    "models"),
                modelsRoot);
        }
        finally
        {
            Environment.SetEnvironmentVariable(AssetsManager.ModelsRootEnvironmentVariable, previousModelsRoot);
            if(Directory.Exists(temporaryRoot))
                Directory.Delete(temporaryRoot, recursive: true);
        }
    }

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
