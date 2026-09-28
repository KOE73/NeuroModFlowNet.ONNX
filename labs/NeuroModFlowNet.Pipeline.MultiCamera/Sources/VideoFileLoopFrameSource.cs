using OpenCvSharp;

namespace NeuroModFlowNet.Pipeline.MultiCamera;

/// <summary>
/// EN: Plays a list of video files back to back and, when <c>loop</c> is set, wraps around to the first file forever.
/// Files that cannot be opened or that yield no frames (for example zero-byte recorder segments) are skipped.
///
/// RU: Проигрывает список видеофайлов подряд и при <c>loop</c> бесконечно начинает сначала. Файлы, которые не
/// открываются или не дают кадров (например, нулевые сегменты рекордера), пропускаются.
/// </summary>
internal sealed class VideoFileLoopFrameSource : IFrameSource
{
    static readonly string[] DefaultPatterns = ["*.ts", "*.mp4", "*.mkv", "*.avi", "*.mov"];

    readonly IReadOnlyList<string> files;
    readonly bool loop;
    readonly VideoAccelerationType acceleration;
    readonly Mat reusableFrame = new();
    VideoCapture? capture;
    int fileIndex = -1;
    int consecutiveOpenFailures;

    VideoFileLoopFrameSource(IReadOnlyList<string> files, bool loop, VideoAccelerationType acceleration, string description)
    {
        this.acceleration = acceleration;
        this.files = files;
        this.loop = loop;
        Description = description;
    }

    public string Description { get; }

    public double Fps { get; private set; }

    public string? CurrentFile => fileIndex >= 0 && fileIndex < files.Count ? files[fileIndex] : null;

    public static VideoFileLoopFrameSource FromFolder(string folder, string? pattern, bool loop, VideoAccelerationType acceleration)
    {
        string[] files = EnumerateFolder(folder, pattern);
        return new VideoFileLoopFrameSource(files, loop, acceleration, $"folder {folder} ({files.Length} files{(loop ? ", loop" : string.Empty)})");
    }

    /// <summary>Non-empty video files of a folder, sorted by name (recorder segments play in time order).</summary>
    public static string[] EnumerateFolder(string folder, string? pattern)
    {
        if(!Directory.Exists(folder))
            throw new DirectoryNotFoundException($"Video folder was not found: {folder}");

        string[] patterns = string.IsNullOrWhiteSpace(pattern)
            ? DefaultPatterns
            : pattern.Split([';', ','], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

        string[] files = patterns
            .SelectMany(filePattern => Directory.EnumerateFiles(folder, filePattern, SearchOption.TopDirectoryOnly))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .Where(static file => new FileInfo(file).Length > 0)
            .OrderBy(static file => file, StringComparer.OrdinalIgnoreCase)
            .ToArray();

        if(files.Length == 0)
            throw new FileNotFoundException($"No non-empty video files matching '{string.Join(';', patterns)}' in {folder}.");

        return files;
    }

    public static VideoFileLoopFrameSource FromFile(string file, bool loop, VideoAccelerationType acceleration)
    {
        if(!File.Exists(file))
            throw new FileNotFoundException($"Video file was not found: {file}", file);

        return new VideoFileLoopFrameSource([file], loop, acceleration, $"file {file}{(loop ? " (loop)" : string.Empty)}");
    }

    /// <summary>Explicit playlist in the given order, for example hand-picked recorder segments that contain bags.</summary>
    public static VideoFileLoopFrameSource FromFiles(IReadOnlyList<string> files, bool loop, VideoAccelerationType acceleration)
    {
        ArgumentNullException.ThrowIfNull(files);
        if(files.Count == 0)
            throw new ArgumentException("At least one video file is required.", nameof(files));

        foreach(string file in files)
        {
            if(!File.Exists(file))
                throw new FileNotFoundException($"Video file was not found: {file}", file);
        }

        string names = string.Join(", ", files.Select(Path.GetFileName));
        return new VideoFileLoopFrameSource(files.ToArray(), loop, acceleration, $"files [{names}]{(loop ? " (loop)" : string.Empty)}");
    }

    public ISourceFrame? ReadNext(CancellationToken cancellationToken)
    {
        while(!cancellationToken.IsCancellationRequested)
        {
            if(capture is null && !OpenNextFile())
                return null;

            if(capture!.Read(reusableFrame) && !reusableFrame.Empty())
            {
                consecutiveOpenFailures = 0;
                return new MatSourceFrame(reusableFrame.Clone());
            }

            // End of file (or decoder error): move on to the next file.
            CloseCurrent();
        }

        return null;
    }

    bool OpenNextFile()
    {
        while(true)
        {
            if(consecutiveOpenFailures >= files.Count)
                throw new InvalidOperationException($"None of the {files.Count} video files in {Description} could be opened.");

            fileIndex++;
            if(fileIndex >= files.Count)
            {
                if(!loop)
                    return false;

                fileIndex = 0;
            }

            VideoCapture? candidate = VideoCaptureFactory.TryOpen(files[fileIndex], acceleration);
            if(candidate is not null)
            {
                capture = candidate;
                Fps = candidate.Fps;
                return true;
            }

            consecutiveOpenFailures++;
        }
    }

    void CloseCurrent()
    {
        capture?.Dispose();
        capture = null;
    }

    public void Dispose()
    {
        CloseCurrent();
        reusableFrame.Dispose();
    }
}
