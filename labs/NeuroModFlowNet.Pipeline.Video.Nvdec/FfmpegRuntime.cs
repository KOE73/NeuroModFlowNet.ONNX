using FFmpeg.AutoGen.Abstractions;
using FFmpeg.AutoGen.Bindings.DynamicallyLoaded;

namespace NeuroModFlowNet.Pipeline.Video.Nvdec;

/// <summary>
/// EN: Process-wide FFmpeg initialisation (native library path, log level) and error helpers. FFmpeg.AutoGen 9.x must
/// be paired with an FFmpeg 9 shared build (avcodec-63); a mismatch fails at load time.
///
/// RU: Инициализация FFmpeg на процесс (путь к нативным библиотекам, уровень лога) и помощники для ошибок.
/// FFmpeg.AutoGen 9.x работает только со сборкой FFmpeg 9 (avcodec-63); несовпадение падает при загрузке.
/// </summary>
public static unsafe class FfmpegRuntime
{
    static readonly object SyncRoot = new();
    static bool initialized;

    public static int ErrorAgain => ffmpeg.AVERROR(ffmpeg.EAGAIN);

    public static void Initialize(string librariesPath)
    {
        lock(SyncRoot)
        {
            if(initialized)
                return;

            if(!File.Exists(Path.Combine(librariesPath, "avcodec-63.dll")))
                throw new FileNotFoundException($"FFmpeg 9 shared libraries (avcodec-63.dll) were not found in '{librariesPath}'.");

            DynamicallyLoadedBindings.LibrariesPath = librariesPath;
            DynamicallyLoadedBindings.Initialize();
            ffmpeg.av_log_set_level(ffmpeg.AV_LOG_ERROR);
            initialized = true;
        }
    }

    public static string Version => ffmpeg.av_version_info();

    public static int Check(int result, string operation)
    {
        if(result < 0)
            throw new InvalidOperationException($"FFmpeg {operation} failed: {Describe(result)} ({result}).");

        return result;
    }

    public static string Describe(int error)
    {
        const int bufferSize = 1024;
        byte* buffer = stackalloc byte[bufferSize];
        ffmpeg.av_strerror(error, buffer, bufferSize);
        return new string((sbyte*)buffer).TrimEnd('\0');
    }
}
