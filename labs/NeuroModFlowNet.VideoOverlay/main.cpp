#include "GpuOverlay.h"
#include "OverlayBenchmark.h"
#include "CodecBenchmark.h"
#include "VideoCodecs.h"

#include <windows.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavdevice/avdevice.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_d3d11va.h>
#include <libavutil/opt.h>
}

#include <algorithm>
#include <charconv>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {
using namespace VideoCodecs;

struct Options {
    std::string input;
    std::string camera;
    std::string rtsp;
    std::string output;
    std::string size{"1920x1080"};
    std::string fps{"30"};
    int frames{};
    int bitrate{6'000'000};
    std::vector<int> benchmarkStreams;
    int benchmarkSeconds{6};
    int matrixOutputs{-1};
    int matrixWindowMs{};
    bool benchmarkStaggered{};
    bool benchmarkCodecs{};
    CodecBenchmarkOutput benchmarkOutput;
    std::string preview;
};

volatile LONG stopRequested{};

BOOL WINAPI ConsoleStop(DWORD signal) {
    if (signal != CTRL_C_EVENT && signal != CTRL_BREAK_EVENT) return FALSE;
    InterlockedExchange(&stopRequested, 1);
    return TRUE;
}

int ShouldStop(void*) { return InterlockedCompareExchange(&stopRequested, 0, 0) != 0; }

std::string Utf8(const wchar_t* text) {
    const int size = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, text, -1, nullptr, 0, nullptr, nullptr);
    if (size == 0) throw std::runtime_error("invalid Unicode text");
    std::string utf8(size, '\0');
    if (!WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, text, -1, utf8.data(), size, nullptr, nullptr))
        throw std::runtime_error("cannot convert Unicode text");
    utf8.pop_back();
    return utf8;
}

void Usage() {
    std::cout <<
        "VideoOverlay --input FILE.mp4 --output FILE.mp4 [--frames N] [--bitrate BPS]\n"
        "VideoOverlay --camera NAME [--size 1920x1080] [--fps 30]\n"
        "             --output FILE.mp4|udp://127.0.0.1:5390?pkt_size=1316 [--frames N] [--bitrate BPS]\n"
        "VideoOverlay --rtsp URL --output FILE.mp4|udp://127.0.0.1:5390?pkt_size=1316 [--frames N]\n"
        "VideoOverlay --input FILE.mp4 --benchmark 1,10,20 [--seconds 6] [--phase synchronized|staggered]\n"
        "VideoOverlay --input FILE.mp4 --full-benchmark 1,2,3 [--seconds 6] [--phase synchronized|staggered]\n"
        "VideoOverlay --input FILE.mp4 --full-benchmark 1,2,3 --matrix-outputs 10 --window-ms 800\n"
        "             [--output-size 1280x720] [--through decode|convert|draw] [--decode-surfaces N]\n"
        "VideoOverlay --input FILE.mp4 --full-benchmark 12 --preview udp://127.0.0.1:5392 [--seconds 120]\n"
        "Inputs must provide progressive 8-bit video. Output is H.264 in MP4 or MPEG-TS over UDP.\n";
}

int PositiveInt(std::string_view value, const std::string& option) {
    int number{};
    auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), number);
    if (error != std::errc{} || end != value.data() + value.size() || number <= 0)
        throw std::runtime_error("expected a positive integer for " + option);
    return number;
}

std::vector<int> BenchmarkCounts(std::string_view list, const std::string& option) {
    std::vector<int> result;
    for (;;) {
        const auto separator = list.find(',');
        result.push_back(PositiveInt(list.substr(0, separator), option));
        if (separator == std::string_view::npos) return result;
        list.remove_prefix(separator + 1);
    }
}

Options Parse(int argc, wchar_t** argv) {
    Options options;
    std::filesystem::path outputPath;
    for (int index = 1; index < argc; index += 2) {
        const auto key = Utf8(argv[index]);
        if (index + 1 >= argc) throw std::runtime_error("missing value for " + key);
        const auto value = Utf8(argv[index + 1]);
        if (key == "--input") options.input = value;
        else if (key == "--camera") options.camera = value;
        else if (key == "--rtsp") options.rtsp = value;
        else if (key == "--size") options.size = value;
        else if (key == "--fps") options.fps = value;
        else if (key == "--output") {
            options.output = value;
            outputPath = argv[index + 1];
        }
        else if (key == "--frames") options.frames = PositiveInt(value, key);
        else if (key == "--bitrate") options.bitrate = PositiveInt(value, key);
        else if (key == "--benchmark" || key == "--full-benchmark") {
            if (!options.benchmarkStreams.empty()) throw std::runtime_error("choose one benchmark mode");
            options.benchmarkStreams = BenchmarkCounts(value, key);
            options.benchmarkCodecs = key == "--full-benchmark";
        }
        else if (key == "--preview") options.preview = value;
        else if (key == "--seconds") options.benchmarkSeconds = PositiveInt(value, key);
        else if (key == "--matrix-outputs") options.matrixOutputs = value == "0" ? 0 : PositiveInt(value, key);
        else if (key == "--window-ms") options.matrixWindowMs = PositiveInt(value, key);
        else if (key == "--first-output") options.benchmarkOutput.firstMatrixOutput = PositiveInt(value, key);
        else if (key == "--decode-surfaces") options.benchmarkOutput.decodeSurfaces = PositiveInt(value, key);
        else if (key == "--output-size") {
            const auto separator = value.find('x');
            if (separator == std::string::npos) throw std::runtime_error("expected --output-size WIDTHxHEIGHT");
            options.benchmarkOutput.width = PositiveInt(std::string_view(value).substr(0, separator), key);
            options.benchmarkOutput.height = PositiveInt(std::string_view(value).substr(separator + 1), key);
            if (options.benchmarkOutput.width % 2 || options.benchmarkOutput.height % 2)
                throw std::runtime_error("--output-size must have even NV12 dimensions");
        }
        else if (key == "--through") {
            if (value == "decode") options.benchmarkOutput.stage = CodecBenchmarkStage::Decode;
            else if (value == "convert") options.benchmarkOutput.stage = CodecBenchmarkStage::Convert;
            else if (value == "draw") options.benchmarkOutput.stage = CodecBenchmarkStage::Draw;
            else throw std::runtime_error("expected --through decode, convert or draw");
        }
        else if (key == "--phase") {
            if (value != "synchronized" && value != "staggered")
                throw std::runtime_error("expected --phase synchronized or staggered");
            options.benchmarkStaggered = value == "staggered";
        }
        else throw std::runtime_error("unknown option " + key);
    }
    if (options.matrixOutputs >= 0 &&
        (!options.benchmarkCodecs || options.matrixOutputs > 10 || !options.preview.empty()))
        throw std::runtime_error("--matrix-outputs requires --full-benchmark, 0-10 outputs and no preview");
    if (options.matrixWindowMs != 0 && (options.matrixOutputs < 0 || options.matrixWindowMs % 40 != 0))
        throw std::runtime_error("--window-ms must be a multiple of 40 in matrix mode (25 FPS)");
    if (options.benchmarkOutput.width && !options.benchmarkCodecs)
        throw std::runtime_error("--output-size requires --full-benchmark");
    if (options.benchmarkOutput.stage != CodecBenchmarkStage::Draw && options.matrixOutputs != 0)
        throw std::runtime_error("--through decode or convert requires --matrix-outputs 0");
    if (options.benchmarkOutput.decodeSurfaces &&
        (!options.benchmarkCodecs || !options.preview.empty() || options.benchmarkOutput.decodeSurfaces > 64))
        throw std::runtime_error("--decode-surfaces requires the timed benchmark and a count between 1 and 64");
    if (options.benchmarkOutput.firstMatrixOutput &&
        options.benchmarkOutput.firstMatrixOutput > options.matrixOutputs)
        throw std::runtime_error("--first-output requires --matrix-outputs at least as large");
    if (!options.preview.empty() && (!options.benchmarkCodecs || options.benchmarkStreams.size() != 1 ||
                                      !options.preview.starts_with("udp://")))
        throw std::runtime_error("--preview requires a UDP URL and one stream count via --full-benchmark");
    if (!options.benchmarkStreams.empty()) {
        if (options.input.empty() || !options.camera.empty() || !options.rtsp.empty() || !options.output.empty())
            throw std::runtime_error("--benchmark requires a file via --input and uses no --output");
        return options;
    }
    const int inputs = !options.input.empty() + !options.camera.empty() + !options.rtsp.empty();
    if (inputs != 1 || options.output.empty())
        throw std::runtime_error("choose one input with --input, --camera or --rtsp and specify --output");
    if (!options.rtsp.empty() && !options.rtsp.starts_with("rtsp://"))
        throw std::runtime_error("--rtsp requires an rtsp:// URL");
    if (!options.output.starts_with("udp://")) {
        if (!options.output.ends_with(".mp4")) throw std::runtime_error("output must be .mp4 or udp://");
        if (std::filesystem::exists(outputPath))
            throw std::runtime_error("output file already exists");
    }
    return options;
}

ComPtr<ID3D11Device> CreateGpu() {
    ComPtr<IDXGIFactory1> factory;
    CheckHr(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "create DXGI factory");
    for (UINT index = 0;; ++index) {
        ComPtr<IDXGIAdapter1> adapter;
        const HRESULT result = factory->EnumAdapters1(index, &adapter);
        if (result == DXGI_ERROR_NOT_FOUND) break;
        CheckHr(result, "enumerate DXGI adapter");
        DXGI_ADAPTER_DESC1 description{};
        CheckHr(adapter->GetDesc1(&description), "read DXGI adapter description");
        if (description.VendorId != 0x10DE || (description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) continue;

        auto device = CreateVideoDevice(adapter.Get());
        std::cout << "GPU: " << Utf8(description.Description) << std::endl;
        return device;
    }
    throw std::runtime_error("no NVIDIA D3D11 adapter found");
}

Input OpenInput(const Options& options) {
    Dictionary settings;
    const AVInputFormat* driver = nullptr;
    std::string source = options.input;
    if (!options.camera.empty()) {
        driver = Require(av_find_input_format("dshow"), "FFmpeg DirectShow capture unavailable");
        source = "video=" + options.camera;
        settings.Set("video_size", options.size);
        settings.Set("framerate", options.fps);
    } else if (!options.rtsp.empty()) {
        driver = Require(av_find_input_format("rtsp"), "FFmpeg RTSP input unavailable");
        source = options.rtsp;
        settings.Set("rtsp_transport", "tcp");
        settings.Set("allowed_media_types", "video");
        settings.Set("timeout", "5000000");
    }
    auto* format = Require(avformat_alloc_context(), "allocate input context");
    format->interrupt_callback = {ShouldStop, nullptr};
    if (!options.camera.empty()) format->video_codec_id = AV_CODEC_ID_H264;
    CheckAv(avformat_open_input(&format, source.c_str(), driver, &settings.value), "open video input");
    return Input(format);
}

class EncodedOutput {
public:
    EncodedOutput(const Options& options, const AVFrame& source, AVRational rate, AVBufferRef* pool)
        : packet_(Require(av_packet_alloc(), "allocate encoded packet")) {
        const bool udp = options.output.starts_with("udp://");
        AVFormatContext* format = nullptr;
        CheckAv(avformat_alloc_output_context2(&format, nullptr, udp ? "mpegts" : "mp4",
                                              options.output.c_str()), "create output muxer");
        mux_.reset(format);
        stream_ = Require(avformat_new_stream(format, nullptr), "create output video stream");
        encoder_ = CreateEncoder(source, rate, pool, options.bitrate, (format->oformat->flags & AVFMT_GLOBALHEADER));
        auto* encoder = encoder_.get();

        CheckAv(avcodec_parameters_from_context(stream_->codecpar, encoder), "write H.264 stream parameters");
        stream_->time_base = encoder->time_base;
        stream_->avg_frame_rate = rate;
        Dictionary settings;
        if (udp) settings.Set("flush_packets", "1");
        else settings.Set("movflags", "+faststart");
        CheckAv(avio_open(&format->pb, options.output.c_str(), AVIO_FLAG_WRITE), "open encoded output");
        CheckAv(avformat_write_header(format, &settings.value), "write output header");
        std::cout << "Output: h264_nvenc, " << (udp ? "MPEG-TS over UDP" : "MP4") << std::endl;
    }

    int64_t Timestamp(int64_t timestamp, AVRational inputTimeBase) {
        int64_t pts = previousPts_ < 0 ? 0 : previousPts_ + FrameDuration();
        if (timestamp != AV_NOPTS_VALUE) {
            const auto scaled = av_rescale_q(timestamp, inputTimeBase, encoder_->time_base);
            if (startPts_ == AV_NOPTS_VALUE) startPts_ = scaled - pts;
            if (scaled - startPts_ > previousPts_) pts = scaled - startPts_;
        }
        return previousPts_ = pts;
    }

    void Write(AVFrame& encoded, int64_t pts) {
        encoded.pts = pts;
        encoded.duration = FrameDuration();
        Send(&encoded);
    }

    void Finish() {
        Send(nullptr);
        CheckAv(av_write_trailer(mux_.get()), "finish encoded output");
    }

private:
    int64_t FrameDuration() const {
        return av_rescale_q(1, av_inv_q(encoder_->framerate), encoder_->time_base);
    }

    void Send(AVFrame* frame) {
        CheckAv(avcodec_send_frame(encoder_.get(), frame), "send GPU frame to NVENC");
        for (;;) {
            const int result = avcodec_receive_packet(encoder_.get(), packet_.get());
            if (result == AVERROR(EAGAIN) || result == AVERROR_EOF) return;
            CheckAv(result, "receive H.264 packet");
            av_packet_rescale_ts(packet_.get(), encoder_->time_base, stream_->time_base);
            packet_->stream_index = stream_->index;
            CheckAv(av_interleaved_write_frame(mux_.get(), packet_.get()), "write H.264 packet");
        }
    }

    std::unique_ptr<AVFormatContext, MuxDeleter> mux_;
    AVStream* stream_{};
    Codec encoder_;
    Packet packet_;
    int64_t startPts_{AV_NOPTS_VALUE};
    int64_t previousPts_{-1};
};

void Run(const Options& options) {
    const auto device = CreateGpu();
    const auto hardware = BindDevice(device.Get());
    const auto input = OpenInput(options);
    // For these inputs the demuxer supplies codec parameters. Avoid a software frame probe.
    const int streamIndex = av_find_best_stream(input.get(), AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    CheckAv(streamIndex, "find input video stream");
    auto* stream = input->streams[streamIndex];
    if (options.benchmarkCodecs) {
        const auto window = options.matrixWindowMs > 0 ? std::chrono::milliseconds(options.matrixWindowMs) :
                            std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::seconds(options.benchmarkSeconds));
        RunCodecBenchmark(device.Get(), hardware.get(), input.get(), stream, options.benchmarkStreams,
                          window, options.benchmarkStaggered, options.bitrate,
                          options.preview, options.matrixOutputs, options.benchmarkOutput, ShouldStop);
        return;
    }
    auto rate = av_guess_frame_rate(input.get(), stream, nullptr);
    if (rate.num <= 0 || rate.den <= 0) rate = {30, 1};
    const auto decoder = CreateDecoder(stream->codecpar, stream->time_base, hardware.get());
    const Packet packet(Require(av_packet_alloc(), "allocate input packet"));
    const Frame decoded(Require(av_frame_alloc(), "allocate decoded frame"));
    const Frame encoded(Require(av_frame_alloc(), "allocate encoder input frame"));
    Buffer encoderPool;
    std::unique_ptr<GpuOverlay> overlay;
    std::unique_ptr<EncodedOutput> output;
    int frameCount = 0;
    int width = 0;
    int height = 0;
    bool bt709 = false;
    bool fullRange = false;
    bool benchmarkComplete = false;
    const bool paceFile = !options.input.empty() && options.output.starts_with("udp://");
    const char* sourceType = !options.camera.empty() ? "DirectShow" : (!options.rtsp.empty() ? "RTSP/TCP" : "file");
    std::chrono::steady_clock::time_point playbackStart{};

    auto limitReached = [&] { return benchmarkComplete || (options.frames > 0 && frameCount >= options.frames); };
    auto receive = [&] {
        while (!limitReached() && !ShouldStop(nullptr)) {
            const int result = avcodec_receive_frame(decoder.get(), decoded.get());
            if (result == AVERROR(EAGAIN) || result == AVERROR_EOF) break;
            CheckAv(result, "receive D3D11VA decoded frame");
            auto decodedAt = std::chrono::steady_clock::now();
            auto* source = CheckGpuFrame(*decoded, device.Get());
            if ((decoded->flags & AV_FRAME_FLAG_INTERLACED) || (decoded->width % 2) || (decoded->height % 2))
                throw std::runtime_error("the lab requires progressive video with even dimensions");
            if (!options.benchmarkStreams.empty()) {
                if (decoded->width != 1920 || decoded->height != 1080)
                    throw std::runtime_error("the benchmark requires a decoded 1920x1080 source");
                RunOverlayBenchmark(device.Get(), source,
                    static_cast<UINT>(reinterpret_cast<uintptr_t>(decoded->data[1])), IsBt709(*decoded),
                    decoded->color_range == AVCOL_RANGE_JPEG, options.benchmarkStreams,
                    options.benchmarkSeconds, options.benchmarkStaggered, ShouldStop);
                benchmarkComplete = true;
                return;
            }
            if (!output) {
                width = decoded->width;
                height = decoded->height;
                bt709 = IsBt709(*decoded);
                fullRange = decoded->color_range == AVCOL_RANGE_JPEG;
                overlay = std::make_unique<GpuOverlay>(device.Get(), width, height, bt709, fullRange);
                encoderPool = CreateEncoderPool(hardware.get(), width, height);
                output = std::make_unique<EncodedOutput>(options, *decoded, rate, encoderPool.get());
                playbackStart = std::chrono::steady_clock::now();
                decodedAt = playbackStart;
                std::cout << "Input: " << avcodec_get_name(decoder->codec_id) << " via " << sourceType
                          << ", " << width << 'x' << height << " @ " << av_q2d(rate) << " fps\n"
                          << "Frames: D3D11VA NV12 -> D3D11 BGRA / Direct2D + DirectWrite -> D3D11 NV12\n"
                          << "Shared NVIDIA device, CPUAccessFlags=0 on video surfaces" << std::endl;
            }
            if (decoded->width != width || decoded->height != height || IsBt709(*decoded) != bt709 ||
                (decoded->color_range == AVCOL_RANGE_JPEG) != fullRange)
                throw std::runtime_error("input dimensions or color space changed during the stream");
            const int64_t pts = output->Timestamp(decoded->best_effort_timestamp, stream->time_base);
            if (paceFile) {
                using namespace std::chrono;
                const auto waitStart = steady_clock::now();
                const auto deadline = playbackStart + microseconds(av_rescale_q(pts, videoTimeBase, {1, 1000000}));
                while (steady_clock::now() < deadline && !ShouldStop(nullptr))
                    std::this_thread::sleep_until(std::min(deadline, steady_clock::now() + milliseconds(20)));
                if (ShouldStop(nullptr)) return;
                decodedAt += steady_clock::now() - waitStart;
            }
            CheckAv(av_hwframe_get_buffer(encoderPool.get(), encoded.get(), 0), "acquire NVENC surface");
            auto* destination = CheckGpuFrame(*encoded, device.Get());
            overlay->Render(source, static_cast<UINT>(reinterpret_cast<uintptr_t>(decoded->data[1])),
                            destination, pts);
            overlay->CompleteHostFrame(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - decodedAt).count());
            output->Write(*encoded, pts);
            ++frameCount;
            av_frame_unref(encoded.get());
            av_frame_unref(decoded.get());
        }
    };

    while (!ShouldStop(nullptr) && !limitReached()) {
        const int result = av_read_frame(input.get(), packet.get());
        if (result == AVERROR_EOF || ShouldStop(nullptr)) break;
        CheckAv(result, "read video packet");
        if (packet->stream_index == streamIndex) {
            CheckAv(avcodec_send_packet(decoder.get(), packet.get()), "send video packet to D3D11VA");
            receive();
        }
        av_packet_unref(packet.get());
    }
    if (!limitReached() && !ShouldStop(nullptr)) {
        CheckAv(avcodec_send_packet(decoder.get(), nullptr), "flush D3D11VA decode");
        receive();
    }
    if (benchmarkComplete) return;
    if (!output) throw std::runtime_error("input produced no D3D11 video frames");
    output->Finish();
    std::cout << "Encoded " << frameCount << " frames." << std::endl;
}
}

int wmain(int argc, wchar_t** argv) {
    if (argc == 1 || (argc == 2 && std::wstring_view(argv[1]) == L"--help")) {
        Usage();
        return 0;
    }
    try {
        const Options options = Parse(argc, argv);
        SetConsoleCtrlHandler(ConsoleStop, TRUE);
        avdevice_register_all();
        Run(options);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "VideoOverlay: " << error.what() << '\n';
        return 1;
    }
}
