#include "CodecBenchmark.h"
#include "CodecPreview.h"
#include "GpuTimeline.h"
#include "CompressedReplay.h"
#include "ParallelCodecBenchmark.h"

#include <conio.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

using namespace VideoCodecs;

namespace {
using namespace std::chrono;
using Clock = steady_clock;
using Moment = Clock::time_point;
constexpr AVRational frameRate{CompressedReplay::framesPerSecond, 1};
constexpr auto period = 1000ms / CompressedReplay::framesPerSecond;
constexpr size_t fixedEncoders = 5;
constexpr size_t maxEncoders = 10;

std::vector<OverlayScene> MakeScenes(unsigned count) {
    std::mt19937 random(std::random_device{}());
    std::uniform_int_distribution<int> kind(0, 3);
    std::uniform_int_distribution<uint32_t> component(85, 250);
    std::uniform_real_distribution<float> angle(-35, 35);
    std::uniform_real_distribution<float> phase(0, 6.28f);
    std::vector<OverlayScene> scenes(count);
    for (unsigned index = 0; index < count; ++index) {
        auto& scene = scenes[index];
        scene.stream = index + 1;
        for (size_t figure = 0; figure < scene.figures.size(); ++figure) {
            auto& entry = scene.figures[figure];
            entry.kind = static_cast<FigureKind>(figure < 4 ? static_cast<int>(figure) : kind(random));
            const auto red = component(random);
            const auto green = component(random);
            const auto blue = component(random);
            entry.color = (red << 16) | (green << 8) | blue;
            entry.angle = angle(random);
            entry.phase = phase(random);
        }
        std::shuffle(scene.figures.begin(), scene.figures.end(), random);
    }
    return scenes;
}

Frame ReadSourceProperties(AVBufferRef* hardware, const AVCodecParameters* parameters, const CompressedReplay& replay,
                           ID3D11Device* device) {
    const auto decoder = CreateDecoder(parameters, videoTimeBase, hardware);
    const auto packet = Packet(Require(av_packet_alloc(), "allocate source validation packet"));
    const auto decoded = Frame(Require(av_frame_alloc(), "allocate source validation frame"));
    for (size_t index = 0; index < replay.Size(); ++index) {
        replay.Read(index, packet.get());
        CheckAv(avcodec_send_packet(decoder.get(), packet.get()), "validate replay decoding");
        av_packet_unref(packet.get());
        const int result = avcodec_receive_frame(decoder.get(), decoded.get());
        if (result == AVERROR(EAGAIN)) continue;
        CheckAv(result, "read replay properties");
        CheckGpuFrame(*decoded, device);
        if ((decoded->flags & AV_FRAME_FLAG_INTERLACED) || decoded->width <= 0 || decoded->height <= 0 ||
            decoded->width % 2 || decoded->height % 2)
            throw std::runtime_error("full benchmark requires progressive NV12 with even dimensions");
        Frame properties(Require(av_frame_alloc(), "allocate replay properties"));
        properties->width = decoded->width;
        properties->height = decoded->height;
        properties->sample_aspect_ratio = decoded->sample_aspect_ratio;
        properties->colorspace = decoded->colorspace;
        properties->color_range = decoded->color_range;
        properties->color_primaries = decoded->color_primaries;
        properties->color_trc = decoded->color_trc;
        return properties;
    }
    throw std::runtime_error("replay did not produce a D3D11VA frame");
}

class CodecBenchmarkCase {
    struct Arrival {
        Moment due;
        size_t generation;
    };
    struct InputStamp {
        Arrival arrival;
        bool decoded{};
    };
    struct Source {
        Codec decoder;
        size_t decoded{};
        std::vector<InputStamp> submitted;
    };
    struct EncoderInput {
        int64_t pts;
        Arrival arrival;
    };
    struct Encoder {
        Codec codec;
        size_t requested{};
        size_t active{};
        int64_t lastPts{-CompressedReplay::frameDuration};
        std::deque<EncoderInput> pending;
        bool hasOutput{};
    };
    struct FenceInput {
        uint64_t value;
        Arrival arrival;
    };

public:
    CodecBenchmarkCase(ID3D11Device* device, AVBufferRef* hardware, const AVCodecParameters* parameters,
                       const AVFrame& source, const AVFrame& properties, const CompressedReplay& replay,
                       std::span<const OverlayScene> scenes, milliseconds window, bool staggered, int bitrate,
                       const std::string& previewUrl, bool matrix, CodecBenchmarkStage stage)
        : device_(device), hardware_(hardware), parameters_(parameters), properties_(properties), source_(source),
          replay_(replay), scenes_(scenes), window_(window), staggered_(staggered), bitrate_(bitrate),
          previewUrl_(previewUrl), matrix_(matrix), stage_(stage), warmFrames_(matrix ? 4 : 10),
          timeline_(device),
          input_(Require(av_packet_alloc(), "allocate input packet")),
          output_(Require(av_packet_alloc(), "allocate output packet")),
          decoded_(Require(av_frame_alloc(), "allocate decoded frame")),
          rendered_(Require(av_frame_alloc(), "allocate rendered frame")),
          frameLimit_(warmFrames_ + static_cast<size_t>(window / period)) {
        if (stage_ != CodecBenchmarkStage::Decode) {
            overlay_ = std::make_unique<GpuOverlay>(device, source.width, source.height, properties.width,
                properties.height, IsBt709(source), source.color_range == AVCOL_RANGE_JPEG);
            pool_ = CreateEncoderPool(hardware, properties.width, properties.height);
        }
        sources_.reserve(scenes.size());
        encoders_.reserve(maxEncoders);
        processedLatency_.reserve(frameLimit_ * scenes.size());
        encodedLatency_.reserve(frameLimit_ * maxEncoders);
    }

    void Run(size_t count, size_t outputs, int (*shouldStop)(void*)) {
        Activate(count, outputs, shouldStop);
        ++generation_;
        submittedInWindow_ = processedInWindow_ = encodedInWindow_ = bytesInWindow_ = 0;
        processedCohort_ = encodedCohort_ = 0;
        processedLatency_.clear();
        encodedLatency_.clear();
        epoch_ = Clock::now();
        begin_ = epoch_ + period * static_cast<int64_t>(warmFrames_);
        end_ = begin_ + window_;
        const auto offered = frameLimit_ * sources_.size();
        size_t next = 0;
        while (!shouldStop(nullptr)) {
            PollProcessed();
            const auto now = Clock::now();
            if (now >= end_) break;
            if (preview_) ReadCommands();
            if (next == offered) {
                timeline_.WaitUntil(end_, FirstPending());
                continue;
            }
            const size_t channel = next % sources_.size();
            const size_t index = next / sources_.size();
            const auto due = Due(channel, index);
            if (now < due) {
                timeline_.WaitUntil(std::min(end_, due), FirstPending());
                continue;
            }
            Feed(channel, {due, generation_});
            if (now >= begin_) ++submittedInWindow_;
            ++next;
        }
        PollProcessed();
        if (shouldStop(nullptr) && !preview_) throw std::runtime_error("full benchmark interrupted");
        PrintResults();
    }

    void Finish() {
        preview_.reset();
        for (size_t channel = 0; channel < sources_.size(); ++channel) {
            SendInput(channel, nullptr, {});
            const auto& source = sources_[channel];
            if (source.decoded != source.submitted.size())
                throw std::runtime_error("input/decoded frame count mismatch for stream " +
                                         std::to_string(channel + 1));
        }
        CloseEncoders(0);
        timeline_.Drain();
    }

private:
    void CloseEncoders(size_t retained) {
        while (encoders_.size() > retained) {
            SendEncoder(encoders_.size() - 1, nullptr);
            if (!encoders_.back().pending.empty())
                throw std::runtime_error("encoder did not return all submitted frames");
            encoders_.pop_back();
        }
    }

    void Activate(size_t count, size_t outputs, int (*shouldStop)(void*)) {
        CloseEncoders(outputs);
        const size_t previous = sources_.size();
        for (size_t channel = previous; channel < count; ++channel) {
            auto& source = sources_.emplace_back();
            source.decoder = CreateDecoder(parameters_, videoTimeBase, hardware_);
        }
        const size_t previousOutputs = encoders_.size();
        while (encoders_.size() < outputs) {
            const size_t channel = encoders_.size() % count;
            auto& encoder = encoders_.emplace_back();
            encoder.codec = CreateEncoder(properties_, frameRate, pool_.get(), bitrate_, false, true);
            encoder.requested = encoder.active = channel;
        }
        if (!preview_ && !previewUrl_.empty()) {
            preview_ = std::make_unique<CodecPreview>(previewUrl_, encoders_.front().codec.get(), encoders_.size());
            std::cout << "Preview commands: view <encoder>, route <encoder 6-10> <input stream>\n"
                      << std::flush;
        }
        // Decoder surfaces are created on the first real packets. Move that first use outside the measurements.
        for (size_t channel = previous; channel < count; ++channel) {
            while (sources_[channel].decoded == 0 && sources_[channel].submitted.size() < replay_.Size()) {
                if (shouldStop(nullptr)) throw std::runtime_error("full benchmark interrupted while warming up");
                Feed(channel, {});
            }
            if (sources_[channel].decoded == 0) throw std::runtime_error("decoder warmup produced no frame");
        }
        if (matrix_) {
            for (size_t slot = previousOutputs; slot < outputs; ++slot) {
                auto& encoder = encoders_[slot];
                for (size_t packets = 0; !encoder.hasOutput && packets < replay_.Size(); ++packets) {
                    if (shouldStop(nullptr)) throw std::runtime_error("matrix interrupted while warming up NVENC");
                    Feed(encoder.active, {});
                }
                if (!encoder.hasOutput) throw std::runtime_error("encoder warmup produced no packet");
            }
        }
        timeline_.Drain();
        PollProcessed();
    }

    void Feed(size_t channel, Arrival arrival) {
        replay_.Read(sources_[channel].submitted.size(), input_.get());
        SendInput(channel, input_.get(), arrival);
        av_packet_unref(input_.get());
    }

    Moment Due(size_t channel, size_t index) const {
        const auto phase = staggered_ ? duration_cast<nanoseconds>(period).count() *
            static_cast<int64_t>(channel) / static_cast<int64_t>(sources_.size()) : 0;
        return epoch_ + period * static_cast<int64_t>(index) + nanoseconds(phase);
    }

    uint64_t FirstPending() const { return fences_.empty() ? 0 : fences_.front().value; }

    void PollProcessed() {
        if (fences_.empty()) return;
        const auto done = timeline_.Completed();
        const auto now = Clock::now();
        while (!fences_.empty() && fences_.front().value <= done) {
            const auto arrival = fences_.front().arrival;
            if ((matrix_ || arrival.generation == generation_) && now >= begin_ && now < end_) {
                ++processedInWindow_;
                if (arrival.generation == generation_) {
                    processedLatency_.push_back(Milliseconds(now - arrival.due));
                    if (arrival.due >= begin_) ++processedCohort_;
                }
            }
            fences_.pop_front();
        }
    }

    void SendInput(size_t channel, AVPacket* packet, Arrival arrival) {
        auto& source = sources_[channel];
        int status = avcodec_send_packet(source.decoder.get(), packet);
        if (status == AVERROR(EAGAIN)) {
            ReceiveFrames(channel);
            status = avcodec_send_packet(source.decoder.get(), packet);
        }
        CheckAv(status, "submit in-memory compressed packet to D3D11VA");
        if (packet) source.submitted.push_back({arrival});
        ReceiveFrames(channel);
    }

    void ReceiveFrames(size_t channel) {
        auto& source = sources_[channel];
        for (;;) {
            const int status = avcodec_receive_frame(source.decoder.get(), decoded_.get());
            if (status == AVERROR(EAGAIN) || status == AVERROR_EOF) return;
            CheckAv(status, "receive D3D11VA frame");
            const auto started = Clock::now();
            const auto pts = decoded_->pts;
            const auto index = replay_.InputIndex(pts);
            if (index >= source.submitted.size() || source.submitted[index].decoded)
                throw std::runtime_error("unexpected or duplicate decoded frame PTS");
            auto& input = source.submitted[index];
            input.decoded = true;
            ++source.decoded;
            const auto arrival = input.arrival;
            auto* texture = CheckGpuFrame(*decoded_, device_);
            if (decoded_->width != source_.width || decoded_->height != source_.height ||
                (decoded_->flags & AV_FRAME_FLAG_INTERLACED))
                throw std::runtime_error("replay dimensions or frame format changed during benchmark");
            if (!overlay_) {
                fences_.push_back({timeline_.Submit(), arrival});
                av_frame_unref(decoded_.get());
                PollProcessed();
                continue;
            }
            CheckAv(av_hwframe_get_buffer(pool_.get(), rendered_.get(), 0), "acquire NV12 result surface");
            auto* destination = CheckGpuFrame(*rendered_, device_);
            overlay_->Render(texture, static_cast<UINT>(reinterpret_cast<uintptr_t>(decoded_->data[1])),
                             destination, pts, /*flush=*/false, &scenes_[channel],
                             stage_ == CodecBenchmarkStage::Draw);
            fences_.push_back({timeline_.Submit(), arrival});
            overlay_->CompleteHostFrame(Milliseconds(Clock::now() - started));
            PollProcessed();
            for (size_t slot = 0; slot < encoders_.size(); ++slot) {
                auto& encoder = encoders_[slot];
                if (encoder.requested != channel) continue;
                const bool switching = encoder.active != channel;
                rendered_->pts = std::max(pts, encoder.lastPts + CompressedReplay::frameDuration);
                rendered_->duration = CompressedReplay::frameDuration;
                rendered_->pict_type = switching ? AV_PICTURE_TYPE_I : AV_PICTURE_TYPE_NONE;
                encoder.pending.push_back({rendered_->pts, arrival});
                SendEncoder(slot, rendered_.get());
                if (switching && preview_)
                    std::cout << "Encoder " << slot + 1 << " routed to stream " << channel + 1
                              << " with forced IDR" << std::endl;
                encoder.active = channel;
                encoder.lastPts = rendered_->pts;
            }
            av_frame_unref(rendered_.get());
            av_frame_unref(decoded_.get());
        }
    }

    void SendEncoder(size_t slot, AVFrame* frame) {
        auto* codec = encoders_[slot].codec.get();
        int status = avcodec_send_frame(codec, frame);
        if (status == AVERROR(EAGAIN)) {
            ReceivePackets(slot);
            status = avcodec_send_frame(codec, frame);
        }
        CheckAv(status, "submit NV12 frame to NVENC");
        ReceivePackets(slot);
    }

    void ReceivePackets(size_t slot) {
        auto& encoder = encoders_[slot];
        for (;;) {
            const int status = avcodec_receive_packet(encoder.codec.get(), output_.get());
            if (status == AVERROR(EAGAIN) || status == AVERROR_EOF) return;
            CheckAv(status, "receive H.264 packet in RAM");
            const auto now = Clock::now();
            if (encoder.pending.empty() || encoder.pending.front().pts != output_->pts || output_->size <= 0)
                throw std::runtime_error("NVENC output did not match its submitted frame");
            const auto arrival = encoder.pending.front().arrival;
            encoder.pending.pop_front();
            encoder.hasOutput = true;
            if ((matrix_ || arrival.generation == generation_) && now >= begin_ && now < end_) {
                ++encodedInWindow_;
                bytesInWindow_ += output_->size;
                if (arrival.generation == generation_) {
                    encodedLatency_.push_back(Milliseconds(now - arrival.due));
                    if (arrival.due >= begin_) ++encodedCohort_;
                }
            }
            if (preview_) preview_->Accept(slot, *output_);
            av_packet_unref(output_.get());
        }
    }

    void ReadCommands() {
        const HANDLE console = GetStdHandle(STD_INPUT_HANDLE);
        DWORD mode{};
        if (GetConsoleMode(console, &mode)) {
            while (_kbhit()) {
                const wchar_t key = _getwch();
                if (key == L'\r') std::cout << '\n';
                else if (key >= 32 && key <= 126) std::cout << static_cast<char>(key) << std::flush;
                if (key == L'\b' && !command_.empty()) command_.pop_back();
                else if (key <= 127) ReadCommandByte(static_cast<char>(key));
            }
        } else {
            DWORD ready{};
            if (!PeekNamedPipe(console, nullptr, 0, nullptr, &ready, nullptr) || ready == 0) return;
            std::array<char, 256> data{};
            DWORD count{};
            if (ReadFile(console, data.data(), std::min<DWORD>(ready, static_cast<DWORD>(data.size())),
                         &count, nullptr))
                for (DWORD index = 0; index < count; ++index) ReadCommandByte(data[index]);
        }
    }

    void ReadCommandByte(char value) {
        if (value != '\n' && value != '\r') {
            if (value >= 32 && command_.size() < 80) command_ += value;
            return;
        }
        if (command_.empty()) return;
        std::istringstream line(command_);
        std::string verb;
        size_t slot{}, input{};
        if (line >> verb >> slot && verb == "view" && slot > 0 && slot <= encoders_.size())
            preview_->Select(slot - 1);
        else if (verb == "route" && line >> input && slot > fixedEncoders && slot <= encoders_.size() &&
                 input > 0 && input <= sources_.size())
            encoders_[slot - 1].requested = input - 1;
        else
            std::cout << "Use: view <encoder> or route <encoder 6-10> <input stream>" << std::endl;
        command_.clear();
    }

    static double Milliseconds(Clock::duration duration) {
        return std::chrono::duration<double, std::milli>(duration).count();
    }

    static void WriteLatency(std::vector<double>& samples) {
        if (samples.empty()) {
            std::cout << ",,,";
            return;
        }
        std::sort(samples.begin(), samples.end());
        for (const double fraction : {0.50, 0.95, 0.99}) {
            const auto index = static_cast<size_t>(std::ceil(fraction * static_cast<double>(samples.size()))) - 1;
            std::cout << ',' << samples[index];
        }
    }

    void PrintResults() {
        const auto offeredFps = sources_.size() * CompressedReplay::framesPerSecond;
        const auto offered = sources_.size() * (frameLimit_ - warmFrames_);
        const auto seconds = duration<double>(window_).count();
        std::cout << sources_.size() << ',' << encoders_.size() << ',' << offeredFps << ','
                  << offered << ',' << submittedInWindow_ << ',' << processedInWindow_ << ',' << std::fixed
                  << std::setprecision(2) << static_cast<double>(processedInWindow_) / seconds << ','
                  << (offered - processedCohort_);
        WriteLatency(processedLatency_);
        std::cout << ',' << encodedInWindow_ << ',' << static_cast<double>(encodedInWindow_) / seconds;
        WriteLatency(encodedLatency_);
        std::cout << ',' << bytesInWindow_ << ',' << encodedCohort_ << ',' << processedCohort_ << std::endl;
    }

    ID3D11Device* device_;
    AVBufferRef* hardware_;
    const AVCodecParameters* parameters_;
    const AVFrame& properties_;
    const AVFrame& source_;
    const CompressedReplay& replay_;
    std::span<const OverlayScene> scenes_;
    milliseconds window_;
    bool staggered_;
    int bitrate_;
    const std::string& previewUrl_;
    bool matrix_;
    CodecBenchmarkStage stage_;
    size_t warmFrames_;
    GpuTimeline timeline_;
    std::unique_ptr<GpuOverlay> overlay_;
    Buffer pool_;
    Packet input_;
    Packet output_;
    Frame decoded_;
    Frame rendered_;
    std::vector<Source> sources_;
    std::vector<Encoder> encoders_;
    std::unique_ptr<CodecPreview> preview_;
    std::deque<FenceInput> fences_;
    size_t frameLimit_;
    size_t generation_{};
    Moment epoch_{}, begin_{}, end_{};
    size_t submittedInWindow_{}, processedInWindow_{}, encodedInWindow_{}, bytesInWindow_{};
    size_t processedCohort_{}, encodedCohort_{};
    std::vector<double> processedLatency_, encodedLatency_;
    std::string command_;
};
}

void RunCodecBenchmark(ID3D11Device* device, AVBufferRef* hardware, AVFormatContext* input, AVStream* stream,
                       std::span<const int> streamCounts, milliseconds window, bool staggered,
                       int bitrate, const std::string& previewUrl, int matrixOutputs, CodecBenchmarkOutput output,
                       int (*shouldStop)(void*)) {
    const CompressedReplay replay(input, stream);
    const auto source = ReadSourceProperties(hardware, stream->codecpar, replay, device);
    const Frame properties(Require(av_frame_alloc(), "allocate encoder properties"));
    CheckAv(av_frame_copy_props(properties.get(), source.get()), "copy encoder color properties");
    properties->width = output.width ? output.width : source->width;
    properties->height = output.height ? output.height : source->height;
    const auto aspect = static_cast<int64_t>(source->width) * properties->height;
    const auto outputAspect = static_cast<int64_t>(source->height) * properties->width;
    av_reduce(&properties->sample_aspect_ratio.num, &properties->sample_aspect_ratio.den,
              aspect * source->sample_aspect_ratio.num, outputAspect * source->sample_aspect_ratio.den, INT_MAX);
    if (std::adjacent_find(streamCounts.begin(), streamCounts.end(), std::greater_equal<int>()) != streamCounts.end())
        throw std::runtime_error("full benchmark input counts must be in increasing order");
    const auto scenes = MakeScenes(static_cast<unsigned>(*std::max_element(streamCounts.begin(), streamCounts.end())));
    std::cout << "Replay: " << replay.Size() << ' ' << avcodec_get_name(stream->codecpar->codec_id)
              << " frames, " << replay.Bytes() << " compressed bytes in RAM; " << source->width << 'x'
              << source->height << " at 25 FPS per stream.\n";
    if (output.stage == CodecBenchmarkStage::Decode) std::cout << "D3D11VA decode only. ";
    else std::cout << "D3D11VA decode, GPU resize/conversions at " << properties->width << 'x' << properties->height
                   << (output.stage == CodecBenchmarkStage::Draw ? ", Direct2D overlay, H.264 NVENC as selected. "
                                                                 : ". ");
    std::cout
              << (staggered ? "Staggered" : "Synchronized") << " arrivals. "
              << (previewUrl.empty() ? "Preview disabled." : "Preview enabled (separate demonstration).")
              << " Window: " << window.count() << " ms.\n";
    if (previewUrl.empty()) {
        RunParallelCodecBenchmark(device, stream->codecpar, *source, *properties, replay, scenes,
                                  streamCounts, window, staggered, bitrate, matrixOutputs,
                                  output.stage, output.decodeSurfaces, output.firstMatrixOutput, shouldStop);
        return;
    }
    std::cout
              << "inputs,encoders,offered_fps,offered_frames,submitted_frames,processed_frames,processed_fps,"
                 "input_cohort_not_ready,process_p50_ms,process_p95_ms,process_p99_ms,encoded_packets,encoded_fps,"
                 "encode_p50_ms,encode_p95_ms,encode_p99_ms,encoded_bytes,encoded_cohort,processed_cohort"
              << std::endl;
    const bool matrix = matrixOutputs >= 0;
    CodecBenchmarkCase measurement(device, hardware, stream->codecpar, *source, *properties, replay, scenes,
                                     window, staggered, bitrate, previewUrl, matrix, output.stage);
    for (const int count : streamCounts) {
        const auto channels = static_cast<size_t>(count);
        const auto outputLimit = matrix ? static_cast<size_t>(matrixOutputs) : std::min(channels, maxEncoders);
        for (size_t outputs = matrix ? 0 : outputLimit; outputs <= outputLimit; ++outputs) {
            try {
                measurement.Run(channels, outputs, shouldStop);
            } catch (const std::exception& error) {
                throw std::runtime_error("full benchmark at " + std::to_string(count) + " inputs and " +
                                         std::to_string(outputs) + " outputs: " + error.what());
            }
        }
    }
    measurement.Finish();
}
