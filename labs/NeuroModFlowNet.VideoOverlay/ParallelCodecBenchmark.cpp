#include "ParallelCodecBenchmark.h"
#include "GpuTimeline.h"

#include <objbase.h>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <exception>
#include <iomanip>
#include <iostream>
#include <latch>
#include <mutex>
#include <thread>
#include <utility>

using namespace VideoCodecs;
using Microsoft::WRL::ComPtr;

namespace {
using namespace std::chrono;
using Clock = steady_clock;
using Moment = Clock::time_point;
constexpr AVRational rate{CompressedReplay::framesPerSecond, 1};
constexpr auto period = 1000ms / CompressedReplay::framesPerSecond;
constexpr size_t maxEncoders = 10;
constexpr size_t warmFrames = 4;

double Elapsed(Moment started, Moment finished) {
    return duration<double, std::milli>(finished - started).count();
}

struct Measurement {
    size_t submitted{};
    uint64_t bytes{};
    std::vector<double> processed;
    std::vector<double> encoded;
};

struct Window {
    explicit Window(size_t streams, size_t outputs, milliseconds duration, bool staggered,
                    int (*shouldStop)(void*))
        : streams(streams), outputs(outputs), duration(duration), staggered(staggered), shouldStop(shouldStop),
          configured(streams), ready(streams), finished(streams) {}
    size_t streams;
    size_t outputs;
    milliseconds duration;
    bool staggered;
    int (*shouldStop)(void*);
    std::latch configured;
    std::latch ready;
    std::latch start{1};
    std::latch finished;
    Moment begin;
};

struct StreamSettings {
    IDXGIAdapter* adapter;
    const AVCodecParameters* parameters;
    const AVFrame& source;
    const AVFrame& output;
    const CompressedReplay& replay;
    int bitrate;
    CodecBenchmarkStage stage;
    int decodeSurfaces;
};

class StreamPipeline {
    struct Encoder {
        Codec codec;
        int64_t nextPts{};
        std::deque<Moment> pending;
    };
    struct Completion {
        uint64_t fence;
        Moment due;
    };

public:
    StreamPipeline(const StreamSettings& settings, const OverlayScene& scene)
        : settings_(settings), scene_(scene), device_(CreateVideoDevice(settings.adapter)),
          hardware_(BindDevice(device_.Get())), decodeSurfaces_(settings.decodeSurfaces),
          decoder_(CreateDecoder(settings.parameters, videoTimeBase, hardware_.get(),
                                 decodeSurfaces_ ? &decodeSurfaces_ : nullptr)),
          timeline_(device_.Get()), input_(Require(av_packet_alloc(), "allocate stream packet")),
          output_(Require(av_packet_alloc(), "allocate encoded packet")),
          decoded_(Require(av_frame_alloc(), "allocate decoded frame")),
          rendered_(Require(av_frame_alloc(), "allocate rendered frame")) {
        if (settings.stage != CodecBenchmarkStage::Decode) {
            const auto& source = settings.source;
            const auto& output = settings.output;
            overlay_ = std::make_unique<GpuOverlay>(device_.Get(), source.width, source.height, output.width,
                output.height, IsBt709(source), source.color_range == AVCOL_RANGE_JPEG);
            pool_ = CreateEncoderPool(hardware_.get(), output.width, output.height);
        }
    }

    void Configure(size_t outputs) {
        while (encoders_.size() > outputs) {
            auto& encoder = encoders_.back();
            Send(encoder, nullptr);
            if (!encoder.pending.empty()) throw std::runtime_error("NVENC drain left pending packets");
            encoders_.pop_back();
        }
        while (encoders_.size() < outputs)
            encoders_.push_back({CreateEncoder(settings_.output, rate, pool_.get(), settings_.bitrate, false)});

        avcodec_flush_buffers(decoder_.get());
        submitted_.clear();
        decodedCount_ = 0;
        begin_ = end_ = {};
        result_ = {};
    }

    void Warmup() {
        for (size_t index = 0; index < warmFrames; ++index) Feed({});
        timeline_.Context()->Flush();
        timeline_.Drain();
        Poll();
        CheckNoPending();
    }

    Measurement Measure(const Window& window, size_t channel) {
        begin_ = window.begin;
        end_ = begin_ + window.duration;
        const auto phase = window.staggered
            ? duration_cast<nanoseconds>(period) * static_cast<int64_t>(channel) / static_cast<int64_t>(window.streams)
            : nanoseconds::zero();
        const auto frameCount = static_cast<size_t>(window.duration / period);
        result_.processed.reserve(frameCount);
        result_.encoded.reserve(frameCount * encoders_.size());
        for (size_t frame = 0; !window.shouldStop(nullptr);) {
            Poll();
            if (Clock::now() >= end_) break;
            const auto due = frame < frameCount ? begin_ + phase + period * static_cast<int64_t>(frame) : end_;
            if (Clock::now() < due) {
                timeline_.WaitUntil(due, completions_.empty() ? 0 : completions_.front().fence);
                continue;
            }
            if (frame == frameCount) break;
            Feed(due);
            ++result_.submitted;
            ++frame;
        }
        timeline_.Context()->Flush();
        timeline_.Drain();
        Poll();
        CheckNoPending();
        return std::move(result_);
    }

private:
    void CheckNoPending() const {
        if (decodedCount_ != submitted_.size())
            throw std::runtime_error("decoder held pictures after submission; require no reorder and zero decode delay");
        for (const auto& encoder : encoders_)
            if (!encoder.pending.empty()) throw std::runtime_error("NVENC held a packet across the window");
    }

    void Poll() {
        if (completions_.empty()) return;
        const auto completed = timeline_.Completed();
        const auto observed = Clock::now();
        while (!completions_.empty() && completions_.front().fence <= completed) {
            const auto due = completions_.front().due;
            if (due >= begin_ && observed < end_) result_.processed.push_back(Elapsed(due, observed));
            completions_.pop_front();
        }
    }

    void Feed(Moment due) {
        settings_.replay.Read(submitted_.size(), input_.get());
        CheckAv(avcodec_send_packet(decoder_.get(), input_.get()), "submit compressed frame to D3D11VA");
        av_packet_unref(input_.get());
        submitted_.push_back(due);
        for (;;) {
            const auto status = avcodec_receive_frame(decoder_.get(), decoded_.get());
            if (status == AVERROR(EAGAIN)) break;
            CheckAv(status, "receive D3D11VA frame");
            if (decoded_->pict_type == AV_PICTURE_TYPE_B)
                throw std::runtime_error("parallel benchmark requires an I/P source (found a B picture)");
            const auto index = settings_.replay.InputIndex(decoded_->pts);
            if (index != decodedCount_ || index >= submitted_.size())
                throw std::runtime_error("unexpected decode order in parallel benchmark");
            ++decodedCount_;
            Process(submitted_[index]);
            av_frame_unref(decoded_.get());
        }
    }

    void Process(Moment due) {
        const auto& source = settings_.source;
        if (decoded_->width != source.width || decoded_->height != source.height ||
            (decoded_->flags & AV_FRAME_FLAG_INTERLACED))
            throw std::runtime_error("source shape or frame format changed during replay");
        auto* texture = CheckGpuFrame(*decoded_, device_.Get());
        if (overlay_) {
            CheckAv(av_hwframe_get_buffer(pool_.get(), rendered_.get(), 0), "acquire per-stream NV12 output");
            auto* destination = CheckGpuFrame(*rendered_, device_.Get());
            overlay_->Render(texture, static_cast<UINT>(reinterpret_cast<uintptr_t>(decoded_->data[1])),
                destination, decoded_->pts, false, &scene_, settings_.stage == CodecBenchmarkStage::Draw);
        }
        completions_.push_back({timeline_.Submit(), due});
        Poll();
        for (auto& encoder : encoders_) {
            rendered_->pts = encoder.nextPts;
            rendered_->duration = CompressedReplay::frameDuration;
            encoder.nextPts += rendered_->duration;
            encoder.pending.push_back(due);
            Send(encoder, rendered_.get());
            Poll();
        }
        av_frame_unref(rendered_.get());
    }

    void Send(Encoder& encoder, AVFrame* frame) {
        CheckAv(avcodec_send_frame(encoder.codec.get(), frame), "submit per-stream frame to NVENC");
        for (;;) {
            const auto status = avcodec_receive_packet(encoder.codec.get(), output_.get());
            if (status == AVERROR(EAGAIN) || status == AVERROR_EOF) return;
            CheckAv(status, "receive per-stream H.264 packet");
            if (encoder.pending.empty() || output_->size <= 0)
                throw std::runtime_error("unexpected NVENC output packet");
            const auto due = encoder.pending.front();
            encoder.pending.pop_front();
            const auto observed = Clock::now();
            if (due >= begin_ && observed < end_) {
                result_.encoded.push_back(Elapsed(due, observed));
                result_.bytes += output_->size;
            }
            av_packet_unref(output_.get());
        }
    }

    const StreamSettings& settings_;
    const OverlayScene& scene_;
    ComPtr<ID3D11Device> device_;
    Buffer hardware_;
    int decodeSurfaces_;
    Codec decoder_;
    GpuTimeline timeline_;
    Packet input_, output_;
    Frame decoded_, rendered_;
    std::unique_ptr<GpuOverlay> overlay_;
    Buffer pool_;
    std::vector<Encoder> encoders_;
    std::deque<Completion> completions_;
    std::vector<Moment> submitted_;
    size_t decodedCount_{};
    Moment begin_{}, end_{};
    Measurement result_;
};

class StreamWorker {
public:
    StreamWorker(const StreamSettings& settings, const OverlayScene& scene, size_t channel)
        : thread_([this, &settings, &scene, channel] { Work(settings, scene, channel); }) {}

    ~StreamWorker() {
        {
            std::lock_guard lock(mutex_);
            stop_ = true;
        }
        wake_.notify_one();
    }

    void Start(Window& window) {
        {
            std::lock_guard lock(mutex_);
            job_ = &window;
        }
        wake_.notify_one();
    }

    Measurement result;
    std::exception_ptr error;
    const char* failedPhase{};

private:
    void Work(const StreamSettings& settings, const OverlayScene& scene, size_t channel) {
        const auto com = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        std::unique_ptr<StreamPipeline> pipeline;
        for (;;) {
            Window* job;
            {
                std::unique_lock lock(mutex_);
                wake_.wait(lock, [&] { return stop_ || job_; });
                if (stop_) break;
                job = std::exchange(job_, nullptr);
            }
            error = {};
            result = {};
            try {
                CheckHr(com, "initialize COM on stream worker");
                if (!pipeline) pipeline = std::make_unique<StreamPipeline>(settings, scene);
                pipeline->Configure(job->outputs / job->streams + (channel < job->outputs % job->streams));
            } catch (...) {
                error = std::current_exception();
                failedPhase = "configure";
            }
            job->configured.arrive_and_wait();
            if (!error) {
                try {
                    pipeline->Warmup();
                } catch (...) {
                    error = std::current_exception();
                    failedPhase = "warmup";
                }
            }
            job->ready.count_down();
            job->start.wait();
            if (!error) {
                try {
                    result = pipeline->Measure(*job, channel);
                } catch (...) {
                    error = std::current_exception();
                    failedPhase = "measure";
                }
            }
            job->finished.count_down();
        }
        pipeline.reset();
        if (SUCCEEDED(com)) CoUninitialize();
    }

    std::mutex mutex_;
    std::condition_variable wake_;
    bool stop_{};
    Window* job_{};
    std::jthread thread_;
};

void WriteLatency(std::vector<double>& samples) {
    if (samples.empty()) {
        std::cout << ",,,";
        return;
    }
    std::sort(samples.begin(), samples.end());
    for (const auto percentile : {0.50, 0.95, 0.99})
        std::cout << ',' << samples[static_cast<size_t>(std::ceil(percentile * samples.size())) - 1];
}

void Print(size_t count, size_t outputs, milliseconds window, std::span<const std::unique_ptr<StreamWorker>> workers) {
    Measurement result;
    for (size_t index = 0; index < workers.size(); ++index) {
        const auto& worker = workers[index];
        if (worker->error) {
            try { std::rethrow_exception(worker->error); }
            catch (const std::exception& error) {
                throw std::runtime_error("stream " + std::to_string(index + 1) + " " +
                                         worker->failedPhase + ": " + error.what());
            }
        }
        result.submitted += worker->result.submitted;
        result.bytes += worker->result.bytes;
        result.processed.insert(result.processed.end(), worker->result.processed.begin(), worker->result.processed.end());
        result.encoded.insert(result.encoded.end(), worker->result.encoded.begin(), worker->result.encoded.end());
    }
    const auto offered = count * static_cast<size_t>(window / period);
    const auto seconds = duration<double>(window).count();
    const auto processed = result.processed.size();
    const auto encoded = result.encoded.size();
    std::cout << count << ',' << outputs << ',' << count * CompressedReplay::framesPerSecond << ','
              << offered << ',' << result.submitted << ',' << processed << ',' << std::fixed << std::setprecision(2)
              << processed / seconds << ',' << offered - processed;
    WriteLatency(result.processed);
    std::cout << ',' << encoded << ',' << encoded / seconds;
    WriteLatency(result.encoded);
    std::cout << ',' << result.bytes << ',' << encoded << ',' << processed << std::endl;
}
}

void RunParallelCodecBenchmark(ID3D11Device* device, const AVCodecParameters* parameters,
                               const AVFrame& source, const AVFrame& output, const CompressedReplay& replay,
                               std::span<const OverlayScene> scenes, std::span<const int> streamCounts,
                               milliseconds window, bool staggered, int bitrate, int matrixOutputs,
                               CodecBenchmarkStage stage, int decodeSurfaces, int firstOutput, int (*shouldStop)(void*)) {
    if (parameters->video_delay || replay.HasReorderedPictures())
        throw std::runtime_error("parallel benchmark requires no B pictures and no reorder in the source");
    ComPtr<IDXGIDevice> dxgi;
    CheckHr(device->QueryInterface(IID_PPV_ARGS(&dxgi)), "get benchmark DXGI adapter");
    ComPtr<IDXGIAdapter> adapter;
    CheckHr(dxgi->GetAdapter(&adapter), "get selected NVIDIA adapter");
    const StreamSettings settings{adapter.Get(), parameters, source, output, replay, bitrate, stage, decodeSurfaces};
    std::vector<std::unique_ptr<StreamWorker>> workers;
    std::cout << "One worker and D3D11 device per input. Each timed window counts only its own frames.\n"
              << "D3D11 decode surfaces per input: "
              << (decodeSurfaces ? std::to_string(decodeSurfaces) : "FFmpeg default") << ".\n"
              << "inputs,encoders,offered_fps,offered_frames,submitted_frames,processed_frames,processed_fps,"
                 "input_cohort_not_ready,process_p50_ms,process_p95_ms,process_p99_ms,encoded_packets,encoded_fps,"
                 "encode_p50_ms,encode_p95_ms,encode_p99_ms,encoded_bytes,encoded_cohort,processed_cohort" << std::endl;
    for (const auto streams : streamCounts) {
        const auto count = static_cast<size_t>(streams);
        while (workers.size() < count) {
            const auto channel = workers.size();
            workers.push_back(std::make_unique<StreamWorker>(settings, scenes[channel], channel));
        }
        const auto limit = matrixOutputs < 0 ? std::min(count, maxEncoders) : static_cast<size_t>(matrixOutputs);
        for (auto outputs = matrixOutputs < 0 ? limit : static_cast<size_t>(firstOutput); outputs <= limit; ++outputs) {
            if (shouldStop(nullptr)) throw std::runtime_error("parallel benchmark interrupted");
            Window job(count, outputs, window, staggered, shouldStop);
            for (const auto& worker : workers) worker->Start(job);
            job.ready.wait();
            job.begin = Clock::now() + 30ms;
            job.start.count_down();
            job.finished.wait();
            if (shouldStop(nullptr)) throw std::runtime_error("parallel benchmark interrupted");
            try { Print(count, outputs, window, workers); }
            catch (const std::exception& error) {
                throw std::runtime_error("parallel benchmark at " + std::to_string(count) + " inputs and " +
                                         std::to_string(outputs) + " outputs: " + error.what());
            }
        }
    }
}
