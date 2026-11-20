#include "OverlayBenchmark.h"
#include "GpuTimeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
using Microsoft::WRL::ComPtr;
using namespace std::chrono;
using Clock = steady_clock;
constexpr int framesPerSecond = 25;
constexpr UINT width = 1920;
constexpr UINT height = 1080;
constexpr auto framePeriod = 40ms;

double Milliseconds(Clock::duration value) {
    return duration<double, std::milli>(value).count();
}

struct StreamResources {
    ComPtr<ID3D11Texture2D> input;
    ComPtr<ID3D11Texture2D> output;
    uint64_t lastWrite{};
};

struct PendingFrame {
    uint64_t completion;
    Clock::time_point readyAt;
    Clock::time_point dispatchedAt;
    double submitMs;
};

struct Samples {
    std::vector<double> latency;
    std::vector<double> queue;
    std::vector<double> submit;
};

std::vector<StreamResources> PrepareStreams(ID3D11Device* device, GpuTimeline& timeline,
                                           ID3D11Texture2D* decoded, UINT slice, int count) {
    D3D11_TEXTURE2D_DESC inputDesc{};
    decoded->GetDesc(&inputDesc);
    inputDesc.ArraySize = 1;
    inputDesc.BindFlags = 0;
    inputDesc.MiscFlags = 0;
    auto outputDesc = inputDesc;
    outputDesc.Width = width;
    outputDesc.Height = height;
    outputDesc.BindFlags = D3D11_BIND_RENDER_TARGET;
    std::vector<StreamResources> streams;
    streams.reserve(count);
    for (int index = 0; index < count; ++index) {
        auto& stream = streams.emplace_back();
        CheckHr(device->CreateTexture2D(&inputDesc, nullptr, &stream.input), "allocate stream input on GPU");
        CheckHr(device->CreateTexture2D(&outputDesc, nullptr, &stream.output), "allocate stream output on GPU");
        timeline.Context()->CopySubresourceRegion(stream.input.Get(), 0, 0, 0, 0, decoded,
            D3D11CalcSubresource(0, slice, inputDesc.MipLevels), nullptr);
    }
    timeline.Submit();
    timeline.Drain();
    return streams;
}

Samples MeasureWindow(std::vector<StreamResources>& streams, GpuOverlay& overlay, GpuTimeline& timeline,
                      seconds duration, int64_t initialPts, bool staggered, int (*shouldStop)(void*)) {
    const size_t offered = streams.size() * framesPerSecond * duration.count();
    Samples samples;
    samples.latency.reserve(offered);
    samples.queue.reserve(offered);
    samples.submit.reserve(offered);
    std::deque<PendingFrame> pending;
    const auto epoch = Clock::now();
    const auto deadline = epoch + duration;
    const auto framePeriodNs = duration_cast<nanoseconds>(framePeriod).count();
    size_t submitted = 0;

    while (!shouldStop(nullptr)) {
        const uint64_t done = timeline.Completed();
        const auto now = Clock::now();
        while (!pending.empty() && pending.front().completion <= done) {
            const auto& frame = pending.front();
            if (now <= deadline) {
                samples.latency.push_back(Milliseconds(now - frame.readyAt));
                samples.queue.push_back(Milliseconds(frame.dispatchedAt - frame.readyAt));
                samples.submit.push_back(frame.submitMs);
            }
            pending.pop_front();
        }
        if (now >= deadline) break;
        if (submitted == offered) {
            timeline.WaitUntil(deadline, pending.empty() ? 0 : pending.front().completion);
            continue;
        }

        const size_t source = submitted % streams.size();
        const auto frameIndex = static_cast<int64_t>(submitted / streams.size());
        const auto phase = staggered ?
            framePeriodNs * static_cast<int64_t>(source) / static_cast<int64_t>(streams.size()) : 0;
        const auto due = epoch + framePeriod * frameIndex + nanoseconds(phase);
        auto& stream = streams[source];
        const bool surfaceFree = stream.lastWrite <= done;
        if (now < due || !surfaceFree) {
            timeline.WaitUntil(surfaceFree ? std::min(deadline, due) : deadline,
                               pending.empty() ? 0 : pending.front().completion);
            continue;
        }

        // Network arrival phase changes scheduling only. All sources render the same content for a given PTS.
        const int64_t pts = initialPts + frameIndex * (90000 / framesPerSecond);
        overlay.Render(stream.input.Get(), 0, stream.output.Get(), pts, /*flush=*/false);
        stream.lastWrite = timeline.Submit();
        const double submitMs = Milliseconds(Clock::now() - now);
        overlay.CompleteHostFrame(submitMs);
        pending.push_back({stream.lastWrite, due, now, submitMs});
        ++submitted;
    }
    timeline.Drain();
    if (shouldStop(nullptr)) throw std::runtime_error("benchmark interrupted");
    return samples;
}

double Percentile(const std::vector<double>& sorted, double fraction) {
    if (sorted.empty()) throw std::runtime_error("no GPU frames completed inside the measurement window");
    const auto index = static_cast<size_t>(std::ceil(fraction * static_cast<double>(sorted.size()))) - 1;
    return sorted[index];
}
}

void RunOverlayBenchmark(ID3D11Device* device, ID3D11Texture2D* decoded, UINT slice, bool bt709,
                         bool fullRange, std::span<const int> streamCounts, int secondsPerCase,
                         bool staggered, int (*shouldStop)(void*)) {
    GpuTimeline timeline(device);
    std::cout << "Benchmark: 1920x1080 at 25 FPS/stream, "
              << (staggered ? "evenly staggered arrivals" : "synchronized arrivals")
              << ", shared overlay and one submission worker, " << secondsPerCase
              << " measured seconds per case\n"
              << "streams,offered_fps,offered_frames,completed_frames,unfinished_frames,completed_fps,"
                 "latency_p50_ms,latency_p95_ms,latency_p99_ms,queue_p95_ms,submit_p50_ms"
              << std::endl;
    for (const int count : streamCounts) {
        auto streams = PrepareStreams(device, timeline, decoded, slice, count);
        GpuOverlay overlay(device, width, height, bt709, fullRange);
        MeasureWindow(streams, overlay, timeline, 1s, 0, staggered, shouldStop);
        auto samples = MeasureWindow(streams, overlay, timeline, seconds(secondsPerCase), 90000,
                                     staggered, shouldStop);
        std::sort(samples.latency.begin(), samples.latency.end());
        std::sort(samples.queue.begin(), samples.queue.end());
        std::sort(samples.submit.begin(), samples.submit.end());
        const auto offered = static_cast<size_t>(count) * framesPerSecond * secondsPerCase;
        const size_t complete = samples.latency.size();
        const double completedFps = static_cast<double>(complete) / secondsPerCase;
        std::cout << count << ',' << count * framesPerSecond << ',' << offered << ',' << complete << ','
                  << offered - complete << ',' << std::fixed << std::setprecision(2) << completedFps << ','
                  << Percentile(samples.latency, 0.50) << ',' << Percentile(samples.latency, 0.95) << ','
                  << Percentile(samples.latency, 0.99) << ',' << Percentile(samples.queue, 0.95) << ','
                  << Percentile(samples.submit, 0.50) << std::endl;
    }
}
