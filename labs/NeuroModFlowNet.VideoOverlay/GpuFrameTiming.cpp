#include "GpuFrameTiming.h"
#include "GpuOverlay.h"

GpuFrameTiming::GpuFrameTiming(ID3D11Device* device) {
    device->GetImmediateContext(&context_);
    const D3D11_QUERY_DESC disjoint{D3D11_QUERY_TIMESTAMP_DISJOINT, 0};
    const D3D11_QUERY_DESC timestamp{D3D11_QUERY_TIMESTAMP, 0};
    for (auto& slot : slots_) {
        CheckHr(device->CreateQuery(&disjoint, &slot.disjoint), "allocate GPU timing disjoint query");
        for (auto& query : slot.timestamps)
            CheckHr(device->CreateQuery(&timestamp, &query), "allocate GPU timestamp query");
    }
}

void GpuFrameTiming::Poll() {
    // A queued frame may finish after the following frame's CPU submission.
    // Never flush or wait for a sample while producing video.
    constexpr auto flags = D3D11_ASYNC_GETDATA_DONOTFLUSH;
    while (pending_ > 0) {
        const auto& slot = slots_[first_];
        D3D11_QUERY_DATA_TIMESTAMP_DISJOINT timing{};
        auto status = context_->GetData(slot.disjoint.Get(), &timing, sizeof(timing), flags);
        if (status == S_FALSE) return;
        CheckHr(status, "read GPU timing frequency");
        if (!timing.Disjoint && timing.Frequency > 0) {
            std::array<UINT64, 4> ticks{};
            for (size_t index = 0; index < ticks.size(); ++index) {
                status = context_->GetData(slot.timestamps[index].Get(), &ticks[index], sizeof(ticks[index]), flags);
                if (status == S_FALSE) return;
                CheckHr(status, "read GPU timestamp");
            }
            const double millisecondsPerTick = 1000.0 / static_cast<double>(timing.Frequency);
            const auto elapsed = [&](size_t start, size_t end) {
                return static_cast<double>(ticks[end] - ticks[start]) * millisecondsPerTick;
            };
            latest_ = Sample{slot.frame, {elapsed(0, 1), elapsed(1, 2), elapsed(2, 3), elapsed(0, 3)}};
        }
        first_ = (first_ + 1) % slots_.size();
        --pending_;
    }
}

void GpuFrameTiming::Begin(uint64_t frame) {
    if (pending_ == slots_.size()) return;
    recording_ = &slots_[(first_ + pending_) % slots_.size()];
    recording_->frame = frame;
    context_->Begin(recording_->disjoint.Get());
    context_->End(recording_->timestamps.front().Get());
}

void GpuFrameTiming::Mark(Point point) {
    if (recording_) context_->End(recording_->timestamps[static_cast<size_t>(point)].Get());
}

void GpuFrameTiming::End(bool flush) {
    if (!recording_) return;
    context_->End(recording_->timestamps.back().Get());
    context_->End(recording_->disjoint.Get());
    // Submit the final marker with this frame so its timestamp cannot slip into the next frame.
    // Flush does not wait for the GPU. Poll still reads only completed queries.
    if (flush) context_->Flush();
    ++pending_;
    recording_ = nullptr;
}
