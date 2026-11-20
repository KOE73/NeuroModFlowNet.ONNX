#pragma once

#include "GpuOverlay.h"

#include <d3d11_4.h>
#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>
#include <stdexcept>

class GpuTimeline {
    using Clock = std::chrono::steady_clock;
    using TimerHandle = std::unique_ptr<void, decltype(&CloseHandle)>;

public:
    explicit GpuTimeline(ID3D11Device* device)
        : alarm_(CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                                      TIMER_MODIFY_STATE | SYNCHRONIZE), CloseHandle),
          ready_(CreateEventW(nullptr, FALSE, FALSE, nullptr), CloseHandle) {
        if (!alarm_ || !ready_) throw std::runtime_error("create benchmark wait handles");
        Microsoft::WRL::ComPtr<ID3D11Device5> device5;
        CheckHr(device->QueryInterface(IID_PPV_ARGS(&device5)), "get benchmark D3D11 fence device");
        CheckHr(device5->CreateFence(0, D3D11_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)),
                "create benchmark GPU fence");
        Microsoft::WRL::ComPtr<ID3D11DeviceContext> immediate;
        device->GetImmediateContext(&immediate);
        CheckHr(immediate.As(&context_), "get benchmark D3D11 immediate context");
    }

    ID3D11DeviceContext4* Context() const { return context_.Get(); }

    uint64_t Submit() {
        CheckHr(context_->Signal(fence_.Get(), ++issued_), "signal completion of video transforms");
        context_->Flush();
        return issued_;
    }

    uint64_t Completed() const {
        const auto value = fence_->GetCompletedValue();
        if (value == std::numeric_limits<uint64_t>::max())
            throw std::runtime_error("benchmark GPU device removed");
        return value;
    }

    void WaitUntil(Clock::time_point deadline, uint64_t firstPending) {
        const auto remaining = deadline - Clock::now();
        if (remaining <= Clock::duration::zero()) return;
        LARGE_INTEGER expiration{};
        expiration.QuadPart = -std::max<int64_t>(
            1, std::chrono::duration_cast<std::chrono::nanoseconds>(remaining).count() / 100);
        CheckWin32(SetWaitableTimer(alarm_.get(), &expiration, 0, nullptr, nullptr, FALSE),
                   "arm benchmark pacing timer");
        HANDLE handles[]{alarm_.get(), ready_.get()};
        if (firstPending != 0) Arm(firstPending);
        CheckWin32(WaitForMultipleObjects(firstPending != 0 ? 2 : 1, handles, FALSE, INFINITE) != WAIT_FAILED,
                   "wait for next frame or GPU completion");
    }

    void Drain() {
        if (Completed() < issued_) {
            Arm(issued_);
            const auto deadline = Clock::now() + std::chrono::seconds(10);
            while (Completed() < issued_ && Clock::now() < deadline)
                CheckWin32(WaitForSingleObject(ready_.get(), 100) != WAIT_FAILED, "wait for benchmark fence");
            if (Completed() < issued_) throw std::runtime_error("benchmark GPU fence timed out");
        }
    }

private:
    static void CheckWin32(BOOL success, const char* operation) {
        if (!success) CheckHr(HRESULT_FROM_WIN32(GetLastError()), operation);
    }

    void Arm(uint64_t value) {
        if (armed_ == value) return;
        CheckHr(fence_->SetEventOnCompletion(value, ready_.get()), "arm benchmark GPU completion event");
        armed_ = value;
    }

    TimerHandle alarm_;
    TimerHandle ready_;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext4> context_;
    Microsoft::WRL::ComPtr<ID3D11Fence> fence_;
    uint64_t issued_{};
    uint64_t armed_{};
};
