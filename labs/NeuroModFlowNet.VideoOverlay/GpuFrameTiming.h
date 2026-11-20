#pragma once

#include <d3d11.h>
#include <wrl/client.h>
#include <array>
#include <cstdint>
#include <optional>

struct FrameStageTimes {
    double toBgra{};
    double draw{};
    double toNv12{};
    double total{};
};

class GpuFrameTiming {
public:
    struct Sample {
        uint64_t frame;
        FrameStageTimes milliseconds;
    };

    enum class Point { ConvertedToBgra = 1, Drawn = 2 };

    explicit GpuFrameTiming(ID3D11Device* device);
    void Poll();
    void Begin(uint64_t frame);
    void Mark(Point point);
    void End(bool flush);
    const std::optional<Sample>& Latest() const { return latest_; }

private:
    struct Slot {
        uint64_t frame{};
        Microsoft::WRL::ComPtr<ID3D11Query> disjoint;
        std::array<Microsoft::WRL::ComPtr<ID3D11Query>, 4> timestamps;
    };

    Microsoft::WRL::ComPtr<ID3D11DeviceContext> context_;
    std::array<Slot, 8> slots_;
    size_t first_{};
    size_t pending_{};
    Slot* recording_{};
    std::optional<Sample> latest_;
};
