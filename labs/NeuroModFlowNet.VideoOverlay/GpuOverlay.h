#pragma once

#include "GpuFrameTiming.h"
#include "OverlayScene.h"

#include <d3d11_1.h>
#include <d2d1_1.h>
#include <dwrite.h>
#include <wrl/client.h>
#include <cstdint>

void CheckHr(HRESULT result, const char* operation);

class GpuOverlay {
public:
    GpuOverlay(ID3D11Device* device, UINT width, UINT height, bool bt709, bool fullRange);
    GpuOverlay(ID3D11Device* device, UINT inputWidth, UINT inputHeight, UINT width, UINT height,
               bool bt709, bool fullRange);
    void Render(ID3D11Texture2D* source, UINT slice, ID3D11Texture2D* destination,
                int64_t timestamp90k, bool flush = true, const OverlayScene* scene = nullptr,
                bool drawOverlay = true);
    void CompleteHostFrame(double decodedToEncoderMs);

private:
    Microsoft::WRL::ComPtr<ID3D11VideoProcessorEnumerator> CreateEnumerator(UINT inputWidth, UINT inputHeight);
    Microsoft::WRL::ComPtr<ID3D11VideoProcessor> CreateProcessor(
        ID3D11VideoProcessorEnumerator* enumerator, UINT inputWidth, UINT inputHeight,
        DXGI_COLOR_SPACE_TYPE inputColor, DXGI_COLOR_SPACE_TYPE outputColor);
    void Convert(ID3D11VideoProcessorEnumerator* enumerator, ID3D11VideoProcessor* processor,
                 ID3D11Texture2D* source, UINT slice,
                 ID3D11Texture2D* destination);
    void Draw(int64_t timestamp90k, const OverlayScene* scene);
    void DrawFigure(FigureKind kind, D2D1_POINT_2F center, const D2D1_RECT_F& region,
                    const D2D1::Matrix3x2F& rotation, uint32_t color, float fill, float outline, float stroke);
    void DrawStreamNumber(unsigned stream);
    void DrawClock(int64_t timestamp90k);
    void DrawTimings(int64_t timestamp90k);

    UINT width_;
    UINT height_;
    GpuFrameTiming gpuTiming_;
    uint64_t frame_{};
    std::optional<int64_t> previousTimestamp_;
    FrameStageTimes hostTiming_;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> rgb_;
    Microsoft::WRL::ComPtr<ID3D11VideoDevice> videoDevice_;
    Microsoft::WRL::ComPtr<ID3D11VideoContext1> videoContext_;
    Microsoft::WRL::ComPtr<ID3D11VideoProcessorEnumerator> toRgbEnumerator_;
    Microsoft::WRL::ComPtr<ID3D11VideoProcessorEnumerator> toNv12Enumerator_;
    Microsoft::WRL::ComPtr<ID3D11VideoProcessor> toRgb_;
    Microsoft::WRL::ComPtr<ID3D11VideoProcessor> toNv12_;
    Microsoft::WRL::ComPtr<ID2D1DeviceContext> drawing_;
    Microsoft::WRL::ComPtr<ID2D1Bitmap1> target_;
    Microsoft::WRL::ComPtr<ID2D1SolidColorBrush> brush_;
    Microsoft::WRL::ComPtr<ID2D1PathGeometry> triangle_;
    Microsoft::WRL::ComPtr<ID2D1PathGeometry> star_;
    Microsoft::WRL::ComPtr<IDWriteTextFormat> text_;
    Microsoft::WRL::ComPtr<IDWriteTextFormat> clock_;
    Microsoft::WRL::ComPtr<IDWriteTextFormat> timingText_;
};
