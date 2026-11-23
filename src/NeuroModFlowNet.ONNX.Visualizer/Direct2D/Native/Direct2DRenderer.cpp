#include <d3d11_4.h>
#include <d3dcompiler.h>
#include <d2d1_1.h>
#include <dwrite.h>
#include <dxgi1_2.h>
#include <wrl/client.h>
#include <cuda.h>
#include <cudaD3D11.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>

using Microsoft::WRL::ComPtr;
using EventHandle = std::unique_ptr<void, decltype(&CloseHandle)>;

namespace {
thread_local std::string lastError;

void Hr(HRESULT result, const char* operation) {
    if (FAILED(result))
        throw std::runtime_error(std::format("{}: HRESULT 0x{:08x}", operation,
                                             static_cast<unsigned long>(result)));
}

void Cuda(CUresult result, const char* operation) {
    if (result == CUDA_SUCCESS) return;
    const char* errorName = nullptr;
    const char* description = nullptr;
    cuGetErrorName(result, &errorName);
    cuGetErrorString(result, &description);
    throw std::runtime_error(std::format("{}: {} ({}) {}", operation, static_cast<int>(result),
                                         errorName ? errorName : "", description ? description : ""));
}

class PrimaryContext {
public:
    explicit PrimaryContext(int deviceOrdinal) {
        Cuda(cuInit(0), "initialize CUDA");
        Cuda(cuDeviceGet(&device, deviceOrdinal), "select CUDA device");
        Cuda(cuDevicePrimaryCtxRetain(&context, device), "retain CUDA primary context");
    }

    ~PrimaryContext() { cuDevicePrimaryCtxRelease(device); }

    CUdevice device{};
    CUcontext context{};
};

class CurrentContext {
public:
    explicit CurrentContext(CUcontext context) {
        Cuda(cuCtxPushCurrent(context), "make CUDA primary context current");
    }

    ~CurrentContext() {
        CUcontext previous{};
        cuCtxPopCurrent(&previous);
    }
};

constexpr const char* shader = R"(
    StructuredBuffer<uint> Pixels : register(t0);

    cbuffer Settings : register(b0) {
        uint Width;
        uint Height;
        uint Padding;
        uint Reserved;
    }

    float4 FullScreen(uint id : SV_VertexID) : SV_POSITION {
        const float2 positions[3] = {
            float2(-1.0, -1.0), float2(-1.0, 3.0), float2(3.0, -1.0)
        };
        return float4(positions[id], 0.0, 1.0);
    }

    float4 Render(float4 position : SV_POSITION) : SV_TARGET {
        uint pixel = Pixels[(uint)position.y * Width + (uint)position.x];
        return float4((pixel >> 16) & 255, (pixel >> 8) & 255, pixel & 255, pixel >> 24) / 255.0;
    }

)";

ComPtr<ID3DBlob> Compile(const char* entry, const char* target) {
    ComPtr<ID3DBlob> bytes;
    ComPtr<ID3DBlob> errors;
    HRESULT result = D3DCompile(shader, std::char_traits<char>::length(shader), nullptr, nullptr, nullptr,
                                entry, target, D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &bytes, &errors);
    if (FAILED(result) && errors) throw std::runtime_error(static_cast<const char*>(errors->GetBufferPointer()));
    Hr(result, "compile D3D shader");
    return bytes;
}

struct ShaderSettings {
    uint32_t width;
    uint32_t height;
    uint32_t padding;
    uint32_t reserved;
};

struct OverlayRectangle {
    float centerX;
    float centerY;
    float width;
    float height;
    float angleDegrees;
    float strokeWidth;
    uint32_t rgba;
    uint32_t labelOffset;
    uint32_t labelLength;
};

struct Renderer {
    Renderer(int deviceOrdinal, uint32_t width, uint32_t height, uint32_t outputWidth, uint32_t outputHeight)
        : cuda(deviceOrdinal), width(width), height(height), outputWidth(outputWidth), outputHeight(outputHeight), bytes(static_cast<size_t>(width) * height * 4),
          event(CreateEventW(nullptr, FALSE, FALSE, nullptr), CloseHandle) {
        if (!event) Hr(HRESULT_FROM_WIN32(GetLastError()), "create D3D11 fence event");
        InitDevice();
        InitResources();
        InitDraw();
        InitConversion();
        CurrentContext current(cuda.context);
        Cuda(cuGraphicsD3D11RegisterResource(&resource, buffer.Get(), CU_GRAPHICS_REGISTER_FLAGS_NONE),
             "register D3D11 buffer for CUDA");
    }

    ~Renderer() {
        if (!resource) return;
        try { Reset(); }
        catch (...) { }
        CUcontext previous{};
        if (cuCtxPushCurrent(cuda.context) != CUDA_SUCCESS) return;
        if (mapped) cuGraphicsUnmapResources(1, &resource, nullptr);
        cuGraphicsUnregisterResource(resource);
        cuCtxPopCurrent(&previous);
    }

    uint64_t Map() {
        if (mapped || prepared) throw std::runtime_error("complete or reset the current D3D11 frame before mapping");
        CurrentContext current(cuda.context);
        Cuda(cuGraphicsMapResources(1, &resource, nullptr), "map D3D11 buffer for CUDA");
        mapped = true;
        try {
            CUdeviceptr pointer{};
            size_t mappedBytes{};
            Cuda(cuGraphicsResourceGetMappedPointer(&pointer, &mappedBytes, resource), "get mapped CUDA pointer");
            if (mappedBytes < bytes) throw std::runtime_error("mapped buffer smaller than BGRA frame");
            return pointer;
        } catch (...) {
            if (cuGraphicsUnmapResources(1, &resource, nullptr) == CUDA_SUCCESS) mapped = false;
            throw;
        }
    }

    ID3D11Device* Device() const { return device.Get(); }

    void PrepareBackground() {
        if (!mapped) throw std::runtime_error("D3D11 buffer must be mapped before drawing");
        Unmap();
        prepared = true;
        DrawBackground();
    }

    void DrawPrepared(ID3D11Texture2D* destination, uint32_t slice,
                      const OverlayRectangle* rectangles, uint32_t count, const wchar_t* labels) {
        BeginDraw();
        for (uint32_t index = 0; index < count; ++index) {
            const auto& box = rectangles[index];
            DrawRectangle(box.centerX, box.centerY, box.width, box.height, box.angleDegrees, box.rgba, box.strokeWidth);
            if (box.labelLength) {
                const float fontSize = text->GetFontSize();
                const float left = std::max(0.0f, box.centerX - box.width * 0.5f);
                const float top = std::max(0.0f, box.centerY - box.height * 0.5f - fontSize - 2);
                DrawText(labels + box.labelOffset, box.labelLength, left, top, static_cast<float>(width) - left,
                         fontSize + 4, box.rgba, 0);
            }
        }
        EndDraw(destination, slice);
    }

    void BeginDraw() {
        if (!prepared || drawingOpen) throw std::runtime_error("prepare the D3D11 background before opening a drawing pass");
        drawing->SetTarget(d2dTarget.Get());
        drawing->BeginDraw();
        drawingOpen = true;
    }

    void DrawRectangle(float centerX, float centerY, float boxWidth, float boxHeight, float angleDegrees,
                       uint32_t rgba, float strokeWidth) {
        RequireDrawing();
        const float left = centerX - boxWidth * 0.5f;
        const float top = centerY - boxHeight * 0.5f;
        SetColor(rgba);
        drawing->SetTransform(D2D1::Matrix3x2F::Rotation(angleDegrees, D2D1::Point2F(centerX, centerY)));
        drawing->DrawRectangle(D2D1::RectF(left, top, left + boxWidth, top + boxHeight), brush.Get(), strokeWidth);
        drawing->SetTransform(D2D1::Matrix3x2F::Identity());
    }

    void DrawLine(float startX, float startY, float endX, float endY, uint32_t rgba, float strokeWidth) {
        RequireDrawing();
        SetColor(rgba);
        drawing->DrawLine(D2D1::Point2F(startX, startY), D2D1::Point2F(endX, endY), brush.Get(), strokeWidth);
    }

    void DrawText(const wchar_t* characters, uint32_t length, float left, float top, float textWidth, float textHeight,
                  uint32_t rgba, float fontSize) {
        RequireDrawing();
        if (!length) return;
        if (!characters || !std::isfinite(fontSize) || fontSize < 0)
            throw std::invalid_argument("invalid Direct2D text or font size");
        IDWriteTextFormat* format = text.Get();
        if (fontSize != 0 && fontSize != text->GetFontSize()) {
            if (!customText || customText->GetFontSize() != fontSize)
                customText = CreateTextFormat(fontSize);
            format = customText.Get();
        }
        SetColor(rgba);
        drawing->DrawText(characters, length, format, D2D1::RectF(left, top, left + textWidth, top + textHeight),
                          brush.Get(), D2D1_DRAW_TEXT_OPTIONS_CLIP);
    }

    void EndDraw(ID3D11Texture2D* destination, uint32_t slice) {
        RequireDrawing();
        Hr(CloseDrawing(), "complete Direct2D overlay");
        ConvertToNv12(destination, slice);
        WaitForGpu();
        prepared = false;
    }

    void Reset() {
        if (mapped) Unmap();
        const HRESULT drawingResult = drawingOpen ? CloseDrawing() : S_OK;
        if (prepared) {
            WaitForGpu();
            prepared = false;
        }
        Hr(drawingResult, "close abandoned Direct2D drawing pass");
    }

private:
    void RequireDrawing() const {
        if (!drawingOpen) throw std::runtime_error("no active Direct2D drawing pass");
    }

    void SetColor(uint32_t rgba) {
        brush->SetColor(D2D1::ColorF(rgba >> 8, (rgba & 255) / 255.0f));
    }

    HRESULT CloseDrawing() {
        const HRESULT result = drawing->EndDraw();
        drawing->SetTarget(nullptr);
        drawingOpen = false;
        return result;
    }

    void Unmap() {
        CurrentContext current(cuda.context);
        Cuda(cuGraphicsUnmapResources(1, &resource, nullptr), "return CUDA buffer to D3D11");
        mapped = false;
    }

    void InitDevice() {
        ComPtr<IDXGIFactory1> factory;
        Hr(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "create DXGI factory");
        for (UINT index = 0;; ++index) {
            ComPtr<IDXGIAdapter1> candidate;
            HRESULT result = factory->EnumAdapters1(index, &candidate);
            if (result == DXGI_ERROR_NOT_FOUND) break;
            Hr(result, "enumerate DXGI adapters");
            CUdevice ordinal{};
            if (cuD3D11GetDevice(&ordinal, candidate.Get()) == CUDA_SUCCESS && ordinal == cuda.device) {
                adapter = candidate;
                break;
            }
        }
        if (!adapter) throw std::runtime_error("no D3D11 adapter matches the selected CUDA device");
        ComPtr<ID3D11DeviceContext> immediate;
        Hr(D3D11CreateDevice(adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, D3D11_CREATE_DEVICE_BGRA_SUPPORT | D3D11_CREATE_DEVICE_VIDEO_SUPPORT,
                             nullptr, 0, D3D11_SDK_VERSION, &device, nullptr, &immediate),
           "create hardware D3D11 GPU device");
        Hr(immediate.As(&context), "get D3D11 fence context");
        ComPtr<ID3D11Device5> fenceDevice;
        Hr(device.As(&fenceDevice), "get D3D11 fence device");
        Hr(fenceDevice->CreateFence(0, D3D11_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "create GPU fence");
    }

    void InitResources() {
        D3D11_BUFFER_DESC bufferDesc{};
        bufferDesc.ByteWidth = static_cast<UINT>(bytes);
        bufferDesc.Usage = D3D11_USAGE_DEFAULT;
        bufferDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        bufferDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        bufferDesc.StructureByteStride = 4;
        Hr(device->CreateBuffer(&bufferDesc, nullptr, &buffer), "create D3D11 GPU-only packed-pixel buffer");
        D3D11_SHADER_RESOURCE_VIEW_DESC viewDesc{};
        viewDesc.Format = DXGI_FORMAT_UNKNOWN;
        viewDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        viewDesc.Buffer.NumElements = width * height;
        Hr(device->CreateShaderResourceView(buffer.Get(), &viewDesc, &bufferView), "view D3D11 packed-pixel buffer");

        D3D11_TEXTURE2D_DESC textureDesc{};
        textureDesc.Width = width;
        textureDesc.Height = height;
        textureDesc.MipLevels = 1;
        textureDesc.ArraySize = 1;
        textureDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        textureDesc.SampleDesc.Count = 1;
        textureDesc.Usage = D3D11_USAGE_DEFAULT;
        textureDesc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        Hr(device->CreateTexture2D(&textureDesc, nullptr, &target), "create GPU-only D3D11 BGRA texture");
        Hr(device->CreateRenderTargetView(target.Get(), nullptr, &targetView), "view BGRA render target");
        D3D11_BUFFER_DESC settingsDesc{};
        settingsDesc.ByteWidth = sizeof(ShaderSettings);
        settingsDesc.Usage = D3D11_USAGE_DEFAULT;
        settingsDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        const ShaderSettings settings{width, height, 0, 0};
        D3D11_SUBRESOURCE_DATA data{};
        data.pSysMem = &settings;
        Hr(device->CreateBuffer(&settingsDesc, &data, &settingsBuffer), "create shader settings buffer");

        D3D11_RASTERIZER_DESC rasterDesc{};
        rasterDesc.FillMode = D3D11_FILL_SOLID;
        rasterDesc.CullMode = D3D11_CULL_NONE;
        rasterDesc.DepthClipEnable = TRUE;
        Hr(device->CreateRasterizerState(&rasterDesc, &rasterizer), "create fullscreen rasterizer");
        auto vertex = Compile("FullScreen", "vs_5_0");
        Hr(device->CreateVertexShader(vertex->GetBufferPointer(), vertex->GetBufferSize(), nullptr, &vertexShader),
           "create fullscreen vertex shader");
        auto copy = Compile("Render", "ps_5_0");
        Hr(device->CreatePixelShader(copy->GetBufferPointer(), copy->GetBufferSize(), nullptr, &copyShader),
           "create packed BGRA pixel shader");
    }

    void InitDraw() {
        ComPtr<IDXGIDevice> dxgiDevice;
        Hr(device.As(&dxgiDevice), "get D3D11 device as DXGI device");
        ComPtr<ID2D1Factory1> factory;
        Hr(D2D1CreateFactory(D2D1_FACTORY_TYPE_MULTI_THREADED, factory.GetAddressOf()), "create Direct2D factory");
        ComPtr<ID2D1Device> d2dDevice;
        Hr(factory->CreateDevice(dxgiDevice.Get(), &d2dDevice), "create Direct2D D3D11 device");
        Hr(d2dDevice->CreateDeviceContext(D2D1_DEVICE_CONTEXT_OPTIONS_NONE, &drawing), "create Direct2D context");
        drawing->SetUnitMode(D2D1_UNIT_MODE_PIXELS);
        drawing->SetTextAntialiasMode(D2D1_TEXT_ANTIALIAS_MODE_GRAYSCALE);

        ComPtr<IDXGISurface> surface;
        Hr(target.As(&surface), "get BGRA texture as DXGI surface");
        auto props = D2D1::BitmapProperties1(D2D1_BITMAP_OPTIONS_TARGET | D2D1_BITMAP_OPTIONS_CANNOT_DRAW,
                                            D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_IGNORE),
                                            96, 96);
        Hr(drawing->CreateBitmapFromDxgiSurface(surface.Get(), &props, &d2dTarget), "bind DXGI target to Direct2D");
        Hr(drawing->CreateSolidColorBrush(D2D1::ColorF(D2D1::ColorF::White), &brush), "create overlay brush");

        Hr(DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, __uuidof(IDWriteFactory),
                               reinterpret_cast<IUnknown**>(writing.GetAddressOf())), "create DirectWrite factory");
        text = CreateTextFormat(std::clamp(height / 48.0f, 14.0f, 36.0f));
    }

    ComPtr<IDWriteTextFormat> CreateTextFormat(float fontSize) {
        ComPtr<IDWriteTextFormat> format;
        Hr(writing->CreateTextFormat(L"Segoe UI", nullptr, DWRITE_FONT_WEIGHT_NORMAL, DWRITE_FONT_STYLE_NORMAL,
                                     DWRITE_FONT_STRETCH_NORMAL, fontSize, L"en-us", &format),
           "create DirectWrite text format");
        return format;
    }

    void InitConversion() {
        Hr(device.As(&videoDevice), "get D3D11 video device");
        Hr(context.As(&videoContext), "get D3D11 video context");
        D3D11_VIDEO_PROCESSOR_CONTENT_DESC description{};
        description.InputFrameFormat = D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE;
        description.InputWidth = width;
        description.InputHeight = height;
        description.OutputWidth = outputWidth;
        description.OutputHeight = outputHeight;
        description.Usage = D3D11_VIDEO_USAGE_PLAYBACK_NORMAL;
        Hr(videoDevice->CreateVideoProcessorEnumerator(&description, &enumerator), "create video processor enumerator");
        ComPtr<ID3D11VideoProcessorEnumerator1> formats;
        Hr(enumerator.As(&formats), "query video format conversion support");
        BOOL supported = FALSE;
        Hr(formats->CheckVideoProcessorFormatConversion(DXGI_FORMAT_B8G8R8A8_UNORM,
            DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709, DXGI_FORMAT_NV12, DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P709, &supported),
            "check BGRA to BT.709 NV12 conversion");
        if (!supported) throw std::runtime_error("GPU does not support the requested BGRA to NV12 format conversion");
        Hr(videoDevice->CreateVideoProcessor(enumerator.Get(), 0, &processor), "create video processor");
        videoContext->VideoProcessorSetStreamColorSpace1(processor.Get(), 0, DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709);
        videoContext->VideoProcessorSetOutputColorSpace1(processor.Get(), DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P709);
        videoContext->VideoProcessorSetStreamAutoProcessingMode(processor.Get(), 0, FALSE);
        videoContext->VideoProcessorSetStreamFrameFormat(processor.Get(), 0, D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE);
        const RECT source{0, 0, static_cast<LONG>(width), static_cast<LONG>(height)};
        const RECT output{0, 0, static_cast<LONG>(outputWidth), static_cast<LONG>(outputHeight)};
        videoContext->VideoProcessorSetStreamSourceRect(processor.Get(), 0, TRUE, &source);
        videoContext->VideoProcessorSetStreamDestRect(processor.Get(), 0, TRUE, &output);
        videoContext->VideoProcessorSetOutputTargetRect(processor.Get(), TRUE, &output);
        D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC input{};
        input.ViewDimension = D3D11_VPIV_DIMENSION_TEXTURE2D;
        Hr(videoDevice->CreateVideoProcessorInputView(target.Get(), enumerator.Get(), &input, &inputView), "view BGRA video input");
    }

    void ConvertToNv12(ID3D11Texture2D* destination, uint32_t slice) {
        if (!destination) throw std::runtime_error("expected a GPU NV12 surface");
        D3D11_TEXTURE2D_DESC description{};
        destination->GetDesc(&description);
        ComPtr<ID3D11Device> owner;
        destination->GetDevice(&owner);
        if (owner.Get() != device.Get() || description.Usage != D3D11_USAGE_DEFAULT || description.CPUAccessFlags != 0 ||
            description.Format != DXGI_FORMAT_NV12 || description.Width != outputWidth || description.Height != outputHeight ||
            slice >= description.ArraySize)
            throw std::runtime_error("expected a compatible NV12 surface on the same GPU device");
        D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC output{};
        if (description.ArraySize > 1) {
            output.ViewDimension = D3D11_VPOV_DIMENSION_TEXTURE2DARRAY;
            output.Texture2DArray.FirstArraySlice = slice;
            output.Texture2DArray.ArraySize = 1;
        } else {
            output.ViewDimension = D3D11_VPOV_DIMENSION_TEXTURE2D;
        }
        ComPtr<ID3D11VideoProcessorOutputView> outputView;
        Hr(videoDevice->CreateVideoProcessorOutputView(destination, enumerator.Get(), &output, &outputView), "view NV12 output");
        D3D11_VIDEO_PROCESSOR_STREAM stream{};
        stream.Enable = TRUE;
        stream.pInputSurface = inputView.Get();
        Hr(videoContext->VideoProcessorBlt(processor.Get(), outputView.Get(), 0, 1, &stream), "convert GPU frame to NV12");
    }

    void DrawBackground() {
        const D3D11_VIEWPORT viewport{0, 0, static_cast<float>(width), static_cast<float>(height), 0, 1};
        context->RSSetState(rasterizer.Get());
        context->RSSetViewports(1, &viewport);
        context->OMSetBlendState(nullptr, nullptr, 0xffffffff);
        context->OMSetDepthStencilState(nullptr, 0);
        auto* view = targetView.Get();
        context->OMSetRenderTargets(1, &view, nullptr);
        context->IASetInputLayout(nullptr);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(vertexShader.Get(), nullptr, 0);
        context->GSSetShader(nullptr, nullptr, 0);
        context->PSSetShader(copyShader.Get(), nullptr, 0);
        auto* pixels = bufferView.Get();
        context->PSSetShaderResources(0, 1, &pixels);
        auto* constants = settingsBuffer.Get();
        context->PSSetConstantBuffers(0, 1, &constants);
        context->Draw(3, 0);
        ID3D11ShaderResourceView* clear = nullptr;
        context->PSSetShaderResources(0, 1, &clear);
        context->OMSetRenderTargets(0, nullptr, nullptr);
    }

    void WaitForGpu() {
        ++fenceValue;
        Hr(context->Signal(fence.Get(), fenceValue), "signal D3D11 work completed");
        Hr(fence->SetEventOnCompletion(fenceValue, event.get()), "arm D3D11 fence event");
        context->Flush();
        const ULONGLONG deadline = GetTickCount64() + 5000;
        while (fence->GetCompletedValue() < fenceValue) {
            const ULONGLONG now = GetTickCount64();
            if (now >= deadline) throw std::runtime_error("timeout while waiting for D3D11 fence");
            const DWORD wait = WaitForSingleObject(event.get(), static_cast<DWORD>(deadline - now));
            if (wait == WAIT_FAILED) Hr(HRESULT_FROM_WIN32(GetLastError()), "wait for D3D11 fence event");
        }
        Hr(device->GetDeviceRemovedReason(), "check D3D11 device");
    }

    PrimaryContext cuda;
    uint32_t width;
    uint32_t height;
    uint32_t outputWidth;
    uint32_t outputHeight;
    size_t bytes;
    EventHandle event;
    CUgraphicsResource resource{};
    bool mapped{};
    bool prepared{};
    bool drawingOpen{};
    uint64_t fenceValue{};
    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext4> context;
    ComPtr<ID3D11Fence> fence;
    ComPtr<ID3D11Buffer> buffer;
    ComPtr<ID3D11ShaderResourceView> bufferView;
    ComPtr<ID3D11Texture2D> target;
    ComPtr<ID3D11RenderTargetView> targetView;
    ComPtr<ID3D11Buffer> settingsBuffer;
    ComPtr<ID3D11RasterizerState> rasterizer;
    ComPtr<ID3D11VertexShader> vertexShader;
    ComPtr<ID3D11PixelShader> copyShader;
    ComPtr<ID2D1DeviceContext> drawing;
    ComPtr<ID2D1Bitmap1> d2dTarget;
    ComPtr<ID2D1SolidColorBrush> brush;
    ComPtr<IDWriteFactory> writing;
    ComPtr<IDWriteTextFormat> text;
    ComPtr<IDWriteTextFormat> customText;
    ComPtr<ID3D11VideoDevice> videoDevice;
    ComPtr<ID3D11VideoContext1> videoContext;
    ComPtr<ID3D11VideoProcessorEnumerator> enumerator;
    ComPtr<ID3D11VideoProcessor> processor;
    ComPtr<ID3D11VideoProcessorInputView> inputView;
};

template<typename Action>
int InvokeDrawing(Action action) {
    try {
        action();
        return 0;
    } catch (const std::exception& error) {
        lastError = error.what();
        return -1;
    }
}
}

extern "C" {

__declspec(dllexport) const char* NmfD2DError() { return lastError.c_str(); }

__declspec(dllexport) Renderer* NmfD2DCreate(int device, int width, int height, int outputWidth, int outputHeight) {
    try {
        const auto valid = [](int dimension) { return dimension > 0 && dimension <= 8192; };
        if (!valid(width) || !valid(height) || !valid(outputWidth) || !valid(outputHeight) ||
            outputWidth % 2 || outputHeight % 2)
            throw std::runtime_error("invalid input or NV12 output dimensions");
        return new Renderer(device, width, height, outputWidth, outputHeight);
    } catch (const std::exception& error) {
        lastError = error.what();
        return nullptr;
    }
}

__declspec(dllexport) ID3D11Device* NmfD2DDevice(Renderer* renderer) { return renderer->Device(); }

__declspec(dllexport) uint64_t NmfD2DMap(Renderer* renderer) {
    try { return renderer->Map(); }
    catch (const std::exception& error) {
        lastError = error.what();
        return 0;
    }
}

__declspec(dllexport) int NmfD2DDraw(Renderer* renderer, ID3D11Texture2D* destination, uint32_t slice,
                                   const OverlayRectangle* rectangles, uint32_t count, const wchar_t* labels) {
    try {
        renderer->PrepareBackground();
        renderer->DrawPrepared(destination, slice, rectangles, count, labels);
        return 0;
    } catch (const std::exception& error) {
        lastError = error.what();
        try { renderer->Reset(); }
        catch (...) { }
        return -1;
    }
}

__declspec(dllexport) int NmfD2DPrepareBackground(Renderer* renderer) {
    try {
        renderer->PrepareBackground();
        return 0;
    } catch (const std::exception& error) {
        lastError = error.what();
        return -1;
    }
}

__declspec(dllexport) int NmfD2DDrawPrepared(Renderer* renderer, ID3D11Texture2D* destination, uint32_t slice,
                                           const OverlayRectangle* rectangles, uint32_t count, const wchar_t* labels) {
    try {
        renderer->DrawPrepared(destination, slice, rectangles, count, labels);
        return 0;
    } catch (const std::exception& error) {
        lastError = error.what();
        return -1;
    }
}

__declspec(dllexport) int NmfD2DBeginDraw(Renderer* renderer) {
    return InvokeDrawing([&] { renderer->BeginDraw(); });
}

__declspec(dllexport) int NmfD2DEndDraw(Renderer* renderer, ID3D11Texture2D* destination, uint32_t slice) {
    return InvokeDrawing([&] { renderer->EndDraw(destination, slice); });
}

__declspec(dllexport) int NmfD2DDrawRectangle(Renderer* renderer, float centerX, float centerY, float width,
                                            float height, float angleDegrees, uint32_t rgba, float strokeWidth) {
    return InvokeDrawing([&] { renderer->DrawRectangle(centerX, centerY, width, height, angleDegrees, rgba, strokeWidth); });
}

__declspec(dllexport) int NmfD2DDrawLine(Renderer* renderer, float startX, float startY, float endX, float endY,
                                       uint32_t rgba, float strokeWidth) {
    return InvokeDrawing([&] { renderer->DrawLine(startX, startY, endX, endY, rgba, strokeWidth); });
}

__declspec(dllexport) int NmfD2DDrawText(Renderer* renderer, const wchar_t* characters, uint32_t length,
                                       float left, float top, float width, float height, uint32_t rgba, float fontSize) {
    return InvokeDrawing([&] { renderer->DrawText(characters, length, left, top, width, height, rgba, fontSize); });
}

__declspec(dllexport) int NmfD2DReset(Renderer* renderer) {
    try {
        renderer->Reset();
        return 0;
    } catch (const std::exception& error) {
        lastError = error.what();
        return -1;
    }
}

__declspec(dllexport) void NmfD2DDestroy(Renderer* renderer) { delete renderer; }

}
