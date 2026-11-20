#include "GpuOverlay.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <format>
#include <iterator>
#include <numbers>
#include <span>
#include <stdexcept>

using Microsoft::WRL::ComPtr;

namespace {
ComPtr<ID2D1PathGeometry> MakePolygon(ID2D1Factory* factory, std::span<const D2D1_POINT_2F> points) {
    ComPtr<ID2D1PathGeometry> geometry;
    CheckHr(factory->CreatePathGeometry(&geometry), "create overlay polygon");
    ComPtr<ID2D1GeometrySink> sink;
    CheckHr(geometry->Open(&sink), "open overlay polygon");
    sink->BeginFigure(points.front(), D2D1_FIGURE_BEGIN_FILLED);
    sink->AddLines(points.data() + 1, static_cast<UINT32>(points.size() - 1));
    sink->EndFigure(D2D1_FIGURE_END_CLOSED);
    CheckHr(sink->Close(), "finish overlay polygon");
    return geometry;
}
}

void CheckHr(HRESULT result, const char* operation) {
    if (FAILED(result)) {
        throw std::runtime_error(std::format("{}: HRESULT 0x{:08X}", operation,
                                             static_cast<unsigned long>(result)));
    }
}

GpuOverlay::GpuOverlay(ID3D11Device* device, UINT width, UINT height, bool bt709, bool fullRange)
    : GpuOverlay(device, width, height, width, height, bt709, fullRange) {}

GpuOverlay::GpuOverlay(ID3D11Device* device, UINT inputWidth, UINT inputHeight, UINT width, UINT height,
                       bool bt709, bool fullRange)
    : width_(width), height_(height), gpuTiming_(device) {
    D3D11_TEXTURE2D_DESC texture{};
    texture.Width = width;
    texture.Height = height;
    texture.MipLevels = 1;
    texture.ArraySize = 1;
    texture.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    texture.SampleDesc.Count = 1;
    texture.Usage = D3D11_USAGE_DEFAULT;
    texture.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    CheckHr(device->CreateTexture2D(&texture, nullptr, &rgb_), "create BGRA GPU render target");

    CheckHr(device->QueryInterface(IID_PPV_ARGS(&videoDevice_)), "get D3D11 video device");
    ComPtr<ID3D11DeviceContext> immediate;
    device->GetImmediateContext(&immediate);
    CheckHr(immediate.As(&videoContext_), "get D3D11.1 video context");

    toRgbEnumerator_ = CreateEnumerator(inputWidth, inputHeight);
    toNv12Enumerator_ = inputWidth == width && inputHeight == height ? toRgbEnumerator_
                                                                    : CreateEnumerator(width, height);

    const auto limitedYuv = bt709 ? DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P709
                                  : DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P601;
    const auto fullYuv = bt709 ? DXGI_COLOR_SPACE_YCBCR_FULL_G22_LEFT_P709
                               : DXGI_COLOR_SPACE_YCBCR_FULL_G22_LEFT_P601;
    constexpr auto fullRgb = DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
    ComPtr<ID3D11VideoProcessorEnumerator1> formatSupport;
    CheckHr(toRgbEnumerator_.As(&formatSupport), "get video processor format support");
    BOOL supported = FALSE;
    CheckHr(formatSupport->CheckVideoProcessorFormatConversion(
        DXGI_FORMAT_NV12, fullRange ? fullYuv : limitedYuv, texture.Format, fullRgb, &supported),
        "check NV12 to BGRA conversion");
    if (!supported) throw std::runtime_error("GPU cannot convert the input NV12 color space to BGRA");
    formatSupport.Reset();
    CheckHr(toNv12Enumerator_.As(&formatSupport), "get NV12 video processor format support");
    CheckHr(formatSupport->CheckVideoProcessorFormatConversion(
        texture.Format, fullRgb, DXGI_FORMAT_NV12, limitedYuv, &supported),
        "check BGRA to NV12 conversion");
    if (!supported) throw std::runtime_error("GPU cannot convert BGRA to limited-range NV12");

    toRgb_ = CreateProcessor(toRgbEnumerator_.Get(), inputWidth, inputHeight,
                             fullRange ? fullYuv : limitedYuv, fullRgb);
    toNv12_ = CreateProcessor(toNv12Enumerator_.Get(), width, height, fullRgb, limitedYuv);

    ComPtr<IDXGIDevice> dxgiDevice;
    CheckHr(device->QueryInterface(IID_PPV_ARGS(&dxgiDevice)), "get DXGI device");
    ComPtr<ID2D1Factory1> factory;
    CheckHr(D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, factory.GetAddressOf()),
            "create Direct2D factory");
    const std::array triangle{D2D1::Point2F(0, -1), D2D1::Point2F(1, 0.86f), D2D1::Point2F(-1, 0.86f)};
    triangle_ = MakePolygon(factory.Get(), triangle);
    std::array<D2D1_POINT_2F, 10> star{};
    for (size_t index = 0; index < star.size(); ++index) {
        const float angle = std::numbers::pi_v<float> * (static_cast<float>(index) / 5.0f - 0.5f);
        const float radius = index % 2 == 0 ? 1.0f : 0.47f;
        star[index] = D2D1::Point2F(radius * std::cos(angle), radius * std::sin(angle));
    }
    star_ = MakePolygon(factory.Get(), star);
    ComPtr<ID2D1Device> drawingDevice;
    CheckHr(factory->CreateDevice(dxgiDevice.Get(), &drawingDevice), "create Direct2D device");
    CheckHr(drawingDevice->CreateDeviceContext(D2D1_DEVICE_CONTEXT_OPTIONS_NONE, &drawing_),
            "create Direct2D drawing context");
    drawing_->SetTextAntialiasMode(D2D1_TEXT_ANTIALIAS_MODE_GRAYSCALE);
    ComPtr<IDXGISurface> surface;
    CheckHr(rgb_.As(&surface), "get BGRA texture as DXGI surface");
    const auto bitmap = D2D1::BitmapProperties1(
        D2D1_BITMAP_OPTIONS_TARGET | D2D1_BITMAP_OPTIONS_CANNOT_DRAW,
        D2D1::PixelFormat(texture.Format, D2D1_ALPHA_MODE_IGNORE), 96, 96);
    CheckHr(drawing_->CreateBitmapFromDxgiSurface(surface.Get(), &bitmap, &target_),
            "bind BGRA texture as Direct2D target");
    CheckHr(drawing_->CreateSolidColorBrush(D2D1::ColorF(D2D1::ColorF::White), &brush_),
            "create drawing brush");

    ComPtr<IDWriteFactory> writing;
    CheckHr(DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, __uuidof(IDWriteFactory),
                               reinterpret_cast<IUnknown**>(writing.GetAddressOf())),
            "create DirectWrite factory");
    CheckHr(writing->CreateTextFormat(L"Segoe UI", nullptr, DWRITE_FONT_WEIGHT_SEMI_BOLD,
                                    DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL,
                                    std::min(height * 0.032f, width * 0.018f), L"en-US", &text_),
            "create DirectWrite label format");
    CheckHr(text_->SetWordWrapping(DWRITE_WORD_WRAPPING_NO_WRAP), "set text wrapping");
    CheckHr(writing->CreateTextFormat(L"Consolas", nullptr, DWRITE_FONT_WEIGHT_BOLD,
                                    DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL,
                                    height * 0.029f, L"en-US", &clock_), "create DirectWrite clock format");
    CheckHr(clock_->SetTextAlignment(DWRITE_TEXT_ALIGNMENT_CENTER), "center clock horizontally");
    CheckHr(clock_->SetParagraphAlignment(DWRITE_PARAGRAPH_ALIGNMENT_CENTER), "center clock vertically");
    CheckHr(clock_->SetWordWrapping(DWRITE_WORD_WRAPPING_NO_WRAP), "set clock wrapping");
    CheckHr(writing->CreateTextFormat(L"Consolas", nullptr, DWRITE_FONT_WEIGHT_NORMAL,
                                    DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL,
                                    std::min(height * 0.020f, width / 76.0f), L"en-US", &timingText_),
            "create DirectWrite timing format");
    CheckHr(timingText_->SetWordWrapping(DWRITE_WORD_WRAPPING_NO_WRAP), "set timing wrapping");
}

ComPtr<ID3D11VideoProcessorEnumerator> GpuOverlay::CreateEnumerator(UINT inputWidth, UINT inputHeight) {
    D3D11_VIDEO_PROCESSOR_CONTENT_DESC content{};
    content.InputFrameFormat = D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE;
    content.InputWidth = inputWidth;
    content.InputHeight = inputHeight;
    content.OutputWidth = width_;
    content.OutputHeight = height_;
    content.Usage = D3D11_VIDEO_USAGE_PLAYBACK_NORMAL;
    ComPtr<ID3D11VideoProcessorEnumerator> enumerator;
    CheckHr(videoDevice_->CreateVideoProcessorEnumerator(&content, &enumerator),
            "create video processor enumerator");
    return enumerator;
}

ComPtr<ID3D11VideoProcessor> GpuOverlay::CreateProcessor(
    ID3D11VideoProcessorEnumerator* enumerator, UINT inputWidth, UINT inputHeight,
    DXGI_COLOR_SPACE_TYPE inputColor, DXGI_COLOR_SPACE_TYPE outputColor) {
    ComPtr<ID3D11VideoProcessor> processor;
    CheckHr(videoDevice_->CreateVideoProcessor(enumerator, 0, &processor),
            "create D3D11 video processor");
    videoContext_->VideoProcessorSetStreamColorSpace1(processor.Get(), 0, inputColor);
    videoContext_->VideoProcessorSetOutputColorSpace1(processor.Get(), outputColor);
    videoContext_->VideoProcessorSetStreamAutoProcessingMode(processor.Get(), 0, FALSE);
    videoContext_->VideoProcessorSetStreamFrameFormat(processor.Get(), 0,
                                                      D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE);
    const RECT area{0, 0, static_cast<LONG>(width_), static_cast<LONG>(height_)};
    const RECT source{0, 0, static_cast<LONG>(inputWidth), static_cast<LONG>(inputHeight)};
    videoContext_->VideoProcessorSetStreamSourceRect(processor.Get(), 0, TRUE, &source);
    videoContext_->VideoProcessorSetStreamDestRect(processor.Get(), 0, TRUE, &area);
    videoContext_->VideoProcessorSetOutputTargetRect(processor.Get(), TRUE, &area);
    return processor;
}

void GpuOverlay::Convert(ID3D11VideoProcessorEnumerator* enumerator, ID3D11VideoProcessor* processor,
                         ID3D11Texture2D* source, UINT slice,
                         ID3D11Texture2D* destination) {
    D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC input{};
    input.ViewDimension = D3D11_VPIV_DIMENSION_TEXTURE2D;
    input.Texture2D.ArraySlice = slice;
    ComPtr<ID3D11VideoProcessorInputView> inputView;
    CheckHr(videoDevice_->CreateVideoProcessorInputView(source, enumerator, &input, &inputView),
            "create video processor input view");
    D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC output{};
    output.ViewDimension = D3D11_VPOV_DIMENSION_TEXTURE2D;
    ComPtr<ID3D11VideoProcessorOutputView> outputView;
    CheckHr(videoDevice_->CreateVideoProcessorOutputView(destination, enumerator, &output, &outputView),
            "create video processor output view");
    D3D11_VIDEO_PROCESSOR_STREAM stream{};
    stream.Enable = TRUE;
    stream.pInputSurface = inputView.Get();
    CheckHr(videoContext_->VideoProcessorBlt(processor, outputView.Get(), 0, 1, &stream),
            "convert video texture on GPU");
}

void GpuOverlay::Draw(int64_t timestamp90k, const OverlayScene* scene) {
    struct Box {
        float angle;
        float stroke;
        float fillAlpha;
        float strokeAlpha;
        UINT32 color;
    };
    constexpr std::array boxes{
        Box{-24, 1.5f, 0.00f, 1.00f, 0x54E8F4},
        Box{-15, 5.0f, 0.25f, 0.90f, 0xF5CE62},
        Box{ -7, 2.5f, 0.45f, 0.80f, 0x7CEF98},
        Box{  8, 7.0f, 0.15f, 1.00f, 0xFF7797},
        Box{ 19, 3.0f, 0.35f, 0.70f, 0xC6A3FF},
        Box{ 27, 6.0f, 0.25f, 0.90f, 0xFF9B68},
        Box{ 13, 2.0f, 0.00f, 0.85f, 0x77BBFF},
        Box{  0, 4.0f, 0.40f, 1.00f, 0xBFEF77},
        Box{-11, 8.0f, 0.20f, 0.75f, 0xF18AF5},
        Box{-30, 3.5f, 0.35f, 1.00f, 0xFCF2DA},
    };
    const float width = static_cast<float>(width_);
    const float height = static_cast<float>(height_);
    const double seconds = static_cast<double>(timestamp90k) / 90000.0;
    drawing_->SetTarget(target_.Get());
    drawing_->BeginDraw();
    for (size_t index = 0; index < boxes.size(); ++index) {
        const auto& box = boxes[index];
        const auto* figure = scene ? &scene->figures[index] : nullptr;
        const double phase = seconds * (1.65 + 0.09 * static_cast<double>(index)) +
                             0.73 * static_cast<double>(index) + (figure ? figure->phase : 0.0f);
        const float horizontal = static_cast<float>(std::sin(phase));
        const float vertical = static_cast<float>(std::cos(phase * 0.83));
        const float size = 1.0f + 0.16f * static_cast<float>(std::sin(phase * 1.17 + 1.2));
        const float angle = (figure ? figure->angle : box.angle) +
                            12.0f * static_cast<float>(std::sin(phase * 0.91 + 0.6));
        const auto center = D2D1::Point2F(
            width * (0.12f + 0.19f * static_cast<float>(index % 5) + 0.016f * horizontal),
            height * (0.35f + 0.31f * static_cast<float>(index / 5) + 0.032f * vertical));
        const float halfWidth = width * 0.065f * size;
        const float halfHeight = height * 0.086f * size;
        const auto region = D2D1::RectF(center.x - halfWidth, center.y - halfHeight,
                                       center.x + halfWidth, center.y + halfHeight);
        const auto boxTransform = D2D1::Matrix3x2F::Rotation(angle, center);
        DrawFigure(figure ? figure->kind : FigureKind::Rectangle, center, region, boxTransform,
                   figure ? figure->color : box.color, box.fillAlpha, box.strokeAlpha, box.stroke * height / 720.0f);

        const float labelLeft = region.left + width * (0.013f + 0.004f * horizontal);
        const float labelTop = region.top + height * (0.027f + 0.010f * vertical);
        const auto label = D2D1::RectF(labelLeft, labelTop, region.right, region.bottom);
        const auto shadow = D2D1::RectF(label.left + 1, label.top + 1, label.right, label.bottom);
        const float labelTilt = 6.0f * static_cast<float>(std::cos(phase * 1.4));
        drawing_->SetTransform(D2D1::Matrix3x2F::Rotation(labelTilt, D2D1::Point2F(labelLeft, labelTop)) *
                               boxTransform);
        wchar_t caption[24]{};
        const auto length = static_cast<UINT32>(swprintf_s(
            caption, std::size(caption), L"%02u  %+d°", static_cast<unsigned>(index + 1),
            static_cast<int>(std::lround(angle))));
        brush_->SetColor(D2D1::ColorF(0x101820, 0.95f));
        drawing_->DrawText(caption, length, text_.Get(), shadow, brush_.Get());
        brush_->SetColor(D2D1::ColorF(0xFFFFFF, 0.90f + 0.10f * vertical));
        drawing_->DrawText(caption, length, text_.Get(), label, brush_.Get());
    }
    DrawClock(timestamp90k);
    if (scene) DrawStreamNumber(scene->stream);
    else DrawTimings(timestamp90k);
    const HRESULT result = drawing_->EndDraw();
    drawing_->SetTarget(nullptr);
    CheckHr(result, "finish Direct2D overlay");
}

void GpuOverlay::DrawFigure(FigureKind kind, D2D1_POINT_2F center, const D2D1_RECT_F& region,
                            const D2D1::Matrix3x2F& rotation, uint32_t color, float fill,
                            float outline, float stroke) {
    drawing_->SetTransform(rotation);
    brush_->SetColor(D2D1::ColorF(color, fill));
    if (kind == FigureKind::Rectangle) {
        drawing_->FillRectangle(region, brush_.Get());
        brush_->SetColor(D2D1::ColorF(color, outline));
        drawing_->DrawRectangle(region, brush_.Get(), stroke);
    } else if (kind == FigureKind::Circle) {
        const float radius = std::min(region.right - center.x, region.bottom - center.y);
        const auto circle = D2D1::Ellipse(center, radius, radius);
        drawing_->FillEllipse(circle, brush_.Get());
        brush_->SetColor(D2D1::ColorF(color, outline));
        drawing_->DrawEllipse(circle, brush_.Get(), stroke);
    } else {
        const float radius = region.bottom - center.y;
        auto* polygon = kind == FigureKind::Triangle ? triangle_.Get() : star_.Get();
        drawing_->SetTransform(D2D1::Matrix3x2F::Scale(radius, radius) *
                               D2D1::Matrix3x2F::Translation(center.x, center.y) * rotation);
        drawing_->FillGeometry(polygon, brush_.Get());
        brush_->SetColor(D2D1::ColorF(color, outline));
        drawing_->DrawGeometry(polygon, brush_.Get(), stroke / radius);
    }
}

void GpuOverlay::DrawStreamNumber(unsigned stream) {
    wchar_t text[24]{};
    const auto length = static_cast<UINT32>(swprintf_s(text, std::size(text), L"STREAM %02u", stream));
    const float height = static_cast<float>(height_);
    const auto region = D2D1::RectF(height * 0.025f, height * 0.088f, height * 0.229f, height * 0.141f);
    brush_->SetColor(D2D1::ColorF(0x101820, 0.84f));
    drawing_->FillRectangle(region, brush_.Get());
    brush_->SetColor(D2D1::ColorF(0xFFFFFF));
    drawing_->DrawText(text, length, clock_.Get(), region, brush_.Get());
}

void GpuOverlay::DrawClock(int64_t timestamp90k) {
    const auto ms = (timestamp90k + 45) / 90;
    wchar_t timecode[32]{};
    const auto length = static_cast<UINT32>(swprintf_s(
        timecode, std::size(timecode), L"%02lld:%02lld:%02lld.%03lld",
        ms / 3600000, ms / 60000 % 60, ms / 1000 % 60, ms % 1000));
    const float width = static_cast<float>(width_);
    const float height = static_cast<float>(height_);
    const float right = width - height * 0.025f;
    const auto region = D2D1::RectF(right - height * 0.233f, height * 0.028f, right, height * 0.081f);
    drawing_->SetTransform(D2D1::Matrix3x2F::Identity());
    brush_->SetColor(D2D1::ColorF(0x101820, 0.75f));
    drawing_->FillRectangle(region, brush_.Get());
    brush_->SetColor(D2D1::ColorF(0xFFFFFF));
    drawing_->DrawText(timecode, length, clock_.Get(), region, brush_.Get());
}

void GpuOverlay::DrawTimings(int64_t timestamp90k) {
    using Line = std::array<wchar_t, 96>;
    std::array<Line, 4> lines{};
    const auto addTimes = [](Line& line, const wchar_t* label, const FrameStageTimes& times) {
        swprintf_s(line.data(), line.size(), L"%-8ls %10.2f %9.2f %10.2f %8.2f",
                   label, times.toBgra, times.draw, times.toNv12, times.total);
    };
    if (previousTimestamp_) {
        swprintf_s(lines[0].data(), lines[0].size(), L"PTS step  %6.2f ms",
                   static_cast<double>(timestamp90k - *previousTimestamp_) / 90.0);
        addTimes(lines[2], L"CPU prev", hostTiming_);
    } else {
        swprintf_s(lines[0].data(), lines[0].size(), L"PTS step      -- ms");
        swprintf_s(lines[2].data(), lines[2].size(), L"%-8ls %10ls %9ls %10ls %8ls",
                   L"CPU prev", L"--", L"--", L"--", L"--");
    }
    swprintf_s(lines[1].data(), lines[1].size(), L"%-8ls %10ls %9ls %10ls %8ls",
               L"ms", L"NV12>BGRA", L"Draw", L"BGRA>NV12", L"Total");
    if (const auto& gpu = gpuTiming_.Latest()) {
        wchar_t label[24]{};
        swprintf_s(label, std::size(label), L"GPU -%lluf",
                   static_cast<unsigned long long>(frame_ - gpu->frame));
        addTimes(lines[3], label, gpu->milliseconds);
    } else {
        swprintf_s(lines[3].data(), lines[3].size(), L"%-8ls %10ls %9ls %10ls %8ls",
                   L"GPU", L"--", L"--", L"--", L"--");
    }

    const float font = timingText_->GetFontSize();
    const float row = font * 1.22f;
    const float padding = font * 0.55f;
    const float bottom = static_cast<float>(height_) - font;
    const float top = bottom - row * static_cast<float>(lines.size()) - 2 * padding;
    const auto region = D2D1::RectF(font, top, font * 30.0f, bottom);
    brush_->SetColor(D2D1::ColorF(0x101820, 0.80f));
    drawing_->FillRectangle(region, brush_.Get());
    for (size_t index = 0; index < lines.size(); ++index) {
        const auto& line = lines[index];
        const float y = top + padding + row * static_cast<float>(index);
        const auto position = D2D1::RectF(region.left + padding, y, region.right - padding, y + row);
        brush_->SetColor(D2D1::ColorF(index < 2 ? 0xB6CDD9 : 0xFFFFFF));
        drawing_->DrawText(line.data(), static_cast<UINT32>(wcslen(line.data())),
                           timingText_.Get(), position, brush_.Get(), D2D1_DRAW_TEXT_OPTIONS_CLIP);
    }
}

void GpuOverlay::Render(ID3D11Texture2D* source, UINT slice, ID3D11Texture2D* destination,
                        int64_t timestamp90k, bool flush, const OverlayScene* scene, bool drawOverlay) {
    using Clock = std::chrono::steady_clock;
    using Milliseconds = std::chrono::duration<double, std::milli>;
    gpuTiming_.Poll();
    gpuTiming_.Begin(++frame_);

    const auto toBgraStart = Clock::now();
    Convert(toRgbEnumerator_.Get(), toRgb_.Get(), source, slice, rgb_.Get());
    const auto toBgraEnd = Clock::now();
    gpuTiming_.Mark(GpuFrameTiming::Point::ConvertedToBgra);

    const auto drawStart = Clock::now();
    if (drawOverlay) Draw(timestamp90k, scene);
    const auto drawEnd = Clock::now();
    gpuTiming_.Mark(GpuFrameTiming::Point::Drawn);

    const auto toNv12Start = Clock::now();
    Convert(toNv12Enumerator_.Get(), toNv12_.Get(), rgb_.Get(), 0, destination);
    const auto toNv12End = Clock::now();
    gpuTiming_.End(flush);

    hostTiming_.toBgra = Milliseconds(toBgraEnd - toBgraStart).count();
    hostTiming_.draw = Milliseconds(drawEnd - drawStart).count();
    hostTiming_.toNv12 = Milliseconds(toNv12End - toNv12Start).count();
    previousTimestamp_ = timestamp90k;
}

void GpuOverlay::CompleteHostFrame(double decodedToEncoderMs) {
    hostTiming_.total = decodedToEncoderMs;
}
