#include "VideoCodecs.h"

extern "C" {
#include <libavutil/hwcontext_d3d11va.h>
#include <libavutil/opt.h>
}

#include <wrl/client.h>
#include <algorithm>

namespace VideoCodecs {

Microsoft::WRL::ComPtr<ID3D11Device> CreateVideoDevice(IDXGIAdapter* adapter) {
    Microsoft::WRL::ComPtr<ID3D11Device> device;
    constexpr UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT | D3D11_CREATE_DEVICE_VIDEO_SUPPORT;
    const auto status = D3D11CreateDevice(adapter, D3D_DRIVER_TYPE_UNKNOWN, nullptr, flags, nullptr, 0,
                                         D3D11_SDK_VERSION, &device, nullptr, nullptr);
    if (FAILED(status)) throw std::runtime_error("create NVIDIA D3D11 device: " + std::to_string(status));
    return device;
}

Buffer BindDevice(ID3D11Device* device) {
    Buffer hardware(Require(av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_D3D11VA), "allocate D3D11 device context"));
    auto* context = reinterpret_cast<AVHWDeviceContext*>(hardware->data);
    auto* native = static_cast<AVD3D11VADeviceContext*>(context->hwctx);
    device->AddRef();
    native->device = device;
    CheckAv(av_hwdevice_ctx_init(hardware.get()), "initialize FFmpeg D3D11 device");
    return hardware;
}

void CheckAv(int result, const char* operation) {
    if (result < 0) {
        char description[AV_ERROR_MAX_STRING_SIZE]{};
        av_strerror(result, description, sizeof(description));
        throw std::runtime_error(std::string(operation) + ": " + description);
    }
}

namespace {
AVPixelFormat HardwareFormat(AVCodecContext* decoder, const AVPixelFormat* formats) {
    for (auto* format = formats; *format != AV_PIX_FMT_NONE; ++format) {
        if (*format != AV_PIX_FMT_D3D11) continue;
        if (!decoder->opaque) return *format;

        AVBufferRef* nativePool{};
        int status = avcodec_get_hw_frames_parameters(decoder, decoder->hw_device_ctx, *format, &nativePool);
        Buffer pool(nativePool);
        if (status >= 0) {
            auto* frames = reinterpret_cast<AVHWFramesContext*>(pool->data);
            frames->initial_pool_size = *static_cast<const int*>(decoder->opaque);
            status = av_hwframe_ctx_init(pool.get());
        }
        if (status < 0) {
            char description[AV_ERROR_MAX_STRING_SIZE]{};
            av_strerror(status, description, sizeof(description));
            av_log(decoder, AV_LOG_ERROR, "Cannot allocate the requested D3D11 decode surface pool: %s\n", description);
            return AV_PIX_FMT_NONE;
        }
        decoder->hw_frames_ctx = pool.release();
        return *format;
    }
    return AV_PIX_FMT_NONE;
}
}

Codec CreateDecoder(const AVCodecParameters* parameters, AVRational packetTimeBase, AVBufferRef* hardware,
                    int* explicitSurfaceCount) {
    const AVCodecID codecId = parameters->codec_id;
    if (codecId != AV_CODEC_ID_H264 && codecId != AV_CODEC_ID_HEVC)
        throw std::runtime_error("selected input must provide H.264 or HEVC video");
    const auto* codec = Require(avcodec_find_decoder(codecId), "video decoder unavailable");
    Codec decoder(Require(avcodec_alloc_context3(codec), "allocate video decoder"));
    CheckAv(avcodec_parameters_to_context(decoder.get(), parameters), "configure video decoder");
    decoder->get_format = HardwareFormat;
    decoder->opaque = explicitSurfaceCount;
    decoder->hw_device_ctx = Require(av_buffer_ref(hardware), "retain decode device");
    decoder->pkt_timebase = packetTimeBase;
    decoder->thread_count = 1;
    CheckAv(avcodec_open2(decoder.get(), codec, nullptr), "open D3D11VA decoder");
    return decoder;
}

ID3D11Texture2D* CheckGpuFrame(const AVFrame& frame, ID3D11Device* device) {
    if (frame.format != AV_PIX_FMT_D3D11 || !frame.data[0] || !frame.hw_frames_ctx)
        throw std::runtime_error("expected a D3D11 GPU frame");
    auto* context = reinterpret_cast<const AVHWFramesContext*>(frame.hw_frames_ctx->data);
    if (context->sw_format != AV_PIX_FMT_NV12)
        throw std::runtime_error("expected an 8-bit NV12 surface");
    auto* texture = reinterpret_cast<ID3D11Texture2D*>(frame.data[0]);
    D3D11_TEXTURE2D_DESC description{};
    texture->GetDesc(&description);
    Microsoft::WRL::ComPtr<ID3D11Device> owner;
    texture->GetDevice(&owner);
    if (owner.Get() != device || description.Usage != D3D11_USAGE_DEFAULT || description.CPUAccessFlags != 0 ||
        description.Format != DXGI_FORMAT_NV12 || reinterpret_cast<uintptr_t>(frame.data[1]) >= description.ArraySize)
        throw std::runtime_error("expected an NV12 texture on the shared GPU device with no CPU access");
    return texture;
}

Buffer CreateEncoderPool(AVBufferRef* hardware, int width, int height) {
    Buffer pool(Require(av_hwframe_ctx_alloc(hardware), "allocate NVENC surface pool"));
    auto* context = reinterpret_cast<AVHWFramesContext*>(pool->data);
    context->format = AV_PIX_FMT_D3D11;
    context->sw_format = AV_PIX_FMT_NV12;
    context->width = width;
    context->height = height;
    static_cast<AVD3D11VAFramesContext*>(context->hwctx)->BindFlags = D3D11_BIND_RENDER_TARGET;
    CheckAv(av_hwframe_ctx_init(pool.get()), "initialize NVENC surface pool");
    return pool;
}

bool IsBt709(const AVFrame& frame) {
    switch (frame.colorspace) {
    case AVCOL_SPC_BT709: return true;
    case AVCOL_SPC_BT470BG:
    case AVCOL_SPC_SMPTE170M: return false;
    case AVCOL_SPC_UNSPECIFIED: return frame.height >= 720;
    default: throw std::runtime_error("input matrix must be BT.601, BT.709 or unspecified");
    }
}

Codec CreateEncoder(const AVFrame& source, AVRational rate, AVBufferRef* pool, int bitrate,
                    bool globalHeader, bool forcedIdr) {
    const auto* codec = Require(avcodec_find_encoder_by_name("h264_nvenc"), "h264_nvenc unavailable");
    Codec encoder(Require(avcodec_alloc_context3(codec), "allocate NVENC encoder"));
    encoder->width = source.width;
    encoder->height = source.height;
    encoder->pix_fmt = AV_PIX_FMT_D3D11;
    encoder->hw_frames_ctx = Require(av_buffer_ref(pool), "retain NVENC surface pool");
    encoder->time_base = videoTimeBase;
    encoder->framerate = rate;
    encoder->sample_aspect_ratio = source.sample_aspect_ratio;
    encoder->bit_rate = bitrate;
    encoder->gop_size = std::max(1, static_cast<int>(av_q2d(rate) + 0.5));
    encoder->max_b_frames = 0;
    encoder->color_range = AVCOL_RANGE_MPEG;
    encoder->colorspace = IsBt709(source) ? AVCOL_SPC_BT709 : AVCOL_SPC_SMPTE170M;
    encoder->color_primaries = source.color_primaries;
    encoder->color_trc = source.color_trc;
    encoder->chroma_sample_location = AVCHROMA_LOC_LEFT;
    if (globalHeader) encoder->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    CheckAv(av_opt_set(encoder->priv_data, "preset", "p1", 0), "set NVENC preset");
    CheckAv(av_opt_set(encoder->priv_data, "tune", "ull", 0), "set NVENC tuning");
    CheckAv(av_opt_set_int(encoder->priv_data, "delay", 0, 0), "set NVENC output delay");
    CheckAv(av_opt_set_int(encoder->priv_data, "zerolatency", 1, 0), "set NVENC zero-latency mode");
    if (forcedIdr)
        CheckAv(av_opt_set_int(encoder->priv_data, "forced-idr", 1, 0), "set NVENC forced IDR mode");
    CheckAv(avcodec_open2(encoder.get(), codec, nullptr), "open NVIDIA H.264 encoder");
    return encoder;
}

}
