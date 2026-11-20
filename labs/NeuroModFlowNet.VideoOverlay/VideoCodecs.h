#pragma once

#include <d3d11.h>
#include <wrl/client.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
}

#include <memory>
#include <stdexcept>
#include <string>

namespace VideoCodecs {

inline constexpr AVRational videoTimeBase{1, 90000};

void CheckAv(int result, const char* operation);

template <typename T> T* Require(T* value, const char* operation) {
    if (!value) throw std::runtime_error(operation);
    return value;
}

template <typename T, void (*Release)(T**)> struct AvDeleter {
    void operator()(T* value) const { Release(&value); }
};

using Codec = std::unique_ptr<AVCodecContext, AvDeleter<AVCodecContext, avcodec_free_context>>;
using Input = std::unique_ptr<AVFormatContext, AvDeleter<AVFormatContext, avformat_close_input>>;
using Buffer = std::unique_ptr<AVBufferRef, AvDeleter<AVBufferRef, av_buffer_unref>>;
using Frame = std::unique_ptr<AVFrame, AvDeleter<AVFrame, av_frame_free>>;
using Packet = std::unique_ptr<AVPacket, AvDeleter<AVPacket, av_packet_free>>;

struct Dictionary {
    AVDictionary* value{};
    ~Dictionary() { av_dict_free(&value); }
    void Set(const char* key, const std::string& text) {
        CheckAv(av_dict_set(&value, key, text.c_str(), 0), key);
    }
};

struct MuxDeleter {
    void operator()(AVFormatContext* mux) const {
        if (!(mux->oformat->flags & AVFMT_NOFILE)) avio_closep(&mux->pb);
        avformat_free_context(mux);
    }
};

Codec CreateDecoder(const AVCodecParameters* parameters, AVRational packetTimeBase, AVBufferRef* hardware,
                    int* explicitSurfaceCount = nullptr);
Microsoft::WRL::ComPtr<ID3D11Device> CreateVideoDevice(IDXGIAdapter* adapter);
Buffer BindDevice(ID3D11Device* device);
ID3D11Texture2D* CheckGpuFrame(const AVFrame& frame, ID3D11Device* device);
Buffer CreateEncoderPool(AVBufferRef* hardware, int width, int height);
bool IsBt709(const AVFrame& frame);
Codec CreateEncoder(const AVFrame& source, AVRational rate, AVBufferRef* pool, int bitrate,
                    bool globalHeader, bool forcedIdr = false);

}
