#pragma once

#include "VideoCodecs.h"

#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

class CodecPreview {
public:
    CodecPreview(const std::string& url, const AVCodecContext* encoder, size_t channels);
    ~CodecPreview();
    void Select(size_t channel);
    void Accept(size_t channel, const AVPacket& packet);

private:
    void WritePackets();

    std::unique_ptr<AVFormatContext, VideoCodecs::MuxDeleter> mux_;
    AVStream* video_{};
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<VideoCodecs::Packet> queue_;
    std::thread writer_;
    size_t channels_{};
    size_t requested_{};
    size_t active_{static_cast<size_t>(-1)};
    int64_t previousPts_{AV_NOPTS_VALUE};
    bool closing_{};
};
