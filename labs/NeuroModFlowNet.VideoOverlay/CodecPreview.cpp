#include "CodecPreview.h"
#include "CompressedReplay.h"

#include <algorithm>
#include <iostream>

using namespace VideoCodecs;

namespace {
constexpr size_t noChannel = static_cast<size_t>(-1);
constexpr size_t maxQueuedPackets = 25;

bool IsRandomAccessPacket(const AVPacket& packet) {
    if (!(packet.flags & AV_PKT_FLAG_KEY)) return false;
    bool sps = false;
    bool pps = false;
    bool idr = false;
    for (int index = 0; index + 3 < packet.size; ++index) {
        const auto* data = packet.data + index;
        if (data[0] != 0 || data[1] != 0 || data[2] != 1) continue;
        const unsigned type = data[3] & 31u;
        sps = sps || type == 7;
        pps = pps || type == 8;
        idr = idr || type == 5;
    }
    return sps && pps && idr;
}
}

CodecPreview::CodecPreview(const std::string& url, const AVCodecContext* encoder, size_t channels)
    : channels_(channels) {
    AVFormatContext* format = nullptr;
    CheckAv(avformat_alloc_output_context2(&format, nullptr, "mpegts", url.c_str()), "create preview muxer");
    mux_.reset(format);
    video_ = Require(avformat_new_stream(format, nullptr), "create preview stream");
    CheckAv(avcodec_parameters_from_context(video_->codecpar, encoder), "copy preview codec parameters");
    video_->time_base = encoder->time_base;
    video_->avg_frame_rate = encoder->framerate;
    Dictionary ioOptions;
    ioOptions.Set("rw_timeout", "1000000");
    CheckAv(avio_open2(&format->pb, url.c_str(), AVIO_FLAG_WRITE, nullptr, &ioOptions.value),
            "open UDP preview");
    Dictionary muxOptions;
    muxOptions.Set("flush_packets", "1");
    CheckAv(avformat_write_header(format, &muxOptions.value), "write preview header");
    writer_ = std::thread(&CodecPreview::WritePackets, this);
}

CodecPreview::~CodecPreview() {
    {
        std::lock_guard lock(mutex_);
        closing_ = true;
        queue_.clear();
    }
    ready_.notify_one();
    writer_.join();
}

void CodecPreview::Select(size_t channel) {
    if (channel >= channels_) throw std::runtime_error("invalid preview encoder");
    requested_ = channel;
}

void CodecPreview::Accept(size_t channel, const AVPacket& packet) {
    if (channel != active_ && channel != requested_) return;
    std::unique_lock lock(mutex_, std::try_to_lock);
    if (!lock) {
        if (channel == active_) active_ = noChannel;
        return;
    }
    if (closing_) return;
    if (queue_.size() == maxQueuedPackets) {
        queue_.clear();
        active_ = noChannel;
    }
    if (channel != active_) {
        if (channel != requested_ || !IsRandomAccessPacket(packet)) return;
        active_ = channel;
        std::cout << "Preview switched on IDR to stream " << active_ + 1 << std::endl;
    }
    Packet copy(Require(av_packet_clone(&packet), "retain preview packet"));
    const int64_t pts = previousPts_ == AV_NOPTS_VALUE ? copy->pts :
                        std::max(copy->pts, previousPts_ + CompressedReplay::frameDuration);
    copy->pts = copy->dts = previousPts_ = pts;
    av_packet_rescale_ts(copy.get(), videoTimeBase, video_->time_base);
    copy->stream_index = video_->index;
    queue_.push_back(std::move(copy));
    lock.unlock();
    ready_.notify_one();
}

void CodecPreview::WritePackets() {
    try {
        for (;;) {
            Packet packet;
            {
                std::unique_lock lock(mutex_);
                ready_.wait(lock, [&] { return closing_ || !queue_.empty(); });
                if (closing_) break;
                packet = std::move(queue_.front());
                queue_.pop_front();
            }
            CheckAv(av_interleaved_write_frame(mux_.get(), packet.get()), "write preview packet");
        }
        CheckAv(av_write_trailer(mux_.get()), "finish preview");
    } catch (const std::exception& error) {
        std::cerr << "Preview: " << error.what() << '\n';
        std::lock_guard lock(mutex_);
        closing_ = true;
        queue_.clear();
    }
}
