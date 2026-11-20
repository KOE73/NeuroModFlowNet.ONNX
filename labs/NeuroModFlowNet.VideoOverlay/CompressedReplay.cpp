#include "CompressedReplay.h"

#include <algorithm>
#include <numeric>

using namespace VideoCodecs;

namespace {
bool IsIdr(const AVPacket& packet, unsigned lengthBytes, bool hevc) {
    const auto* cursor = packet.data;
    size_t remaining = packet.size;
    while (remaining >= lengthBytes) {
        size_t size = 0;
        for (unsigned index = 0; index < lengthBytes; ++index)
            size = (size << 8) | cursor[index];
        cursor += lengthBytes;
        remaining -= lengthBytes;
        if (size == 0 || size > remaining) throw std::runtime_error("invalid NAL length in benchmark source");
        const unsigned type = hevc ? (cursor[0] >> 1) & 63u : cursor[0] & 31u;
        if (hevc ? (type == 19 || type == 20) : type == 5) return true;
        cursor += size;
        remaining -= size;
    }
    return false;
}
}

CompressedReplay::CompressedReplay(AVFormatContext* input, const AVStream* stream) {
    const auto* parameters = stream->codecpar;
    const bool hevc = parameters->codec_id == AV_CODEC_ID_HEVC;
    if ((!hevc && parameters->codec_id != AV_CODEC_ID_H264) ||
        parameters->extradata_size < (hevc ? 23 : 7) || parameters->extradata[0] != 1)
        throw std::runtime_error("full benchmark requires a H.264 or HEVC MP4 source");
    const unsigned lengthBytes = 1u + (parameters->extradata[hevc ? 21 : 4] & 3u);
    constexpr size_t minimumFrames = 100;
    bool reachedBoundary = false;
    Packet packet(Require(av_packet_alloc(), "allocate replay input packet"));
    for (;;) {
        const int result = av_read_frame(input, packet.get());
        if (result == AVERROR_EOF) break;
        CheckAv(result, "load benchmark packet");
        if (packet->stream_index == stream->index) {
            const bool idr = (packet->flags & AV_PKT_FLAG_KEY) && IsIdr(*packet, lengthBytes, hevc);
            if (idr && packets_.size() >= minimumFrames) {
                reachedBoundary = true;
                break;
            }
            if (idr || !packets_.empty()) {
                if (packet->pts == AV_NOPTS_VALUE)
                    throw std::runtime_error("benchmark source has video packets without PTS");
                bytes_ += packet->size;
                packets_.emplace_back(Require(av_packet_clone(packet.get()), "retain compressed frame in RAM"));
            }
        }
        av_packet_unref(packet.get());
    }
    if (!reachedBoundary)
        throw std::runtime_error("source must contain at least 100 frames bounded by IDR frames");

    // Keep decode order for submission and presentation order for matching encoded output.
    inputByPresentation_.resize(packets_.size());
    std::iota(inputByPresentation_.begin(), inputByPresentation_.end(), size_t{0});
    std::sort(inputByPresentation_.begin(), inputByPresentation_.end(), [&](size_t left, size_t right) {
        return packets_[left]->pts < packets_[right]->pts;
    });
    if (inputByPresentation_.front() != 0)
        throw std::runtime_error("replay requires the first IDR to be first in presentation order");
    presentationByInput_.resize(packets_.size());
    for (size_t index = 0; index < packets_.size(); ++index) {
        const size_t nativeIndex = inputByPresentation_[index];
        if (index > 0 && packets_[nativeIndex]->pts == packets_[inputByPresentation_[index - 1]]->pts)
            throw std::runtime_error("benchmark source contains duplicate PTS");
        presentationByInput_[nativeIndex] = index;
        reorderDepth_ = std::max(reorderDepth_, static_cast<int64_t>(nativeIndex) - static_cast<int64_t>(index));
    }
}

void CompressedReplay::Read(size_t index, AVPacket* packet) const {
    const auto slot = index % packets_.size();
    CheckAv(av_packet_ref(packet, packets_[slot].get()), "reference in-memory compressed frame");
    const size_t displayIndex = index - slot + presentationByInput_[slot];
    packet->pts = static_cast<int64_t>(displayIndex) * frameDuration;
    packet->dts = (static_cast<int64_t>(index) - reorderDepth_) * frameDuration;
    packet->duration = frameDuration;
    packet->pos = -1;
}

size_t CompressedReplay::InputIndex(int64_t pts) const {
    if (pts < 0 || pts % frameDuration != 0)
        throw std::runtime_error("unexpected frame PTS in full benchmark");
    const size_t displayIndex = static_cast<size_t>(pts / frameDuration);
    const size_t slot = displayIndex % packets_.size();
    return displayIndex - slot + inputByPresentation_[slot];
}
