#pragma once

#include "VideoCodecs.h"

#include <cstddef>
#include <cstdint>
#include <vector>

class CompressedReplay {
public:
    CompressedReplay(AVFormatContext* input, const AVStream* stream);
    void Read(size_t index, AVPacket* packet) const;
    size_t InputIndex(int64_t pts) const;
    size_t Size() const { return packets_.size(); }
    size_t Bytes() const { return bytes_; }
    bool HasReorderedPictures() const { return reorderDepth_ != 0; }

    static constexpr int framesPerSecond = 25;
    static constexpr int64_t frameDuration = 90000 / framesPerSecond;

private:
    std::vector<VideoCodecs::Packet> packets_;
    std::vector<size_t> presentationByInput_;
    std::vector<size_t> inputByPresentation_;
    size_t bytes_{};
    int64_t reorderDepth_{};
};
