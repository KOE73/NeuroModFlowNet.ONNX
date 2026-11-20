#pragma once

#include "VideoCodecs.h"

#include <chrono>
#include <span>
#include <string>

enum class CodecBenchmarkStage { Decode, Convert, Draw };

struct CodecBenchmarkOutput {
    int width{};
    int height{};
    CodecBenchmarkStage stage{CodecBenchmarkStage::Draw};
    int decodeSurfaces{};
    int firstMatrixOutput{};
};

void RunCodecBenchmark(ID3D11Device* device, AVBufferRef* hardware, AVFormatContext* input, AVStream* stream,
                       std::span<const int> streamCounts, std::chrono::milliseconds window, bool staggered,
                       int bitrate, const std::string& previewUrl, int matrixOutputs, CodecBenchmarkOutput output,
                       int (*shouldStop)(void*));
