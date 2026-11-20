#pragma once

#include "CodecBenchmark.h"
#include "CompressedReplay.h"
#include "OverlayScene.h"

void RunParallelCodecBenchmark(ID3D11Device* device, const AVCodecParameters* parameters,
                               const AVFrame& source, const AVFrame& output, const CompressedReplay& replay,
                               std::span<const OverlayScene> scenes, std::span<const int> streamCounts,
                               std::chrono::milliseconds window, bool staggered, int bitrate, int matrixOutputs,
                               CodecBenchmarkStage stage, int decodeSurfaces, int firstOutput, int (*shouldStop)(void*));
