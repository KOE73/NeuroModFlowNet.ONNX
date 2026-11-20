#pragma once

#include <d3d11.h>
#include <span>

void RunOverlayBenchmark(ID3D11Device* device, ID3D11Texture2D* decoded, UINT slice, bool bt709,
                         bool fullRange, std::span<const int> streamCounts, int seconds,
                         bool staggered, int (*shouldStop)(void*));
