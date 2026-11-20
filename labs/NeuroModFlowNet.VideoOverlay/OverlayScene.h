#pragma once

#include <array>
#include <cstdint>

enum class FigureKind { Rectangle, Circle, Triangle, Star };

struct OverlayFigure {
    FigureKind kind{};
    uint32_t color{};
    float angle{};
    float phase{};
};

struct OverlayScene {
    unsigned stream{};
    std::array<OverlayFigure, 10> figures;
};
