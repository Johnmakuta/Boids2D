#pragma once
#include "boids_sim.hpp"

namespace BoidsRender {
    void setOrtho2D(int W, int H);
    void drawBoidsTriangles(const float2h* pos, const float2h* vel, int N);
}