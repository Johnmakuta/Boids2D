#pragma once
#include <cuda_runtime.h>

struct BoidParams {
    int   N = 250;
    int   W = 1000;
    int   H = 700;

    float dtFixed     = 1.0f/120.0f;

    float maxSpeed    = 160.0f;
    float maxForce    = 220.0f;
    float neighborRad = 70.0f;
    float sepRad      = 26.0f;

    float wAlign      = 1.0f;
    float wCohesion   = 0.85f;
    float wSeparation = 1.55f;

    bool  wrapEdges   = true;
};

// simple host vector type for positions/velocities used by renderer
struct float2h { float x, y; };

class BoidsCudaSim {
public:
    BoidsCudaSim() = default;
    ~BoidsCudaSim();

    BoidsCudaSim(const BoidsCudaSim&) = delete;
    BoidsCudaSim& operator=(const BoidsCudaSim&) = delete;

    void init(const BoidParams& p);
    void shutdown();

    // Randomize boids on GPU
    void resetRandom(unsigned seed);

    // Update W/H (for edge wrap/bounce)
    void setViewport(int W, int H);

    // Launch kernel(s) and start async D2H stage (streams+event)
    void simulateAndStageAsync(float dt);

    // If the staged copy is finished, returns true and outputs pointers to CPU-stable data.
    // Pointers remain valid until next successful pollCopyReady().
    bool pollCopyReady(const float2h*& outPos, const float2h*& outVel) const;

    const BoidParams& params() const { return m_p; }

private:
    BoidParams m_p{};

    // device ping-pong
    float2 *d_posA=nullptr, *d_posB=nullptr;
    float2 *d_velA=nullptr, *d_velB=nullptr;
    bool usingAasInput = true;

    // streams/events
    cudaStream_t simStream  = nullptr;
    cudaStream_t copyStream = nullptr;
    cudaEvent_t  simDoneEvent = nullptr;

    // double-buffer pinned staging
    float2h* h_posPinned[2] = {nullptr,nullptr};
    float2h* h_velPinned[2] = {nullptr,nullptr};
    int hostBufIndex = 0;

    bool inited = false;
};