#include "boids_sim.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>

#define CUDA_CHECK(x) do {                                  \
    cudaError_t err = (x);                                   \
    if (err != cudaSuccess) {                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(err));\
        std::exit(1);                                        \
    }                                                        \
} while(0)

__device__ __forceinline__ float2 operator+(float2 a, float2 b){ return {a.x+b.x, a.y+b.y}; }
__device__ __forceinline__ float2 operator-(float2 a, float2 b){ return {a.x-b.x, a.y-b.y}; }
__device__ __forceinline__ float2 operator*(float2 a, float s){ return {a.x*s, a.y*s}; }
__device__ __forceinline__ float2 operator/(float2 a, float s){ return {a.x/s, a.y/s}; }
__device__ __forceinline__ void operator+=(float2 &a, float2 b){ a.x+=b.x; a.y+=b.y; }

__device__ __forceinline__ float dot2(float2 a, float2 b){ return a.x*b.x + a.y*b.y; }
__device__ __forceinline__ float len2(float2 v){ return dot2(v,v); }
__device__ __forceinline__ float len1(float2 v){ return sqrtf(len2(v)); }

__device__ __forceinline__ float2 normalized(float2 v){
    float l = len1(v);
    if(l <= 1e-6f) {
        return {0,0};
    }
    return v / l;
}

__device__ __forceinline__ float2 limitMag(float2 v, float maxMag){
    float l2 = len2(v);
    float mm2 = maxMag*maxMag;
    if(l2 <= mm2) {
        return v;
    }
    float l = sqrtf(l2);
    return v * (maxMag / (l + 1e-6f));
}

__device__ __forceinline__ float2 steerTowards(float2 desiredVel, float2 currentVel, float maxForce){
    float2 steer = desiredVel - currentVel;
    return limitMag(steer, maxForce);
}

__global__ void boidsStepKernel(
    const float2* __restrict__ posIn,
    const float2* __restrict__ velIn,
    float2* __restrict__ posOut,
    float2* __restrict__ velOut,
    int N, int W, int H,
    float dt,
    float maxSpeed, float maxForce,
    float neighborRad, float sepRad,
    float wAlign, float wCohesion, float wSeparation,
    int wrapEdgesInt
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) {
        return;
    }

    float2 meP = posIn[i];
    float2 meV = velIn[i];

    float2 sumVel = {0,0};
    float2 sumPos = {0,0};
    float2 sep    = {0,0};
    int cnt = 0, cntSep = 0;

    float nR2 = neighborRad * neighborRad;
    float sR2 = sepRad * sepRad;

    for(int j=0; j<N; j++) {
        if(j==i) {
            continue;
        }

        float2 d = posIn[j] - meP;

        if(wrapEdgesInt){
            if(d.x >  W*0.5f) {
                d.x -= W;
            }
            if(d.x < -W*0.5f) {
                d.x += W;
            }
            if(d.y >  H*0.5f) {
                d.y -= H;
            }
            if(d.y < -H*0.5f) {
                d.y += H;
            }
        }

        float dist2 = d.x*d.x + d.y*d.y;
        if(dist2 < nR2) {
            sumVel += velIn[j];
            sumPos += (meP + d);
            cnt++;

            if(dist2 < sR2 && dist2 > 1e-6f) {
                float invd = rsqrtf(dist2);
                sep += (d * (-invd));
                cntSep++;
            }
        }
    }

    float2 acc = {0,0};

    if(cnt > 0){
        float2 avgVel = sumVel / (float)cnt;
        float2 desiredA = normalized(avgVel) * maxSpeed;
        acc += steerTowards(desiredA, meV, maxForce) * wAlign;

        float2 center = sumPos / (float)cnt;
        float2 toCenter = center - meP;
        float2 desiredC = normalized(toCenter) * maxSpeed;
        acc += steerTowards(desiredC, meV, maxForce) * wCohesion;
    }

    if(cntSep > 0){
        float2 desiredS = normalized(sep) * maxSpeed;
        acc += steerTowards(desiredS, meV, maxForce) * wSeparation;
    }

    float2 newV = limitMag(meV + acc * dt, maxSpeed);
    float2 newP = meP + newV * dt;

    if(wrapEdgesInt){
        if(newP.x < 0) {
            newP.x += W;
        }
        if(newP.x >= W) {
            newP.x -= W;
        }
        if(newP.y < 0) {
            newP.y += H;
        }
        if(newP.y >= H) {
            newP.y -= H;
        }
    } else {
        if(newP.x < 0) { 
            newP.x = 0;        
            newV.x *= -1; 
        }
        if(newP.x > W) { 
            newP.x = (float)W; 
            newV.x *= -1; 
        }
        if(newP.y < 0) { 
            newP.y = 0;        
            newV.y *= -1; 
        }
        if(newP.y > H) { 
            newP.y = (float)H; 
            newV.y *= -1; 
        }
    }

    posOut[i] = newP;
    velOut[i] = newV;
}

BoidsCudaSim::~BoidsCudaSim(){
    shutdown();
}

void BoidsCudaSim::init(const BoidParams& p){
    shutdown();
    m_p = p;

    CUDA_CHECK(cudaMalloc(&d_posA, m_p.N*sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&d_posB, m_p.N*sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&d_velA, m_p.N*sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&d_velB, m_p.N*sizeof(float2)));

    CUDA_CHECK(cudaStreamCreate(&simStream));
    CUDA_CHECK(cudaStreamCreate(&copyStream));
    CUDA_CHECK(cudaEventCreateWithFlags(&simDoneEvent, cudaEventDisableTiming));

    for(int i=0;i<2;i++){
        CUDA_CHECK(cudaMallocHost(&h_posPinned[i], m_p.N*sizeof(float2h)));
        CUDA_CHECK(cudaMallocHost(&h_velPinned[i], m_p.N*sizeof(float2h)));
    }

    usingAasInput = true;
    hostBufIndex = 0;
    inited = true;
}

void BoidsCudaSim::shutdown(){
    if(!inited) return;

    if(d_posA) cudaFree(d_posA);
    if(d_posB) cudaFree(d_posB);
    if(d_velA) cudaFree(d_velA);
    if(d_velB) cudaFree(d_velB);
    d_posA=d_posB=d_velA=d_velB=nullptr;

    if(simDoneEvent) cudaEventDestroy(simDoneEvent);
    if(simStream) cudaStreamDestroy(simStream);
    if(copyStream) cudaStreamDestroy(copyStream);
    simDoneEvent=nullptr; simStream=nullptr; copyStream=nullptr;

    for(int i=0;i<2;i++){
        if(h_posPinned[i]) {
            cudaFreeHost(h_posPinned[i]);
        }
        if(h_velPinned[i]) {
            cudaFreeHost(h_velPinned[i]);
        }
        h_posPinned[i]=h_velPinned[i]=nullptr;
    }

    inited = false;
}

static inline float frand(unsigned& state, float a, float b){
    // simple LCG for reproducible host-side random
    state = 1664525u * state + 1013904223u;
    float u = (state & 0x00FFFFFF) / float(0x01000000);
    return a + (b-a)*u;
}

void BoidsCudaSim::resetRandom(unsigned seed){
    std::vector<float2> pos(m_p.N), vel(m_p.N);
    unsigned st = seed ? seed : 1u;

    for(int i=0; i<m_p.N; i++) {
        pos[i] = { frand(st, 0.0f, (float)m_p.W), frand(st, 0.0f, (float)m_p.H) };
        float a = frand(st, 0.0f, 6.2831853f);
        float s = frand(st, 40.0f, m_p.maxSpeed);
        vel[i] = { std::cos(a)*s, std::sin(a)*s };
    }

    CUDA_CHECK(cudaMemcpy(d_posA, pos.data(), m_p.N*sizeof(float2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_velA, vel.data(), m_p.N*sizeof(float2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_posB, pos.data(), m_p.N*sizeof(float2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_velB, vel.data(), m_p.N*sizeof(float2), cudaMemcpyHostToDevice));

    // also prime pinned buffers so renderer has something immediately
    for(int k=0;k<2;k++){
        for(int i=0; i<m_p.N; i++) {
            h_posPinned[k][i] = {pos[i].x, pos[i].y};
            h_velPinned[k][i] = {vel[i].x, vel[i].y};
        }
    }
}

void BoidsCudaSim::setViewport(int W, int H){
    m_p.W = std::max(1, W);
    m_p.H = std::max(1, H);
}

void BoidsCudaSim::simulateAndStageAsync(float dt){
    float2 *posIn  = usingAasInput ? d_posA : d_posB;
    float2 *velIn  = usingAasInput ? d_velA : d_velB;
    float2 *posOut = usingAasInput ? d_posB : d_posA;
    float2 *velOut = usingAasInput ? d_velB : d_velA;

    dim3 block(256);
    dim3 grid((m_p.N + block.x - 1) / block.x);

    boidsStepKernel<<<grid, block, 0, simStream>>>(
        posIn, velIn, posOut, velOut,
        m_p.N, m_p.W, m_p.H,
        dt,
        m_p.maxSpeed, m_p.maxForce,
        m_p.neighborRad, m_p.sepRad,
        m_p.wAlign, m_p.wCohesion, m_p.wSeparation,
        m_p.wrapEdges ? 1 : 0
    );
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(simDoneEvent, simStream));

    int nextHost = hostBufIndex ^ 1;
    CUDA_CHECK(cudaStreamWaitEvent(copyStream, simDoneEvent, 0));
    CUDA_CHECK(cudaMemcpyAsync(h_posPinned[nextHost], posOut, m_p.N*sizeof(float2h), cudaMemcpyDeviceToHost, copyStream));
    CUDA_CHECK(cudaMemcpyAsync(h_velPinned[nextHost], velOut, m_p.N*sizeof(float2h), cudaMemcpyDeviceToHost, copyStream));

    usingAasInput = !usingAasInput;
    hostBufIndex = nextHost;
}

bool BoidsCudaSim::pollCopyReady(const float2h*& outPos, const float2h*& outVel) const{
    if(cudaStreamQuery(copyStream) != cudaSuccess) {
        return false;
    }
    outPos = h_posPinned[hostBufIndex];
    outVel = h_velPinned[hostBufIndex];
    return true;
}