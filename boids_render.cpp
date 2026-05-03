#include "boids_render.hpp"
#include <GL/freeglut.h>
#include <cmath>

namespace BoidsRender {

void setOrtho2D(int W, int H){
    glViewport(0,0,W,H);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, W, 0, H);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

static inline void drawOne(const float2h& p, const float2h& v){
    float vx=v.x, vy=v.y;
    float l = std::sqrt(vx*vx + vy*vy);
    if(l < 1e-6f){ vx=1.0f; vy=0.0f; l=1.0f; }
    vx/=l; vy/=l;

    float size = 7.5f;
    float fx = vx * size;
    float fy = vy * size;
    float rx = -vy * (size * 0.55f);
    float ry =  vx * (size * 0.55f);

    float p0x = p.x + fx,            p0y = p.y + fy;
    float p1x = p.x - fx*0.65f + rx, p1y = p.y - fy*0.65f + ry;
    float p2x = p.x - fx*0.65f - rx, p2y = p.y - fy*0.65f - ry;

    glVertex2f(p0x,p0y);
    glVertex2f(p1x,p1y);
    glVertex2f(p2x,p2y);
}

void drawBoidsTriangles(const float2h* pos, const float2h* vel, int N){
    glBegin(GL_TRIANGLES);
    for(int i=0;i<N;i++) drawOne(pos[i], vel[i]);
    glEnd();
}

} // namespace