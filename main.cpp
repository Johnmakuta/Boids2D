#include <GL/freeglut.h>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <cstring>

#include "boids_sim.hpp"
#include "boids_render.hpp"

static int gW = 1000, gH = 700;

static BoidParams gParams;
static BoidsCudaSim gSim;

// CPU draw copies
static std::vector<float2h> gDrawPos;
static std::vector<float2h> gDrawVel;

static void display(){
    glClear(GL_COLOR_BUFFER_BIT);
    glColor3f(0.95f, 0.95f, 1.0f);

    if(!gDrawPos.empty()) {
        BoidsRender::drawBoidsTriangles(gDrawPos.data(), gDrawVel.data(), (int)gDrawPos.size());
    }

    glutSwapBuffers();
}

static void reshape(int w, int h){
    gW = std::max(1, w);
    gH = std::max(1, h);

    gSim.setViewport(gW, gH);
    BoidsRender::setOrtho2D(gW, gH);
}

static void keyboard(unsigned char key, int, int){
    switch(key){
        case 27: 
            std::exit(0); 
            break;
        case 'r': 
            gSim.resetRandom((unsigned)std::time(nullptr)); 
            break;
        case 'w':
            gParams.wrapEdges = !gParams.wrapEdges;
            // update sim params
            gSim.shutdown();
            gSim.init(gParams);
            gSim.resetRandom((unsigned)std::time(nullptr));
            break;
        default: 
            break;
    }
}

static void timer(int){
    static int lastMs = glutGet(GLUT_ELAPSED_TIME);
    int nowMs = glutGet(GLUT_ELAPSED_TIME);
    float frameDt = (nowMs - lastMs) / 1000.0f;
    lastMs = nowMs;
    frameDt = std::min(frameDt, 0.05f);

    static float acc = 0.0f;
    acc += frameDt;

    // If copy finished, grab latest staged data into our draw buffers
    const float2h* pPos=nullptr;
    const float2h* pVel=nullptr;
    if(gSim.pollCopyReady(pPos, pVel)) {
        std::memcpy(gDrawPos.data(), pPos, gParams.N*sizeof(float2h));
        std::memcpy(gDrawVel.data(), pVel, gParams.N*sizeof(float2h));
    }

    while(acc >= gParams.dtFixed) {
        gSim.simulateAndStageAsync(gParams.dtFixed);
        acc -= gParams.dtFixed;
    }

    glutPostRedisplay();
    glutTimerFunc(0, timer, 0);
}

int main(int argc, char** argv){
    std::srand((unsigned)std::time(nullptr));

    gParams.N = 250;
    gParams.W = gW;
    gParams.H = gH;

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(gW, gH);
    glutCreateWindow("2D Boids (CUDA sim + freeglut render)");

    glClearColor(0.06f, 0.06f, 0.08f, 1.0f);

    gSim.init(gParams);
    gSim.resetRandom((unsigned)std::time(nullptr));

    gDrawPos.resize(gParams.N);
    gDrawVel.resize(gParams.N);

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutTimerFunc(0, timer, 0);

    glutMainLoop();
    return 0;
}