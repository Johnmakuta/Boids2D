#include <GL/freeglut.h>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <algorithm>

struct Vec2 {
    float x=0, y=0;
    Vec2() = default;
    Vec2(float X, float Y): x(X), y(Y) {}
    Vec2 operator+(const Vec2& r) const { return {x+r.x, y+r.y}; }
    Vec2 operator-(const Vec2& r) const { return {x-r.x, y-r.y}; }
    Vec2 operator*(float s) const { return {x*s, y*s}; }
    Vec2 operator/(float s) const { return {x/s, y/s}; }
    Vec2& operator+=(const Vec2& r){ x+=r.x; y+=r.y; return *this; }
    Vec2& operator-=(const Vec2& r){ x-=r.x; y-=r.y; return *this; }
    Vec2& operator*=(float s){ x*=s; y*=s; return *this; }
};

static inline float dot(const Vec2& a, const Vec2& b){ return a.x*b.x + a.y*b.y; }
static inline float len2(const Vec2& v){ return dot(v,v); }
static inline float len(const Vec2& v){ return std::sqrt(len2(v)); }

static inline Vec2 normalized(const Vec2& v){
    float l = len(v);
    if(l <= 1e-6f) return {0,0};
    return v / l;
}

static inline Vec2 limitMag(const Vec2& v, float maxMag){
    float l2 = len2(v);
    if(l2 <= maxMag*maxMag) return v;
    float l = std::sqrt(l2);
    return v * (maxMag / (l + 1e-6f));
}

static inline float frand(float a, float b){
    return a + (b-a) * (float(std::rand())/float(RAND_MAX));
}

struct Boid {
    Vec2 pos;
    Vec2 vel;
};

static int gW = 1000, gH = 700;
static std::vector<Boid> boids;

static int   N             = 250;
static float maxSpeed      = 160.0f;   // units per second
static float maxForce      = 220.0f;   // steering accel cap
static float neighborRad   = 70.0f;
static float sepRad        = 26.0f;

static float wAlign        = 1.0f;
static float wCohesion     = 0.85f;
static float wSeparation   = 1.55f;

static float dtFixed = 1.0f/120.0f; // fixed simulation step

static bool wrapEdges = true;

static void resetBoids(){
    boids.clear();
    boids.reserve(N);
    for(int i=0;i<N;i++){
        Boid b;
        b.pos = { frand(0.0f, (float)gW), frand(0.0f, (float)gH) };
        float a = frand(0.0f, 6.2831853f);
        float s = frand(40.0f, maxSpeed);
        b.vel = { std::cos(a)*s, std::sin(a)*s };
        boids.push_back(b);
    }
}

static Vec2 steerTowards(const Vec2& desiredVel, const Vec2& currentVel){
    Vec2 steer = desiredVel - currentVel;
    return limitMag(steer, maxForce);
}

static void stepSim(float dt){
    // Copy positions/velocities for synchronous update
    std::vector<Boid> prev = boids;

    for(int i=0;i<(int)boids.size();i++){
        const Boid& me = prev[i];

        Vec2 sumVel(0,0);
        Vec2 sumPos(0,0);
        Vec2 sep(0,0);
        int  cnt = 0;
        int  cntSep = 0;

        float nR2 = neighborRad*neighborRad;
        float sR2 = sepRad*sepRad;

        for(int j=0;j<(int)prev.size();j++){
            if(i==j) continue;
            Vec2 d = prev[j].pos - me.pos;

            // If wrapping, choose shortest toroidal delta
            if(wrapEdges){
                if(d.x >  gW*0.5f) d.x -= gW;
                if(d.x < -gW*0.5f) d.x += gW;
                if(d.y >  gH*0.5f) d.y -= gH;
                if(d.y < -gH*0.5f) d.y += gH;
            }

            float dist2 = len2(d);
            if(dist2 < nR2){
                sumVel += prev[j].vel;
                sumPos += (me.pos + d); // neighbor position in same "image"
                cnt++;
                if(dist2 < sR2 && dist2 > 1e-6f){
                    // push away inversely proportional to distance
                    sep += (d * (-1.0f)) / (std::sqrt(dist2));
                    cntSep++;
                }
            }
        }

        Vec2 acc(0,0);

        if(cnt > 0){
            // Alignment: match average heading
            Vec2 avgVel = sumVel / (float)cnt;
            Vec2 desired = normalized(avgVel) * maxSpeed;
            acc += steerTowards(desired, me.vel) * wAlign;

            // Cohesion: steer to center of mass
            Vec2 center = sumPos / (float)cnt;
            Vec2 toCenter = center - me.pos;
            Vec2 desiredC = normalized(toCenter) * maxSpeed;
            acc += steerTowards(desiredC, me.vel) * wCohesion;
        }

        if(cntSep > 0){
            Vec2 desiredS = normalized(sep) * maxSpeed;
            acc += steerTowards(desiredS, me.vel) * wSeparation;
        }

        // Integrate
        Boid& b = boids[i];
        b.vel = limitMag(me.vel + acc * dt, maxSpeed);
        b.pos = me.pos + b.vel * dt;

        // Borders
        if(wrapEdges){
            if(b.pos.x < 0) b.pos.x += gW;
            if(b.pos.x >= gW) b.pos.x -= gW;
            if(b.pos.y < 0) b.pos.y += gH;
            if(b.pos.y >= gH) b.pos.y -= gH;
        } else {
            // bounce
            if(b.pos.x < 0){ b.pos.x = 0; b.vel.x *= -1; }
            if(b.pos.x > gW){ b.pos.x = (float)gW; b.vel.x *= -1; }
            if(b.pos.y < 0){ b.pos.y = 0; b.vel.y *= -1; }
            if(b.pos.y > gH){ b.pos.y = (float)gH; b.vel.y *= -1; }
        }
    }
}

static void drawBoid(const Boid& b){
    Vec2 dir = normalized(b.vel);
    if(len2(dir) < 1e-6f) dir = {1,0};

    // Triangle in direction of velocity
    float size = 7.5f;
    Vec2 forward = dir * size;
    Vec2 right   = Vec2(-dir.y, dir.x) * (size * 0.55f);

    Vec2 p0 = b.pos + forward;
    Vec2 p1 = b.pos - forward*0.65f + right;
    Vec2 p2 = b.pos - forward*0.65f - right;

    glBegin(GL_TRIANGLES);
    glVertex2f(p0.x, p0.y);
    glVertex2f(p1.x, p1.y);
    glVertex2f(p2.x, p2.y);
    glEnd();
}

static void display(){
    glClear(GL_COLOR_BUFFER_BIT);

    glColor3f(0.95f, 0.95f, 1.0f);
    for(const auto& b : boids) drawBoid(b);

    glutSwapBuffers();
}

static void reshape(int w, int h){
    gW = std::max(1, w);
    gH = std::max(1, h);

    glViewport(0,0,gW,gH);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    // pixel-like coordinates: (0,0) bottom-left, (w,h) top-right
    gluOrtho2D(0, gW, 0, gH);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

static void keyboard(unsigned char key, int, int){
    switch(key){
        case 27: std::exit(0); break; // ESC
        case 'r': resetBoids(); break;
        case 'w': wrapEdges = !wrapEdges; break;
        case '+': N = std::min(2000, N+25); resetBoids(); break;
        case '-': N = std::max(25,   N-25); resetBoids(); break;
        default: break;
    }
}

static void timer(int){
    // fixed-step accumulator
    static int lastMs = glutGet(GLUT_ELAPSED_TIME);
    int nowMs = glutGet(GLUT_ELAPSED_TIME);
    float frameDt = (nowMs - lastMs) / 1000.0f;
    lastMs = nowMs;

    // avoid spiral of death on stalls
    frameDt = std::min(frameDt, 0.05f);

    static float acc = 0.0f;
    acc += frameDt;
    while(acc >= dtFixed){
        stepSim(dtFixed);
        acc -= dtFixed;
    }

    glutPostRedisplay();
    glutTimerFunc(0, timer, 0);
}

int main(int argc, char** argv){
    std::srand((unsigned)std::time(nullptr));

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(gW, gH);
    glutCreateWindow("2D Boids (freeglut)");

    glClearColor(0.06f, 0.06f, 0.08f, 1.0f);

    resetBoids();

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutTimerFunc(0, timer, 0);

    glutMainLoop();
    return 0;
}