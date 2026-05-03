CUDACXX   := nvcc

NVCCFLAGS := -O2 -std=c++17

LDLIBS    := -lglut -lGL -lGLU

OBJS := main.o boids_render.o boids_sim.o
TARGET := boids_cuda

all: $(TARGET)

main.o: main.cpp boids_sim.hpp boids_render.hpp
	$(CUDACXX) $(NVCCFLAGS) -c $< -o $@

boids_render.o: boids_render.cpp boids_render.hpp boids_sim.hpp
	$(CUDACXX) $(NVCCFLAGS) -c $< -o $@

boids_sim.o: boids_sim.cu boids_sim.hpp
	$(CUDACXX) $(NVCCFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(CUDACXX) $(NVCCFLAGS) $(OBJS) $(LDLIBS) -o $@

clean:
	rm -f $(TARGET) $(OBJS)