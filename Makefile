CXX      := g++
CXXFLAGS := -O2 -std=c++17 -Wall -Wextra -pedantic
LDLIBS   := -lglut -lGL -lGLU
TARGET   := boids
SRC      := boids_freeglut.cpp
OBJ      := $(SRC:.cpp=.o)

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) $(OBJ)