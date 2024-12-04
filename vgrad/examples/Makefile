CXX = clang++

ifeq ($(OS),Windows_NT)
	CXXFLAGS = -std=c++23 -O2 -fopenmp=libgomp
	LDFLAGS =
	EXT = .exe
else
	CXXFLAGS = -std=c++23 -O2 -Xclang -fopenmp
	LDFLAGS = -lomp
	EXT = .out
endif

INCLUDES = -I../include -I../../typehint/include -I/opt/homebrew/opt/libomp/include
LIBS = -L/opt/homebrew/opt/libomp/lib

all: mnist regression

mnist:
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIBS) mnist.cpp -o mnist$(EXT) $(LDFLAGS)

regression:
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIBS) regression.cpp -o regression$(EXT) $(LDFLAGS)

.PHONY: clean

clean:
	rm -f *$(EXT)