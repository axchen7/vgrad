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

all: measure_unary measure_matmul measure_matmul_batch measure_transpose

measure_unary:
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIBS) measure_unary.cpp -o measure_unary$(EXT) $(LDFLAGS)

measure_matmul:
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIBS) measure_matmul.cpp -o measure_matmul$(EXT) $(LDFLAGS)

measure_matmul_batch:
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIBS) measure_matmul_batch.cpp -o measure_matmul_batch$(EXT) $(LDFLAGS)

measure_transpose:
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIBS) measure_transpose.cpp -o measure_transpose$(EXT) $(LDFLAGS)

.PHONY: clean

clean:
	rm -f *$(EXT)