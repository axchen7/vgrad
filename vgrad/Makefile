CXX = clang++
CXXFLAGS = -std=c++23 -O3 -Xclang -fopenmp
LDFLAGS = -lomp
INCLUDES = -I/opt/homebrew/opt/libomp/include
LIBS = -L/opt/homebrew/opt/libomp/lib

mnist:
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIBS) mnist.cpp -o mnist.out $(LDFLAGS)

.PHONY: clean

clean:
	rm -f *.out