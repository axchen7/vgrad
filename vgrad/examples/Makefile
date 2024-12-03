CXX = clang++
CXXFLAGS = -std=c++23 -O2 -Xclang -fopenmp
LDFLAGS = -lomp
INCLUDES = -I../include -I../../typehint/include -I/opt/homebrew/opt/libomp/include
LIBS = -L/opt/homebrew/opt/libomp/lib

mnist:
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIBS) mnist.cpp -o mnist.out $(LDFLAGS)

regression:
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIBS) regression.cpp -o regression.out $(LDFLAGS)

.PHONY: clean

clean:
	rm -f *.out