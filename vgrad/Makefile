CXX = clang++
CXXFLAGS = -std=c++23 -O2 -Xclang -fopenmp
LDFLAGS = -lomp
INCLUDES = -Iinclude -I../typehint/include -I/opt/homebrew/opt/libomp/include
LIBS = -L/opt/homebrew/opt/libomp/lib

mnist:
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIBS) mnist.cpp -o mnist.out $(LDFLAGS)

profile:
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIBS) profile.cpp -o profile.out $(LDFLAGS)

regression:
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIBS) regression.cpp -o regression.out $(LDFLAGS)

cx_test:
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIBS) cx_test.cpp -o cx_test.out $(LDFLAGS)

.PHONY: clean

clean:
	rm -f *.out