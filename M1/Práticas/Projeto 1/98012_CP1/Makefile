
.PHONY: all clean

TARGETS = fastDetectorCuda fastDetectorOpenMP testDiffs
CPPFLAGS = -O3 -Wall 

all: $(TARGETS)

fastDetectorCuda: fastDetectorCuda.cu
	nvcc -arch=sm_30 -O3 -ICommon $^ -o $@

fastDetectorOpenMP: fastDetectorOpenMP.cpp
	g++ -fopenmp $(CPPFLAGS) -ICommon $^ -o $@

testDiffs: testDiffs.cpp
	g++ $(CPPFLAGS) -ICommon $^ -o $@

clean:
	rm -f $(TARGETS) 
	rm -f referenceCuda.pgm resultCuda.pgm referenceOpenMP.pgm resultOpenMP.pgm
