
TARGETS = arradd1 arradd2 arradd3 arradd4 arradd5

.PHONY: all clean

all: $(TARGETS)


arradd1: arradd1.cu
	nvcc -arch=sm_30 arradd1.cu -o arradd1

arradd2: arradd2.cu
	nvcc -arch=sm_30 arradd2.cu -o arradd2

arradd3: arradd3.cu
	nvcc -arch=sm_30 arradd3.cu -o arradd3

arradd4: arradd4.cu
	nvcc -arch=sm_30 arradd4.cu -o arradd4

arradd5: arradd5.cu
	nvcc -arch=sm_30 arradd5.cu -o arradd5

clean:
	rm $(TARGETS)
