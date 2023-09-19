
CFLAGS   = -Wall
CXXFLAGS = -Wall

.PHONY: all clean

all: sumarray

sumarray: sumarray.o
	g++ $(CFLAGS) -o $@ $^


clean:
	rm -f *.o sumarray

