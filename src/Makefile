CC= g++
COPTS= -Wall -O3

NVCC=nvcc
NVLIBS=-L/usr/local/cuda/lib64 -lcuda -lcudart
NVINCLUDES=-I/usr/local/cuda/include
NVOPTS=-O3 -arch sm_20

all: main

main: main.o utils.o sim.o
	$(CC) main.o utils.o sim.o -o main $(COPTS)

main.o: main.cpp
	$(CC) -c main.cpp $(COPTS)

utils.o: utils.cpp utils.h
	$(CC) -c utils.cpp $(COPTS)

utils_gpu.o: utils_gpu.cu utils.h
	$(NVCC) -c utils_gpu.cu $(NVINCLUDES) $(NVOPTS)

sim.o: sim.cpp sim.h
	$(CC) -c sim.cpp $(COPTS)

clean:
	rm -f *.o main
