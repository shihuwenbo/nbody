CC= g++
COPTS= -Wall -O3

all: main

main: main.o utils.o sim.o
	$(CC) main.o utils.o sim.o -o main $(COPTS)\
	    -lsfml-graphics -lsfml-window -lsfml-system

main.o: main.cpp
	$(CC) -c main.cpp $(COPTS)

utils.o: utils.cpp utils.h
	$(CC) -c utils.cpp $(COPTS)

sim.o: sim.cpp sim.h
	$(CC) -c sim.cpp $(COPTS)

clean:
	rm -f *.o main

run:
	./main
